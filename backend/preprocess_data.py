import torch
import torch.nn.functional as F
import torch_geometric.data as GData
import pandapower as pp
import pandapower.networks as nw
import numpy as np
import h5py
import os
from glob import glob
from tqdm import tqdm
import time

# --- Configuration ---
HPC_DATA_DIR = "hpc_data"
OUTPUT_DIR = "processed_data"
OUTPUT_GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

def get_base_grid_data():
    """
    Loads the base PEGASE 9241 grid *once* to get topology (edge_index)
    and line parameters (edge_attr).
    """
    print("Loading base grid (case9241pegase) for topology...")
    try:
        net = nw.case9241pegase()
    except Exception as e:
        print(f"FATAL: Could not load case9241pegase. Error: {e}")
        raise e

    # --- 1. Get Node -> Integer Mapping ---
    bus_map = {bus_id: i for i, bus_id in enumerate(net.bus.index.values)}
    num_nodes = len(bus_map)

    # --- 2. Get Edge Index (Grid Topology) ---
    from_buses_pp = net.line.from_bus.values
    to_buses_pp = net.line.to_bus.values

    from_buses = np.vectorize(bus_map.get)(from_buses_pp)
    to_buses = np.vectorize(bus_map.get)(to_buses_pp)

    edge_index = torch.tensor(np.array([from_buses, to_buses]), dtype=torch.long)

    # GNNs need undirected graphs: add reverse edges
    edge_index_rev = torch.tensor(np.array([to_buses, from_buses]), dtype=torch.long)
    edge_index_full = torch.cat([edge_index, edge_index_rev], dim=1)

    # --- 3. Get Edge Features (Line Parameters) ---
    r_pu = torch.tensor(net.line.r_ohm_per_km.values, dtype=torch.float)
    x_pu = torch.tensor(net.line.x_ohm_per_km.values, dtype=torch.float)
    c_nf = torch.tensor(net.line.c_nf_per_km.values, dtype=torch.float)
    g_us = torch.tensor(net.line.g_us_per_km.values, dtype=torch.float)

    edge_attr = torch.stack([r_pu, x_pu, c_nf, g_us], dim=1)
    edge_attr_full = torch.cat([edge_attr, edge_attr], dim=0)

    num_physical_lines = len(net.line)

    print(f"Base grid loaded. Nodes: {len(bus_map)}, Edges: {edge_index_full.shape[1]}")
    # Return num_physical_lines to help map contingency IDs to edge indices
    return edge_index_full, edge_attr_full, net.bus.index.values, num_physical_lines

def process_all_simulations():
    """
    Loops through all HDF5 files, extracts data, and builds
    torch_geometric.data.Data objects.
    """
    print("--- Starting Data Preprocessing (WITH BILLIONAIRE FIX) ---")

    # 1. Get base grid topology
    edge_index, edge_attr, pp_bus_indices, num_physical_lines = get_base_grid_data()

    # 2. Get static node features (bus types)
    net = nw.case9241pegase()

    bus_type_mapping = {'PQ': 0, 'PV': 1, 'Slack': 2, 1: 0, 2: 1, 3: 2}
    bus_types_mapped = net.bus.type.map(bus_type_mapping).fillna(0)

    bus_type = torch.tensor(bus_types_mapped.values, dtype=torch.long)
    bus_type_onehot = F.one_hot(bus_type, num_classes=3).float()

    # 3. Find all simulation files
    file_names = glob(os.path.join(HPC_DATA_DIR, "sample_*.h5"))
    if not file_names:
        print(f"FATAL: No 'sample_*.h5' files found in {HPC_DATA_DIR}.")
        return

    print(f"Found {len(file_names)} simulation files to process.")

    os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)
    print(f"Saving individual graphs to: {OUTPUT_GRAPH_DIR}")

    # Create reverse mapping
    bus_map_reverse = {bus_id: i for i, bus_id in enumerate(pp_bus_indices)}

    processed_count = 0
    converged_count = 0
    diverged_count = 0

    for i, f_name in enumerate(tqdm(file_names, desc="Processing files", mininterval=2.0, ncols=100)):
        try:
            with h5py.File(f_name, 'r') as f:

                # 4. Get Node Features (X)
                num_buses = len(pp_bus_indices)
                p_load = np.zeros(num_buses)
                q_load = np.zeros(num_buses)
                p_gen = np.zeros(num_buses)

                # Map load values
                load_bus_indices = net.load.bus.values
                load_bus_mapped = np.vectorize(bus_map_reverse.get)(load_bus_indices)
                p_load[load_bus_mapped] = f['inputs']['load_p_mw'][:]
                q_load[load_bus_mapped] = f['inputs']['load_q_mvar'][:]

                # Map gen values
                gen_bus_indices = net.gen.bus.values
                gen_bus_mapped = np.vectorize(bus_map_reverse.get)(gen_bus_indices)
                p_gen[gen_bus_mapped] = f['inputs']['gen_p_mw'][:]

                # Get pre-contingency state
                v_pre_mag = f['inputs']['pre_bus_vm_pu'][:]
                v_pre_ang = f['inputs']['pre_bus_va_degree'][:]

                # Stack all node features: [num_nodes, 8]
                x = torch.tensor(np.stack([
                    p_load,
                    q_load,
                    p_gen,
                    v_pre_mag,
                    v_pre_ang
                ], axis=1), dtype=torch.float)

                # Add the static one-hot bus types
                x = torch.cat([x, bus_type_onehot], dim=1)

                # 5. Get Target (Y_classify)
                y_class = torch.tensor([
                    f['outputs'].attrs['is_voltage_violation'],
                    f['outputs'].attrs['is_thermal_violation']
                ], dtype=torch.float)

                # 6. Get Target (Y_estimate)
                y_est_v_mag = f['outputs']['bus_vm_pu'][:]
                y_est_v_ang = f['outputs']['bus_va_degree'][:]

                y_estimate = torch.tensor(np.stack([
                    y_est_v_mag,
                    y_est_v_ang
                ], axis=1), dtype=torch.float)

                # 7. Get Contingency Info & BUILD MASK
                contingency_id = f['inputs'].attrs['contingency_line_id']

                # --- THE BILLIONAIRE FIX (Dynamic Edge Masking) ---
                # Initialize mask as all 1s (Active)
                edge_mask = torch.ones(edge_index.shape[1], dtype=torch.float)

                if contingency_id != -1:
                    # In get_base_grid_data, we stacked [forward_edges, reverse_edges]
                    # Forward edges indices: 0 to num_physical_lines - 1
                    # Reverse edges indices: num_physical_lines to 2*num_physical_lines - 1
                    
                    # The contingency_id corresponds to the index in net.line
                    forward_idx = int(contingency_id)
                    reverse_idx = int(contingency_id) + num_physical_lines
                    
                    # Safety check
                    if reverse_idx < edge_index.shape[1]:
                        edge_mask[forward_idx] = 0.0
                        edge_mask[reverse_idx] = 0.0
                
                # Check convergence
                converged = bool(f['outputs'].attrs['converged'])

                if converged:
                    converged_count += 1
                else:
                    diverged_count += 1
                    # print(f"\n🔴 Found DIVERGED sample: task_id {i+1} (file {i:06d})")

                # Create the Data object
                data = GData.Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    edge_mask=edge_mask, # <--- NEW: Dynamic Mask
                    y_classify=y_class.unsqueeze(0),
                    y_estimate=y_estimate,
                    converged=torch.tensor([converged], dtype=torch.bool)
                )

                # Handle non-converged cases (store NaNs)
                if not converged:
                    data.y_estimate.fill_(float('nan'))

                # Save the single graph file
                save_path = os.path.join(OUTPUT_GRAPH_DIR, f"graph_{i+1:06d}.pt")
                torch.save(data, save_path)
                processed_count += 1

        except Exception as e:
            print(f"\nFailed to process file {f_name}. Error: {e}")
            print("FATAL: Stopping preprocessing due to file error.")
            raise e

    if processed_count == 0:
        print("FATAL: No data was successfully processed.")
        return

    print(f"\nSuccessfully processed and saved {processed_count} files to {OUTPUT_GRAPH_DIR}.")
    print(f"\n📊 CONVERGENCE STATISTICS:")
    print(f"  ✅ Converged samples: {converged_count} ({converged_count/processed_count*100:.2f}%)")
    print(f"  🔴 Diverged samples:  {diverged_count} ({diverged_count/processed_count*100:.2f}%)")
    print("--- Preprocessing Complete ---")

if __name__ == "__main__":
    start_time = time.time()
    process_all_simulations()
    end_time = time.time()
    print(f"Total processing time: {(end_time - start_time) / 60.0:.2f} minutes.")
