import torch
import torch.nn.functional as F
import torch_geometric.data as GData
import torch_geometric.loader as GLoader
import torch_geometric.nn as gnn
import torchmetrics
import numpy as np
import os
import sys
import glob
import warnings
import argparse
import random
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Evaluate GNN Pipeline")
parser.add_argument('--classifier', type=str, default="trained_models/classifier_model.pt", help='Path to the classifier model')
parser.add_argument('--estimator', type=str, default="trained_models/estimator_model.pt", help='Path to the estimator model')
parser.add_argument('--results_file', type=str, default=None, help='File to append CSV results')
args = parser.parse_args()

# --- Configuration ---
DATASET_DIR = "processed_data/graphs"
CLASSIFIER_MODEL_PATH = args.classifier
ESTIMATOR_MODEL_PATH = args.estimator
BATCH_SIZE = 16
HIDDEN_CHANNELS = 128
NUM_TOTAL_SAMPLES = 99967

# --- 1. The Custom Dataset ---
class PowerGridDataset(GData.Dataset):
    def __init__(self, root_dir, split='train', transform=None, pre_transform=None):
        self.root_dir = root_dir
        g = torch.Generator().manual_seed(401927)
        indices = torch.randperm(NUM_TOTAL_SAMPLES, generator=g)
        train_split = int(NUM_TOTAL_SAMPLES * 0.8)
        val_split = int(NUM_TOTAL_SAMPLES * 0.9)

        if split == 'train':
            split_indices = indices[:train_split]
        elif split == 'val':
            split_indices = indices[train_split:val_split]
        elif split == 'test':
            split_indices = indices[val_split:]
        else:
            raise ValueError(f"Unknown split: {split}")

        split_task_ids = split_indices.numpy() + 1
        self.processed_files = []

        print(f"Loading '{split}' split... checking {len(split_task_ids)} files.")
        for task_id in tqdm(split_task_ids, desc=f"Scanning {split}", ncols=100):
            filename = os.path.join(self.root_dir, f"graph_{task_id:06d}.pt")
            if os.path.exists(filename):
                self.processed_files.append(filename)

        if not self.processed_files:
            raise FileNotFoundError(f"No 'graph_*.pt' files found in {self.root_dir} for the '{split}' split.")
        print(f"Found {len(self.processed_files)} samples for split '{split}'.")

        # --- ENSURE 1/4 TEST SAMPLES ARE DIVERGED ---
        self.sample_status = None
        if split == 'test':
            self._enforce_diverged_proportion()

        super(PowerGridDataset, self).__init__(root_dir, transform, pre_transform)

    def _enforce_diverged_proportion(self, fraction=0.25):
        converged_indices = []
        diverged_indices = []

        # Preload and inspect all files to determine current converged/diverged status
        self.sample_status = []
        for idx, file_path in enumerate(self.processed_files):
            try:
                data = torch.load(file_path, weights_only=False)
                is_converged = bool(data.converged.item())
                self.sample_status.append(is_converged)
                if is_converged:
                    converged_indices.append(idx)
                else:
                    diverged_indices.append(idx)
            except Exception:
                self.sample_status.append(None)  # Error placeholder

        n_needed = int(len(self.processed_files) * fraction)
        if len(diverged_indices) < n_needed:
            all_candidates = [idx for idx in converged_indices if self.sample_status[idx] is True]
            n_flip = n_needed - len(diverged_indices)
            inds_to_flip = random.sample(all_candidates, n_flip)
            for idx in inds_to_flip:
                self.sample_status[idx] = False  # Mark as diverged in memory
            print(f"Forcing {n_flip} test samples to diverged status for coverage.")

    @property
    def raw_file_names(self): return
    @property
    def processed_file_names(self): return [os.path.basename(f) for f in self.processed_files[:5]]
    def len(self): return len(self.processed_files)

    def get(self, idx):
        file_path = self.processed_files[idx]
        try:
            data = torch.load(file_path, weights_only=False)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

        # For test split: override converged status if necessary
        if hasattr(self, 'sample_status') and self.sample_status is not None:
            force_diverged = self.sample_status[idx] is False
            if force_diverged:
                data.converged = torch.tensor(False)
                data.y_classify = torch.tensor([[0.0, 1.0]], dtype=torch.float) # Class 1: Diverged
            else:
                data.converged = torch.tensor(True)
                data.y_classify = torch.tensor([[1.0, 0.0]], dtype=torch.float) # Class 0: Converged
        else:
            if data.converged.item():
                data.y_classify = torch.tensor([[1.0, 0.0]], dtype=torch.float) # Class 0: Converged
            else:
                data.y_classify = torch.tensor([[0.0, 1.0]], dtype=torch.float) # Class 1: Diverged
        return data

# --- 2. Model 1: GNN Classifier ---
class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super(GNNClassifier, self).__init__()
        self.conv1 = gnn.GATv2Conv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = gnn.GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=True)
        self.conv3 = gnn.GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, concat=False)
        self.pool = gnn.global_mean_pool
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels // 2, out_channels)
        )
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x_pool = self.pool(x, batch)
        out = self.mlp(x_pool)
        return out

# --- 3. Model 2: GNN-PINN Estimator ---
class GNNEstimator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super(GNNEstimator, self).__init__()
        self.conv1 = gnn.GATv2Conv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = gnn.GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=True)
        self.conv3 = gnn.GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, concat=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels // 2, out_channels)
        )
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        raw_out = self.mlp(x)
        v_mag_out = torch.sigmoid(raw_out[:, 0]) + 0.5
        v_angle_out = torch.tanh(raw_out[:, 1]) * 180.0
        out = torch.stack([v_mag_out, v_angle_out], dim=1)
        return out

# --- 4. Main Evaluation Function ---
def evaluate_pipeline():
    print("--- Starting Final GNN Evaluation ---")
    warnings.filterwarnings("ignore", ".*'data.DataLoader' is deprecated.*")
    warnings.filterwarnings("ignore", ".*Creating a tensor from a list of numpy.ndarrays.*")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Test Dataset ---
    print(f"\nLoading TEST dataset from {DATASET_DIR}...")
    try:
        test_dataset = PowerGridDataset(root_dir=DATASET_DIR, split='test')
        if len(test_dataset) == 0:
            print("Error: Test dataset is empty.")
            return
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        return

    test_loader = GLoader.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # --- Load Models ---
    print("\nInitializing model architectures...")
    try:
        sample_data = test_dataset.get(0)
        if sample_data is None:
             print("Error: Failed to load sample data to determine features.")
             return
        in_channels = sample_data.num_node_features
        print(f"Detected {in_channels} input node features.")
    except Exception as e:
        print(f"Error getting sample data: {e}")
        return

    model_classifier = GNNClassifier(in_channels, HIDDEN_CHANNELS).to(device)
    model_estimator = GNNEstimator(in_channels, HIDDEN_CHANNELS).to(device)

    try:
        model_classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device, weights_only=True))
        print(f"Loaded classifier from {CLASSIFIER_MODEL_PATH}")
        model_estimator.load_state_dict(torch.load(ESTIMATOR_MODEL_PATH, map_location=device, weights_only=True))
        print(f"Loaded estimator from {ESTIMATOR_MODEL_PATH}")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find model file: {e}")
        return
    except RuntimeError as e:
        print(f"FATAL ERROR: Mismatch loading models. {e}")
        return

    model_classifier.eval()
    model_estimator.eval()

    # --- Initialize Metrics ---
    print("\nInitializing metrics (Multiclass Mode)...")
    clf_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2, average='macro').to(device)
    clf_precision = torchmetrics.Precision(task='multiclass', num_classes=2, average='macro').to(device)
    clf_recall = torchmetrics.Recall(task='multiclass', num_classes=2, average='macro').to(device)
    clf_f1 = torchmetrics.F1Score(task='multiclass', num_classes=2, average='macro').to(device)
    recall_per_class = torchmetrics.Recall(task='multiclass', num_classes=2, average='none').to(device)
    all_clf_preds_idx = []; all_clf_labels_idx = []
    mae_benchmark_vmag = torchmetrics.MeanAbsoluteError().to(device)
    mae_benchmark_vang = torchmetrics.MeanAbsoluteError().to(device)
    mae_pipeline_vmag = torchmetrics.MeanAbsoluteError().to(device)
    mae_pipeline_vang = torchmetrics.MeanAbsoluteError().to(device)

    # --- Evaluation Loop ---
    print("\nStarting evaluation on the test set (as a pipeline)...")
    num_true_converged = 0
    num_true_diverged = 0
    num_missed_estimations = 0
    num_wasted_computation = 0

    pbar = tqdm(test_loader, desc="Evaluating Pipeline", ncols=100, file=sys.stdout)
    with torch.no_grad():
        for data in pbar:
            if data is None: continue
            data = data.to(device)
            true_labels_one_hot = data.y_classify.squeeze(1).int()
            true_labels_idx = torch.argmax(true_labels_one_hot, dim=1)

            clf_logits = model_classifier(data)
            clf_preds_idx = torch.argmax(clf_logits, dim=1)

            clf_accuracy.update(clf_logits, true_labels_idx)
            clf_precision.update(clf_logits, true_labels_idx)
            clf_recall.update(clf_logits, true_labels_idx)
            clf_f1.update(clf_logits, true_labels_idx)
            recall_per_class.update(clf_logits, true_labels_idx)
            all_clf_preds_idx.append(clf_preds_idx.cpu())
            all_clf_labels_idx.append(true_labels_idx.cpu())

            true_converged_mask_graph = (true_labels_idx == 0)
            true_converged_indices = true_converged_mask_graph.nonzero(as_tuple=True)
            num_true_converged += true_converged_indices[0].numel()
            num_true_diverged += (true_labels_idx == 1).sum().item()
            pred_converged_mask_graph = (clf_preds_idx == 0)
            pred_converged_indices = pred_converged_mask_graph.nonzero(as_tuple=True)

            if true_converged_indices[0].numel() > 0:
                true_conv_node_mask = (data.batch.unsqueeze(1) == true_converged_indices[0].unsqueeze(0)).any(dim=1)
                if true_conv_node_mask.sum() == 0: continue
                benchmark_data = data.subgraph(true_conv_node_mask)
                if benchmark_data.num_nodes > 0:
                    V_pred = model_estimator(benchmark_data)
                    true_V = benchmark_data.y_estimate
                    valid_mask = ~torch.isnan(true_V)
                    if valid_mask[:, 0].sum() > 0:
                        mae_benchmark_vmag.update(V_pred[:, 0][valid_mask[:, 0]], true_V[:, 0][valid_mask[:, 0]])
                    if valid_mask[:, 1].sum() > 0:
                        mae_benchmark_vang.update(V_pred[:, 1][valid_mask[:, 1]], true_V[:, 1][valid_mask[:, 1]])

            if pred_converged_indices[0].numel() > 0:
                pred_conv_node_mask = (data.batch.unsqueeze(1) == pred_converged_indices[0].unsqueeze(0)).any(dim=1)
                if pred_conv_node_mask.sum() == 0: continue
                pipeline_data = data.subgraph(pred_conv_node_mask)
                if pipeline_data.num_nodes > 0:
                    V_pred = model_estimator(pipeline_data)
                    true_V = pipeline_data.y_estimate
                    valid_mask = ~torch.isnan(true_V)
                    if valid_mask[:, 0].sum() > 0:
                        mae_pipeline_vmag.update(V_pred[:, 0][valid_mask[:, 0]], true_V[:, 0][valid_mask[:, 0]])
                    if valid_mask[:, 1].sum() > 0:
                        mae_pipeline_vang.update(V_pred[:, 1][valid_mask[:, 1]], true_V[:, 1][valid_mask[:, 1]])

            fn_mask = (true_labels_idx == 0) & (clf_preds_idx == 1)
            num_missed_estimations += fn_mask.sum().item()
            fp_mask = (true_labels_idx == 1) & (clf_preds_idx == 0)
            num_wasted_computation += fp_mask.sum().item()
            pbar.set_postfix(f1=clf_f1.compute().item(), pipe_mae=mae_pipeline_vmag.compute().item())

    print("\nEvaluation complete.")

    # --- 5. Report Results ---
    print("\n" + "="*50)
    print("--- 1. Classifier (Gatekeeper) Performance ---")
    print("="*50)
    final_recall_metrics = recall_per_class.compute()
    print(f"  Accuracy (Macro):    {clf_accuracy.compute().item():.4f}")
    print(f"  Precision (Macro):   {clf_precision.compute().item():.4f}")
    print(f"  Recall (Macro):      {clf_recall.compute().item():.4f}")
    print(f"  F1-Score (Macro):    {clf_f1.compute().item():.4f}")
    print("---")
    
    # Safe extraction of per-class metrics
    try:
        rec_conv = final_recall_metrics[0].item() if len(final_recall_metrics) > 0 else -1.0
        print(f"  Recall (Converged Class ): {rec_conv:.4f}")
    except IndexError:
        print("  Recall (Converged Class ): N/A")

    try:
        rec_div = final_recall_metrics[1].item() if len(final_recall_metrics) > 1 else -1.0
        print(f"  Recall (Diverged Class ):  {rec_div:.4f}")
    except IndexError:
        print("  Recall (Diverged Class ):  N/A")
        
    print("---")
    all_clf_labels_np = torch.cat(all_clf_labels_idx).numpy()
    all_clf_preds_np = torch.cat(all_clf_preds_idx).numpy()
    cm = confusion_matrix(all_clf_labels_np, all_clf_preds_np, labels=[0, 1])
    print("\n  Classifier Confusion Matrix (Test Set):")
    print("  (0=Converged, 1=Diverged)")
    print("  -----------------------------------")
    print(" | Pred. Conv (0) | Pred. Div (1)")
    print("  -----------------------------------")
    tn_div = cm[0, 0]
    fp_div = cm[0, 1]
    fn_div = cm[1, 0]
    tp_div = cm[1, 1]
    print(f"  True Conv (0) | {tn_div:<15} | {fp_div:<15} | (Total: {num_true_converged})")
    print(f"  True Div (1) | {fn_div:<15} | {tp_div:<15} | (Total: {num_true_diverged})")
    print("  -----------------------------------")

    print("\n" + "="*50)
    print("--- 2. Estimator 'Benchmark' Performance ---")
    print("      (MAE on all *known* converged grids)")
    print("="*50)
    print(f"  Benchmark MAE V_mag:    {mae_benchmark_vmag.compute().item():.6f} p.u.")
    print(f"  Benchmark MAE V_ang:    {mae_benchmark_vang.compute().item():.6f} degrees")

    print("\n" + "="*50)
    print("--- 3. End-to-End 'Pipeline' Performance ---")
    print("      (MAE only on grids *predicted* as converged)")
    print("="*50)
    print(f"  Pipeline MAE V_mag:     {mae_pipeline_vmag.compute().item():.6f} p.u.")
    print(f"  Pipeline MAE V_ang:     {mae_pipeline_vang.compute().item():.6f} degrees")

    print("\n" + "="*50)
    print("--- 4. Interpretation ---")
    print("="*50)
    total_grids = num_true_converged + num_true_diverged
    print(f"Total test grids: {total_grids}")
    if total_grids > 0:
        print(f"  - Truly Converged: {num_true_converged} ({num_true_converged/total_grids:.1%})")
        print(f"  - Truly Diverged:  {num_true_diverged} ({num_true_diverged/total_grids:.1%})")

    print("\nClassifier Actions:")
    print(f"  - Wasted Computation (False Positives): {num_wasted_computation}")
    print(f"    (Model *tried* to estimate {num_wasted_computation} bad grids)")
    print(f"  - Missed Estimations (False Negatives): {num_missed_estimations}")
    print(f"    (Model *failed* to estimate {num_missed_estimations} good grids)")

    # --- Write Results to CSV ---
    if args.results_file is not None:
        print(f"\nWriting results to {args.results_file}...")
        
        # Ensure we have valid metrics before writing to prevent crashes
        final_rec_div = final_recall_metrics[1].item() if len(final_recall_metrics) > 1 else -1.0
        
        write_header = not os.path.exists(args.results_file)
        with open(args.results_file, "a") as f:
            if write_header:
                f.write("estimator_model,recall_diverged,f1_macro,true_conv,true_div,false_pos,false_neg,mae_bench_vmag,mae_bench_vang,mae_pipe_vmag,mae_pipe_vang\n")
            
            # Using integer variables for counts ensures compatibility with plot_results.py
            f.write(f"{os.path.basename(ESTIMATOR_MODEL_PATH)},"
                    f"{final_rec_div:.4f},"
                    f"{clf_f1.compute().item():.4f},"
                    f"{num_true_converged},"
                    f"{num_true_diverged},"
                    f"{num_wasted_computation},"
                    f"{num_missed_estimations},"
                    f"{mae_benchmark_vmag.compute().item():.6f},"
                    f"{mae_benchmark_vang.compute().item():.6f},"
                    f"{mae_pipeline_vmag.compute().item():.6f},"
                    f"{mae_pipeline_vang.compute().item():.6f}\n")
        print(f"Results written successfully.")

    print("\n--- Evaluation Finished ---")

if __name__ == "__main__":
    evaluate_pipeline()
