// app.js - Main application logic and event handlers

// Application state
const AppState = {
        currentSystem: 'pegase',
    gridData: null,
    contingencies: [],
    currentResults: null,
    apiConnected: false
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    loadContingencies();
    updateSystemInfo();
    
    // Check API connection
    checkAPIConnection();
    
    Visualization.updateStatus({
        gridState: 'No Data',
        contingencies: 100000,
        inference: '0.4ms/step',
        api: 'Checking...'
    });
}

function setupEventListeners() {
    // File upload
    const fileInput = document.getElementById('fileUpload');
    fileInput.addEventListener('change', handleFileUpload);
    
    // Drag and drop
    const fileUploadDiv = document.querySelector('.file-upload');
    fileUploadDiv.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUploadDiv.style.borderColor = 'var(--accent-teal)';
    });
    
    fileUploadDiv.addEventListener('dragleave', () => {
        fileUploadDiv.style.borderColor = 'var(--border-color)';
    });
    
    fileUploadDiv.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadDiv.style.borderColor = 'var(--border-color)';
        if (e.dataTransfer.files.length) {
            handleFileUpload({ target: { files: e.dataTransfer.files } });
        }
    });
    
    // Analysis buttons
    document.getElementById('analyzeSingle').addEventListener('click', analyzeSingleContingency);
    document.getElementById('analyzeScreening').addEventListener('click', runBulkScreening);
    document.getElementById('exportResults').addEventListener('click', exportResults);

        // System selector
        document.getElementById('systemSelect').addEventListener('change', handleSystemChange);
    
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => switchTab(button.dataset.tab));
    });
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const content = e.target.result;
            let gridData;
            
            if (file.name.endsWith('.json')) {
                gridData = JSON.parse(content);
            } else if (file.name.endsWith('.csv')) {
                gridData = parseCSV(content);
            } else {
                throw new Error('Unsupported file format. Use CSV or JSON.');
            }
            
            processGridData(gridData);
            Visualization.showAlert(`Grid data loaded: ${file.name}`, 'success');
        } catch (error) {
            Visualization.showAlert(`Error loading file: ${error.message}`, 'error');
        }
    };
    
    reader.readAsText(file);
}

function parseCSV(content) {
    // Simple CSV parser for grid data
    const lines = content.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    // Assume format: bus_id,voltage,power,type OR from_bus,to_bus,impedance
    const buses = [];
    const lines_data = [];
    
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const values = lines[i].split(',').map(v => v.trim());
        
        if (headers.includes('bus_id')) {
            buses.push({
                id: parseInt(values[0]) || i-1,
                voltage: parseFloat(values[1]) || 1.0,
                power: parseFloat(values[2]) || 0.0
            });
        } else if (headers.includes('from_bus')) {
            lines_data.push({
                from_bus: parseInt(values[0]) || 0,
                to_bus: parseInt(values[1]) || 1,
                impedance: parseFloat(values[2]) || 0.01
            });
        }
    }
    
    return { buses, lines: lines_data };
}

function processGridData(data) {
    AppState.gridData = data;
    
    // Update UI
    document.getElementById('gridStatus').textContent = 'Data Loaded';
    document.getElementById('gridStatus').className = 'status-indicator status-success';
    
    // Update system info
    if (data.buses) {
        document.getElementById('busCount').textContent = data.buses.length.toLocaleString();
    }
    if (data.lines) {
        document.getElementById('lineCount').textContent = data.lines.length.toLocaleString();
    }
    
    // Render initial visualization
    Visualization.renderNetworkGraph(data);
    Visualization.updateStatus({ gridState: 'Loaded' });
    
    // Enable analysis buttons
    document.getElementById('analyzeSingle').disabled = false;
    document.getElementById('analyzeScreening').disabled = false;
}

async function loadContingencies() {
    try {
        const contingencies = await API.getContingencies();
        AppState.contingencies = contingencies;
        
        // Populate dropdown
        const select = document.getElementById('contingencySelect');
        select.innerHTML = '<option value="">Select a contingency...</option>' +
            contingencies.slice(0, 100).map(c => 
                `<option value="${c.id}">${c.name || c.id} - ${c.type || 'N-1'}</option>`
            ).join('');
    } catch (error) {
        console.warn('Could not load contingencies:', error);
        // Use simulated contingencies
        AppState.contingencies = generateSimulatedContingencies();
    }
}

function generateSimulatedContingencies() {
    const contingencies = [];
    for (let i = 1; i <= 100000; i++) {
        contingencies.push({
            id: `N1_${i}`,
            name: `Line ${i} Outage`,
            type: 'N-1',
            line_id: i
        });
    }
    return contingencies;
}

async function analyzeSingleContingency() {
    const select = document.getElementById('contingencySelect');
    const contingencyId = select.value;
    
    if (!contingencyId) {
        Visualization.showAlert('Please select a contingency', 'warning');
        return;
    }
    
    if (!AppState.gridData) {
        Visualization.showAlert('Please upload grid state data first', 'warning');
        return;
    }
    
    try {
        Visualization.showAlert('Analyzing contingency...', 'info');
        const startTime = performance.now();
        
        const results = await API.analyzeSingle(AppState.gridData, contingencyId);
        const duration = ((performance.now() - startTime) / 1000).toFixed(3);
        
        AppState.currentResults = results;
        
        // Update visualization
        Visualization.renderNetworkGraph(AppState.gridData, results);
        Visualization.updateRiskTable({ contingencies: [results] });
        
        Visualization.showAlert(`Analysis complete in ${duration}s`, 'success');
        Visualization.updateStatus({ inference: `${duration}s` });
    } catch (error) {
        Visualization.showAlert(`Analysis failed: ${error.message}`, 'error');
    }
}

async function runBulkScreening() {
    if (!AppState.gridData) {
        Visualization.showAlert('Please upload grid state data first', 'warning');
        return;
    }
    
    try {
        Visualization.showAlert('Running 100k N-1 screening... This may take ~40 seconds', 'info');
        const startTime = performance.now();
        
        const results = await API.runScreening(AppState.gridData);
        const duration = ((performance.now() - startTime) / 1000).toFixed(1);
        
        AppState.currentResults = results;
        
        // Update visualization with top risks
        Visualization.updateRiskTable(results);
        
        Visualization.showAlert(`Screening complete! Analyzed ${results.total || 100000} contingencies in ${duration}s`, 'success');
        Visualization.updateStatus({ 
            inference: `${duration}s total`,
            contingencies: results.total || 100000
        });
    } catch (error) {
        Visualization.showAlert(`Screening failed: ${error.message}`, 'error');
    }
}

function exportResults() {
    if (!AppState.currentResults) {
        Visualization.showAlert('No results to export', 'warning');
        return;
    }
    
    const data = JSON.stringify(AppState.currentResults, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `gridos_results_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    Visualization.showAlert('Results exported successfully', 'success');
}

function switchTab(tabName) {
    // Update button states
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    // Update content visibility
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === tabName);
    });
}

function handleSystemChange(event) {
        const system = event.target.value;
        AppState.currentSystem = system;

        // Update UI
        const systemNames = {
                    'pegase': 'PEGASE 9241',
                    'ieee118': 'IEEE 118-Bus',
                    'ieee300': 'IEEE 300-Bus'
                            };

        const systemCounts = {
                    'pegase': { buses: 9241, lines: 16049 },
                    'ieee118': { buses: 118, lines: 186 },
                    'ieee300': { buses: 300, lines: 411 }
                            };

        // Update header
        document.getElementById('systemName').textContent = systemNames[system];

        // Update system counts
        const counts = systemCounts[system];
        document.getElementById('busCount').textContent = counts.buses.toLocaleString();
        document.getElementById('lineCount').textContent = counts.lines.toLocaleString();

        // Clear current data
        AppState.gridData = null;
        AppState.currentResults = null;
        Visualization.clear();

        Visualization.showAlert(`Switched to ${systemNames[system]}. Please upload grid state data.`, 'info');
    }

async function checkAPIConnection() {
    try {
        const status = await API.checkConnection();
        AppState.apiConnected = status.connected;
        Visualization.updateStatus({ 
            api: status.connected ? 'Connected' : 'Simulated' 
        });
    } catch (error) {
        AppState.apiConnected = false;
        Visualization.updateStatus({ api: 'Simulated' });
    }
}

function updateSystemInfo() {
    // Update system information display
    document.getElementById('busCount').textContent = '9,241';
    document.getElementById('lineCount').textContent = '16,049';
}

async function analyzeBusContingency(busId) {
            if (!AppState.gridData) {
                            Visualization.showAlert('Please upload grid state data first', 'warning');
                            return;
                        }

            try {
                            Visualization.showAlert(`Analyzing Bus ${busId} contingency...`, 'info');
                            const startTime = performance.now();

                            // Create a simulated contingency for the bus
                            const contingencyId = `BUS_${busId}_OUTAGE`;
                            const results = await API.analyzeSingle(AppState.gridData, contingencyId);
                            const duration = ((performance.now() - startTime) / 1000).toFixed(3);

                            AppState.currentResults = results;

                            // Update visualization
                            Visualization.renderNetworkGraph(AppState.gridData, results);
                            Visualization.updateRiskTable({ contingencies: [results] });

                            Visualization.showAlert(`Bus ${busId} analysis complete in ${duration}s`, 'success');
                        } catch (error) {
                            Visualization.showAlert(`Analysis failed: ${error.message}`, 'error');
                        }
        }

// Export for debugging
window.App = {
    state: AppState,
    loadContingencies,
    analyzeSingleContingency,
    runBulkScreening,
            analyzeBusContingency
};
