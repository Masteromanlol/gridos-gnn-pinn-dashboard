// ===== API LAYER =====

const API_BASE_URL = '/api';  // Update with actual backend URL
let currentGridData = null;

// Simulated API for demo purposes
const DEMO_MODE = true;

// Upload grid state file
async function uploadGridState(file) {
    try {
        const text = await file.text();
        const extension = file.name.split('.').pop();
        
        let data;
        if (extension === 'csv') {
            data = parseCSV(text);
        } else if (extension === 'json') {
            data = parseJSON(text);
        } else {
            throw new Error('Unsupported file format');
        }
        
        currentGridData = data;
        showAlert(`Grid state loaded: ${file.name}`, 'success');
        updateStatus('Loaded', null, null, DEMO_MODE ? 'Simulated' : 'Connected');
        
        // Populate contingency dropdown
        populateContingencies(data);
        
        return data;
    } catch (error) {
        showAlert(`Error loading file: ${error.message}`, 'error');
        throw error;
    }
}

// Populate contingency dropdown
function populateContingencies(data) {
    const select = document.getElementById('contingencySelect');
    select.innerHTML = '<option>Select a contingency...</option>';
    
    // Generate sample contingencies based on grid data
    const numContingencies = data.data ? Math.min(data.data.length, 50) : 50;
    for (let i = 1; i <= numContingencies; i++) {
        const option = document.createElement('option');
        option.value = `line_${i}`;
        option.textContent = `Line ${i} Outage`;
        select.appendChild(option);
    }
}

// Analyze single contingency
async function analyzeSingleContingency(contingencyId) {
    if (DEMO_MODE) {
        return simulateSingleAnalysis(contingencyId);
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze/single`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gridData: currentGridData,
                contingency: contingencyId
            })
        });
        
        if (!response.ok) throw new Error('Analysis failed');
        return await response.json();
    } catch (error) {
        showAlert(`Analysis error: ${error.message}`, 'error');
        throw error;
    }
}

// Run N-1 screening
async function runScreening() {
    if (DEMO_MODE) {
        return simulateScreening();
    }
    
    try {
        showAlert('Starting 100k N-1 screening...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/analyze/screening`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gridData: currentGridData })
        });
        
        if (!response.ok) throw new Error('Screening failed');
        return await response.json();
    } catch (error) {
        showAlert(`Screening error: ${error.message}`, 'error');
        throw error;
    }
}

// Simulated single analysis
function simulateSingleAnalysis(contingencyId) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const result = {
                contingency: contingencyId,
                violations: Math.random() > 0.7 ? [
                    { bus: Math.floor(Math.random() * 9241), type: 'Voltage', severity: Math.random() * 0.3 }
                ] : [],
                inferenceTime: Math.random() * 0.6 + 0.2
            };
            resolve(result);
        }, 500);
    });
}

// Simulated screening
function simulateScreening() {
    return new Promise((resolve) => {
        showAlert('Running GNN-PINN accelerated screening...', 'info');
        
        setTimeout(() => {
            const results = {
                totalContingencies: 100000,
                analysisTime: 38.5,
                highRiskCount: Math.floor(Math.random() * 150) + 50,
                contingencies: []
            };
            
            // Generate high-risk contingencies
            for (let i = 0; i < results.highRiskCount; i++) {
                results.contingencies.push({
                    id: `line_${Math.floor(Math.random() * 16049)}`,
                    type: ['Line', 'Transformer', 'Generator'][Math.floor(Math.random() * 3)],
                    riskScore: (Math.random() * 3 + 7).toFixed(2),
                    violations: Math.floor(Math.random() * 5) + 1
                });
            }
            
            // Sort by risk score
            results.contingencies.sort((a, b) => parseFloat(b.riskScore) - parseFloat(a.riskScore));
            
            resolve(results);
        }, 2000);
    });
}
