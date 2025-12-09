// ===== UTILITY FUNCTIONS =====

// Format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Format time duration
function formatDuration(ms) {
    if (ms < 1000) return `${ms.toFixed(2)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertsDiv = document.getElementById('alerts');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    alertsDiv.appendChild(alert);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// Clear all alerts
function clearAlerts() {
    const alertsDiv = document.getElementById('alerts');
    alertsDiv.innerHTML = '';
}

// Update status bar
function updateStatus(gridState, contingencies, inference, api) {
    if (gridState) document.getElementById('statusGridState').textContent = gridState;
    if (contingencies) document.getElementById('statusContingencies').textContent = formatNumber(contingencies);
    if (inference) document.getElementById('statusInference').textContent = formatDuration(inference);
    if (api) document.getElementById('statusAPI').textContent = api;
}

// Tab switching functionality
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Show corresponding content
            const tabName = button.getAttribute('data-tab');
            document.getElementById(tabName).classList.add('active');
        });
    });
}

// Parse CSV file
function parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const row = {};
        headers.forEach((header, index) => {
            row[header] = values[index]?.trim();
        });
        data.push(row);
    }
    
    return { headers, data };
}

// Parse JSON file
function parseJSON(text) {
    try {
        return JSON.parse(text);
    } catch (e) {
        console.error('Invalid JSON:', e);
        return null;
    }
}

// Debounce function for performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export data to CSV
function exportToCSV(data, filename) {
    if (!data || data.length === 0) {
        showAlert('No data to export', 'warning');
        return;
    }
    
    const headers = Object.keys(data[0]);
    const csvContent = [
        headers.join(','),
        ...data.map(row => headers.map(h => row[h]).join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'export.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showAlert('Data exported successfully', 'success');
}

// Calculate risk score
function calculateRiskScore(violations) {
    if (!violations || violations.length === 0) return 0;
    return violations.reduce((sum, v) => sum + (v.severity || 1), 0);
}

// Sort table data
function sortTableData(data, key, ascending = true) {
    return data.sort((a, b) => {
        const aVal = a[key];
        const bVal = b[key];
        if (ascending) {
            return aVal > bVal ? 1 : -1;
        }
        return aVal < bVal ? 1 : -1;
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
});
