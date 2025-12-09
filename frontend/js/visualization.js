// visualization.js - Network visualization and risk display module

const Visualization = {
    // Network graph visualization using Plotly
    renderNetworkGraph: function(gridData, contingencyResults = null) {
        const networkGraph = document.getElementById('networkGraph');
        
        if (!gridData || !gridData.buses || !gridData.lines) {
            networkGraph.innerHTML = '<p style="text-align:center;color:#666;padding:2rem;">No grid data loaded</p>';
            return;
        }

        // Create network layout
        const nodes = gridData.buses.map((bus, idx) => ({
            x: Math.cos(2 * Math.PI * idx / gridData.buses.length),
            y: Math.sin(2 * Math.PI * idx / gridData.buses.length),
            text: `Bus ${bus.id || idx}`,
            marker: {
                size: 8,
                color: this.getNodeColor(bus, contingencyResults),
                line: { width: 1, color: 'rgba(167,169,169,0.3)' }
            }
        }));

        // Create edges
        const edges = {
            x: [],
            y: [],
            mode: 'lines',
            line: { color: 'rgba(32,128,128,0.3)', width: 1 },
            hoverinfo: 'none'
        };

        gridData.lines.forEach(line => {
            const from = line.from_bus || 0;
            const to = line.to_bus || 1;
            if (from < nodes.length && to < nodes.length) {
                edges.x.push(nodes[from].x, nodes[to].x, null);
                edges.y.push(nodes[from].y, nodes[to].y, null);
            }
        });

        const data = [
            edges,
            {
                x: nodes.map(n => n.x),
                y: nodes.map(n => n.y),
                text: nodes.map(n => n.text),
                mode: 'markers',
                marker: nodes[0].marker,
                hoverinfo: 'text'
            }
        ];

        const layout = {
            showlegend: false,
            hovermode: 'closest',
            margin: { t: 0, r: 0, l: 0, b: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: { visible: false, range: [-1.5, 1.5] },
            yaxis: { visible: false, range: [-1.5, 1.5] },
            height: 400
        };

        Plotly.newPlot(networkGraph, data, layout, { responsive: true, displayModeBar: false });
    },

    getNodeColor: function(bus, results) {
        if (!results || !results.violations) return 'rgb(32,128,128)';
        
        const violation = results.violations.find(v => v.bus_id === bus.id);
        if (!violation) return 'rgb(34,197,94)'; // green - safe
        
        const severity = violation.severity || 0;
        if (severity > 0.8) return 'rgb(255,84,89)'; // red - critical
        if (severity > 0.5) return 'rgb(255,159,67)'; // orange - warning
        return 'rgb(255,159,67)';
    },

    // Update risk table with contingency results
    updateRiskTable: function(results) {
        const tbody = document.getElementById('riskTableBody');
        
        if (!results || !results.contingencies || results.contingencies.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:#666;padding:1rem;">No analysis results yet</td></tr>';
            return;
        }

        // Sort by risk score descending
        const sorted = [...results.contingencies].sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0));
        const top20 = sorted.slice(0, 20);

        tbody.innerHTML = top20.map(cont => `
            <tr>
                <td>${this.escapeHtml(cont.name || cont.id || 'Unknown')}</td>
                <td>${this.escapeHtml(cont.type || 'N-1')}</td>
                <td>
                    <span class="risk-badge risk-${this.getRiskClass(cont.risk_score)}">
                        ${(cont.risk_score || 0).toFixed(3)}
                    </span>
                </td>
                <td>${cont.violations || 0}</td>
            </tr>
        `).join('');
    },

    getRiskClass: function(score) {
        if (score > 0.8) return 'critical';
        if (score > 0.5) return 'warning';
        if (score > 0.2) return 'moderate';
        return 'low';
    },

    escapeHtml: function(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    // Show alert message
    showAlert: function(message, type = 'info') {
        const alerts = document.getElementById('alerts');
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.innerHTML = `
            ${this.getAlertIcon(type)}
            <span>${this.escapeHtml(message)}</span>
            <button class="alert-close" onclick="this.parentElement.remove()">✕</button>
        `;
        alerts.appendChild(alertDiv);

        // Auto-remove after 5 seconds
        setTimeout(() => alertDiv.remove(), 5000);
    },

    getAlertIcon: function(type) {
        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };
        return icons[type] || icons.info;
    },

    // Update status bar
    updateStatus: function(status) {
        if (status.gridState) {
            document.getElementById('statusGridState').textContent = status.gridState;
        }
        if (status.contingencies !== undefined) {
            document.getElementById('statusContingencies').textContent = status.contingencies.toLocaleString();
        }
        if (status.inference) {
            document.getElementById('statusInference').textContent = status.inference;
        }
        if (status.api) {
            document.getElementById('statusAPI').textContent = status.api;
        }
    },

    // Clear all visualizations
    clear: function() {
        document.getElementById('networkGraph').innerHTML = '';
        document.getElementById('riskTableBody').innerHTML = '<tr><td colspan="4" style="text-align:center;color:#666;padding:1rem;">System idle. Upload state or run screening.</td></tr>';
        document.getElementById('alerts').innerHTML = '';
    }
};

// Export for use in app.js
window.Visualization = Visualization;
