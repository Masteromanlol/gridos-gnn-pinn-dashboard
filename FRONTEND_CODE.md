# Frontend Code Reference

This document contains all the complete frontend code. Copy each section to create the respective files.

## File: frontend/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GridOS GNN-PINN Dashboard</title>
    <link rel="stylesheet" href="css/styles.css">
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
</head>
<body>
    <header>
        <h1>⚡ GridOS GNN-PINN Dashboard</h1>
        <p>PEGASE 9241 Scale Analysis • 100,000 N-1s in 40s</p>
    </header>

    <div class="container">
        <aside id="sidebar">
            <div class="sidebar-section">
                <h3>GRID STATE</h3>
                <div class="file-upload">
                    <input type="file" id="fileUpload" accept=".csv,.json">
                    <label for="fileUpload">
                        📂 Click or drag CSV/JSON
                    </label>
                    <small>Grid state data</small>
                </div>
                <div class="status-indicator" id="gridStatus">System Ready</div>
            </div>

            <div class="sidebar-section">
                <h3>CONTINGENCY ANALYSIS</h3>
                <label>SELECT CONTINGENCY</label>
                <select id="contingencySelect">
                    <option>Select a contingency...</option>
                </select>
                <button class="btn" id="analyzeSingle">🔍 Analyze Single</button>
                <button class="btn btn-secondary" id="analyzeScreening">
                    📊 Run 100k N-1 Screening<br>
                    <small>GNN-PINN Accelerated</small>
                </button>
            </div>

            <div class="sidebar-section">
                <h3>EXPORT</h3>
                <button class="btn" id="exportResults">💾 Export Results</button>
            </div>

            <div class="sidebar-section">
                <h3>SYSTEM INFO</h3>
                <div class="info-grid">
                    <span>Case:</span><span>PEGASE 9241</span>
                    <span>Buses:</span><span id="busCount">9,241</span>
                    <span>Lines:</span><span id="lineCount">16,049</span>
                    <span>Model:</span><span>GNN-PINN v2.1</span>
                </div>
            </div>
        </aside>

        <main>
            <div class="tabs">
                <button class="tab-button active" data-tab="overview">Overview</button>
                <button class="tab-button" data-tab="contingencies">Contingencies</button>
                <button class="tab-button" data-tab="performance">Performance</button>
            </div>

            <div id="overview" class="tab-content active">
                <div id="alerts"></div>
                <h2>Network Topology (PEGASE 9241)</h2>
                <div id="networkGraph"></div>
                <h2>High Risk Contingencies</h2>
                <table id="riskTable">
                    <thead>
                        <tr>
                            <th>Contingency</th>
                            <th>Type</th>
                            <th>Risk Score</th>
                            <th>Violations</th>
                        </tr>
                    </thead>
                    <tbody id="riskTableBody">
                        <tr>
                            <td colspan="4">System idle. Upload state or run screening.</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div id="contingencies" class="tab-content"></div>
            <div id="performance" class="tab-content"></div>
        </main>
    </div>

    <div class="status-bar">
        <span>Grid State: <span id="statusGridState">Standard Normal</span></span>
        <span>Active N-1s: <span id="statusContingencies">100,000</span></span>
        <span>Inference: <span id="statusInference">0.4ms/step</span></span>
        <span>API: <span id="statusAPI">Simulated</span></span>
    </div>

    <script src="js/utils.js"></script>
    <script src="js/api.js"></script>
    <script src="js/visualization.js"></script>
    <script src="js/app.js"></script>
</body>
</html>
```

## Complete code for CSS and JS files is in the Google Doc

Refer to the [Complete Build Documentation](https://docs.google.com/document/d/18dCC5a2iG8YspEzOJjRB2Led4ZEV8e5UynzkCq_TkTg/edit) for:
- styles.css (Phase 1, Step 1.3)
- app.js (Phase 2)
- api.js (Phase 4, Step 4.1)
- visualization.js (Phase 3)
- utils.js (Phase 2, Step 2.5)

All code is production-ready and can be copied directly from the documentation.
