"""GridOS GNN-PINN Dashboard Backend
FastAPI server for power grid contingency analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import time
import json

app = FastAPI(
    title="GridOS GNN-PINN Dashboard API",
    description="Real-time Power Grid Contingency Analysis using GNN and PINN",
    version="2.1.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class GridState(BaseModel):
    buses: List[Dict[str, Any]]
    lines: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

class ContingencyResult(BaseModel):
    id: str
    name: str
    type: str
    risk_score: float
    violations: int
    severity: float
    affected_buses: Optional[List[int]] = None

class ScreeningResults(BaseModel):
    total: int
    duration: float
    contingencies: List[ContingencyResult]
    summary: Dict[str, Any]

# Simulated GNN-PINN model
class GNN_PINN_Model:
    """Simulated GNN-PINN model for demonstration"""
    
    def __init__(self):
        self.inference_time = 0.0004  # 0.4ms per inference
    
    def analyze_contingency(self, grid_state: GridState, contingency_id: str) -> ContingencyResult:
        """Simulate single contingency analysis"""
        time.sleep(self.inference_time)
        
        # Simulate risk calculation
        risk_score = np.random.beta(2, 5)  # Skewed towards lower risk
        violations = int(np.random.poisson(risk_score * 10))
        severity = risk_score if violations > 0 else 0.0
        
        return ContingencyResult(
            id=contingency_id,
            name=f"Line Outage {contingency_id}",
            type="N-1",
            risk_score=risk_score,
            violations=violations,
            severity=severity,
            affected_buses=[int(i) for i in np.random.choice(len(grid_state.buses), min(violations, 5), replace=False)] if violations > 0 else []
        )
    
    def run_screening(self, grid_state: GridState, n_contingencies: int = 100000) -> ScreeningResults:
        """Simulate bulk contingency screening"""
        start_time = time.time()
        
        # Generate results for top risky contingencies
        contingencies = []
        for i in range(min(200, n_contingencies)):  # Generate top 200 results
            risk_score = np.random.beta(2, 5)
            violations = int(np.random.poisson(risk_score * 10))
            
            contingencies.append(ContingencyResult(
                id=f"N1_{i+1}",
                name=f"Line {i+1} Outage",
                type="N-1",
                risk_score=risk_score,
                violations=violations,
                severity=risk_score if violations > 0 else 0.0
            ))
        
        # Sort by risk score
        contingencies.sort(key=lambda x: x.risk_score, reverse=True)
        
        # Simulate processing time (should be ~40 seconds for 100k)
        processing_time = n_contingencies * self.inference_time
        time.sleep(min(processing_time, 2.0))  # Cap at 2 seconds for demo
        
        duration = time.time() - start_time
        
        # Calculate summary statistics
        critical = sum(1 for c in contingencies if c.risk_score > 0.8)
        high = sum(1 for c in contingencies if 0.5 < c.risk_score <= 0.8)
        moderate = sum(1 for c in contingencies if 0.2 < c.risk_score <= 0.5)
        
        return ScreeningResults(
            total=n_contingencies,
            duration=duration,
            contingencies=contingencies[:50],  # Return top 50
            summary={
                "critical_count": critical,
                "high_count": high,
                "moderate_count": moderate,
                "avg_risk": float(np.mean([c.risk_score for c in contingencies])),
                "total_violations": sum(c.violations for c in contingencies)
            }
        )

# Initialize model
model = GNN_PINN_Model()

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "GridOS GNN-PINN Dashboard API",
        "version": "2.1.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "GNN-PINN v2.1",
        "inference_time": "0.4ms/step"
    }

@app.get("/api/contingencies")
async def get_contingencies():
    """Get list of available contingencies"""
    # Generate contingency list
    contingencies = [
        {
            "id": f"N1_{i}",
            "name": f"Line {i} Outage",
            "type": "N-1",
            "line_id": i
        }
        for i in range(1, 101)  # Return first 100 for dropdown
    ]
    return contingencies

@app.post("/api/upload")
async def upload_grid_state(file: UploadFile = File(...)):
    """Upload and validate grid state file"""
    try:
        content = await file.read()
        
        if file.filename.endswith('.json'):
            data = json.loads(content)
        else:
            raise HTTPException(status_code=400, detail="Only JSON files supported")
        
        # Validate grid data structure
        if 'buses' not in data or 'lines' not in data:
            raise HTTPException(status_code=400, detail="Invalid grid data format")
        
        return {
            "status": "success",
            "filename": file.filename,
            "buses_count": len(data.get('buses', [])),
            "lines_count": len(data.get('lines', []))
        }
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_single(grid_state: GridState, contingency_id: str):
    """Analyze single contingency"""
    try:
        result = model.analyze_contingency(grid_state, contingency_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/screening")
async def run_screening(grid_state: GridState, n_contingencies: int = 100000):
    """Run bulk N-1 contingency screening"""
    try:
        results = model.run_screening(grid_state, n_contingencies)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """Get model statistics"""
    return {
        "model_type": "GNN-PINN Hybrid",
        "architecture": "3-layer GCN with Physics Constraints",
        "inference_speed": "0.4ms per contingency",
        "throughput": "2,500 contingencies/second",
        "accuracy": "95%+ correlation with AC power flow",
        "benchmark": "PEGASE 9241 (9,241 buses, 16,049 lines)"
    }

    # AI Assistant endpoints
@app.post("/api/assistant/query")
async def query_assistant(request: Dict[str, Any]):
    """Query the rnj-1 AI assistant with natural language"""
    try:
        from ai_assistant import get_assistant
        assistant = get_assistant()
        
        question = request.get("question", "")
        context = request.get("context")
        
        if not question:
            raise HTTPException(status_code=400, details="Question is required")
        
        response = assistant.query(question, grid_context=context)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, details=str(e))

@app.post("/api/assistant/explain")
async def explain_contingency(result: ContingencyResult):
    """Get natural language explanation of contingency result"""
    try:
        from ai_assistant import get_assistant
        assistant = get_assistant()
        explanation = assistant.explain_result(result.dict())
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, details=str(e))

@app.get("/api/assistant/status")
async def check_assistant_status():
    """Check if Ollama and rnj-1 are available"""
    try:
        from ai_assistant import get_assistant
        assistant = get_assistant()
        status = assistant.check_ollama_status()
        return status
    except Exception as e:
        return {
            "available": False,
            "message": f"Error checking Ollama status: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
