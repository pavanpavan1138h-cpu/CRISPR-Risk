from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

from ml_model import CRISPRModel
from risk_calculator import calculate_brs

app = FastAPI(title="CRISPR Biosafety Risk Assessor")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model
model = CRISPRModel()
# Pre-train on startup
model.train()

# Request Model
class PredictionRequest(BaseModel):
    grna: str

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    grna = request.grna.upper()
    
    # Validation for flexible length
    if not (18 <= len(grna) <= 25) or not all(c in "ATGC" for c in grna):
        raise HTTPException(status_code=400, detail="Invalid gRNA sequence. Must be 18-25 nucleotides (A, T, G, C).")
    
    try:
        # 1. Get Off-Target Predictions
        off_targets = model.predict(grna)
        
        # 2. Calculate BRS for each off-target
        enriched_results = []
        for site in off_targets:
            risk_data = calculate_brs(site["off_target_prob"])
            
            # Merge dicts
            result_item = {**site, **risk_data}
            enriched_results.append(result_item)
            
        # Sort by BRS descending (High risk first)
        enriched_results.sort(key=lambda x: x["brs_score"], reverse=True)
            
        return {
            "query_grna": grna,
            "results": enriched_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount Static Files (Frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
