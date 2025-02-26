# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import logging
from typing import List, Dict
import os

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS Configuration (consider restricting origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
REQUIRED_CREDIT_COLS = ['id', 'amount', 'interest_rate', 'effective_date']
REQUIRED_SWAP_COLS = ['id', 'amount', 'interest_rate', 'effective_date']

def validate_dataframe(df: pd.DataFrame, required_cols: List[str], file_type: str) -> None:
    """Validate dataframe structure and content"""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise HTTPException(400, f"{file_type} file missing columns: {', '.join(missing)}")
    if df.empty:
        raise HTTPException(400, f"{file_type} file is empty")

def match_hedges(credits: pd.DataFrame, swaps: pd.DataFrame) -> List[Dict]:
    """Core matching algorithm combining datasets"""
    return pd.concat([credits, swaps], ignore_index=True)\
             .fillna('')\
             .astype(str)\
             .to_dict(orient='records')

@app.post("/api/match")
async def process_match(
    credit_file: UploadFile = File(...),
    swap_file: UploadFile = File(...),
    regulation: str = Form(...),
    threshold_lower: float = Form(...),
    threshold_upper: float = Form(...),
    infeasible: str = Form(...)
):
    try:
        # Validate file types
        if not (credit_file.filename.endswith('.xlsx') and swap_file.filename.endswith('.xlsx')):
            raise HTTPException(400, "Only .xlsx files are supported")

        # Process credit file
        credit_content = await credit_file.read()
        credit_df = pd.read_excel(credit_content, engine='openpyxl')
        validate_dataframe(credit_df, REQUIRED_CREDIT_COLS, "Credit")

        # Process swap file
        swap_content = await swap_file.read()
        swap_df = pd.read_excel(swap_content, engine='openpyxl')
        validate_dataframe(swap_df, REQUIRED_SWAP_COLS, "Swap")

        # Run matching algorithm
        results = match_hedges(credit_df, swap_df)
        return JSONResponse(content={"results": results})

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(500, "Internal server error")
@app.post("/api/preview")
async def preview_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_excel(content, nrows=10, engine='openpyxl')
        return JSONResponse(content={
            "columns": df.columns.tolist(),
            "data": df.fillna('').astype(str).head(5).to_dict(orient='records')
        })
    except Exception as e:
        logger.error(f"Preview error: {str(e)}", exc_info=True)
        raise HTTPException(500, "Error previewing file")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)