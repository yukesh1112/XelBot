from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import uvicorn
import io
import google.generativeai as genai
import logging
import os

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Store uploaded dataset globally
dataframe = None

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Configure Gemini AI (Replace with actual API Key)
GEMINI_API_KEY = "AIzaSyA0ePTaWI-Up6oTuG_R_B-Pvave0UwumjM"
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Define request model for chatbot queries
class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    """🏠 Home Route: Check API Status"""
    return {"message": "🚀 XelBot API is running!"}

@app.post("/upload/")
async def upload_data(file: UploadFile = File(...)):
    """📂 Upload CSV Dataset"""
    global dataframe
    try:
        # ✅ Read CSV file
        contents = await file.read()
        dataframe = pd.read_csv(io.BytesIO(contents))
        logging.info(f"✅ Dataset uploaded: {file.filename}")

        return {
            "message": "✅ Dataset uploaded successfully!",
            "columns": list(dataframe.columns),
            "rows": len(dataframe),
        }

    except Exception as e:
        logging.error(f"❌ Error reading dataset: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

@app.post("/chatbot/")
def chatbot(query: QueryRequest):
    """🤖 Chatbot logic for Dataset Analysis & AI Responses"""
    global dataframe

    # ✅ Check if dataset is loaded
    if dataframe is None:
        raise HTTPException(status_code=400, detail="⚠ No dataset uploaded! Please upload a dataset first.")

    query_text = query.query.lower()

    # ✅ Call Gemini AI for Natural Language Understanding with Dataset Context
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        
        # Create dataset context for AI
        dataset_context = f"""
DATASET ANALYSIS CONTEXT:
- Dataset shape: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns
- Columns: {', '.join(dataframe.columns)}
- Data types: {dict(dataframe.dtypes)}
- Sample data: {dataframe.head(2).to_string()}

USER QUESTION: {query.query}

As XelBot, analyze this specific dataset and provide insights based on the actual data shown above. Focus only on what's in this dataset."""
        
        gemini_response = model.generate_content(dataset_context).text
    except Exception as e:
        logging.error(f"❌ Gemini AI Error: {str(e)}")
        gemini_response = f"⚠️ Gemini AI Error: {str(e)}"

    # ✅ Dataset Insights Processing
    dataset_response = "🔍 Searching dataset..."
    try:
        if "columns" in query_text:
            dataset_response = f"📊 The dataset contains these columns: {', '.join(dataframe.columns)}"

        elif "rows" in query_text or "size" in query_text:
            dataset_response = f"📊 The dataset has {dataframe.shape[0]} rows and {dataframe.shape[1]} columns."

        elif "mean" in query_text or "average" in query_text:
            numeric_cols = dataframe.select_dtypes(include=["number"]).columns
            if numeric_cols.any():
                avg_values = dataframe[numeric_cols].mean().to_dict()
                dataset_response = f"📈 Mean values: {avg_values}"
            else:
                dataset_response = "⚠ No numerical columns found for averaging."

        elif "sales" in query_text:
            if "region" in dataframe.columns:
                region_sales = dataframe.groupby("region")["sales"].sum()
                top_region = region_sales.idxmax()
                dataset_response = f"🏆 The region with the highest sales is **{top_region}**."
            else:
                dataset_response = "⚠ Your dataset does not contain a 'region' column."

        elif "product" in query_text:
            if "product" in dataframe.columns and "sales" in dataframe.columns:
                product_sales = dataframe.groupby("product")["sales"].sum()
                top_product = product_sales.idxmax()
                dataset_response = f"🏆 The product with the highest sales is **{top_product}**."
            else:
                dataset_response = "⚠ Your dataset does not contain 'product' and 'sales' columns."

        else:
            dataset_response = "⚠️ Unable to find a dataset-specific answer. Try a different question."

    except Exception as e:
        logging.error(f"❌ Dataset Analysis Error: {str(e)}")
        dataset_response = f"⚠️ Dataset Analysis Error: {str(e)}"

    return {
        "gemini_response": gemini_response,
        "dataset_response": dataset_response
    }

@app.get("/download/")
def download_dataset():
    """📥 Download Processed Dataset"""
    global dataframe
    if dataframe is None:
        raise HTTPException(status_code=400, detail="⚠ No dataset available to download!")

    file_path = "processed_dataset.csv"
    dataframe.to_csv(file_path, index=False)
    return {"message": "✅ Processed dataset saved!", "file_path": file_path}

# ✅ Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
