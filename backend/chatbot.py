from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import google.generativeai as genai
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI()

# Global variable to store dataset
dataframe = None

# ✅ Set up Google Gemini AI (Replace with your API Key)
GEMINI_API_KEY = "AIzaSyDMoRoUKWBryd7wO8mJy-g1Oidns5GU43A"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Define request model for chatbot queries
class QueryRequest(BaseModel):
    query: str

# ✅ Define request model for dataset upload
class FileUploadRequest(BaseModel):
    file_path: str

@app.get("/")
def home():
    """🏠 Home Route"""
    return {"message": "XelBot API is running!"}

@app.post("/upload/")
def upload_data(request: FileUploadRequest):
    """📂 Upload Dataset from File Path"""
    global dataframe
    file_path = request.file_path

    # ✅ Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="File not found! Please provide a valid file path.")

    try:
        # ✅ Load dataset (supports CSV & Excel)
        if file_path.endswith(".csv"):
            dataframe = pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            dataframe = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format! Use CSV or Excel.")

        return {
            "message": "✅ Dataset uploaded successfully!",
            "columns": list(dataframe.columns),
            "rows": len(dataframe),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

@app.get("/chatbot/")
def chatbot(query: str):
    """🤖 Chatbot to analyze dataset & use Gemini AI"""
    global dataframe

    # ✅ Check if dataset is loaded
    if dataframe is None:
        raise HTTPException(status_code=400, detail="⚠ No dataset uploaded! Upload a dataset first.")

    query_lower = query.lower()

    # ✅ Dataset Analysis
    if "columns" in query_lower:
        return {"response": f"📊 The dataset contains these columns: {', '.join(dataframe.columns)}"}

    elif "rows" in query_lower or "size" in query_lower:
        return {"response": f"📊 The dataset has {dataframe.shape[0]} rows and {dataframe.shape[1]} columns."}

    elif "average" in query_lower or "mean" in query_lower:
        numeric_cols = dataframe.select_dtypes(include=["number"]).columns
        if numeric_cols.any():
            avg_values = dataframe[numeric_cols].mean().to_dict()
            return {"response": f"📈 Mean values: {avg_values}"}
        else:
            return {"response": "⚠ No numerical columns found for averaging."}

    # ✅ Generate response using Gemini AI
    else:
        try:
            model = genai.GenerativeModel("gemini-pro")
            gemini_response = model.generate_content(query).text
            return {"response": gemini_response}
        except Exception as e:
            return {"response": f"⚠ Gemini AI Error: {str(e)}"}

# ✅ Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
