"""
Enhanced Backend API for XelBot Professional Data Analytics
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn
import io
import google.generativeai as genai
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from advanced_analytics import AdvancedAnalytics, BusinessIntelligence

# Initialize FastAPI App
app = FastAPI(
    title="XelBot Professional Data Analytics API",
    description="AI-Powered Data Analytics Chatbot Backend",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
datasets = {}  # Store multiple datasets
analysis_cache = {}  # Cache analysis results

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure Gemini AI
GEMINI_API_KEY = "AIzaSyA0ePTaWI-Up6oTuG_R_B-Pvave0UwumjM"
genai.configure(api_key=GEMINI_API_KEY)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    dataset_id: Optional[str] = "default"

class AnalysisRequest(BaseModel):
    dataset_id: str = "default"
    analysis_type: str = "complete"  # complete, quick, custom

class ChatMessage(BaseModel):
    message: str
    timestamp: datetime
    type: str  # user, bot

class DatasetInfo(BaseModel):
    id: str
    name: str
    rows: int
    columns: int
    upload_time: datetime
    size_mb: float

@app.get("/")
def home():
    """🏠 API Status and Information"""
    return {
        "message": "🚀 XelBot Professional Data Analytics API",
        "version": "2.0.0",
        "status": "active",
        "features": [
            "Advanced Data Analytics",
            "Business Intelligence",
            "Natural Language Queries",
            "Predictive Analytics",
            "Professional Reports"
        ]
    }

@app.post("/upload/")
async def upload_dataset(file: UploadFile = File(...), dataset_id: str = "default"):
    """📂 Upload and Process Dataset"""
    try:
        # Read and validate file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Store dataset
        datasets[dataset_id] = {
            "dataframe": df,
            "metadata": {
                "filename": file.filename,
                "upload_time": datetime.now(),
                "rows": len(df),
                "columns": len(df.columns),
                "size_mb": round(len(contents) / (1024 * 1024), 2),
                "column_names": list(df.columns),
                "data_types": dict(df.dtypes.astype(str))
            }
        }
        
        # Run initial analysis
        analytics = AdvancedAnalytics(df)
        initial_profile = analytics.data_profiling()
        
        # Cache initial analysis
        analysis_cache[dataset_id] = {
            "profile": initial_profile,
            "timestamp": datetime.now()
        }
        
        logging.info(f"✅ Dataset uploaded: {file.filename} ({dataset_id})")
        
        return {
            "message": "✅ Dataset uploaded and analyzed successfully!",
            "dataset_id": dataset_id,
            "metadata": datasets[dataset_id]["metadata"],
            "initial_insights": initial_profile["overview"],
            "data_quality_score": _calculate_quality_score(initial_profile)
        }
        
    except Exception as e:
        logging.error(f"❌ Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/analyze/")
def analyze_dataset(request: AnalysisRequest):
    """🔍 Comprehensive Dataset Analysis"""
    if request.dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        df = datasets[request.dataset_id]["dataframe"]
        
        # Perform analysis based on type
        analytics = AdvancedAnalytics(df)
        bi = BusinessIntelligence(df)
        
        if request.analysis_type == "complete":
            results = {
                "profile": analytics.data_profiling(),
                "business_insights": analytics.business_insights(),
                "predictive_analysis": analytics.predictive_analysis(),
                "recommendations": analytics.generate_recommendations(),
                "executive_summary": bi.executive_summary()
            }
        elif request.analysis_type == "quick":
            results = {
                "profile": analytics.data_profiling(),
                "quick_insights": analytics.business_insights()[:3]
            }
        else:
            results = {"message": "Custom analysis not implemented yet"}
        
        # Cache results
        analysis_cache[request.dataset_id] = {
            **analysis_cache.get(request.dataset_id, {}),
            "full_analysis": results,
            "timestamp": datetime.now()
        }
        
        return {
            "dataset_id": request.dataset_id,
            "analysis_type": request.analysis_type,
            "results": results,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logging.error(f"❌ Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/chat/")
def chat_with_data(query: QueryRequest):
    """💬 Intelligent Chat with Data"""
    if query.dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        df = datasets[query.dataset_id]["dataframe"]
        
        # Generate AI response with comprehensive context
        ai_response = _generate_intelligent_response(query.query, df, query.dataset_id)
        
        # Generate specific data insights
        data_insights = _generate_data_insights(query.query, df)
        
        return {
            "query": query.query,
            "dataset_id": query.dataset_id,
            "ai_response": ai_response,
            "data_insights": data_insights,
            "timestamp": datetime.now(),
            "suggestions": _generate_follow_up_questions(query.query, df)
        }
        
    except Exception as e:
        logging.error(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/datasets/")
def list_datasets():
    """📋 List All Uploaded Datasets"""
    dataset_list = []
    for dataset_id, data in datasets.items():
        dataset_list.append({
            "id": dataset_id,
            "metadata": data["metadata"],
            "has_analysis": dataset_id in analysis_cache
        })
    
    return {
        "datasets": dataset_list,
        "total_count": len(dataset_list)
    }

@app.get("/insights/{dataset_id}")
def get_insights(dataset_id: str):
    """💡 Get Cached Insights for Dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="No analysis found. Run analysis first.")
    
    return {
        "dataset_id": dataset_id,
        "insights": analysis_cache[dataset_id],
        "metadata": datasets[dataset_id]["metadata"]
    }

@app.post("/report/")
def generate_report(dataset_id: str = "default", report_type: str = "executive"):
    """📊 Generate Professional Report"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset_id not in analysis_cache or "full_analysis" not in analysis_cache[dataset_id]:
        raise HTTPException(status_code=400, detail="Complete analysis required first")
    
    try:
        analysis_results = analysis_cache[dataset_id]["full_analysis"]
        metadata = datasets[dataset_id]["metadata"]
        
        if report_type == "executive":
            report = _generate_executive_report(analysis_results, metadata)
        else:
            report = {"message": "Other report types not implemented yet"}
        
        return {
            "dataset_id": dataset_id,
            "report_type": report_type,
            "report": report,
            "generated_at": datetime.now()
        }
        
    except Exception as e:
        logging.error(f"❌ Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.delete("/dataset/{dataset_id}")
def delete_dataset(dataset_id: str):
    """🗑️ Delete Dataset and Analysis"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Remove dataset and analysis
    del datasets[dataset_id]
    if dataset_id in analysis_cache:
        del analysis_cache[dataset_id]
    
    return {"message": f"Dataset {dataset_id} deleted successfully"}

# Helper functions
def _generate_intelligent_response(query: str, df: pd.DataFrame, dataset_id: str) -> str:
    """Generate intelligent AI response with business context"""
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        
        # Get cached analysis if available
        cached_analysis = analysis_cache.get(dataset_id, {})
        
        # Create comprehensive business context
        analytics = AdvancedAnalytics(df)
        insights = analytics.business_insights()
        
        context = f"""
You are XelBot, a senior data analytics consultant with 10+ years of experience. 
You're analyzing a client's business data to provide actionable insights.

CLIENT'S DATASET:
- Size: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns)}
- Business Domain: {_detect_business_domain(df.columns)}

SAMPLE DATA:
{df.head(3).to_string()}

KEY BUSINESS METRICS:
{chr(10).join(insights[:5]) if insights else "Analyzing business metrics..."}

CLIENT QUESTION: "{query}"

CONSULTANT INSTRUCTIONS:
1. Act as a senior business consultant, not just a data analyst
2. Provide specific, actionable business recommendations
3. Include relevant KPIs and metrics from the actual data
4. Suggest concrete next steps for business improvement
5. Use professional business language
6. If data visualization would help, recommend specific chart types
7. Focus on ROI and business impact
8. Be conversational but authoritative

Professional Response:"""
        
        response = model.generate_content(context)
        return response.text
        
    except Exception as e:
        return f"I apologize, but I encountered an issue analyzing your data: {str(e)}. Let me try a different approach to help you."

def _generate_data_insights(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Generate specific data insights based on query"""
    insights = {}
    
    # Keyword-based analysis
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['revenue', 'sales', 'profit', 'income']):
        revenue_cols = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['revenue', 'sales', 'price', 'amount', 'value'])]
        if revenue_cols:
            insights['financial'] = {
                'total_revenue': df[revenue_cols[0]].sum() if revenue_cols[0] in df.select_dtypes(include=['number']).columns else None,
                'avg_revenue': df[revenue_cols[0]].mean() if revenue_cols[0] in df.select_dtypes(include=['number']).columns else None
            }
    
    if any(word in query_lower for word in ['trend', 'growth', 'time', 'period']):
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            insights['temporal'] = {'trend_analysis_available': True, 'date_columns': date_cols}
    
    if any(word in query_lower for word in ['customer', 'user', 'client']):
        customer_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['customer', 'user', 'client', 'id'])]
        if customer_cols:
            insights['customer'] = {
                'unique_customers': df[customer_cols[0]].nunique(),
                'customer_columns': customer_cols
            }
    
    return insights

def _generate_follow_up_questions(query: str, df: pd.DataFrame) -> List[str]:
    """Generate relevant follow-up questions"""
    suggestions = []
    
    # Based on data columns
    if any('revenue' in col.lower() or 'sales' in col.lower() for col in df.columns):
        suggestions.extend([
            "What are my top revenue-generating segments?",
            "Show me revenue trends over time",
            "Which products/services have the highest profit margins?"
        ])
    
    if any('customer' in col.lower() or 'user' in col.lower() for col in df.columns):
        suggestions.extend([
            "What's my customer retention rate?",
            "Who are my most valuable customers?",
            "What are the customer behavior patterns?"
        ])
    
    # General business questions
    suggestions.extend([
        "What are the key performance indicators in my data?",
        "Identify any anomalies or outliers",
        "What recommendations do you have for business growth?"
    ])
    
    return suggestions[:4]  # Return top 4 suggestions

def _detect_business_domain(columns: List[str]) -> str:
    """Detect business domain based on column names"""
    columns_lower = [col.lower() for col in columns]
    
    if any(word in ' '.join(columns_lower) for word in ['revenue', 'sales', 'profit', 'price']):
        return "Sales & Revenue"
    elif any(word in ' '.join(columns_lower) for word in ['customer', 'user', 'client']):
        return "Customer Analytics"
    elif any(word in ' '.join(columns_lower) for word in ['product', 'inventory', 'stock']):
        return "Product Management"
    elif any(word in ' '.join(columns_lower) for word in ['marketing', 'campaign', 'ad']):
        return "Marketing Analytics"
    else:
        return "General Business"

def _calculate_quality_score(profile: Dict) -> float:
    """Calculate data quality score"""
    score = 100.0
    
    # Deduct for missing values
    total_cells = profile['overview']['rows'] * profile['overview']['columns']
    missing_pct = (profile['overview']['missing_values'] / total_cells) * 100
    score -= missing_pct * 2
    
    # Deduct for duplicates
    duplicate_pct = (profile['overview']['duplicate_rows'] / profile['overview']['rows']) * 100
    score -= duplicate_pct * 3
    
    return max(0.0, min(100.0, round(score, 1)))

def _generate_executive_report(analysis_results: Dict, metadata: Dict) -> Dict:
    """Generate executive summary report"""
    return {
        "title": "Executive Data Analytics Report",
        "generated_at": datetime.now().isoformat(),
        "dataset_info": {
            "filename": metadata["filename"],
            "size": f"{metadata['rows']:,} rows × {metadata['columns']} columns",
            "upload_date": metadata["upload_time"].isoformat()
        },
        "executive_summary": analysis_results.get("executive_summary", {}),
        "key_findings": analysis_results.get("business_insights", [])[:5],
        "recommendations": analysis_results.get("recommendations", []),
        "data_quality": {
            "score": analysis_results["executive_summary"]["data_quality_score"],
            "assessment": "Excellent" if analysis_results["executive_summary"]["data_quality_score"] > 90 else "Good" if analysis_results["executive_summary"]["data_quality_score"] > 70 else "Needs Improvement"
        }
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
