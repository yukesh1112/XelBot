"""
Enhanced Professional UI for XelBot Data Analytics Chatbot
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from advanced_analytics import AdvancedAnalytics, BusinessIntelligence
import json
from datetime import datetime

def generate_ai_response(question, df):
    """Generate intelligent AI response with data context"""
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        
        # Create comprehensive data context
        analytics = AdvancedAnalytics(df)
        profile = analytics.data_profiling()
        insights = analytics.business_insights()
        
        # Get basic data info
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        context = f"""
You are XelBot, a professional data analytics consultant. Analyze the user's question about their business data.

DATASET CONTEXT:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Numeric Columns: {', '.join(numeric_cols) if numeric_cols else 'None'}
- Categorical Columns: {', '.join(categorical_cols) if categorical_cols else 'None'}
- Data Types: {dict(df.dtypes)}
- Sample Data:
{df.head(3).to_string()}

DATA QUALITY:
- Missing Values: {df.isnull().sum().sum()}
- Duplicate Rows: {df.duplicated().sum()}

BUSINESS INSIGHTS:
{chr(10).join(insights[:5]) if insights else 'No insights generated yet'}

USER QUESTION: {question}

INSTRUCTIONS:
1. Act as a professional data analyst consultant
2. Provide specific, actionable insights based on the actual data
3. Include relevant statistics and metrics
4. Give business recommendations when appropriate
5. Be conversational but professional
6. If the question requires visualization, suggest what charts would be helpful
7. Focus on business value and ROI

Response:"""
        
        response = model.generate_content(context)
        return response.text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Feature Correlations")
        st.plotly_chart(fig, use_container_width=True)

def generate_executive_report(results):
    """Generate professional executive report"""
    # Prefer recomputing from the current dataframe to ensure fresh, complete info
    df = st.session_state.get("current_df")

    exec_summary = {}
    dataset_size_str = "Unknown (run complete analysis to populate dataset details)"
    memory_usage_str = "Unknown"
    data_quality_score = 0
    key_insights = []
    recommendations = []

    if df is not None:
        try:
            bi = BusinessIntelligence(df)
            exec_summary = bi.executive_summary()

            data_overview = exec_summary.get('data_overview', {})
            rows = data_overview.get('rows')
            columns = data_overview.get('columns')
            memory_usage_str = data_overview.get('memory_usage', memory_usage_str)

            if rows is not None and columns is not None:
                try:
                    dataset_size_str = f"{int(rows):,} rows, {int(columns)} columns"
                except Exception:
                    pass

            data_quality_score = exec_summary.get('data_quality_score', 0)

            analytics = AdvancedAnalytics(df)
            full_insights = analytics.business_insights()
            full_recs = analytics.generate_recommendations()

            key_insights = exec_summary.get('key_insights', full_insights[:5])
            recommendations = exec_summary.get('recommendations', full_recs[:3])
        except Exception:
            exec_summary = {}

    # Fallback to whatever is in results if recomputation failed or df is missing
    if not exec_summary and isinstance(results, dict):
        exec_summary = results.get('executive_summary', {})
        data_overview = exec_summary.get('data_overview', results.get('profile', {}).get('overview', {})) or {}
        rows = data_overview.get('rows')
        columns = data_overview.get('columns')
        memory_usage = data_overview.get('memory_usage')

        if rows is not None and columns is not None:
            try:
                dataset_size_str = f"{int(rows):,} rows, {int(columns)} columns"
            except Exception:
                pass
        memory_usage_str = memory_usage if memory_usage is not None else memory_usage_str
        data_quality_score = exec_summary.get('data_quality_score', data_quality_score)

        if not key_insights:
            key_insights = exec_summary.get('key_insights', results.get('insights', []))
        if not recommendations:
            recommendations = exec_summary.get('recommendations', results.get('recommendations', []))

    report = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
        <h1 style="color: #667eea; text-align: center;">📊 XelBot Analytics Report</h1>
        <p style="text-align: center; color: #666;">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        
        <h2>📋 Executive Summary</h2>
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p><strong>Data Quality Score:</strong> {data_quality_score}%</p>
            <p><strong>Dataset Size:</strong> {dataset_size_str}</p>
            <p><strong>Memory Usage:</strong> {memory_usage_str}</p>
        </div>
        
        <h2>💡 Key Insights</h2>
        <ul>
    """
    
    for insight in key_insights:
        report += f"<li>{insight}</li>"
    
    report += """
        </ul>
        
        <h2>🎯 Strategic Recommendations</h2>
        <ol>
    """
    
    for rec in recommendations:
        report += f"<li>{rec}</li>"
    
    report += """
        </ol>
        
        <div style="margin-top: 2rem; padding: 1rem; background: #e3f2fd; border-radius: 8px;">
            <p><strong>🤖 Generated by XelBot</strong> - Your AI-Powered Data Analytics Assistant</p>
            <p>This report provides actionable insights to help drive your business decisions.</p>
        </div>
    </div>
    """
    
    return report

# Configure Streamlit page
st.set_page_config(
    page_title="XelBot - Professional Data Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Gemini AI
GEMINI_API_KEY = "AIzaSyA0ePTaWI-Up6oTuG_R_B-Pvave0UwumjM"
genai.configure(api_key=GEMINI_API_KEY)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .recommendation-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #e5e7eb;
    }
    .user-message {
        background: rgba(37, 99, 235, 0.35);
        border-left: 4px solid #60a5fa;
    }
    .bot-message {
        background: rgba(124, 58, 237, 0.35);
        border-left: 4px solid #a855f7;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with default values
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {
            'profile': {},
            'insights': [],
            'predictions': {},
            'recommendations': [],
            'executive_summary': {
                'data_quality_score': 0,
                'key_metrics': {},
                'top_insights': []
            }
        }
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None

# Initialize the session state
init_session_state()

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 XelBot - Professional Data Analytics Chatbot</h1>
    <p>Your AI-Powered Data Analyst | Replace expensive consultants with intelligent automation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 Analytics Dashboard")
    
    # File upload
    st.subheader("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file for analysis",
        type=['csv'],
        help="Upload your business data for comprehensive analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Show loading state
            with st.spinner('📥 Loading and processing your data...'):
                # Read the file with error handling for different encodings
                try:
                    df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    # Try with different encodings if default fails
                    encodings = ['utf-8', 'latin1', 'windows-1252', 'iso-8859-1']
                    for encoding in encodings:
                        try:
                            uploaded_file.seek(0)  # Reset file pointer
                            df = pd.read_csv(uploaded_file, encoding=encoding)
                            break
                        except:
                            continue
                    else:
                        raise Exception("Could not read file with any standard encoding. Please save the file as UTF-8 and try again.")
                
                # Basic data validation
                if df.empty:
                    raise ValueError("The uploaded file is empty.")
                
                # Clean column names (remove leading/trailing spaces)
                df.columns = df.columns.str.strip()
                
                # Store the dataframe in session state
                st.session_state.current_df = df
                
                # Clear any previous analysis results (reset to default structure)
                if not ('analysis_results' in st.session_state and st.session_state.analysis_results.get('profile', {}).get('overview')):
                    st.session_state.analysis_results = {
                        'profile': {},
                        'insights': [],
                        'predictions': {},
                        'recommendations': [],
                        'executive_summary': {
                            'data_quality_score': 0,
                            'key_metrics': {},
                            'top_insights': []
                        }
                    }
                
                st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                
                # Quick data preview
                st.subheader("👀 Data Preview")
                st.dataframe(df.head(3), use_container_width=True)
                
                # Show basic info
                with st.expander("📋 Dataset Information"):
                    st.write(f"**File Name:** {uploaded_file.name}")
                    st.write(f"**Total Rows:** {len(df):,}")
                    st.write(f"**Total Columns:** {len(df.columns)}")
                    st.write("\n**Column Types:**")
                    st.json(df.dtypes.astype(str).to_dict())
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.error("Please check that your file is a valid CSV and try again.")
            st.info("💡 Tip: Try opening and saving your file with a different application to ensure it's properly formatted.")
            # Clear any problematic state
            if 'current_df' in st.session_state:
                del st.session_state.current_df
    
    # Analysis options
    if st.session_state.current_df is not None:
        st.subheader("🔍 Analysis Options")
        
        if st.button("🚀 Run Complete Analysis", use_container_width=True):
            with st.spinner("Analyzing your data..."):
                analytics = AdvancedAnalytics(st.session_state.current_df)
                bi = BusinessIntelligence(st.session_state.current_df)
                
                st.session_state.analysis_results = {
                    'profile': analytics.data_profiling(),
                    'insights': analytics.business_insights(),
                    'predictions': analytics.predictive_analysis(),
                    'recommendations': analytics.generate_recommendations(),
                    'executive_summary': bi.executive_summary()
                }
            st.success("Analysis complete!")
        
        # Quick stats
        if 'analysis_results' in st.session_state and st.session_state.analysis_results:
            st.subheader("📈 Quick Stats")
            results = st.session_state.analysis_results
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data Quality", f"{results['executive_summary']['data_quality_score']}%")
            with col2:
                st.metric("Insights Found", len(results['insights']))

# Main content area
if st.session_state.current_df is None:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ## 🎯 Welcome to XelBot
        
        **Your Problem**: Need data insights but can't afford expensive data analysts?
        
        **Our Solution**: AI-powered analytics chatbot that provides professional-grade analysis
        
        ### 🚀 What XelBot Can Do:
        - **📊 Comprehensive Data Analysis** - Statistical profiling and quality assessment
        - **💡 Business Insights** - Revenue, growth, and performance analysis
        - **🤖 Natural Language Queries** - Ask questions in plain English
        - **📈 Predictive Analytics** - Forecasting and trend analysis
        - **📋 Professional Reports** - Export-ready business intelligence
        - **🎯 Actionable Recommendations** - Data-driven business advice
        
        ### 📁 Get Started:
        1. Upload your CSV data file
        2. Run complete analysis or ask specific questions
        3. Get professional insights instantly
        
        **Replace your data analyst today!** 🚀
        """)

else:
    # Main dashboard with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Analysis", "📊 Dashboard", "📈 Insights", "📋 Reports"])
    
    with tab1:
        st.subheader("💬 Chat with Your Data")
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        for chat in st.session_state.chat_history:
            if chat['type'] == 'user':
                st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong> {chat["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">🤖 <strong>XelBot:</strong> {chat["message"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        user_question = st.text_input(
            "Ask me anything about your data:",
            placeholder="e.g., 'What are my top performing products?' or 'Show me revenue trends'"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🚀 Ask XelBot", use_container_width=True):
                if user_question:
                    # Add user message to history
                    st.session_state.chat_history.append({
                        'type': 'user',
                        'message': user_question,
                        'timestamp': datetime.now()
                    })
                    
                    # Generate AI response
                    with st.spinner("Analyzing..."):
                        response = generate_ai_response(user_question, st.session_state.current_df)
                    
                    # Add bot response to history
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'message': response,
                        'timestamp': datetime.now()
                    })
                    
                    st.rerun()
        
        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Suggested questions
        st.subheader("💡 Suggested Questions")
        suggestions = [
            "Analyze my data and give me key insights",
            "What are the trends in my data?",
            "Show me the most important metrics",
            "What recommendations do you have for my business?",
            "Identify any data quality issues",
            "What patterns do you see in my data?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                    st.session_state.chat_history.append({
                        'type': 'user',
                        'message': suggestion,
                        'timestamp': datetime.now()
                    })
                    
                    with st.spinner("Analyzing..."):
                        response = generate_ai_response(suggestion, st.session_state.current_df)
                    
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'message': response,
                        'timestamp': datetime.now()
                    })
                    
                    st.rerun()
    
    with tab2:
        st.subheader("📊 Data Dashboard")
        
        if st.session_state.current_df is not None:
            # Display basic dataset info
            st.subheader("📋 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Data Quality", f"{st.session_state.analysis_results.get('executive_summary', {}).get('data_quality_score', 0)}%")
            with col2:
                st.metric("📈 Total Rows", f"{len(st.session_state.current_df):,}")
            with col3:
                st.metric("📋 Columns", len(st.session_state.current_df.columns))
            with col4:
                st.metric("💾 Size", f"{(st.session_state.current_df.memory_usage(deep=True).sum() / (1024 * 1024)):.2f} MB")
            
            # Display visualizations
            st.subheader("📈 Data Visualizations")
            try:
                # Simple visualization of the first few numeric columns
                numeric_cols = st.session_state.current_df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) > 0:
                    # Show basic statistics
                    st.write("### Basic Statistics")
                    st.dataframe(st.session_state.current_df.describe())
                    
                    # Show distribution of first numeric column
                    st.write(f"### Distribution of {numeric_cols[0]}")
                    fig = px.histogram(st.session_state.current_df, x=numeric_cols[0], 
                                     title=f"Distribution of {numeric_cols[0]}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # If there are at least 2 numeric columns, show scatter plot
                    if len(numeric_cols) >= 2:
                        st.write(f"### {numeric_cols[0]} vs {numeric_cols[1]}")
                        fig = px.scatter(st.session_state.current_df, x=numeric_cols[0], y=numeric_cols[1],
                                       title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns found for visualization.")
                    
            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")
                st.warning("Some visualizations could not be generated. The data might need cleaning.")

            # Executive summary cards (only if analysis results are available)
            if 'analysis_results' in st.session_state and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                profile_overview = results.get('profile', {}).get('overview', {})
                rows = profile_overview.get('rows', len(st.session_state.current_df))
                columns = profile_overview.get('columns', len(st.session_state.current_df.columns))
                memory_usage = profile_overview.get(
                    'memory_usage',
                    f"{(st.session_state.current_df.memory_usage(deep=True).sum() / (1024 * 1024)):.2f} MB"
                )

                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Data Quality</h3>
                        <h2>{results['executive_summary']['data_quality_score']}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📈 Total Rows</h3>
                        <h2>{rows:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📋 Columns</h3>
                        <h2>{columns}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>💾 Size</h3>
                        <h2>{memory_usage}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data visualization section is now handled in the tab2 section above
                st.info("👆 Check the 'Dashboard' tab to view data visualizations")
        
        else:
            st.info("👆 Upload and analyze a dataset first to see the dashboard")
    
    with tab3:
        st.subheader("💡 Business Insights")
        
        results = st.session_state.analysis_results if 'analysis_results' in st.session_state else None
        has_insights = bool(results and results.get('insights'))
        has_recommendations = bool(results and results.get('recommendations'))
        has_predictions = bool(results and results.get('predictions', {}).get('correlations'))

        if not (has_insights or has_recommendations or has_predictions):
            st.info("No insights yet. Click 'Run Complete Analysis' in the sidebar to generate business insights.")
        else:
            key_tab, rec_tab, pred_tab = st.tabs(["🔍 Key Insights", "🎯 Recommendations", "� Predictive Insights"])

            with key_tab:
                if has_insights:
                    for idx, insight in enumerate(results['insights'], start=1):
                        st.markdown(f'<div class="insight-box"><strong>Insight {idx}:</strong> {insight}</div>', unsafe_allow_html=True)
                else:
                    st.info("No key insights were generated yet.")

            with rec_tab:
                if has_recommendations:
                    for idx, rec in enumerate(results['recommendations'], start=1):
                        st.markdown(f'<div class="recommendation-box"><strong>Recommendation {idx}:</strong> {rec}</div>', unsafe_allow_html=True)
                else:
                    st.info("No recommendations are available yet.")

            with pred_tab:
                if has_predictions:
                    st.markdown("### Correlation-based Predictive Insights")
                    for corr in results['predictions']['correlations']:
                        st.write(f"**{corr['variables']}**: Correlation = {corr['correlation']}")
                else:
                    st.info("No strong predictive correlations were detected yet.")
    
    with tab4:
        st.subheader("📋 Professional Reports")
        
        if st.session_state.current_df is None:
            st.info("Upload a dataset in the sidebar to generate an executive report.")
        else:
            results = st.session_state.get('analysis_results', {})

            # Generate report directly from the current dataframe (with results as optional context)
            if st.button("📄 Generate Executive Report"):
                report = generate_executive_report(results)
                st.markdown(report, unsafe_allow_html=True)
                
                # Download option
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"xelbot_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )

if __name__ == "__main__":
    st.write("XelBot Professional Data Analytics Chatbot is running!")
