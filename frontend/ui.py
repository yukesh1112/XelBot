import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------- SETUP GEMINI AI -----------------
GEMINI_API_KEY = "AIzaSyA0ePTaWI-Up6oTuG_R_B-Pvave0UwumjM"  # 🔥 Updated API key
genai.configure(api_key=GEMINI_API_KEY)

# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(page_title="XelBot - AI Data Assistant", layout="wide")

# ----------------- STYLING -----------------
st.markdown("""
    <style>
        .main { background-color: #f4f4f4; }
        h1 { color: #007BFF; text-align: center; }
        .stButton button { width: 100%; font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

# ----------------- HEADER SECTION -----------------
st.markdown("<h1>🤖 XelBot - AI Data Assistant</h1>", unsafe_allow_html=True)
st.write("### 📊 Upload a dataset and let AI analyze it for you!")

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "💬 Chatbot", "📈 Data Insights"])

# ----------------- SIDEBAR - CHAT HISTORY -----------------
st.sidebar.subheader("📝 Chat History")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    st.sidebar.write(chat)

# ----------------- FILE UPLOAD SECTION -----------------
st.sidebar.subheader("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# ----------------- GLOBAL DATAFRAME -----------------
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# ----------------- HOME PAGE -----------------
if page == "🏠 Home":
    st.write("### Welcome to XelBot - Your AI-Powered Data Assistant!")
    st.write("""
    - 🔍 Upload your dataset and get instant insights.
    - 🤖 Chat with XelBot to ask AI-powered questions.
    - 📊 Visualize trends and discover valuable patterns.
    """)
    
    if uploaded_file:
        st.success("✅ Dataset uploaded successfully!")

# ----------------- CHATBOT PAGE -----------------
elif page == "💬 Chatbot":
    if df is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        st.write("### 💬 Chat with XelBot")
        question = st.text_input("Ask a question about your data:")

        if st.button("Ask XelBot"):
            if question.strip():
                with st.spinner("Thinking..."):
                    # Get AI-generated response from Gemini with dataset context
                    try:
                        model = genai.GenerativeModel("models/gemini-2.0-flash")
                        
                        # Create dataset summary for AI context
                        dataset_summary = f"""
DATASET CONTEXT:
- Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns)}
- Data types: {dict(df.dtypes)}
"""
                        
                        # Add sample data if dataset is not too large
                        if len(df) <= 1000:
                            dataset_summary += f"\nSample data (first 3 rows):\n{df.head(3).to_string()}"
                        
                        # Create focused prompt
                        enhanced_prompt = f"""
You are XelBot, a data analysis assistant. Analyze the user's question about their specific dataset.

{dataset_summary}

USER QUESTION: {question}

INSTRUCTIONS:
1. Focus ONLY on the uploaded dataset shown above
2. Provide specific insights based on the actual data
3. If the question is about sales/products but the dataset doesn't contain that info, redirect to what the dataset actually contains
4. Be concise and data-focused
5. Suggest relevant questions they could ask about their actual data

Answer:"""
                        
                        gemini_response = model.generate_content(enhanced_prompt).text
                    except Exception as e:
                        gemini_response = f"⚠️ Gemini AI Error: {str(e)}"

                    # Dataset Analysis - Answer based on CSV
                    dataset_response = "🔍 Searching dataset..."
                    try:
                        numeric_cols = df.select_dtypes(include=["number"]).columns
                        categorical_cols = df.select_dtypes(include=["object"]).columns

                        # Enhanced responses for social media content data
                        if "content type" in question.lower() or "types" in question.lower():
                            if "Type" in df.columns:
                                type_counts = df["Type"].value_counts()
                                dataset_response = f"📊 Content types: {dict(type_counts)}"
                            else:
                                dataset_response = "⚠️ Content type data not available."

                        elif "category" in question.lower() or "categories" in question.lower():
                            if "Category" in df.columns:
                                category_counts = df["Category"].value_counts().head(5)
                                dataset_response = f"📈 Top 5 categories: {dict(category_counts)}"
                            else:
                                dataset_response = "⚠️ Category data not available."

                        elif "most common" in question.lower() or "popular" in question.lower():
                            if "Category" in df.columns:
                                most_common = df["Category"].value_counts().idxmax()
                                count = df["Category"].value_counts().iloc[0]
                                dataset_response = f"🏆 Most popular category: **{most_common}** ({count} posts)"
                            elif len(categorical_cols) > 0:
                                most_common = df[categorical_cols[0]].value_counts().idxmax()
                                dataset_response = f"🔍 Most frequent in '{categorical_cols[0]}': **{most_common}**"
                            else:
                                dataset_response = "⚠️ No categorical data available."

                        elif "photo" in question.lower() or "video" in question.lower():
                            if "Type" in df.columns:
                                type_counts = df["Type"].value_counts()
                                dataset_response = f"📸 Content breakdown: {dict(type_counts)}"
                            else:
                                dataset_response = "⚠️ Content type data not available."

                        elif "columns" in question.lower():
                            dataset_response = f"📊 Dataset columns: {', '.join(df.columns)}"

                        elif "rows" in question.lower() or "size" in question.lower():
                            dataset_response = f"📊 Dataset has {len(df)} rows and {len(df.columns)} columns"

                        else:
                            # Smart general analysis based on question keywords
                            insights = []
                            
                            # Analyze based on question content
                            if any(word in question.lower() for word in ["analyze", "summary", "overview", "insights"]):
                                # Comprehensive analysis
                                analysis_parts = []
                                
                                # Content type analysis
                                if "Type" in df.columns:
                                    type_counts = df["Type"].value_counts()
                                    analysis_parts.append(f"Content types: {dict(type_counts)}")
                                
                                # Category analysis  
                                if "Category" in df.columns:
                                    top_categories = df["Category"].value_counts().head(3)
                                    analysis_parts.append(f"Top categories: {dict(top_categories)}")
                                
                                # URL analysis
                                if "URL" in df.columns:
                                    url_count = df["URL"].notna().sum()
                                    analysis_parts.append(f"Content with URLs: {url_count}/{len(df)}")
                                
                                dataset_response = f"📊 Dataset Analysis: {' | '.join(analysis_parts)}"
                            
                            else:
                                # Default insights
                                if "Type" in df.columns:
                                    top_type = df["Type"].value_counts().idxmax()
                                    insights.append(f"Most common content type: {top_type}")
                                if "Category" in df.columns:
                                    top_category = df["Category"].value_counts().idxmax()
                                    insights.append(f"Most popular category: {top_category}")
                                
                                if insights:
                                    dataset_response = f"💡 Quick insights: {' | '.join(insights)}"
                                else:
                                    dataset_response = "⚠️ Try asking: 'analyze my dataset' or 'what insights can you provide?'"

                    except Exception as e:
                        dataset_response = f"⚠️ Dataset Analysis Error: {str(e)}"

                    # Display Responses
                    st.success(f"🤖 **Gemini AI:** {gemini_response}")
                    st.info(f"📊 **Dataset Insight:** {dataset_response}")

                    # Save chat history
                    st.session_state.chat_history.append(f"🧠 You: {question}\n🤖 Gemini: {gemini_response}\n📊 Dataset: {dataset_response}")

            else:
                st.warning("⚠️ Please enter a question!")

# ----------------- DATA INSIGHTS PAGE -----------------
elif page == "📈 Data Insights":
    if df is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        st.write("### 📊 Data Visualization & Insights")

        # Select Column for Analysis
        selected_col = st.selectbox("📊 Select a column to visualize:", df.columns)

        # Show Histogram
        st.subheader("📊 Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

        # Show Correlation Heatmap (If enough numeric data)
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 1:
            st.subheader("📈 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("⚠️ Not enough numeric data for a correlation heatmap.")

        # Top 5 Products by Sales
        if "product" in df.columns and "sales" in df.columns:
            st.subheader("🏆 Top 5 Products by Sales")
            top_products = df.groupby("product")["sales"].sum().nlargest(5)
            st.bar_chart(top_products)

        # Sales by Region
        if "region" in df.columns and "sales" in df.columns:
            st.subheader("🌍 Sales by Region")
            region_sales = df.groupby("region")["sales"].sum()
            st.bar_chart(region_sales)

# ----------------- END -----------------