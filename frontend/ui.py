import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------- SETUP GEMINI AI -----------------
GEMINI_API_KEY = "AIzaSyBhjd38FR2xIOWml7JBFYCixngF1nkQ5zQ"  # 🔥 Replace with actual key
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
                    # Get AI-generated response from Gemini
                    try:
                        model = genai.GenerativeModel("gemini-pro")
                        gemini_response = model.generate_content(question).text
                    except Exception as e:
                        gemini_response = f"⚠️ Gemini AI Error: {str(e)}"

                    # Dataset Analysis - Answer based on CSV
                    dataset_response = "🔍 Searching dataset..."
                    try:
                        numeric_cols = df.select_dtypes(include=["number"]).columns
                        categorical_cols = df.select_dtypes(include=["object"]).columns

                        # Predefined responses based on keywords
                        if "highest profit" in question.lower():
                            if "profit" in df.columns:
                                max_profit = df["profit"].max()
                                dataset_response = f"💰 The highest profit is **{max_profit}**."
                            elif "revenue" in df.columns and "cost" in df.columns:
                                df["profit"] = df["revenue"] - df["cost"]
                                max_profit = df["profit"].max()
                                dataset_response = f"💰 Calculated profit: The highest profit is **{max_profit}**."
                            else:
                                dataset_response = "⚠️ Profit data not available."

                        elif "average profit" in question.lower():
                            if "profit" in df.columns:
                                avg_profit = df["profit"].mean()
                                dataset_response = f"💰 The average profit is **{avg_profit:.2f}**."
                            elif "revenue" in df.columns and "cost" in df.columns:
                                df["profit"] = df["revenue"] - df["cost"]
                                avg_profit = df["profit"].mean()
                                dataset_response = f"💰 Calculated average profit: **{avg_profit:.2f}**."
                            else:
                                dataset_response = "⚠️ Profit data not available."

                        elif "highest sales" in question.lower():
                            if "product" in df.columns and "sales" in df.columns:
                                top_product = df.loc[df["sales"].idxmax(), "product"]
                                max_sales = df["sales"].max()
                                dataset_response = f"🏆 The product with the highest sales is **{top_product}** with **{max_sales}** sales."
                            else:
                                dataset_response = "⚠️ Sales data not available."

                        elif "most common" in question.lower():
                            if len(categorical_cols) > 0:
                                most_common = df[categorical_cols[0]].value_counts().idxmax()
                                dataset_response = f"🔍 The most frequent category in column '{categorical_cols[0]}' is **{most_common}**."
                            else:
                                dataset_response = "⚠️ No categorical data available."

                        else:
                            dataset_response = "⚠️ Unable to find a dataset-specific answer. Try a different question."

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