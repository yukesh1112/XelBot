# 🤖 XelBot – AI-Powered Data Analytics Chatbot

XelBot is an AI-powered data analytics assistant designed to help business owners, shopkeepers, and analysts make sense of their data. Upload a dataset and chat with XelBot in natural language to get insights, visualizations, and statistics — instantly.

---

## 🚀 Features

- 🧠 Chat-based interface for natural language queries  
- 📊 Automated data insights & visualizations  
- 📂 Upload CSV datasets and explore them easily  
- 🧮 Supports summary statistics, filtering, grouping, and more  
- 🔌 Built with Python, FastAPI, and DeepSeek for intelligent query handling  
- 💼 Designed for business use with Point-of-Sale (POS) integration vision

---

## 📁 Project Structure

xelbot/ │ ├── backend/ │ ├── chatbot.py # FastAPI backend for chatbot │ ├── data_processor.py # Data handling and analysis │ ├── ui/ │ └── ui.py # Frontend user interface │ ├── datasets/ # Sample datasets (user-uploaded or static) ├── README.md # You're reading it :) └── requirements.txt # Python dependencies


---

## ⚙️ Tech Stack

- **Frontend:** Python UI (Tkinter / Streamlit / Custom HTML - update accordingly)  
- **Backend:** FastAPI  
- **AI Model:** DeepSeek or OpenAI for NL-to-Query conversion  
- **Data Processing:** Pandas, Matplotlib, Seaborn  
- **Future Scope:** POS System Integration, Dashboard View, SQL Query Export

---

## 🧪 Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/yukesh1112/XelBot.git
   cd XelBot
2.Install dependencies

pip install -r requirements.txt

3.Run the backend

uvicorn backend.chatbot:app --reload

4.Run the frontend

python ui/ui.py

![xelbot_logo](https://github.com/user-attachments/assets/954695c4-83b2-481f-a2d3-ac9a4fd6e226)

Xelbot Interface :

![Screenshot (150)](https://github.com/user-attachments/assets/37c74f28-286c-4072-b8ad-4cce0e503aa0)



