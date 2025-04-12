# Use Python 3.9 base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your project to the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose necessary ports
EXPOSE 8000 8501

# Start both FastAPI and Streamlit using a process manager
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port 8000 & streamlit run frontend/ui.py --server.port 8501 --server.address 0.0.0.0"]
