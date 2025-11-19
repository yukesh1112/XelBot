"""
XelBot Professional Data Analytics Chatbot Launcher
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def install_requirements():
    """Install enhanced requirements"""
    print("🔧 Installing enhanced requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_enhanced.txt"], 
                      check=True, capture_output=True)
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def start_backend():
    """Start the enhanced backend server"""
    print("🚀 Starting XelBot Backend API...")
    try:
        # Start backend in background
        backend_process = subprocess.Popen([
            sys.executable, "enhanced_backend.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if process is still running
        if backend_process.poll() is None:
            print("✅ Backend API started successfully on http://127.0.0.1:8000")
            return backend_process
        else:
            print("❌ Backend failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the enhanced frontend"""
    print("🎨 Starting XelBot Frontend...")
    try:
        # Start Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "enhanced_ui.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 XelBot shutdown requested by user")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🤖 XelBot Professional Data Analytics Chatbot")
    print("   Replace expensive data analysts with AI!")
    print("=" * 60)
    
    # Check if enhanced files exist
    required_files = ["enhanced_ui.py", "enhanced_backend.py", "advanced_analytics.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all enhanced files are in the current directory.")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually.")
        return
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Cannot start without backend. Exiting.")
        return
    
    try:
        # Start frontend (this will block)
        start_frontend()
    finally:
        # Cleanup: terminate backend when frontend closes
        if backend_process and backend_process.poll() is None:
            print("🛑 Shutting down backend...")
            backend_process.terminate()
            backend_process.wait()
        
        print("👋 XelBot shutdown complete!")

if __name__ == "__main__":
    main()
