#!/usr/bin/env python3
"""
Quick launcher for the Bird Species Identification Web App
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import flask
        import torch
        import torchvision
        import PIL
        import numpy
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories."""
    dirs = ['static/uploads', 'templates']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")

def start_app():
    """Start the Flask application."""
    print("🚀 Starting Bird Species Identification Web App...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("-" * 50)
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development'
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

def main():
    """Main launcher function."""
    print("🐦 Bird Species Identification Web App Launcher")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    if not check_dependencies():
        answer = input("🔧 Install dependencies now? (y/n): ")
        if answer.lower() in ['y', 'yes']:
            print("📦 Installing dependencies...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Dependencies installed")
        else:
            print("❌ Cannot start without dependencies")
            return
    
    # Create directories
    create_directories()
    
    # Start app
    start_app()

if __name__ == "__main__":
    main()
