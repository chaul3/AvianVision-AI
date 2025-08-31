#!/usr/bin/env python3
"""
Real Bird Identification App Launcher

This script starts the web application with the actual ResNet50-based
bird identification pipeline instead of the mock demo.
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import torch
        import torchvision
        import flask
        from PIL import Image
        import numpy as np
        print("✅ All dependencies found!")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  Running on CPU (CUDA not available)")
        
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip3 install -r requirements.txt")
        return False

def setup_environment():
    """Setup the environment for the real app."""
    # Create necessary directories
    upload_dir = Path("static/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Upload directory ready: {upload_dir}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"ℹ️  Available memory: {memory.available / (1024**3):.1f} GB")
        
        if memory.available < 2 * (1024**3):  # Less than 2GB
            print("⚠️  Warning: Low memory detected. Consider using demo mode.")
    except ImportError:
        pass

def main():
    """Main launcher function."""
    print("🐦 Real Bird Identification App")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Import and start the real app
    print("\n🚀 Starting real bird identification pipeline...")
    print("📍 Open your browser to: http://localhost:5001")
    print("🧠 Real Mode: Using ResNet50 + attribute analysis")
    print("💡 Upload bird images for actual AI-powered identification!")
    print("-" * 60)
    
    # Import the app
    from app import app
    
    # Start the real app
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\n👋 Shutting down bird identification app...")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("Try the demo version: python3 demo_app.py")

if __name__ == '__main__':
    main()
