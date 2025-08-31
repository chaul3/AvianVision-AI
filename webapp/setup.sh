#!/bin/bash

# Bird Species Identification Web App Setup Script

echo "🐦 Setting up Bird Species Identification Web App..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p static/uploads
mkdir -p templates

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the web app:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
echo "   2. Run the app: python app.py"
echo "   3. Open your browser to: http://localhost:5000"
echo ""
echo "📝 The app will start on http://localhost:5000"
echo "   Upload a bird image and get instant species identification!"
