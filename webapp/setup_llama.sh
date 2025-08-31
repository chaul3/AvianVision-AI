#!/bin/bash
# Llama 3 Setup Script for Bird Identification App

echo "🦙 Setting up Llama 3 for Bird Identification"
echo "============================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Installing..."
    brew install ollama
fi

echo "✅ Ollama found at: $(which ollama)"

# Start Ollama service in background
echo "🚀 Starting Ollama service..."
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# Wait for service to start
echo "⏳ Waiting for Ollama to start..."
sleep 3

# Pull Llama 3 model
echo "📥 Downloading Llama 3 model (this may take 5-10 minutes)..."
echo "💡 Model size: ~4.7GB - ensure you have enough disk space"

if ollama pull llama3; then
    echo "✅ Llama 3 downloaded successfully!"
else
    echo "❌ Failed to download Llama 3. Trying Llama 3.1..."
    if ollama pull llama3.1; then
        echo "✅ Llama 3.1 downloaded successfully!"
    else
        echo "⚠️ Could not download Llama models. The app will use fallback LLM."
        echo "📝 You can manually try: ollama pull llama3"
    fi
fi

# Test the model
echo "🧪 Testing Llama model..."
TEST_RESPONSE=$(ollama generate llama3 "What is a bird?" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✅ Llama 3 is working correctly!"
    echo "🎯 Sample response: ${TEST_RESPONSE:0:100}..."
else
    echo "⚠️ Llama test failed, but the model may still work"
fi

echo ""
echo "🎉 Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔹 Ollama Service: Running (PID: $OLLAMA_PID)"
echo "🔹 Llama Model: Ready for bird identification"
echo "🔹 API Endpoint: http://localhost:11434"
echo ""
echo "🚀 Ready to start your bird identification app!"
echo "   Run: python3 app.py"
echo ""
echo "💡 To manually stop Ollama later: pkill ollama"
