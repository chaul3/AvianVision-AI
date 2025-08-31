#!/usr/bin/env python3
"""
Quick test script for the fast Llama 3 integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from huggingface_hub import InferenceClient
    print("✅ Hugging Face Hub imported successfully")
    
    # Test the API connection
    client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", timeout=10)
    print("✅ Inference client created")
    
    # Quick test
    response = client.text_generation(
        "Identify this bird: small songbird with red breast",
        max_new_tokens=50,
        return_full_text=False
    )
    print(f"✅ API test successful!")
    print(f"Sample response: {response[:100]}...")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Note: Hugging Face API may require authentication or have rate limits")
    print("The app will fallback to mock mode if API is unavailable")

print("\n🚀 Ready to start the app with fast Llama 3!")
print("Run: python3 app.py")
