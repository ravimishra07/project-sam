#!/usr/bin/env python3
"""
Test script to verify LM Studio connection
"""

import requests
import json

def test_lmstudio_connection():
    """Test if LM Studio is running and accessible"""
    try:
        # Test models endpoint
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ LM Studio is running!")
            print(f"Available models: {models}")
            return True
        else:
            print(f"❌ LM Studio responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to LM Studio")
        print("💡 Make sure:")
        print("   1. LM Studio app is open")
        print("   2. A model is loaded")
        print("   3. Local server is started (click 'Start Server')")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_chat_completion():
    """Test a simple chat completion"""
    try:
        url = "http://localhost:1234/v1/chat/completions"
        payload = {
            "model": "local-model",  # This is the default model name
            "messages": [
                {"role": "user", "content": "Hello! Can you respond with just 'Hello from LM Studio'?"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ Chat test successful!")
            print(f"Response: {content}")
            return True
        else:
            print(f"❌ Chat test failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat test error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing LM Studio connection...")
    print("=" * 40)
    
    if test_lmstudio_connection():
        print("\nTesting chat completion...")
        test_chat_completion()
    else:
        print("\nPlease start LM Studio first!") 