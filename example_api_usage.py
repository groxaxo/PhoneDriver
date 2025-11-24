#!/usr/bin/env python3
"""
Example script demonstrating how to use PhoneDriver with an external API provider.

This example shows that no NVIDIA dependencies (torch, CUDA) are required when
using external API providers like OpenAI, Azure OpenAI, or local inference servers.

Requirements:
- pillow
- gradio
- requests

No torch or transformers required!
"""

import json

def example_api_configuration():
    """Example configuration for API provider."""
    
    print("="*60)
    print("PhoneDriver with External API Provider")
    print("="*60)
    print("\nThis example demonstrates using PhoneDriver without NVIDIA dependencies.")
    print("No torch, CUDA, or transformers required!\n")
    
    # Example 1: OpenAI API
    openai_config = {
        "provider": "api",
        "model_name": "gpt-4-vision-preview",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "your-openai-api-key-here",
        "temperature": 0.1,
        "max_tokens": 512,
        "step_delay": 1.5,
        "screen_width": 1080,
        "screen_height": 2340
    }
    
    print("Example 1: OpenAI API Configuration")
    print("-" * 40)
    print(json.dumps(openai_config, indent=2))
    
    # Example 2: Local inference server (vLLM, Text Generation WebUI, etc.)
    local_server_config = {
        "provider": "api",
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "api_base_url": "http://localhost:8000/v1",
        "api_key": "not-needed-for-local",
        "temperature": 0.1,
        "max_tokens": 512,
        "step_delay": 1.5,
        "screen_width": 1080,
        "screen_height": 2340
    }
    
    print("\n\nExample 2: Local Inference Server Configuration")
    print("-" * 40)
    print(json.dumps(local_server_config, indent=2))
    
    # Example 3: Azure OpenAI
    azure_config = {
        "provider": "api",
        "model_name": "gpt-4-vision",
        "api_base_url": "https://your-resource.openai.azure.com/openai/deployments/your-deployment/v1",
        "api_key": "your-azure-api-key",
        "temperature": 0.1,
        "max_tokens": 512,
        "step_delay": 1.5,
        "screen_width": 1080,
        "screen_height": 2340
    }
    
    print("\n\nExample 3: Azure OpenAI Configuration")
    print("-" * 40)
    print(json.dumps(azure_config, indent=2))
    
    print("\n\n" + "="*60)
    print("Usage Instructions:")
    print("="*60)
    print("""
1. Save one of these configurations to config.json
2. Update the api_key with your actual API key
3. Run the UI: python ui.py
4. Or use CLI: python phone_agent.py "your task here"

Benefits of API provider:
✓ No NVIDIA GPU required
✓ No CUDA installation needed
✓ No large model downloads
✓ Works on any machine (Windows, Mac, Linux)
✓ Lower RAM requirements
✓ Can use cloud providers for better vision models
    """)


if __name__ == "__main__":
    example_api_configuration()
