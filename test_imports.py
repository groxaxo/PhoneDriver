#!/usr/bin/env python3
"""
Simple test to verify import behavior in different scenarios.
"""

import sys

def test_imports():
    """Test that imports work correctly."""
    
    print("Test 1: Basic import of qwen_vl_agent module")
    try:
        import qwen_vl_agent
        print("✓ Module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import module: {e}")
        return False
    
    print("\nTest 2: Import QwenVLAgent class")
    try:
        from qwen_vl_agent import QwenVLAgent
        print("✓ QwenVLAgent class imported successfully")
    except Exception as e:
        print(f"✗ Failed to import QwenVLAgent: {e}")
        return False
    
    print("\nTest 3: Check conditional imports")
    print(f"  - torch available: {qwen_vl_agent.torch is not None}")
    print(f"  - transformers available: {qwen_vl_agent.Qwen3VLForConditionalGeneration is not None}")
    print(f"  - process_vision_info available: {qwen_vl_agent.process_vision_info is not None}")
    print(f"  - requests available: {qwen_vl_agent.requests is not None}")
    
    print("\nTest 4: API provider initialization (no models loaded)")
    try:
        agent = QwenVLAgent(
            provider="api",
            model_name="test-model",
            api_base_url="https://api.example.com/v1",
            api_key="test-key"
        )
        print("✓ API provider initialized without loading models")
        print(f"  - Provider: {agent.provider}")
        print(f"  - Model name: {agent.model_name}")
        print(f"  - Model object: {agent.model}")
        print(f"  - Processor object: {agent.processor}")
    except Exception as e:
        print(f"✗ Failed to initialize API provider: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All import tests passed!")
    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
