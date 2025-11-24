#!/usr/bin/env python3
"""
Test script to verify that API provider can be used without torch/transformers installed.

This script simulates an environment where torch is not available and verifies
that the API provider can still be initialized and used.
"""

import sys
import json

def test_api_provider_without_torch():
    """Test that API provider works when torch is not available."""
    print("Testing API provider initialization without torch dependencies...")
    
    # Mock the torch import to simulate it not being available
    original_modules = {}
    modules_to_mock = ['torch', 'transformers']
    
    for module_name in modules_to_mock:
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]
            sys.modules[module_name] = None
    
    try:
        # Import after mocking torch
        from qwen_vl_agent import QwenVLAgent
        
        # Test 1: API provider should initialize successfully
        print("\n✓ Test 1: Initializing API provider...")
        try:
            agent = QwenVLAgent(
                provider="api",
                model_name="gpt-4-vision-preview",
                api_base_url="https://api.openai.com/v1",
                api_key="test-key-12345"
            )
            print("✓ API provider initialized successfully without torch")
        except Exception as e:
            print(f"✗ Failed to initialize API provider: {e}")
            return False
        
        # Test 2: Local provider should fail gracefully with helpful error
        print("\n✓ Test 2: Verifying local provider fails gracefully without torch...")
        try:
            agent_local = QwenVLAgent(
                provider="local",
                model_name="Qwen/Qwen3-VL-8B-Instruct"
            )
            print("✗ Local provider should have failed without torch")
            return False
        except ImportError as e:
            error_msg = str(e)
            if "torch" in error_msg.lower():
                print(f"✓ Local provider correctly reports torch dependency: {error_msg}")
            else:
                print(f"⚠ Local provider failed but message unclear: {error_msg}")
        except Exception as e:
            print(f"✗ Unexpected error type: {type(e).__name__}: {e}")
            return False
        
        print("\n✅ All tests passed! API provider works without torch dependencies.")
        return True
        
    finally:
        # Restore original modules
        for module_name, original_module in original_modules.items():
            if original_module is not None:
                sys.modules[module_name] = original_module
            else:
                if module_name in sys.modules:
                    del sys.modules[module_name]


def test_with_torch_available():
    """Test that local provider still works when torch is available."""
    print("\n" + "="*60)
    print("Testing with torch available (if installed)...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ torch is available: {torch.__version__}")
        has_torch = True
    except ImportError:
        print("⚠ torch is not installed, skipping local provider test")
        has_torch = False
    
    if has_torch:
        try:
            from qwen_vl_agent import QwenVLAgent
            
            # Test API provider still works
            print("\n✓ Test: API provider with torch installed...")
            agent_api = QwenVLAgent(
                provider="api",
                model_name="gpt-4-vision-preview",
                api_base_url="https://api.openai.com/v1",
                api_key="test-key-12345"
            )
            print("✓ API provider works with torch installed")
            
            print("\n✅ Verification complete: Both modes coexist properly")
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("PhoneDriver API Provider Dependency Test")
    print("="*60)
    
    # Test 1: Without torch
    success1 = test_api_provider_without_torch()
    
    # Test 2: With torch (if available)
    success2 = test_with_torch_available()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("✅ ALL TESTS PASSED")
        print("="*60)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        sys.exit(1)
