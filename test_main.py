#!/usr/bin/env python3
"""
Test script for steganography functionality.
This will create a test image and verify that hide/extract works properly.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.getcwd())

from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel


def create_test_image(path: Path, width: int = 200, height: int = 200):
    """Create a test PNG image with random colors."""
    # Create random RGB image
    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    img.save(path, 'PNG')
    print(f"Created test image: {path}")


def test_steganography():
    """Test basic steganography functionality."""
    print("=" * 60)
    print("Testing InvisioVault Steganography Engine")
    print("=" * 60)
    
    # Create test files
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    carrier_path = test_dir / "carrier.png"
    stego_path = test_dir / "stego.png"
    test_data = b"Hello, this is a test message for steganography!"
    
    # Create test image
    create_test_image(carrier_path)
    
    # Initialize steganography engine
    stego_engine = SteganographyEngine()
    
    print(f"\n1. Testing image validation...")
    if not stego_engine.validate_image_format(carrier_path):
        print("❌ Image validation failed!")
        return False
    print("✅ Image validation passed")
    
    print(f"\n2. Testing capacity calculation...")
    capacity = stego_engine.calculate_capacity(carrier_path)
    print(f"✅ Image capacity: {capacity} bytes ({capacity/1024:.2f} KB)")
    
    if capacity < len(test_data):
        print(f"❌ Image too small for test data ({len(test_data)} bytes)")
        return False
    
    # Test 1: Sequential hiding/extraction (no randomization)
    print(f"\n3. Testing sequential hide/extract...")
    success = stego_engine.hide_data(carrier_path, test_data, stego_path, randomize=False)
    if not success:
        print("❌ Hide operation failed!")
        return False
    print("✅ Hide operation successful")
    
    extracted_data = stego_engine.extract_data(stego_path, randomize=False)
    if extracted_data is None:
        print("❌ Extract operation failed!")
        return False
    
    if extracted_data != test_data:
        print("❌ Extracted data doesn't match original!")
        print(f"Original: {test_data}")
        print(f"Extracted: {extracted_data}")
        return False
    print("✅ Sequential extraction successful")
    
    # Test 2: Randomized hiding/extraction 
    print(f"\n4. Testing randomized hide/extract...")
    stego_path_rand = test_dir / "stego_rand.png"
    test_seed = 12345
    
    success = stego_engine.hide_data(carrier_path, test_data, stego_path_rand, 
                                   randomize=True, seed=test_seed)
    if not success:
        print("❌ Randomized hide operation failed!")
        return False
    print("✅ Randomized hide operation successful")
    
    extracted_data_rand = stego_engine.extract_data(stego_path_rand, 
                                                  randomize=True, seed=test_seed)
    if extracted_data_rand is None:
        print("❌ Randomized extract operation failed!")
        return False
    
    if extracted_data_rand != test_data:
        print("❌ Randomized extracted data doesn't match original!")
        print(f"Original: {test_data}")
        print(f"Extracted: {extracted_data_rand}")
        return False
    print("✅ Randomized extraction successful")
    
    # Test 3: Try extracting with wrong seed (should fail)
    print(f"\n5. Testing wrong seed extraction (should fail)...")
    try:
        wrong_extracted = stego_engine.extract_data(stego_path_rand, 
                                                  randomize=True, seed=54321)
        if wrong_extracted is not None and wrong_extracted == test_data:
            print("❌ Wrong seed extraction should have failed!")
            return False
        print("✅ Wrong seed extraction correctly failed")
    except Exception:
        print("✅ Wrong seed extraction correctly failed (with exception)")
    
    # Test 4: Test with encryption
    print(f"\n6. Testing with encryption...")
    encryption_engine = EncryptionEngine(SecurityLevel.HIGH)
    password = os.environ.get('TEST_PASSWORD', 'temp_test_pass_123')
    
    encrypted_data = encryption_engine.encrypt_with_metadata(test_data, password)
    stego_path_enc = test_dir / "stego_encrypted.png"
    
    success = stego_engine.hide_data(carrier_path, encrypted_data, stego_path_enc, 
                                   randomize=True, seed=test_seed)
    if not success:
        print("❌ Encrypted hide operation failed!")
        return False
    
    extracted_enc_data = stego_engine.extract_data(stego_path_enc, 
                                                 randomize=True, seed=test_seed)
    if extracted_enc_data is None:
        print("❌ Encrypted extract operation failed!")
        return False
    
    # Decrypt and verify
    decrypted_data = encryption_engine.decrypt_with_metadata(extracted_enc_data, password)
    if decrypted_data != test_data:
        print("❌ Decrypted data doesn't match original!")
        return False
    print("✅ Encrypted hide/extract successful")
    
    print(f"\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Steganography is working correctly.")
    print("=" * 60)
    
    # Clean up test files
    for file_path in test_dir.glob("*.png"):
        file_path.unlink()
    test_dir.rmdir()
    
    return True


if __name__ == "__main__":
    success = test_steganography()
    sys.exit(0 if success else 1)
