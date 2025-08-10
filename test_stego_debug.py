#!/usr/bin/env python3
"""
Simple diagnostic test for steganography engine.
"""

import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from core.steganography_engine import SteganographyEngine


def create_test_image(width=800, height=600, filename="test_carrier.png"):
    """Create a test carrier image."""
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    img.save(filename, format='PNG')
    return filename


def test_basic_steganography():
    """Test basic steganography without seeding first."""
    print("Testing basic steganography...")
    
    temp_dir = None
    try:
        # Create test environment
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test carrier image
        carrier_image = create_test_image(800, 600, str(temp_dir / "carrier.png"))
        
        # Initialize engine
        stego_engine = SteganographyEngine()
        
        # Test data
        test_data = b"Hello, world! This is a simple test."
        print(f"Test data: {test_data}")
        print(f"Test data size: {len(test_data)} bytes")
        
        output_path = temp_dir / "simple_test.png"
        
        # Test capacity
        capacity = stego_engine.calculate_capacity(carrier_image)
        print(f"Image capacity: {capacity} bytes")
        
        if len(test_data) > capacity:
            print("❌ Test data too large for image")
            return False
        
        # Hide without randomization (sequential)
        print("Hiding data sequentially...")
        success = stego_engine.hide_data(
            Path(carrier_image),
            test_data,
            output_path,
            randomize=False
        )
        
        if not success:
            print("❌ Failed to hide data")
            return False
        
        print("✅ Data hidden successfully")
        
        # Extract without randomization
        print("Extracting data sequentially...")
        extracted_data = stego_engine.extract_data(
            output_path,
            randomize=False
        )
        
        if not extracted_data:
            print("❌ Failed to extract data")
            return False
        
        print("✅ Data extracted successfully")
        print(f"Extracted data: {extracted_data}")
        
        # Verify
        if extracted_data != test_data:
            print(f"❌ Data mismatch: expected {test_data}, got {extracted_data}")
            return False
        
        print("✅ Data integrity verified")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


def test_seeded_steganography():
    """Test seeded steganography."""
    print("\nTesting seeded steganography...")
    
    temp_dir = None
    try:
        # Create test environment
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test carrier image
        carrier_image = create_test_image(800, 600, str(temp_dir / "carrier_seeded.png"))
        
        # Initialize engine
        stego_engine = SteganographyEngine()
        
        # Test data
        test_data = b"This is seeded test data."
        seed = 12345
        
        print(f"Test data: {test_data}")
        print(f"Using seed: {seed}")
        
        output_path = temp_dir / "seeded_test.png"
        
        # Hide with randomization
        print("Hiding data with seeding...")
        success = stego_engine.hide_data(
            Path(carrier_image),
            test_data,
            output_path,
            randomize=True,
            seed=seed
        )
        
        if not success:
            print("❌ Failed to hide data with seed")
            return False
        
        print("✅ Data hidden successfully with seed")
        
        # Extract with same seed
        print("Extracting data with same seed...")
        extracted_data = stego_engine.extract_data(
            output_path,
            randomize=True,
            seed=seed
        )
        
        if not extracted_data:
            print("❌ Failed to extract data with seed")
            return False
        
        print("✅ Data extracted successfully with seed")
        
        # Verify
        if extracted_data != test_data:
            print(f"❌ Data mismatch: expected {test_data}, got {extracted_data}")
            return False
        
        print("✅ Seeded data integrity verified")
        return True
        
    except Exception as e:
        print(f"❌ Seeded test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


if __name__ == "__main__":
    print("Running diagnostic tests for steganography engine...")
    
    test1_success = test_basic_steganography()
    test2_success = test_seeded_steganography()
    
    overall_success = test1_success and test2_success
    
    if overall_success:
        print("\n🎉 ALL DIAGNOSTIC TESTS PASSED!")
        print("Basic steganography engine is working correctly.")
    else:
        print("\n❌ SOME DIAGNOSTIC TESTS FAILED!")
        print("There may be an issue with the steganography engine.")
    
    exit(0 if overall_success else 1)
