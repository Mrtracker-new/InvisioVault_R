#!/usr/bin/env python3
"""
Quick test to verify the MEGA-FAST instant detection is working correctly.
This should find the exact encrypted size in Stage 1 and return immediately.
"""

import time
import hashlib
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel

def create_test_carrier(width=1500, height=1000):
    """Create a test carrier image."""
    # Create simple RGB noise image
    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array, 'RGB')

def test_instant_detection():
    """Test if Stage 1 instant detection works for our exact use case."""
    print("🔬 TESTING MEGA-FAST INSTANT DETECTION")
    print("=" * 50)
    
    stego_engine = SteganographyEngine()
    encryption_engine = EncryptionEngine(SecurityLevel.HIGH)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create carrier
        carrier_img = create_test_carrier()
        carrier_path = temp_dir / "test_carrier.png"
        carrier_img.save(carrier_path, "PNG")
        
        # Create 94.8KB test file
        test_data = b"PDF_HEADER" + b"X" * (int(94.8 * 1024) - 50) + b"PDF_FOOTER"
        password = "Test_PDF_2024!"
        
        print(f"📄 Original data: {len(test_data):,} bytes ({len(test_data)/1024:.1f} KB)")
        
        # Encrypt
        encrypted_data = encryption_engine.encrypt_with_metadata(test_data, password)
        print(f"🔒 Encrypted data: {len(encrypted_data):,} bytes ({len(encrypted_data)/1024:.1f} KB)")
        
        # Generate seed
        seed = int.from_bytes(hashlib.sha256(password.encode()).digest()[:4], 'big') % (2**32)
        print(f"🎲 Seed: {seed}")
        
        # Hide data
        stego_path = temp_dir / "test_stego.png"
        hide_start = time.time()
        success = stego_engine.hide_data(carrier_path, encrypted_data, stego_path, randomize=True, seed=seed)
        hide_time = time.time() - hide_start
        
        if not success:
            print("❌ Hiding failed!")
            return
        
        print(f"📥 Hidden in {hide_time:.3f}s")
        
        # TEST: Extract with mega-fast algorithm
        print("\n🚀 TESTING MEGA-FAST EXTRACTION...")
        extract_start = time.time()
        
        extracted = stego_engine.extract_data(stego_path, randomize=True, seed=seed)
        
        extract_time = time.time() - extract_start
        
        if extracted is None:
            print(f"❌ EXTRACTION FAILED after {extract_time:.3f}s")
            return
        
        print(f"✅ EXTRACTED in {extract_time:.3f}s!")
        print(f"📦 Extracted size: {len(extracted):,} bytes")
        
        # Verify data integrity
        decrypted = encryption_engine.decrypt_with_metadata(extracted, password)
        
        if decrypted == test_data:
            print(f"✅ DATA INTEGRITY: Perfect!")
            print(f"🚀 Throughput: {len(test_data)/extract_time:,.0f} bytes/second")
            
            # Performance assessment
            if extract_time < 1.0:
                print("🎉 EXCELLENT! Sub-second extraction achieved!")
            elif extract_time < 2.0:
                print("✅ VERY GOOD! Under 2 seconds.")
            elif extract_time < 5.0:
                print("👍 GOOD! Under 5 seconds.")
            else:
                print("⚠️ SLOW - Needs more optimization.")
            
            return True
        else:
            print("❌ DATA CORRUPTION!")
            return False

if __name__ == "__main__":
    test_instant_detection()
