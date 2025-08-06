"""
Test script to verify the optimized randomized LSB extraction algorithm
Tests both hide and extract operations with different data sizes
"""

import time
import hashlib
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel

def create_test_image(width=1200, height=800):
    """Create a noisy test image suitable for steganography."""
    # Create a random RGB image with good noise characteristics
    img_array = np.random.randint(50, 205, (height, width, 3), dtype=np.uint8)
    # Add some patterns to make it more realistic
    for i in range(0, height, 50):
        img_array[i:i+10, :] = np.random.randint(100, 255, (10, width, 3))
    
    img = Image.fromarray(img_array, 'RGB')
    return img

def test_randomized_extraction():
    """Test the optimized randomized LSB extraction algorithm."""
    print("🚀 Testing Optimized Randomized LSB Extraction")
    print("=" * 60)
    
    # Initialize engines
    stego_engine = SteganographyEngine()
    encryption_engine = EncryptionEngine(SecurityLevel.STANDARD)
    
    # Test cases with different data sizes
    test_cases = [
        ("Small text file", b"This is a small test message for LSB steganography testing! " * 2),  # ~122 bytes
        ("Medium file (1KB)", b"X" * 1024),  # 1KB
        ("Large file (5KB)", b"Y" * 5000),   # 5KB  
        ("Text document", "Hello World! This is a longer test document that will be encrypted and hidden using randomized LSB positioning. Let's see if the optimized extraction can find it quickly!".encode('utf-8'))
    ]
    
    password = "test_password_123"
    
    # Generate consistent seed from password
    seed_hash = hashlib.sha256(password.encode('utf-8')).digest()
    seed = int.from_bytes(seed_hash[:4], byteorder='big') % (2**32)
    
    print(f"Using password: {password}")
    print(f"Generated seed: {seed}")
    print()
    
    results = []
    
    for test_name, test_data in test_cases:
        print(f"🧪 Testing: {test_name} ({len(test_data)} bytes)")
        print("-" * 50)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                carrier_path = temp_dir / "carrier.png"
                stego_path = temp_dir / "stego.png"
                
                # Create test image
                test_img = create_test_image()
                test_img.save(carrier_path, "PNG")
                print("  ✅ Created test image")
                
                # Encrypt the data (realistic usage scenario)
                print("  🔒 Encrypting data...")
                start_time = time.time()
                encrypted_data = encryption_engine.encrypt_with_metadata(test_data, password)
                encrypt_time = time.time() - start_time
                
                if not encrypted_data:
                    print("  ❌ Failed to encrypt data")
                    continue
                
                print(f"  ✅ Encrypted in {encrypt_time:.3f}s - Size: {len(encrypted_data)} bytes (overhead: +{len(encrypted_data) - len(test_data)} bytes)")
                
                # Hide the encrypted data with randomization
                print("  📥 Hiding data with randomized LSB...")
                start_time = time.time()
                success = stego_engine.hide_data(
                    carrier_path=carrier_path,
                    data=encrypted_data,
                    output_path=stego_path,
                    randomize=True,
                    seed=seed
                )
                hide_time = time.time() - start_time
                
                if not success:
                    print("  ❌ Failed to hide data")
                    continue
                
                print(f"  ✅ Hidden in {hide_time:.3f}s")
                
                # Extract the data (this is what we're testing)
                print("  📤 Extracting data with optimized algorithm...")
                start_time = time.time()
                extracted_data = stego_engine.extract_data(
                    stego_path=stego_path,
                    randomize=True,
                    seed=seed
                )
                extract_time = time.time() - start_time
                
                if extracted_data is None:
                    print("  ❌ Failed to extract data")
                    continue
                
                print(f"  ✅ Extracted in {extract_time:.3f}s")
                
                # Decrypt and verify
                print("  🔓 Decrypting extracted data...")
                decrypted_data = encryption_engine.decrypt_with_metadata(extracted_data, password)
                
                if decrypted_data != test_data:
                    print("  ❌ Data corruption detected!")
                    continue
                
                print("  ✅ Data integrity verified - perfect match!")
                
                # Calculate performance metrics
                throughput = len(test_data) / extract_time if extract_time > 0 else 0
                print(f"  📊 Performance: {throughput:.0f} bytes/second")
                
                # Test wrong seed (should fail)
                wrong_seed = seed + 1
                wrong_extracted = stego_engine.extract_data(
                    stego_path=stego_path,
                    randomize=True,
                    seed=wrong_seed
                )
                
                if wrong_extracted is None:
                    print("  ✅ Correctly failed with wrong seed")
                else:
                    print("  ⚠️  WARNING: Wrong seed extracted data (potential issue)")
                
                results.append({
                    'name': test_name,
                    'size': len(test_data),
                    'encrypted_size': len(encrypted_data),
                    'hide_time': hide_time,
                    'extract_time': extract_time,
                    'throughput': throughput,
                    'success': True
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'name': test_name,
                'size': len(test_data),
                'success': False
            })
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r.get('success', False)]
    
    if successful:
        avg_extract_time = sum(r['extract_time'] for r in successful) / len(successful)
        total_throughput = sum(r['throughput'] for r in successful) / len(successful)
        
        print(f"✅ Successful tests: {len(successful)}/{len(results)}")
        print(f"⚡ Average extraction time: {avg_extract_time:.3f} seconds")
        print(f"🚀 Average throughput: {total_throughput:.0f} bytes/second")
        print()
        print("Detailed Results:")
        
        for r in successful:
            print(f"  {r['name']:<20} | {r['size']:>6} bytes → {r['encrypted_size']:>6} bytes | Extract: {r['extract_time']:>6.3f}s | Rate: {r['throughput']:>8.0f} B/s")
        
        # Performance assessment
        if avg_extract_time < 0.5:
            print("\n🎉 EXCELLENT! Ultra-fast extraction performance")
        elif avg_extract_time < 1.0:
            print("\n✅ VERY GOOD! Fast extraction performance")
        elif avg_extract_time < 2.0:
            print("\n✅ GOOD! Reasonable extraction performance")
        else:
            print("\n⚠️  SLOW: May need further optimization")
    else:
        print("❌ No successful tests - algorithm needs debugging")
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    test_randomized_extraction()
