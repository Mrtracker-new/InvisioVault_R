"""
Test script to verify large file support (like 94.8KB PDF) with randomized LSB positioning
"""

import time
import hashlib
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel

def create_large_test_image(width=2400, height=1600):
    """Create a large image suitable for hiding large files."""
    # Create a realistic RGB image with good noise characteristics
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient patterns
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [
                int(80 + (x / width) * 100 + (y / height) * 75),
                int(120 + (y / height) * 100 + np.sin(x * 0.01) * 20),  
                int(150 + ((x + y) / (width + height)) * 80 + np.cos(y * 0.01) * 30)
            ]
    
    # Add significant noise for better steganography
    noise = np.random.randint(-30, 30, (height, width, 3))
    img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Add some textures and patterns
    for i in range(0, height, 100):
        for j in range(0, width, 100):
            # Add small textured blocks
            block_noise = np.random.randint(-50, 50, (min(100, height-i), min(100, width-j), 3))
            img_array[i:i+100, j:j+100] = np.clip(
                img_array[i:i+100, j:j+100].astype(int) + block_noise, 0, 255
            ).astype(np.uint8)
    
    return Image.fromarray(img_array, 'RGB')

def test_large_file_support():
    """Test randomized LSB positioning with large files like PDF (94.8KB)."""
    print("🔬 Testing Large File Support (94.8KB PDF simulation)")
    print("=" * 65)
    
    # Initialize engines
    stego_engine = SteganographyEngine()
    encryption_engine = EncryptionEngine(SecurityLevel.HIGH)
    
    # Create test data simulating a 94.8KB PDF
    pdf_size = int(94.8 * 1024)  # 94.8KB in bytes
    test_pdf_data = b"PDF_HEADER" + b"X" * (pdf_size - 50) + b"PDF_FOOTER"
    
    password = "LargePDF_Test2024!"
    
    # Generate consistent seed from password
    seed_hash = hashlib.sha256(password.encode('utf-8')).digest()
    seed = int.from_bytes(seed_hash[:4], byteorder='big') % (2**32)
    
    print(f"📄 Simulating PDF file: {len(test_pdf_data):,} bytes ({len(test_pdf_data)/1024:.1f} KB)")
    print(f"🔑 Password: {password}")
    print(f"🎲 Generated seed: {seed}")
    print()
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            carrier_path = temp_dir / "large_photo.png"
            stego_path = temp_dir / "large_photo_with_pdf.png"
            
            # Create large test image (2400x1600 = 11.52MB capacity)
            print("🖼️  Creating large test image...")
            start_time = time.time()
            test_img = create_large_test_image()
            test_img.save(carrier_path, "PNG")
            create_time = time.time() - start_time
            
            # Check capacity
            capacity = stego_engine.calculate_capacity(carrier_path)
            print(f"✅ Large image created in {create_time:.2f}s")
            print(f"📏 Image size: {test_img.size[0]}x{test_img.size[1]} pixels")
            print(f"💾 Image capacity: {capacity:,} bytes ({capacity/1024:.1f} KB)")
            
            if capacity < len(test_pdf_data):
                print(f"❌ Image capacity too small! Need {len(test_pdf_data):,} bytes, have {capacity:,}")
                return
            
            # Encrypt the PDF data
            print("🔒 Encrypting PDF data...")
            start_time = time.time()
            encrypted_data = encryption_engine.encrypt_with_metadata(test_pdf_data, password)
            encrypt_time = time.time() - start_time
            
            if not encrypted_data:
                print("❌ Encryption failed!")
                return
                
            print(f"✅ Encrypted in {encrypt_time:.3f}s")
            print(f"📦 Encrypted size: {len(encrypted_data):,} bytes ({len(encrypted_data)/1024:.1f} KB)")
            print(f"📊 Encryption overhead: +{len(encrypted_data) - len(test_pdf_data):,} bytes")
            
            # Hide the encrypted PDF with randomization
            print("📥 Hiding encrypted PDF with randomized LSB...")
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
                print("❌ Failed to hide PDF data!")
                return
                
            print(f"✅ PDF hidden successfully in {hide_time:.3f}s")
            
            # Extract the PDF data (this is the critical test)
            print("📤 Extracting PDF with optimized randomized algorithm...")
            start_time = time.time()
            extracted_encrypted = stego_engine.extract_data(
                stego_path=stego_path,
                randomize=True,
                seed=seed
            )
            extract_time = time.time() - start_time
            
            if extracted_encrypted is None:
                print(f"❌ Extraction failed after {extract_time:.2f} seconds!")
                print("   This indicates the algorithm needs further optimization for large files.")
                return
                
            print(f"✅ PDF extracted in {extract_time:.3f}s")
            
            # Decrypt and verify
            print("🔓 Decrypting extracted PDF...")
            start_time = time.time()
            decrypted_pdf = encryption_engine.decrypt_with_metadata(extracted_encrypted, password)
            decrypt_time = time.time() - start_time
            
            if decrypted_pdf != test_pdf_data:
                print("❌ PDF data corrupted during extraction!")
                return
                
            print(f"✅ PDF decrypted successfully in {decrypt_time:.3f}s")
            
            # Performance metrics
            total_time = hide_time + extract_time + encrypt_time + decrypt_time
            throughput = len(test_pdf_data) / extract_time if extract_time > 0 else 0
            
            print("\n" + "=" * 65)
            print("📊 LARGE FILE PERFORMANCE RESULTS")
            print("=" * 65)
            print(f"📄 File size: {len(test_pdf_data):,} bytes ({len(test_pdf_data)/1024:.1f} KB)")
            print(f"🔒 Encrypt time: {encrypt_time:.3f}s")
            print(f"📥 Hide time: {hide_time:.3f}s") 
            print(f"📤 Extract time: {extract_time:.3f}s")
            print(f"🔓 Decrypt time: {decrypt_time:.3f}s")
            print(f"⏱️  Total workflow: {total_time:.3f}s")
            print(f"🚀 Extraction throughput: {throughput:,.0f} bytes/second")
            print(f"💾 Extraction throughput: {throughput/1024:.1f} KB/s")
            
            # Performance assessment
            if extract_time < 1.0:
                print("\n🎉 EXCELLENT! Large file extraction is very fast")
                print("   Your 94.8KB PDF should extract quickly!")
            elif extract_time < 3.0:
                print("\n✅ GOOD! Large file extraction is reasonably fast") 
                print("   Your PDF extraction should complete in a few seconds.")
            elif extract_time < 10.0:
                print("\n⚠️  ACCEPTABLE: Large file extraction works but is slow")
                print("   PDF extraction may take several seconds.")
            else:
                print("\n❌ SLOW: Large file extraction needs optimization")
                print("   Consider using sequential mode for large files.")
            
            # Test wrong seed (security verification)
            print("\n🔒 Testing security with wrong seed...")
            wrong_seed = seed + 1
            wrong_extracted = stego_engine.extract_data(
                stego_path=stego_path,
                randomize=True,
                seed=wrong_seed
            )
            
            if wrong_extracted is None:
                print("✅ Security verified: Wrong seed correctly failed")
            else:
                print("⚠️  WARNING: Wrong seed extracted data (potential issue)")
            
            print("\n🎯 CONCLUSION FOR YOUR 94.8KB PDF:")
            if extract_time < 3.0:
                print("   ✅ Your PDF should extract quickly with randomized positioning!")
                print("   ✅ The algorithm is now optimized for large files.")
            else:
                print("   ⚠️  Your PDF may take longer to extract than expected.")
                print("   💡 Consider using sequential mode for better performance.")
                
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_large_file_support()
