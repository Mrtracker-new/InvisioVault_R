#!/usr/bin/env python3
"""
Test script specifically for large PDF files (94.8KB) with optimized randomized LSB positioning
This script simulates your exact use case and measures performance improvements.
"""

import time
import hashlib
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel

def create_pdf_sized_carrier_image(width=3000, height=2000):
    """Create a large carrier image suitable for 94.8KB files."""
    print(f"🖼️  Creating {width}x{height} carrier image (capacity ~{width*height*3//8//1024}KB)...")
    
    # Create realistic image with good entropy
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate complex patterns for better hiding capacity
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Multi-layered pattern generation
    pattern1 = np.sin(x * 0.01) * np.cos(y * 0.01) * 127 + 128
    pattern2 = np.sin(x * 0.005 + y * 0.003) * 63 + 64
    pattern3 = np.random.randint(-40, 40, (height, width))
    
    img_array[:, :, 0] = np.clip(pattern1 + pattern3, 0, 255)
    img_array[:, :, 1] = np.clip(pattern2 + pattern3 + 50, 0, 255)
    img_array[:, :, 2] = np.clip((pattern1 + pattern2) / 2 + pattern3 + 100, 0, 255)
    
    # Add noise for better steganography
    noise = np.random.randint(-20, 20, img_array.shape)
    img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array, 'RGB')

def simulate_pdf_data(size_kb=94.8):
    """Simulate PDF file data with realistic structure."""
    size_bytes = int(size_kb * 1024)
    
    # PDF-like header and footer
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    footer = b"\nstartxref\n12345\n%%EOF\n"
    
    # Fill with semi-random data that mimics PDF content
    middle_size = size_bytes - len(header) - len(footer)
    
    # Create realistic PDF-like content
    content_parts = [
        b"1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n",
        b"2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n",
        b"3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\n"
    ]
    
    # Repeat and pad to reach target size
    content = b"".join(content_parts * (middle_size // len(b"".join(content_parts)) + 1))
    content = content[:middle_size]
    
    return header + content + footer

def test_pdf_optimization():
    """Test the optimized randomized LSB positioning with 94.8KB PDF simulation."""
    print("=" * 80)
    print("🚀 TESTING MEGA-OPTIMIZED LSB POSITIONING FOR 94.8KB PDF FILES")
    print("=" * 80)
    
    # Initialize engines
    stego_engine = SteganographyEngine()
    encryption_engine = EncryptionEngine(SecurityLevel.HIGH)
    
    # Create realistic PDF-sized data
    pdf_data = simulate_pdf_data(94.8)  # 94.8KB exactly
    password = "MyLarge_PDF_Password_2024!"
    
    # Generate consistent seed from password (same as UI)
    seed_hash = hashlib.sha256(password.encode('utf-8')).digest()
    seed = int.from_bytes(seed_hash[:4], byteorder='big') % (2**32)
    
    print(f"📄 PDF simulation: {len(pdf_data):,} bytes ({len(pdf_data)/1024:.1f} KB)")
    print(f"🔑 Password: {password}")
    print(f"🎲 Seed: {seed}")
    print()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        carrier_path = temp_dir / "large_carrier.png"
        stego_path = temp_dir / "stego_with_pdf.png"
        
        # Create carrier image
        start_time = time.time()
        carrier_img = create_pdf_sized_carrier_image()
        carrier_img.save(carrier_path, "PNG")
        create_time = time.time() - start_time
        
        capacity = stego_engine.calculate_capacity(carrier_path)
        print(f"✅ Carrier created in {create_time:.2f}s")
        print(f"📏 Image: {carrier_img.size[0]}x{carrier_img.size[1]} pixels")
        print(f"💾 Capacity: {capacity:,} bytes ({capacity/1024:.1f} KB)")
        
        if capacity < len(pdf_data):
            print(f"❌ ERROR: Insufficient capacity! Need {len(pdf_data):,}, have {capacity:,}")
            return
        
        print(f"✅ Capacity check: OK (margin: {capacity-len(pdf_data):,} bytes)")
        print()
        
        # Test 1: Encrypt the PDF data
        print("🔒 STEP 1: Encrypting PDF data...")
        start_time = time.time()
        encrypted_data = encryption_engine.encrypt_with_metadata(pdf_data, password)
        encrypt_time = time.time() - start_time
        
        if not encrypted_data:
            print("❌ Encryption failed!")
            return
        
        print(f"✅ Encrypted in {encrypt_time:.3f}s")
        print(f"📦 Encrypted size: {len(encrypted_data):,} bytes ({len(encrypted_data)/1024:.1f} KB)")
        print(f"📊 Overhead: +{len(encrypted_data) - len(pdf_data):,} bytes")
        print()
        
        # Test 2: Hide with randomized positioning
        print("🔥 STEP 2: Hiding with MEGA-OPTIMIZED randomized LSB...")
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
            print("❌ Hiding failed!")
            return
        
        print(f"✅ Hidden successfully in {hide_time:.3f}s")
        print(f"🎯 Hiding throughput: {len(encrypted_data)/hide_time:,.0f} bytes/second")
        print()
        
        # Test 3: Extract with optimized algorithm (THE CRITICAL TEST)
        print("⚡ STEP 3: Extracting with MEGA-OPTIMIZED algorithm...")
        start_time = time.time()
        extracted_encrypted = stego_engine.extract_data(
            stego_path=stego_path,
            randomize=True,
            seed=seed
        )
        extract_time = time.time() - start_time
        
        if extracted_encrypted is None:
            print(f"❌ EXTRACTION FAILED after {extract_time:.2f} seconds!")
            print("   The optimization may need further tuning.")
            return
        
        print(f"✅ EXTRACTION SUCCESSFUL in {extract_time:.3f}s")
        print(f"🚀 Extraction throughput: {len(extracted_encrypted)/extract_time:,.0f} bytes/second")
        print(f"📊 Extracted size: {len(extracted_encrypted):,} bytes")
        print()
        
        # Test 4: Decrypt and verify
        print("🔓 STEP 4: Decrypting and verifying...")
        start_time = time.time()
        decrypted_pdf = encryption_engine.decrypt_with_metadata(extracted_encrypted, password)
        decrypt_time = time.time() - start_time
        
        if decrypted_pdf != pdf_data:
            print("❌ Data corruption detected!")
            print(f"Original: {len(pdf_data)} bytes")
            print(f"Decrypted: {len(decrypted_pdf) if decrypted_pdf else 0} bytes")
            return
        
        print(f"✅ Decrypted and verified in {decrypt_time:.3f}s")
        print(f"✅ Data integrity: PERFECT MATCH")
        print()
        
        # Performance Summary
        total_time = hide_time + extract_time + encrypt_time + decrypt_time
        
        print("=" * 80)
        print("📊 MEGA-OPTIMIZATION PERFORMANCE RESULTS")
        print("=" * 80)
        print(f"📄 File size: {len(pdf_data):,} bytes ({len(pdf_data)/1024:.1f} KB)")
        print(f"🔒 Encryption: {encrypt_time:.3f}s ({len(pdf_data)/encrypt_time:,.0f} B/s)")
        print(f"📥 Hide (randomized): {hide_time:.3f}s ({len(encrypted_data)/hide_time:,.0f} B/s)")
        print(f"📤 Extract (optimized): {extract_time:.3f}s ({len(extracted_encrypted)/extract_time:,.0f} B/s)")
        print(f"🔓 Decryption: {decrypt_time:.3f}s ({len(decrypted_pdf)/decrypt_time:,.0f} B/s)")
        print(f"⏱️  Total workflow: {total_time:.3f}s")
        print()
        
        # Performance Assessment
        if extract_time < 0.5:
            status = "🎉 EXCELLENT!"
            message = "Your 94.8KB PDF will extract almost instantly!"
        elif extract_time < 1.5:
            status = "✅ VERY GOOD!"
            message = "Your PDF will extract in about 1 second or less."
        elif extract_time < 3.0:
            status = "✅ GOOD!"
            message = "Your PDF will extract in a few seconds."
        elif extract_time < 10.0:
            status = "⚠️  ACCEPTABLE"
            message = "PDF extraction will take several seconds."
        else:
            status = "❌ NEEDS MORE WORK"
            message = "Extraction is still too slow for large files."
        
        print(f"🎯 PERFORMANCE RATING: {status}")
        print(f"💡 For your 94.8KB PDF: {message}")
        
        # Optimization details
        throughput_mbps = (len(extracted_encrypted) / extract_time) / (1024 * 1024)
        print(f"🚀 Extraction speed: {throughput_mbps:.2f} MB/s")
        
        # Test security (wrong password should fail)
        print()
        print("🔒 SECURITY TEST: Testing with wrong seed...")
        wrong_seed = seed + 12345
        wrong_extracted = stego_engine.extract_data(stego_path, randomize=True, seed=wrong_seed)
        
        if wrong_extracted is None:
            print("✅ Security verified: Wrong seed correctly failed")
        else:
            print("⚠️  WARNING: Wrong seed should have failed but didn't")
        
        print()
        print("🎊 OPTIMIZATION TEST COMPLETE!")
        print(f"🎯 Your 94.8KB PDF should now extract in ~{extract_time:.2f} seconds")
        
        if extract_time < 2.0:
            print("✨ OPTIMIZATION SUCCESS: Large file performance is now excellent!")
        else:
            print("💡 Consider using sequential mode for even faster performance.")

if __name__ == "__main__":
    test_pdf_optimization()
