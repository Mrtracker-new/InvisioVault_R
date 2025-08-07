#!/usr/bin/env python3
"""
MEGA-FAST Large File Test Suite
Tests the revolutionary ultra-fast LSB positioning algorithm for MB-sized files.

This script demonstrates the massive performance improvements for:
- 94.8KB PDFs (your specific use case)
- 1MB+ image files
- Multi-MB archive files
- Various document formats

Expected performance: Sub-second extraction for files up to several MB.
"""

import time
import hashlib
import tempfile
import random
from pathlib import Path
from PIL import Image
import numpy as np

from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel

def create_ultra_large_carrier(width=4000, height=3000):
    """Create an ultra-large carrier image for MB-scale files."""
    print(f"🖼️  Creating {width}x{height} ultra-large carrier...")
    
    # Create high-entropy image for optimal hiding capacity
    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Add structured patterns for realism
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Multi-layered realistic patterns
    pattern1 = np.sin(x * 0.001) * np.cos(y * 0.001) * 50 + 128
    pattern2 = np.sin(x * 0.002 + y * 0.003) * 30 + 64
    texture_noise = np.random.randint(-60, 60, (height, width, 3))
    
    # Blend patterns with base random data
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.7 + pattern1 + texture_noise[:, :, 0] * 0.3, 0, 255)
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 0.7 + pattern2 + texture_noise[:, :, 1] * 0.3, 0, 255) 
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.7 + (pattern1 + pattern2) / 2 + texture_noise[:, :, 2] * 0.3, 0, 255)
    
    capacity_mb = (width * height * 3) // 8 // 1024 // 1024
    print(f"📊 Ultra-large carrier: ~{capacity_mb} MB capacity")
    
    return Image.fromarray(img_array.astype(np.uint8), 'RGB')

def create_test_file_data(size_kb, file_type="PDF"):
    """Create realistic test file data of specified size."""
    size_bytes = int(size_kb * 1024)
    
    if file_type == "PDF":
        # Create PDF-like structure
        header = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        footer = b"\nstartxref\n" + str(size_bytes - 100).encode() + b"\n%%EOF\n"
        
        # PDF objects and content
        content_chunks = [
            b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n",
            b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n",
            b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n",
            b"4 0 obj\n<</Length 50>>\nstream\nBT /F1 12 Tf 72 720 Td (Hello World) Tj ET\nendstream\nendobj\n",
            b"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n \n"
        ]
        
        # Fill middle with realistic PDF content
        middle_size = size_bytes - len(header) - len(footer)
        content = b"".join(content_chunks * (middle_size // len(b"".join(content_chunks)) + 1))
        content = content[:middle_size]
        
        return header + content + footer
        
    elif file_type == "IMAGE":
        # Create image-like binary data
        header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        # Random "image" data
        image_data = bytes([random.randint(0, 255) for _ in range(size_bytes - len(header) - 20)])
        footer = b"\x00\x00\x00\x00IEND\xaeB`\x82"
        return header + image_data + footer
        
    elif file_type == "ARCHIVE":
        # Create ZIP-like archive data
        header = b"PK\x03\x04"
        # Random compressed-like data with patterns
        archive_data = bytes([random.randint(0, 255) for _ in range(size_bytes - len(header) - 20)])
        footer = b"PK\x05\x06\x00\x00\x00\x00"
        return header + archive_data + footer
        
    else:  # TEXT
        # Create text document
        base_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50
        full_text = (base_text * (size_bytes // len(base_text) + 1))[:size_bytes]
        return full_text.encode('utf-8')

def test_mega_large_files():
    """Test ultra-fast algorithm with massive files (MB scale)."""
    print("=" * 80)
    print("🚀 MEGA-FAST LARGE FILE TEST SUITE")
    print("   Testing revolutionary LSB optimizations for MB-sized files")
    print("=" * 80)
    
    # Initialize engines
    stego_engine = SteganographyEngine()
    encryption_engine = EncryptionEngine(SecurityLevel.HIGH)
    
    # Test scenarios - from your 94.8KB use case to multi-MB files
    test_scenarios = [
        {"name": "94.8KB PDF (Your Use Case)", "size_kb": 94.8, "file_type": "PDF", "expected_time": 1.0},
        {"name": "200KB Office Document", "size_kb": 200, "file_type": "PDF", "expected_time": 1.5},
        {"name": "500KB Image File", "size_kb": 500, "file_type": "IMAGE", "expected_time": 2.0},
        {"name": "1MB Archive File", "size_kb": 1024, "file_type": "ARCHIVE", "expected_time": 3.0},
        {"name": "2MB Large Document", "size_kb": 2048, "file_type": "PDF", "expected_time": 4.0},
        {"name": "3MB Media File", "size_kb": 3072, "file_type": "IMAGE", "expected_time": 5.0},
    ]
    
    results = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create one ultra-large carrier for all tests
        print("\n🔨 Creating ultra-large carrier image...")
        start_time = time.time()
        carrier_img = create_ultra_large_carrier(4000, 3000)  # ~36MB capacity
        carrier_path = temp_dir / "ultra_carrier.png"
        carrier_img.save(carrier_path, "PNG")
        carrier_creation_time = time.time() - start_time
        
        capacity = stego_engine.calculate_capacity(carrier_path)
        print(f"✅ Ultra-carrier created in {carrier_creation_time:.2f}s")
        print(f"📏 Size: {carrier_img.size[0]}x{carrier_img.size[1]} pixels")
        print(f"💾 Capacity: {capacity:,} bytes ({capacity/1024/1024:.1f} MB)")
        print()
        
        # Test each scenario
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"🧪 TEST {i}/{len(test_scenarios)}: {scenario['name']}")
            print("-" * 50)
            
            # Create test data
            test_data = create_test_file_data(scenario['size_kb'], scenario['file_type'])
            password = f"MegaFast_Test_{scenario['size_kb']}KB_2024!"
            
            # Generate seed from password
            seed_hash = hashlib.sha256(password.encode('utf-8')).digest()
            seed = int.from_bytes(seed_hash[:4], byteorder='big') % (2**32)
            
            print(f"📄 File: {len(test_data):,} bytes ({len(test_data)/1024:.1f} KB)")
            print(f"🔑 Seed: {seed}")
            
            stego_path = temp_dir / f"stego_{scenario['size_kb']}kb.png"
            
            try:
                # Step 1: Encrypt
                print("🔒 Encrypting...")
                encrypt_start = time.time()
                encrypted_data = encryption_engine.encrypt_with_metadata(test_data, password)
                encrypt_time = time.time() - encrypt_start
                
                if not encrypted_data:
                    print("❌ Encryption failed!")
                    continue
                
                print(f"   ✅ Encrypted in {encrypt_time:.3f}s ({len(encrypted_data):,} bytes)")
                
                # Step 2: Hide with randomized positioning
                print("📥 Hiding with MEGA-FAST randomized LSB...")
                hide_start = time.time()
                success = stego_engine.hide_data(
                    carrier_path=carrier_path,
                    data=encrypted_data,
                    output_path=stego_path,
                    randomize=True,
                    seed=seed
                )
                hide_time = time.time() - hide_start
                
                if not success:
                    print("❌ Hiding failed!")
                    continue
                
                print(f"   ✅ Hidden in {hide_time:.3f}s ({len(encrypted_data)/hide_time:,.0f} B/s)")
                
                # Step 3: Extract with MEGA-FAST algorithm
                print("📤 Extracting with MEGA-FAST algorithm...")
                extract_start = time.time()
                extracted_encrypted = stego_engine.extract_data(
                    stego_path=stego_path,
                    randomize=True,
                    seed=seed
                )
                extract_time = time.time() - extract_start
                
                if extracted_encrypted is None:
                    print(f"❌ EXTRACTION FAILED after {extract_time:.2f}s!")
                    results.append({
                        "name": scenario["name"],
                        "size_kb": scenario["size_kb"],
                        "status": "FAILED",
                        "extract_time": extract_time,
                        "expected_time": scenario["expected_time"]
                    })
                    continue
                
                print(f"   ✅ Extracted in {extract_time:.3f}s ({len(extracted_encrypted)/extract_time:,.0f} B/s)")
                
                # Step 4: Decrypt and verify
                print("🔓 Decrypting and verifying...")
                decrypt_start = time.time()
                decrypted_data = encryption_engine.decrypt_with_metadata(extracted_encrypted, password)
                decrypt_time = time.time() - decrypt_start
                
                if decrypted_data != test_data:
                    print("❌ Data corruption detected!")
                    continue
                
                print(f"   ✅ Verified in {decrypt_time:.3f}s - Data integrity perfect!")
                
                # Performance assessment
                total_time = encrypt_time + hide_time + extract_time + decrypt_time
                throughput_mbps = (len(test_data) / extract_time) / (1024 * 1024)
                
                if extract_time <= scenario["expected_time"]:
                    performance = "🎉 EXCELLENT"
                elif extract_time <= scenario["expected_time"] * 1.5:
                    performance = "✅ VERY GOOD"
                elif extract_time <= scenario["expected_time"] * 2.0:
                    performance = "✅ GOOD"
                else:
                    performance = "⚠️  SLOWER THAN EXPECTED"
                
                print(f"\n📊 PERFORMANCE: {performance}")
                print(f"   Extract time: {extract_time:.3f}s (expected ≤{scenario['expected_time']:.1f}s)")
                print(f"   Throughput: {throughput_mbps:.2f} MB/s")
                print(f"   Total workflow: {total_time:.3f}s")
                
                results.append({
                    "name": scenario["name"],
                    "size_kb": scenario["size_kb"],
                    "status": "SUCCESS",
                    "extract_time": extract_time,
                    "expected_time": scenario["expected_time"],
                    "throughput_mbps": throughput_mbps,
                    "total_time": total_time,
                    "performance": performance
                })
                
            except Exception as e:
                print(f"❌ Test failed with error: {e}")
                results.append({
                    "name": scenario["name"],
                    "size_kb": scenario["size_kb"],
                    "status": "ERROR",
                    "error": str(e)
                })
            
            print("\n")
        
        # Final performance summary
        print("=" * 80)
        print("📊 MEGA-FAST PERFORMANCE RESULTS SUMMARY")
        print("=" * 80)
        
        successful_tests = [r for r in results if r["status"] == "SUCCESS"]
        failed_tests = [r for r in results if r["status"] != "SUCCESS"]
        
        if successful_tests:
            print("✅ SUCCESSFUL TESTS:")
            print("-" * 50)
            for result in successful_tests:
                print(f"  {result['name']}")
                print(f"    Size: {result['size_kb']} KB")
                print(f"    Extract Time: {result['extract_time']:.3f}s (target: ≤{result['expected_time']:.1f}s)")
                print(f"    Throughput: {result['throughput_mbps']:.2f} MB/s")
                print(f"    Performance: {result['performance']}")
                print()
        
        if failed_tests:
            print("❌ FAILED TESTS:")
            print("-" * 50)
            for result in failed_tests:
                print(f"  {result['name']}: {result['status']}")
                if "error" in result:
                    print(f"    Error: {result['error']}")
                print()
        
        # Overall assessment
        success_rate = len(successful_tests) / len(results) * 100
        if successful_tests:
            avg_extract_time = sum(r["extract_time"] for r in successful_tests) / len(successful_tests)
            avg_throughput = sum(r["throughput_mbps"] for r in successful_tests) / len(successful_tests)
            
            print(f"🎯 OVERALL PERFORMANCE:")
            print(f"   Success Rate: {success_rate:.1f}% ({len(successful_tests)}/{len(results)} tests)")
            print(f"   Average Extract Time: {avg_extract_time:.3f}s")
            print(f"   Average Throughput: {avg_throughput:.2f} MB/s")
            
            if success_rate >= 90 and avg_extract_time <= 3.0:
                print(f"\n🏆 MEGA-OPTIMIZATION SUCCESS!")
                print(f"   The algorithm now handles MB-sized files with excellent performance!")
                print(f"   Your 94.8KB PDFs should extract in ~{[r['extract_time'] for r in successful_tests if '94.8KB' in r['name']][0]:.2f} seconds.")
            elif success_rate >= 70:
                print(f"\n✅ GOOD PERFORMANCE!")
                print(f"   The algorithm handles large files well with room for improvement.")
            else:
                print(f"\n⚠️  NEEDS MORE OPTIMIZATION")
                print(f"   Some large files still take longer than expected.")
        
        print("\n🎊 MEGA-FAST LARGE FILE TESTING COMPLETE!")
        print(f"🔍 Total tests: {len(results)}")
        print(f"✅ Successful: {len(successful_tests)}")
        print(f"❌ Failed: {len(failed_tests)}")

def quick_pdf_test():
    """Quick test specifically for your 94.8KB PDF use case."""
    print("\n" + "="*60)
    print("⚡ QUICK 94.8KB PDF TEST")
    print("="*60)
    
    stego_engine = SteganographyEngine()
    encryption_engine = EncryptionEngine(SecurityLevel.STANDARD)  # Faster for quick test
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create smaller carrier for speed
        carrier_img = create_ultra_large_carrier(2000, 1500)  # Still plenty of capacity
        carrier_path = temp_dir / "quick_carrier.png"
        carrier_img.save(carrier_path, "PNG")
        
        # Create 94.8KB PDF
        pdf_data = create_test_file_data(94.8, "PDF")
        password = "Quick_PDF_Test_2024!"
        seed = int.from_bytes(hashlib.sha256(password.encode()).digest()[:4], 'big') % (2**32)
        
        print(f"📄 PDF size: {len(pdf_data):,} bytes ({len(pdf_data)/1024:.1f} KB)")
        
        # Full workflow test
        start_total = time.time()
        
        # Encrypt
        encrypted = encryption_engine.encrypt_with_metadata(pdf_data, password)
        
        # Hide
        stego_path = temp_dir / "quick_stego.png"
        stego_engine.hide_data(carrier_path, encrypted, stego_path, randomize=True, seed=seed)
        
        # Extract (the critical test)
        extract_start = time.time()
        extracted = stego_engine.extract_data(stego_path, randomize=True, seed=seed)
        extract_time = time.time() - extract_start
        
        # Decrypt
        decrypted = encryption_engine.decrypt_with_metadata(extracted, password)
        
        total_time = time.time() - start_total
        
        if decrypted == pdf_data:
            print(f"✅ SUCCESS! PDF extracted in {extract_time:.3f} seconds")
            print(f"⚡ Total workflow: {total_time:.3f} seconds")
            print(f"🚀 Throughput: {len(pdf_data)/extract_time:,.0f} bytes/second")
            
            if extract_time < 1.0:
                print("🎉 EXCELLENT - Sub-second extraction!")
            elif extract_time < 2.0:
                print("✅ VERY GOOD - Under 2 seconds!")
            elif extract_time < 5.0:
                print("👍 GOOD - Under 5 seconds!")
            else:
                print("⚠️  Could be faster, but functional")
        else:
            print("❌ FAILED - Data corruption or extraction failure")

if __name__ == "__main__":
    # Run quick test first
    quick_pdf_test()
    
    # Run full test suite
    test_mega_large_files()
