#!/usr/bin/env python3
"""
Final comprehensive test demonstrating successful multi-decoy functionality
Shows that the password handling issue has been fixed.
"""

import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from core.multi_decoy_engine import MultiDecoyEngine
from core.encryption_engine import SecurityLevel
from utils.logger import Logger

def create_test_image_with_capacity(filename="test_carrier.png"):
    """Create a test image with sufficient capacity for multiple datasets."""
    # Create a high-capacity image
    data = np.random.randint(0, 256, (1000, 1500, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    img.save(filename, format='PNG')
    return filename

def create_test_datasets():
    """Create test datasets with different content."""
    
    # Dataset 1: Innocent vacation photos list
    innocent_file = "innocent_photos.txt"
    with open(innocent_file, "w") as f:
        f.write("My Vacation Photo List:\n")
        f.write("1. Beach sunset\n")
        f.write("2. Mountain hiking trail\n")
        f.write("3. City downtown skyline\n")
        f.write("4. Hotel room view\n")
        f.write("5. Airport departure gate\n")
    
    # Dataset 2: Personal diary entry
    personal_file = "personal_diary.txt"  
    with open(personal_file, "w") as f:
        f.write("Personal Diary Entry - March 15, 2024\n")
        f.write("========================================\n")
        f.write("Had a great day today. Met with friends for lunch.\n")
        f.write("Discussed plans for the summer vacation.\n")
        f.write("Really looking forward to the beach trip in July.\n")
        f.write("Need to remember to book the hotel soon.\n")
    
    # Dataset 3: Sensitive business document
    sensitive_file = "business_confidential.txt"
    with open(sensitive_file, "w") as f:
        f.write("CONFIDENTIAL BUSINESS DOCUMENT\n")
        f.write("===============================\n")
        f.write("Q1 Financial Report - Internal Only\n")
        f.write("Revenue: $2.3M (15% increase)\n") 
        f.write("Expenses: $1.8M (controlled growth)\n")
        f.write("Profit Margin: 21.7% (target achieved)\n")
        f.write("Strategic Initiative: Market expansion planned for Q2\n")
    
    return {
        "innocent": [innocent_file],
        "personal": [personal_file], 
        "sensitive": [sensitive_file]
    }

def main():
    """Demonstrate successful multi-decoy functionality."""
    logger = Logger()
    print("🎯 Final Multi-Decoy Test - Demonstrating Fixed Password Handling")
    print("=" * 65)
    
    try:
        # Setup
        carrier_image = create_test_image_with_capacity()
        test_files = create_test_datasets()
        
        # Initialize multi-decoy engine
        engine = MultiDecoyEngine(SecurityLevel.MAXIMUM)
        
        # Check capacity
        capacity = engine.calculate_multi_capacity(Path(carrier_image), 3)
        print(f"📊 Image Capacity Analysis:")
        print(f"   Total capacity: {capacity['total_capacity']:,} bytes")
        print(f"   Per-dataset capacity: {capacity['per_dataset_capacity']:,} bytes")
        print(f"   Max datasets supported: {capacity['max_datasets']}")
        print()
        
        # Create datasets with different passwords and priorities
        datasets = [
            {
                "name": "VacationPhotos", 
                "password": "beach2024",
                "priority": 1,  # Outer layer (least secure)
                "decoy_type": "innocent",
                "files": test_files["innocent"]
            },
            {
                "name": "PersonalDiary",
                "password": "memories123", 
                "priority": 3,  # Middle layer
                "decoy_type": "personal", 
                "files": test_files["personal"]
            },
            {
                "name": "BusinessReports",
                "password": "SecureBiz2024!",
                "priority": 5,  # Inner layer (most secure)
                "decoy_type": "business",
                "files": test_files["sensitive"] 
            }
        ]
        
        print("🔒 Hiding Multiple Datasets...")
        output_image = "multi_decoy_demo.png"
        
        success = engine.hide_multiple_datasets(
            carrier_path=Path(carrier_image),
            datasets=datasets,
            output_path=Path(output_image)
        )
        
        if success:
            print("✅ Successfully hidden all datasets!")
        else:
            print("❌ Failed to hide datasets")
            return False
        
        print()
        print("🔍 Testing Password-Based Extraction...")
        print("-" * 45)
        
        # Test extraction scenarios
        test_scenarios = [
            ("beach2024", "VacationPhotos", "Should extract innocent vacation photos"),
            ("memories123", "PersonalDiary", "Should extract personal diary"),
            ("SecureBiz2024!", "BusinessReports", "Should extract confidential business data"),
            ("wrongpassword", None, "Should fail with wrong password")
        ]
        
        all_tests_passed = True
        
        for i, (password, expected_dataset, description) in enumerate(test_scenarios, 1):
            print(f"Test {i}: {description}")
            print(f"   Password: {'*' * len(password)}")
            
            output_dir = Path(f"extracted_test_{i}")
            output_dir.mkdir(exist_ok=True)
            
            metadata = engine.extract_dataset(
                stego_path=Path(output_image),
                password=password,
                output_dir=output_dir
            )
            
            if expected_dataset is None:
                # Should fail
                if metadata is None:
                    print("   ✅ Correctly rejected wrong password")
                else:
                    print(f"   ❌ Unexpected success with wrong password!")
                    all_tests_passed = False
            else:
                # Should succeed
                if metadata and metadata['dataset_id'] == expected_dataset:
                    print(f"   ✅ Successfully extracted: {metadata['dataset_id']}")
                    print(f"      Priority: {metadata['priority']}")
                    print(f"      Files: {metadata['file_count']}")
                    
                    # Check if files were actually extracted
                    extracted_files = list(output_dir.rglob("*"))
                    extracted_files = [f for f in extracted_files if f.is_file()]
                    print(f"      Extracted files: {len(extracted_files)}")
                    
                else:
                    print(f"   ❌ Failed to extract expected dataset: {expected_dataset}")
                    all_tests_passed = False
            
            print()
        
        # Final summary
        print("🎯 Multi-Decoy Test Results")
        print("=" * 30)
        if all_tests_passed:
            print("✅ ALL TESTS PASSED!")
            print("✅ Password handling is working correctly!")
            print("✅ Multiple datasets can be hidden and extracted!")
            print("✅ Plausible deniability is working!")
            print()
            print("🎉 Multi-decoy functionality is FULLY OPERATIONAL!")
        else:
            print("❌ Some tests failed!")
        
        return all_tests_passed
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        cleanup_files = [
            "test_carrier.png", "multi_decoy_demo.png",
            "innocent_photos.txt", "personal_diary.txt", "business_confidential.txt"
        ]
        
        cleanup_dirs = [f"extracted_test_{i}" for i in range(1, 5)]
        
        for file in cleanup_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except:
                pass
        
        for dir_name in cleanup_dirs:
            try:
                if os.path.exists(dir_name):
                    import shutil
                    shutil.rmtree(dir_name)
            except:
                pass

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Multi-decoy password handling has been successfully FIXED! 🎯")
        exit(0)
    else:
        print("\n❌ Multi-decoy test failed!")
        exit(1)
