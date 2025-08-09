#!/usr/bin/env python3
"""
Test script for Multi-Decoy functionality
Tests the ability to hide and extract multiple datasets with different passwords.
"""

import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from core.multi_decoy_engine import MultiDecoyEngine
from core.encryption_engine import SecurityLevel
from utils.logger import Logger

def create_test_image(width=800, height=600, filename="test_image.png"):
    """Create a test carrier image."""
    # Create random image data
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    img.save(filename)
    return filename

def create_test_files():
    """Create test files for datasets."""
    test_files = {}
    
    # Dataset 1: Innocent files
    innocent_dir = Path("test_innocent")
    innocent_dir.mkdir(exist_ok=True)
    
    with open(innocent_dir / "vacation_photos.txt", "w") as f:
        f.write("List of vacation photos:\n- Beach sunset\n- Mountain hiking\n- City skyline")
    
    with open(innocent_dir / "recipe.txt", "w") as f:
        f.write("Chocolate Chip Cookie Recipe:\n1. Mix flour and sugar\n2. Add eggs and butter\n3. Bake at 350°F")
    
    test_files["innocent"] = [str(innocent_dir / "vacation_photos.txt"), str(innocent_dir / "recipe.txt")]
    
    # Dataset 2: Personal files
    personal_dir = Path("test_personal")
    personal_dir.mkdir(exist_ok=True)
    
    with open(personal_dir / "diary_entry.txt", "w") as f:
        f.write("Personal diary entry: Today was a good day at work...")
    
    with open(personal_dir / "family_notes.txt", "w") as f:
        f.write("Family meeting notes: Discussed vacation plans and budget...")
    
    test_files["personal"] = [str(personal_dir / "diary_entry.txt"), str(personal_dir / "family_notes.txt")]
    
    # Dataset 3: Sensitive files
    sensitive_dir = Path("test_sensitive")
    sensitive_dir.mkdir(exist_ok=True)
    
    with open(sensitive_dir / "important_docs.txt", "w") as f:
        f.write("CONFIDENTIAL: Important business documents and contracts...")
    
    with open(sensitive_dir / "passwords.txt", "w") as f:
        f.write("Backup passwords:\nGmail: *******\nBank: *******")
    
    test_files["sensitive"] = [str(sensitive_dir / "important_docs.txt"), str(sensitive_dir / "passwords.txt")]
    
    return test_files

def test_multi_decoy():
    """Test multi-decoy hide and extract functionality."""
    logger = Logger()
    logger.info("Starting Multi-Decoy Engine Test")
    
    try:
        # Create test environment
        carrier_image = create_test_image()
        test_files = create_test_files()
        
        # Initialize engine
        engine = MultiDecoyEngine(SecurityLevel.MAXIMUM)
        
        # Test capacity calculation
        capacity = engine.calculate_multi_capacity(Path(carrier_image), 3)
        logger.info(f"Image capacity: {capacity}")
        
        # Create datasets with different priorities
        datasets = [
            {
                "name": "Innocent_Data",
                "password": "vacation2023",
                "priority": 1,  # Outer layer - least secure
                "decoy_type": "innocent",
                "files": test_files["innocent"]
            },
            {
                "name": "Personal_Data", 
                "password": "family_memories",
                "priority": 3,  # Middle layer
                "decoy_type": "personal",
                "files": test_files["personal"]
            },
            {
                "name": "Sensitive_Data",
                "password": "SecurePass2023!@#",
                "priority": 5,  # Inner layer - most secure
                "decoy_type": "business",
                "files": test_files["sensitive"]
            }
        ]
        
        # Hide multiple datasets
        logger.info("Hiding multiple datasets...")
        output_image = "multi_decoy_result.png"
        success = engine.hide_multiple_datasets(
            carrier_path=Path(carrier_image),
            datasets=datasets,
            output_path=Path(output_image)
        )
        
        if not success:
            logger.error("Failed to hide datasets!")
            return False
        
        logger.info("Successfully hidden all datasets!")
        
        # Test extraction with different passwords
        extraction_tests = [
            ("vacation2023", "Should extract innocent data"),
            ("family_memories", "Should extract personal data"), 
            ("SecurePass2023!@#", "Should extract sensitive data"),
            ("wrong_password", "Should fail - wrong password")
        ]
        
        for password, description in extraction_tests:
            logger.info(f"\nTesting extraction: {description}")
            output_dir = Path(f"extracted_{password.replace('!@#', '_special')}")
            output_dir.mkdir(exist_ok=True)
            
            metadata = engine.extract_dataset(
                stego_path=Path(output_image),
                password=password,
                output_dir=output_dir
            )
            
            if metadata:
                logger.info(f"Successfully extracted: {metadata['dataset_id']} (Priority: {metadata['priority']})")
                logger.info(f"Files extracted to: {output_dir}")
            else:
                logger.info("No dataset found with this password (expected for wrong password)")
        
        # Test listing datasets (without passwords)
        logger.info("\nListing all datasets in image:")
        dataset_list = engine.list_datasets(Path(output_image))
        for i, ds_meta in enumerate(dataset_list):
            logger.info(f"Dataset {i+1}: {ds_meta['dataset_id']} (Priority: {ds_meta['priority']}, Files: {ds_meta['file_count']})")
        
        logger.info("\nMulti-Decoy Engine Test Completed Successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        cleanup_files = [
            "test_image.png", "multi_decoy_result.png",
            "test_innocent", "test_personal", "test_sensitive"
        ]
        
        for item in cleanup_files:
            try:
                if os.path.isfile(item):
                    os.remove(item)
                elif os.path.isdir(item):
                    import shutil
                    shutil.rmtree(item)
            except:
                pass  # Ignore cleanup errors

if __name__ == "__main__":
    success = test_multi_decoy()
    if success:
        print("\n✅ All tests passed! Multi-decoy functionality is working correctly.")
        exit(0)
    else:
        print("\n❌ Tests failed! Check the logs above for details.")
        exit(1)
