#!/usr/bin/env python3
"""
Simple test to verify multi-decoy hide and extract functionality
"""

import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from core.multi_decoy_engine import MultiDecoyEngine
from core.encryption_engine import SecurityLevel
from utils.logger import Logger

def create_simple_test_image(filename="simple_test.png"):
    """Create a simple test image with high capacity."""
    # Create a larger image for more capacity
    data = np.random.randint(0, 256, (1200, 1600, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    img.save(filename, format='PNG')
    return filename

def create_simple_test_files():
    """Create simple test files."""
    test_files = {}
    
    # Dataset 1: Simple text file
    test1_path = "simple_test1.txt"
    with open(test1_path, "w") as f:
        f.write("This is dataset 1 - simple test data.")
    test_files["dataset1"] = [test1_path]
    
    # Dataset 2: Another text file  
    test2_path = "simple_test2.txt"
    with open(test2_path, "w") as f:
        f.write("This is dataset 2 - different test data.")
    test_files["dataset2"] = [test2_path]
    
    return test_files

def test_simple_multi_decoy():
    """Test basic multi-decoy functionality."""
    logger = Logger()
    logger.info("Starting Simple Multi-Decoy Test")
    
    try:
        # Create test environment
        carrier_image = create_simple_test_image()
        test_files = create_simple_test_files()
        
        # Initialize engine
        engine = MultiDecoyEngine(SecurityLevel.MAXIMUM)
        
        # Test capacity
        capacity = engine.calculate_multi_capacity(Path(carrier_image), 2)
        logger.info(f"Image capacity: {capacity}")
        
        # Create simple datasets
        datasets = [
            {
                "name": "SimpleDataset1",
                "password": "password123",
                "priority": 1,
                "decoy_type": "standard",
                "files": test_files["dataset1"]
            },
            {
                "name": "SimpleDataset2", 
                "password": "secret456",
                "priority": 2,
                "decoy_type": "standard",
                "files": test_files["dataset2"]
            }
        ]
        
        # Hide datasets
        logger.info("Hiding datasets...")
        output_image = "simple_result.png"
        success = engine.hide_multiple_datasets(
            carrier_path=Path(carrier_image),
            datasets=datasets,
            output_path=Path(output_image)
        )
        
        if not success:
            logger.error("Failed to hide datasets!")
            return False
        
        logger.info("Successfully hidden datasets!")
        
        # Test extraction with first password
        logger.info("\nTesting extraction with first password...")
        output_dir = Path("extracted_simple1")
        output_dir.mkdir(exist_ok=True)
        
        metadata = engine.extract_dataset(
            stego_path=Path(output_image),
            password="password123",
            output_dir=output_dir
        )
        
        if metadata:
            logger.info(f"SUCCESS: Extracted dataset {metadata['dataset_id']}")
        else:
            logger.error("FAILED: Could not extract dataset with first password")
            return False
        
        # Test extraction with second password
        logger.info("\nTesting extraction with second password...")
        output_dir2 = Path("extracted_simple2")
        output_dir2.mkdir(exist_ok=True)
        
        metadata2 = engine.extract_dataset(
            stego_path=Path(output_image),
            password="secret456", 
            output_dir=output_dir2
        )
        
        if metadata2:
            logger.info(f"SUCCESS: Extracted dataset {metadata2['dataset_id']}")
        else:
            logger.error("FAILED: Could not extract dataset with second password")
            return False
        
        # Test with wrong password
        logger.info("\nTesting extraction with wrong password...")
        output_dir3 = Path("extracted_wrong")
        output_dir3.mkdir(exist_ok=True)
        
        metadata3 = engine.extract_dataset(
            stego_path=Path(output_image),
            password="wrongpassword",
            output_dir=output_dir3
        )
        
        if metadata3:
            logger.error("UNEXPECTED: Extracted dataset with wrong password!")
            return False
        else:
            logger.info("SUCCESS: Correctly rejected wrong password")
        
        logger.info("\nAll tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        cleanup_files = [
            "simple_test.png", "simple_result.png",
            "simple_test1.txt", "simple_test2.txt",
            "extracted_simple1", "extracted_simple2", "extracted_wrong"
        ]
        
        for item in cleanup_files:
            try:
                if os.path.isfile(item):
                    os.remove(item)
                elif os.path.isdir(item):
                    import shutil
                    shutil.rmtree(item)
            except:
                pass

if __name__ == "__main__":
    success = test_simple_multi_decoy()
    if success:
        print("\n✅ Simple multi-decoy test passed!")
        exit(0)
    else:
        print("\n❌ Simple multi-decoy test failed!")
        exit(1)
