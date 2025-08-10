#!/usr/bin/env python3
"""
Quick test to verify extraction success message shows correct files.
"""

import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from core.multi_decoy_engine import MultiDecoyEngine
from core.encryption_engine import SecurityLevel
from utils.logger import Logger

def create_test_image(filename="test_extraction.png"):
    """Create a test carrier image."""
    data = np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    img.save(filename, format='PNG')
    return filename

def test_extraction_message():
    """Test that extraction shows correct file names."""
    logger = Logger()
    logger.info("Testing extraction message accuracy...")
    
    temp_dir = None
    try:
        # Create test environment
        carrier_image = create_test_image()
        
        # Create temp directory with test files
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create specific test files
        test_file1 = temp_dir / "document1.txt"
        test_file2 = temp_dir / "document2.txt"
        
        with open(test_file1, "w") as f:
            f.write("This is document 1 content.")
        
        with open(test_file2, "w") as f:
            f.write("This is document 2 content.")
        
        # Initialize multi-decoy engine
        engine = MultiDecoyEngine(SecurityLevel.MAXIMUM)
        
        # Create a simple dataset
        dataset = {
            "name": "TestDocuments",
            "password": "test123456",
            "priority": 1,
            "decoy_type": "standard",
            "files": [str(test_file1), str(test_file2)]
        }
        
        # Hide the dataset
        stego_image = "test_stego.png"
        success = engine.hide_multiple_datasets(
            carrier_path=Path(carrier_image),
            datasets=[dataset],
            output_path=Path(stego_image)
        )
        
        if not success:
            logger.error("Failed to hide dataset")
            return False
        
        logger.info("✅ Dataset hidden successfully")
        
        # Create output directory with some existing files (like the issue you experienced)
        output_dir = Path("test_extraction_output")
        output_dir.mkdir(exist_ok=True)
        
        # Add some existing files to the output directory
        existing_file1 = output_dir / "existing_file1.png"
        existing_file2 = output_dir / "existing_file2.key"
        
        with open(existing_file1, "w") as f:
            f.write("This is an existing file that should NOT appear in extraction results")
        
        with open(existing_file2, "w") as f:
            f.write("Another existing file")
        
        logger.info("Created existing files in output directory")
        
        # Extract the dataset
        metadata = engine.extract_dataset(
            stego_path=Path(stego_image),
            password="test123456",
            output_dir=output_dir
        )
        
        if not metadata:
            logger.error("Failed to extract dataset")
            return False
        
        logger.info("✅ Dataset extracted successfully")
        
        # Check the metadata for extracted files
        if 'extracted_files' in metadata:
            logger.info("Extracted files information found in metadata:")
            for file_info in metadata['extracted_files']:
                logger.info(f"  - {file_info['name']} at {file_info['path']}")
            
            # Verify we got the correct files (not the existing ones)
            extracted_names = [f['name'] for f in metadata['extracted_files']]
            expected_names = ['document1.txt', 'document2.txt']
            
            if set(extracted_names) == set(expected_names):
                logger.info("✅ SUCCESS: Extraction message will show correct files!")
                logger.info(f"Expected files: {expected_names}")
                logger.info(f"Extracted files: {extracted_names}")
                return True
            else:
                logger.error(f"❌ MISMATCH: Expected {expected_names}, got {extracted_names}")
                return False
        else:
            logger.error("❌ No extracted_files information in metadata")
            return False
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        cleanup_items = [
            "test_extraction.png", "test_stego.png", 
            "test_extraction_output"
        ]
        
        for item in cleanup_items:
            try:
                if os.path.isfile(item):
                    os.remove(item)
                elif os.path.isdir(item):
                    shutil.rmtree(item)
            except:
                pass
        
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

if __name__ == "__main__":
    success = test_extraction_message()
    if success:
        print("\n✅ TEST PASSED!")
        print("The extraction success message fix is working correctly.")
        print("You should now see only the files that were actually extracted,")
        print("not all files in the output directory.")
    else:
        print("\n❌ TEST FAILED!")
        print("The extraction message still has issues.")
    
    exit(0 if success else 1)
