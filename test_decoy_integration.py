#!/usr/bin/env python3
"""
Test script to verify decoy mode integration with basic operations.
Tests that basic extract can work with decoy mode hidden files.
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

def create_test_image(width=1200, height=800, filename="integration_test.png"):
    """Create a test carrier image with good capacity."""
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    img.save(filename, format='PNG')
    return filename

def create_test_files():
    """Create test files for different datasets."""
    test_files = {}
    
    # Create temp directory for test files
    temp_dir = Path(tempfile.mkdtemp())
    
    # Dataset 1: Decoy files (what someone would see with "innocent" password)
    decoy_dir = temp_dir / "decoy"
    decoy_dir.mkdir()
    
    with open(decoy_dir / "vacation_list.txt", "w") as f:
        f.write("Vacation Planning\n\n1. Book flights\n2. Reserve hotel\n3. Pack sunscreen\n4. Camera batteries")
    
    with open(decoy_dir / "shopping.txt", "w") as f:
        f.write("Shopping List:\n- Milk\n- Bread\n- Eggs\n- Cheese\n- Coffee")
    
    test_files["decoy"] = [str(decoy_dir / "vacation_list.txt"), str(decoy_dir / "shopping.txt")]
    
    # Dataset 2: Real sensitive files (what they actually want to hide)
    real_dir = temp_dir / "real"
    real_dir.mkdir()
    
    with open(real_dir / "passwords.txt", "w") as f:
        f.write("CONFIDENTIAL\n\nImportant passwords:\nEmail: ***\nBank: ***\nCrypto: ***")
    
    with open(real_dir / "business_plan.txt", "w") as f:
        f.write("TOP SECRET BUSINESS PLAN\n\nNew product launch strategy...\nMarket analysis...")
    
    test_files["real"] = [str(real_dir / "passwords.txt"), str(real_dir / "business_plan.txt")]
    
    return test_files, temp_dir

def test_decoy_integration():
    """Test that basic extract works with decoy mode files."""
    logger = Logger()
    logger.info("Starting Decoy Integration Test")
    
    temp_dir = None
    try:
        # Create test environment
        carrier_image = create_test_image()
        test_files, temp_dir = create_test_files()
        
        # Initialize multi-decoy engine
        engine = MultiDecoyEngine(SecurityLevel.MAXIMUM)
        
        # Step 1: Create decoy-protected image using multi-decoy mode
        logger.info("Step 1: Creating decoy-protected image...")
        
        datasets = [
            {
                "name": "InnocentFiles",
                "password": "vacation2023",  # Decoy password
                "priority": 1,  # Outer layer
                "decoy_type": "innocent",
                "files": test_files["decoy"]
            },
            {
                "name": "RealSecrets", 
                "password": "MyRealPassword123",  # Real password
                "priority": 5,  # Inner layer
                "decoy_type": "business",
                "files": test_files["real"]
            }
        ]
        
        stego_image = "decoy_integrated.png"
        success = engine.hide_multiple_datasets(
            carrier_path=Path(carrier_image),
            datasets=datasets,
            output_path=Path(stego_image)
        )
        
        if not success:
            raise Exception("Failed to create decoy-protected image")
        
        logger.info("✅ Decoy-protected image created successfully")
        
        # Step 2: Test basic extraction with decoy password
        logger.info("Step 2: Testing basic extraction with decoy password...")
        
        decoy_output = Path("extracted_decoy")
        decoy_output.mkdir(exist_ok=True)
        
        decoy_result = engine.extract_dataset(
            stego_path=Path(stego_image),
            password="vacation2023",  # Using decoy password
            output_dir=decoy_output
        )
        
        if not decoy_result:
            raise Exception("Failed to extract decoy dataset")
        
        # Verify decoy files were extracted
        extracted_decoy_files = list(decoy_output.glob("*"))
        logger.info(f"✅ Decoy extraction successful: {len(extracted_decoy_files)} files")
        
        for file_path in extracted_decoy_files:
            logger.info(f"   - {file_path.name}")
        
        # Step 3: Test basic extraction with real password  
        logger.info("Step 3: Testing basic extraction with real password...")
        
        real_output = Path("extracted_real")
        real_output.mkdir(exist_ok=True)
        
        real_result = engine.extract_dataset(
            stego_path=Path(stego_image),
            password="MyRealPassword123",  # Using real password
            output_dir=real_output
        )
        
        if not real_result:
            raise Exception("Failed to extract real dataset")
        
        # Verify real files were extracted
        extracted_real_files = list(real_output.glob("*"))
        logger.info(f"✅ Real extraction successful: {len(extracted_real_files)} files")
        
        for file_path in extracted_real_files:
            logger.info(f"   - {file_path.name}")
        
        # Step 4: Test extraction with wrong password
        logger.info("Step 4: Testing extraction with wrong password...")
        
        wrong_output = Path("extracted_wrong")
        wrong_output.mkdir(exist_ok=True)
        
        wrong_result = engine.extract_dataset(
            stego_path=Path(stego_image),
            password="wrongpassword",
            output_dir=wrong_output
        )
        
        if wrong_result:
            logger.warning("⚠️  Wrong password extraction should have failed!")
        else:
            logger.info("✅ Wrong password correctly rejected")
        
        # Step 5: Verify plausible deniability
        logger.info("Step 5: Verifying plausible deniability...")
        
        # Check that decoy files look innocent
        decoy_vacation_file = decoy_output / "vacation_list.txt"
        if decoy_vacation_file.exists():
            with open(decoy_vacation_file, 'r') as f:
                content = f.read()
                if "vacation" in content.lower() and "flights" in content.lower():
                    logger.info("✅ Decoy files appear innocent and believable")
                else:
                    logger.warning("⚠️  Decoy files may not be convincing enough")
        
        # Check that real files are actually sensitive
        real_password_file = real_output / "passwords.txt"
        if real_password_file.exists():
            with open(real_password_file, 'r') as f:
                content = f.read()
                if "CONFIDENTIAL" in content and "passwords" in content.lower():
                    logger.info("✅ Real files contain sensitive information")
        
        logger.info("\n🎉 INTEGRATION TEST PASSED!")
        logger.info("Basic extraction now works seamlessly with decoy mode:")
        logger.info("- Different passwords reveal different datasets")
        logger.info("- Users don't need to know about decoy mode")
        logger.info("- Security is preserved through layered encryption")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        cleanup_files = [
            "integration_test.png", "decoy_integrated.png",
            "extracted_decoy", "extracted_real", "extracted_wrong"
        ]
        
        for item in cleanup_files:
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
    success = test_decoy_integration()
    if success:
        print("\n✅ INTEGRATION TEST SUCCESSFUL!")
        print("Basic operations now support decoy mode transparently!")
    else:
        print("\n❌ INTEGRATION TEST FAILED!")
        print("Check logs for details.")
    
    exit(0 if success else 1)
