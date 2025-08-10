#!/usr/bin/env python3
"""
Test to verify two-factor authentication seeding fix.
Tests that fragments are correctly hidden and extracted with proper seeding.
"""

import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel
from utils.logger import Logger


def create_test_image(width=800, height=600, filename="test_carrier.png"):
    """Create a test carrier image."""
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    img.save(filename, format='PNG')
    return filename


def test_fragment_seeding():
    """Test that fragment seeding works correctly for two-factor authentication."""
    logger = Logger()
    logger.info("Testing two-factor fragment seeding...")
    
    temp_dir = None
    try:
        # Create test environment
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test carrier image
        carrier_image = create_test_image(1200, 800, str(temp_dir / "carrier.png"))
        
        # Initialize engines
        stego_engine = SteganographyEngine()
        encryption_engine = EncryptionEngine(SecurityLevel.MAXIMUM)
        
        # Simulate two-factor distribution process
        password = "TestPassword123"
        test_data = b"This is test data for two-factor authentication fragment testing."
        
        # Encrypt the test data
        encrypted_data = encryption_engine.encrypt_with_metadata(test_data, password)
        logger.info(f"Encrypted data size: {len(encrypted_data)} bytes")
        
        # Create fragment metadata (simulating what the dialog does)
        fragment_index = 0
        metadata = {
            'fragment_index': fragment_index,
            'total_fragments': 2,
            'fragment_size': len(encrypted_data),
            'redundancy_level': 2
        }
        
        # Combine metadata and fragment (simulating dialog process)
        metadata_str = str(metadata).encode('utf-8')
        combined_data = len(metadata_str).to_bytes(4, 'big') + metadata_str + encrypted_data
        
        logger.info(f"Combined data size: {len(combined_data)} bytes")
        
        # Test hiding with the same seed calculation as the dialog
        fragment_seed = hash(password + str(fragment_index)) % (2**32)
        logger.info(f"Using fragment seed: {fragment_seed}")
        
        output_path = temp_dir / "fragment_01.png"
        
        # Hide the fragment
        success = stego_engine.hide_data(
            Path(carrier_image),
            combined_data,
            output_path,
            randomize=True,
            seed=fragment_seed
        )
        
        if not success:
            raise Exception("Failed to hide fragment data")
        
        logger.info("✅ Fragment hidden successfully")
        
        # Test extraction with the same seed
        extracted_data = stego_engine.extract_data(
            output_path,
            randomize=True,
            seed=fragment_seed
        )
        
        if not extracted_data:
            raise Exception("Failed to extract fragment data")
        
        logger.info("✅ Fragment extracted successfully")
        
        # Verify the extracted data matches what we put in
        if extracted_data != combined_data:
            raise Exception(f"Data mismatch: expected {len(combined_data)}, got {len(extracted_data)}")
        
        logger.info("✅ Data integrity verified")
        
        # Test metadata extraction
        metadata_size = int.from_bytes(extracted_data[:4], 'big')
        extracted_metadata_str = extracted_data[4:4+metadata_size].decode('utf-8')
        extracted_fragment_data = extracted_data[4+metadata_size:]
        
        extracted_metadata = eval(extracted_metadata_str)
        
        if extracted_metadata['fragment_index'] != fragment_index:
            raise Exception("Metadata fragment index mismatch")
        
        logger.info("✅ Metadata extracted correctly")
        
        # Test decryption of the fragment
        decrypted_data = encryption_engine.decrypt_with_metadata(extracted_fragment_data, password)
        
        if decrypted_data != test_data:
            raise Exception("Decrypted data doesn't match original")
        
        logger.info("✅ Fragment decryption successful")
        
        logger.info("🎉 TWO-FACTOR SEEDING TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"Two-factor seeding test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


def test_seeding_mismatch():
    """Test that wrong seeds fail to extract data correctly."""
    logger = Logger()
    logger.info("Testing seeding mismatch scenarios...")
    
    temp_dir = None
    try:
        # Create test environment
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test carrier image
        carrier_image = create_test_image(800, 600, str(temp_dir / "carrier2.png"))
        
        # Initialize engines
        stego_engine = SteganographyEngine()
        
        # Test data
        password = "TestPassword123"
        test_data = b"Test data for seeding mismatch"
        
        # Hide with one seed
        correct_seed = hash(password + "0") % (2**32)
        output_path = temp_dir / "seeded.png"
        
        success = stego_engine.hide_data(
            Path(carrier_image),
            test_data,
            output_path,
            randomize=True,
            seed=correct_seed
        )
        
        if not success:
            raise Exception("Failed to hide data with correct seed")
        
        logger.info("✅ Data hidden with correct seed")
        
        # Try to extract with wrong seed
        wrong_seed = hash(password + "1") % (2**32)
        
        extracted_data = stego_engine.extract_data(
            output_path,
            randomize=True,
            seed=wrong_seed
        )
        
        if extracted_data:
            # If we get data with wrong seed, it should be garbage, not our original data
            if extracted_data == test_data:
                raise Exception("Wrong seed extracted correct data - this shouldn't happen!")
            logger.info("✅ Wrong seed returned garbage data as expected")
        else:
            logger.info("✅ Wrong seed returned None as expected")
        
        # Try to extract with correct seed
        correct_extracted = stego_engine.extract_data(
            output_path,
            randomize=True,
            seed=correct_seed
        )
        
        if not correct_extracted:
            raise Exception("Failed to extract with correct seed")
        
        if correct_extracted != test_data:
            raise Exception("Correct seed didn't return correct data")
        
        logger.info("✅ Correct seed extracted correct data")
        
        logger.info("🎉 SEEDING MISMATCH TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"Seeding mismatch test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


if __name__ == "__main__":
    print("Testing two-factor authentication seeding fix...")
    
    test1_success = test_fragment_seeding()
    test2_success = test_seeding_mismatch()
    
    overall_success = test1_success and test2_success
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("Two-factor authentication seeding is now working correctly.")
        print("The 'No data found in fragment' error should be resolved.")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Check the logs for details.")
    
    exit(0 if overall_success else 1)
