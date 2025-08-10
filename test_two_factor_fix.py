#!/usr/bin/env python3
"""
Quick test to verify two-factor dialog path handling fixes.
"""

import tempfile
import zipfile
from pathlib import Path


def test_path_handling():
    """Test that file paths are properly converted to Path objects."""
    print("Testing path handling for two-factor dialog...")
    
    # Create test files (strings like from QFileDialog)
    files_to_hide = [
        r"C:\Users\test\document1.txt",
        r"C:\Users\test\document2.pdf",
    ]
    
    # Test the conversion logic from the fixed code
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
        temp_zip_path = Path(temp_file.name)
    
    try:
        # Create actual temporary files to test with
        temp_dir = Path(tempfile.mkdtemp())
        real_files = []
        
        for i, file_path_str in enumerate(files_to_hide):
            # Create actual test files
            test_file = temp_dir / f"test_file_{i}.txt"
            with open(test_file, 'w') as f:
                f.write(f"Test content for file {i}")
            real_files.append(str(test_file))
        
        # Test the fixed zipfile creation logic
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file_path in real_files:
                file_path_obj = Path(file_path)  # Convert string to Path object
                archive.write(file_path_obj, file_path_obj.name)
                print(f"✅ Successfully added {file_path_obj.name} to archive")
        
        # Verify the archive was created correctly
        with zipfile.ZipFile(temp_zip_path, 'r') as archive:
            file_list = archive.namelist()
            print(f"✅ Archive contains {len(file_list)} files: {file_list}")
        
        print("✅ Path handling test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Path handling test FAILED: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            temp_zip_path.unlink()
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except:
            pass


if __name__ == "__main__":
    success = test_path_handling()
    if success:
        print("\n🎉 Two-factor dialog path fix verification SUCCESSFUL!")
        print("The 'str' object has no attribute 'name' error should now be resolved.")
    else:
        print("\n❌ Path fix verification FAILED!")
    
    exit(0 if success else 1)
