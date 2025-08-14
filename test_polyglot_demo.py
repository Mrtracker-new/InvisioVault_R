#!/usr/bin/env python3
"""
Polyglot File Demo and Testing Script
This script demonstrates how to create and test polyglot files properly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add InVisioVault to path
sys.path.insert(0, str(Path(__file__).parent))

from core.self_executing_engine import SelfExecutingEngine
from utils.logger import Logger

def create_simple_test_executable():
    """Create a simple test executable for demonstration."""
    print("🔧 Creating test executable...")
    
    # Create a simple batch file that works on Windows
    test_script_content = '''@echo off
echo Hello from the embedded executable!
echo This proves the polyglot file is working as an executable.
echo Current directory: %CD%
echo Current time: %TIME%
pause
'''
    
    # Save as .bat file
    test_exe_path = Path(__file__).parent / "temp_test.bat"
    with open(test_exe_path, 'w') as f:
        f.write(test_script_content)
    
    print(f"✅ Created test executable: {test_exe_path}")
    return str(test_exe_path)

def create_sample_image():
    """Create a simple test image."""
    print("🖼️ Creating sample image...")
    
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple 200x200 image
        img = Image.new('RGB', (200, 200), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw some text
        draw.text((50, 80), "Test Image", fill='darkblue')
        draw.text((30, 100), "for Polyglot Demo", fill='darkblue')
        
        # Save as PNG
        test_img_path = Path(__file__).parent / "temp_test_image.png"
        img.save(test_img_path)
        
        print(f"✅ Created test image: {test_img_path}")
        return str(test_img_path)
        
    except ImportError:
        print("❌ PIL not available, using alternative method...")
        
        # Create a minimal BMP file manually
        test_img_path = Path(__file__).parent / "temp_test_image.bmp"
        
        # Simple 2x2 pixel BMP file (54 bytes header + 16 bytes data)
        bmp_data = bytearray([
            # BMP Header (14 bytes)
            0x42, 0x4D,  # Signature "BM"
            0x46, 0x00, 0x00, 0x00,  # File size (70 bytes)
            0x00, 0x00,  # Reserved
            0x00, 0x00,  # Reserved
            0x36, 0x00, 0x00, 0x00,  # Offset to pixel data (54 bytes)
            
            # DIB Header (40 bytes)
            0x28, 0x00, 0x00, 0x00,  # DIB header size
            0x02, 0x00, 0x00, 0x00,  # Width (2 pixels)
            0x02, 0x00, 0x00, 0x00,  # Height (2 pixels)
            0x01, 0x00,  # Planes
            0x18, 0x00,  # Bits per pixel (24)
            0x00, 0x00, 0x00, 0x00,  # Compression
            0x10, 0x00, 0x00, 0x00,  # Image size
            0x13, 0x0B, 0x00, 0x00,  # X pixels per meter
            0x13, 0x0B, 0x00, 0x00,  # Y pixels per meter
            0x00, 0x00, 0x00, 0x00,  # Colors used
            0x00, 0x00, 0x00, 0x00,  # Important colors
            
            # Pixel data (16 bytes)
            0xFF, 0x00, 0x00,  # Red pixel
            0x00, 0xFF, 0x00,  # Green pixel
            0x00, 0x00,  # Padding
            0x00, 0x00, 0xFF,  # Blue pixel
            0xFF, 0xFF, 0x00,  # Yellow pixel
            0x00, 0x00   # Padding
        ])
        
        with open(test_img_path, 'wb') as f:
            f.write(bmp_data)
            
        print(f"✅ Created minimal BMP image: {test_img_path}")
        return str(test_img_path)

def test_polyglot_creation():
    """Test the polyglot file creation process."""
    print("\n🚀 Testing Polyglot File Creation")
    print("=" * 50)
    
    # Initialize the engine
    engine = SelfExecutingEngine()
    
    try:
        # Create test files
        image_path = create_sample_image()
        exe_path = create_simple_test_executable()
        
        # Create polyglot file
        output_path = Path(__file__).parent / "test_polyglot.exe"
        
        print(f"\n📝 Creating polyglot file...")
        print(f"   Image: {image_path}")
        print(f"   Executable: {exe_path}")
        print(f"   Output: {output_path}")
        
        success = engine.create_polyglot_executable(
            image_path=image_path,
            executable_path=exe_path,
            output_path=str(output_path)
        )
        
        if success:
            print(f"✅ Polyglot file created successfully!")
            print(f"   Location: {output_path}")
            print(f"   Size: {output_path.stat().st_size} bytes")
            
            # Test analysis
            print(f"\n🔍 Testing polyglot analysis...")
            result = engine.extract_and_execute(
                image_path=str(output_path),
                execution_mode='safe'
            )
            
            print(f"Analysis result: {result}")
            
            # Instructions for user
            print(f"\n📋 HOW TO TEST YOUR POLYGLOT FILE:")
            print(f"1. Double-click: {output_path}")
            print(f"   → Should run as executable and show 'Hello from embedded executable!'")
            print(f"2. Open with image viewer: {output_path}")
            print(f"   → Should display the test image")
            print(f"3. The same file works as BOTH image and executable!")
            
        else:
            print("❌ Failed to create polyglot file")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temporary files
        cleanup_files = [
            "temp_test.bat",
            "temp_test_image.png", 
            "temp_test_image.bmp"
        ]
        
        for file in cleanup_files:
            file_path = Path(__file__).parent / file
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"🗑️  Cleaned up: {file}")
                except:
                    pass

if __name__ == "__main__":
    print("🎯 InVisioVault Polyglot File Demo")
    print("This script will create and test a polyglot file.")
    print("The polyglot will work as both an image AND executable.")
    
    input("\nPress Enter to start the demo...")
    
    test_polyglot_creation()
    
    print("\n✨ Demo completed!")
    print("If successful, you now have 'test_polyglot.exe' that works as both image and executable.")
