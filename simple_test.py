"""
Simple Anti-Detection Test
Basic test to verify the core functionality is working.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def simple_test():
    """Run a simple test focusing on core functionality."""
    
    print("🔧 Simple InVisioVault Test")
    print("=" * 40)
    
    # Test basic imports
    print("\n📦 Testing imports...")
    try:
        from core.steganography_engine import SteganographyEngine
        from core.encryption_engine import EncryptionEngine, SecurityLevel
        print("✅ Basic modules imported successfully")
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False
    
    # Test enhanced imports (optional)
    try:
        from core.enhanced_steganography_engine import EnhancedSteganographyEngine
        enhanced_available = True
        print("✅ Enhanced modules imported successfully")
    except ImportError as e:
        print(f"⚠️ Enhanced modules not available: {e}")
        enhanced_available = False
    
    # Create simple test data
    print("\n📄 Creating test data...")
    test_message = "Hello! This is a secret test message for steganography."
    test_data = test_message.encode('utf-8')
    password = "test123"
    print(f"✅ Test message: {len(test_data)} bytes")
    
    # Create simple test image
    print("\n🖼️ Creating test image...")
    img_size = (400, 300)  # Small but adequate
    test_img_array = np.random.randint(50, 205, (*img_size, 3), dtype=np.uint8)  # Avoid extremes
    
    carrier_path = Path("test_carrier.png")
    output_path = Path("test_stego.png")
    
    # Save test image
    img = Image.fromarray(test_img_array)
    img.save(carrier_path, 'PNG')
    print(f"✅ Test carrier created: {carrier_path}")
    
    try:
        # Test 1: Basic SteganographyEngine
        print("\n🔍 Testing Basic SteganographyEngine...")
        basic_engine = SteganographyEngine()
        
        # Test capacity
        capacity = basic_engine.calculate_capacity(carrier_path)
        print(f"  📊 Image capacity: {capacity:,} bytes")
        
        if len(test_data) > capacity:
            print(f"  ⚠️ Test data too large, reducing size...")
            test_data = test_data[:capacity//2]  # Use half capacity to be safe
        
        # Test hiding (without randomization for simplicity)
        print(f"  🔒 Hiding {len(test_data)} bytes...")
        success = basic_engine.hide_data(
            carrier_path=carrier_path,
            data=test_data,
            output_path=output_path,
            randomize=False  # Disable randomization for simple test
        )
        
        if success:
            print("  ✅ Basic hiding successful!")
            
            # Test extraction
            print("  📤 Extracting data...")
            extracted_data = basic_engine.extract_data(
                stego_path=output_path,
                randomize=False
            )
            
            if extracted_data and extracted_data == test_data:
                print("  ✅ Basic extraction successful!")
                print("  ✅ Data integrity verified!")
                print(f"  📝 Extracted message: {extracted_data.decode('utf-8')}")
                basic_test_passed = True
            else:
                print("  ❌ Basic extraction failed!")
                basic_test_passed = False
        else:
            print("  ❌ Basic hiding failed!")
            basic_test_passed = False
        
        # Test 2: Enhanced Engine (if available)
        enhanced_test_passed = True
        if enhanced_available:
            print("\n🛡️ Testing Enhanced SteganographyEngine...")
            try:
                enhanced_engine = EnhancedSteganographyEngine(use_anti_detection=False)  # Start with fast mode
                
                enhanced_output = Path("test_enhanced_stego.png")
                
                # Test enhanced hiding
                print("  🔒 Testing enhanced hiding...")
                success = enhanced_engine.hide_data_enhanced(
                    carrier_path=carrier_path,
                    data=test_data,
                    output_path=enhanced_output,
                    password=password,
                    use_anti_detection=False  # Fast mode first
                )
                
                if success:
                    print("  ✅ Enhanced hiding successful!")
                    
                    # Test enhanced extraction
                    print("  📤 Testing enhanced extraction...")
                    extracted = enhanced_engine.extract_data_enhanced(
                        stego_path=enhanced_output,
                        password=password,
                        use_anti_detection=False
                    )
                    
                    if extracted and extracted == test_data:
                        print("  ✅ Enhanced extraction successful!")
                        print("  ✅ Enhanced data integrity verified!")
                    else:
                        print("  ❌ Enhanced extraction failed!")
                        enhanced_test_passed = False
                else:
                    print("  ❌ Enhanced hiding failed!")
                    enhanced_test_passed = False
                    
            except Exception as e:
                print(f"  ❌ Enhanced test error: {e}")
                enhanced_test_passed = False
        
        # Summary
        print("\n" + "=" * 40)
        print("📋 TEST SUMMARY")
        print("=" * 40)
        
        if basic_test_passed:
            print("✅ Basic steganography: WORKING")
        else:
            print("❌ Basic steganography: FAILED")
        
        if enhanced_available:
            if enhanced_test_passed:
                print("✅ Enhanced steganography: WORKING")
            else:
                print("❌ Enhanced steganography: FAILED")
        else:
            print("⚠️ Enhanced steganography: NOT AVAILABLE")
        
        overall_success = basic_test_passed and (enhanced_test_passed if enhanced_available else True)
        
        if overall_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("\nYour InVisioVault setup is working correctly!")
            if enhanced_available:
                print("Enhanced anti-detection features are available.")
            else:
                print("Install OpenCV for enhanced anti-detection features:")
                print("  pip install opencv-python>=4.8.0")
        else:
            print("\n❌ SOME TESTS FAILED!")
            print("Check the errors above for troubleshooting.")
        
        return overall_success
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            if carrier_path.exists():
                carrier_path.unlink()
            if output_path.exists():
                print(f"\n💾 Basic stego image: {output_path}")
            enhanced_output = Path("test_enhanced_stego.png") 
            if enhanced_output.exists():
                print(f"💾 Enhanced stego image: {enhanced_output}")
                print("\n🔍 You can test these images with external steganalysis tools!")
        except:
            pass


if __name__ == "__main__":
    success = simple_test()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ SIMPLE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("""
Next steps to test anti-detection:

1. Install OpenCV (if not done):
   pip install opencv-python>=4.8.0

2. Run the enhanced test:
   python quick_test.py

3. Test with external tools:
   - Download StegExpose or zsteg
   - Test your generated images
   - Compare detection rates

4. Use the enhanced UI:
   python -c "from ui.dialogs.enhanced_hide_files_dialog import EnhancedHideFilesDialog; import sys; from PySide6.QtWidgets import QApplication; app=QApplication(sys.argv); dialog=EnhancedHideFilesDialog(); dialog.show(); app.exec()"
""")
    else:
        print("\n❌ TESTS FAILED - CHECK ERRORS ABOVE")
        print("\nBasic troubleshooting:")
        print("1. Make sure you're in the InVisioVault directory")
        print("2. Check that all required packages are installed")
        print("3. Try with a different test image")
        print("4. Check the logs for specific error messages")
