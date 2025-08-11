"""
Quick Anti-Detection Test
Simple test to verify that anti-detection steganography is working.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def quick_test():
    """Run a quick test to verify anti-detection is working."""
    
    print("🕵️‍♂️ InVisioVault Quick Anti-Detection Test")
    print("=" * 50)
    
    # Check imports
    print("\n📦 Checking imports...")
    try:
        from core.enhanced_steganography_engine import EnhancedSteganographyEngine
        from PIL import Image
        import numpy as np
        print("✅ Core modules imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError:
        print("❌ OpenCV missing. Install with: pip install opencv-python>=4.8.0")
        return False
    
    # Create test data
    print("\n📄 Creating test data...")
    test_data = b"This is a secret message for anti-detection testing! " * 20
    password = "test_password_123"
    print(f"✅ Test data: {len(test_data)} bytes")
    
    # Create test image
    print("\n🖼️ Creating test carrier image...")
    test_img = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    carrier_path = Path("temp_test_carrier.png")
    output_path = Path("temp_test_stego.png")
    
    img = Image.fromarray(test_img)
    img.save(carrier_path, 'PNG')
    print(f"✅ Test carrier created: {carrier_path}")
    
    try:
        # Test enhanced steganography engine
        print("\n🛡️ Testing Enhanced Anti-Detection Engine...")
        enhanced_engine = EnhancedSteganographyEngine(use_anti_detection=True)
        
        # Hide data
        print("  🔒 Hiding data with anti-detection...")
        success = enhanced_engine.hide_data_enhanced(
            carrier_path=carrier_path,
            data=test_data,
            output_path=output_path,
            password=password,
            use_anti_detection=True
        )
        
        if success:
            print("  ✅ Hiding successful!")
            
            # Extract data
            print("  📤 Extracting data...")
            extracted = enhanced_engine.extract_data_enhanced(
                stego_path=output_path,
                password=password,
                use_anti_detection=True
            )
            
            if extracted == test_data:
                print("  ✅ Extraction successful!")
                print("  ✅ Data integrity verified!")
                
                # Test steganalysis resistance
                if hasattr(enhanced_engine, 'test_against_steganalysis'):
                    print("  🔍 Testing steganalysis resistance...")
                    results = enhanced_engine.test_against_steganalysis(output_path)
                    
                    overall = results.get('overall_assessment', {})
                    risk_score = overall.get('average_detection_risk', 1.0)
                    likely_detected = overall.get('likely_detected_by_any_tool', True)
                    
                    print(f"  📊 Risk score: {risk_score:.3f}")
                    print(f"  🚨 Likely detected by tools: {'YES' if likely_detected else 'NO'}")
                    
                    # Show individual tool results
                    tools = results.get('tool_simulation', {})
                    if tools:
                        print("  🔍 Tool-specific results:")
                        for tool_name, tool_result in tools.items():
                            detected = tool_result.get('likely_detected', True)
                            confidence = tool_result.get('confidence', 0)
                            status = "🔴 DETECTED" if detected else "🟢 SAFE"
                            print(f"    • {tool_name}: {status} (confidence: {confidence:.3f})")
                    
                    # Final assessment
                    if risk_score < 0.3 and not likely_detected:
                        print("\n🎉 EXCELLENT! Anti-detection is working very well!")
                    elif risk_score < 0.6:
                        print("\n⚠️ MODERATE: Anti-detection has some effectiveness")
                    else:
                        print("\n❌ POOR: Anti-detection needs improvement")
                    
                    return True
                else:
                    print("  ⚠️ Steganalysis testing not available")
                    print("\n✅ Basic functionality working, but can't test detection resistance")
                    return True
            else:
                print("  ❌ Data integrity check failed!")
                return False
        else:
            print("  ❌ Hiding failed!")
            return False
            
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
                print(f"\n💾 Steganographic image saved as: {output_path}")
                print("   You can test this image with external steganalysis tools!")
        except:
            pass


def show_external_testing_guide():
    """Show how to test with external steganalysis tools."""
    
    print("\n" + "=" * 70)
    print("🔍 EXTERNAL STEGANALYSIS TESTING GUIDE")
    print("=" * 70)
    
    print("""
To verify your anti-detection is working with REAL tools:

1️⃣ INSTALL STEGANALYSIS TOOLS:

   StegExpose (Java-based):
   • Download: https://github.com/b3dk7/StegExpose
   • Usage: java -jar StegExpose.jar temp_test_stego.png
   
   zsteg (Ruby gem):
   • Install: gem install zsteg
   • Usage: zsteg temp_test_stego.png
   
   StegSeek (if you use JPEG/Steghide format):
   • Install: https://github.com/RickdeJager/stegseek
   • Usage: stegseek temp_test_stego.png wordlist.txt

2️⃣ TEST YOUR IMAGES:

   Compare results between:
   • Standard steganography image (should be detected)
   • Anti-detection enhanced image (should NOT be detected)

3️⃣ INTERPRET RESULTS:

   ✅ GOOD: Tools find nothing or report low confidence
   ❌ BAD: Tools confidently detect steganography
   ⚠️ MODERATE: Tools detect but with low confidence

4️⃣ ONLINE TESTING:

   • Upload to online steganalysis tools
   • Compare detection rates
   • Test with multiple tools for comprehensive validation

5️⃣ CREATE BASELINE:

   • Test with standard LSB steganography (should be detected)
   • Test with your anti-detection version (should be safe)
   • Compare detection rates
""")


if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n✅ Quick test completed successfully!")
        show_external_testing_guide()
    else:
        print("\n❌ Quick test failed!")
        print("\nTroubleshooting steps:")
        print("1. Make sure OpenCV is installed: pip install opencv-python>=4.8.0")
        print("2. Verify you're running from the InVisioVault directory")
        print("3. Check that all required modules are available")
        print("4. Run the comprehensive test: python test_anti_detection.py")
