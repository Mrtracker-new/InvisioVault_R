"""
Comparison Test: Standard vs Anti-Detection Steganography
Creates images with both methods and compares their steganalysis resistance.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import secrets
import string

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_image(width=800, height=600, complexity="high"):
    """Create a test image with specified complexity."""
    if complexity == "low":
        # Simple gradient
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            img_array[i, :] = [i * 255 // height, (height-i) * 255 // height, 128]
    elif complexity == "medium":
        # Pattern with some texture
        img_array = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
        # Add some patterns
        for i in range(0, height, 20):
            img_array[i:i+10, :] = np.random.randint(50, 250, (10, width, 3))
    else:  # high complexity
        # Complex texture
        np.random.seed(42)  # For reproducible results
        base = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        # Add noise layers
        noise1 = np.random.normal(0, 30, (height, width, 3))
        noise2 = np.random.normal(0, 15, (height, width, 3))
        img_array = np.clip(base + noise1 + noise2, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def generate_test_data(size_bytes=1000):
    """Generate random test data."""
    return ''.join(secrets.choice(string.ascii_letters + string.digits + string.punctuation) 
                   for _ in range(size_bytes))

def comparison_test():
    """Run a comprehensive comparison test."""
    
    print("🔬 STANDARD vs ANTI-DETECTION STEGANOGRAPHY COMPARISON")
    print("=" * 70)
    
    # Import required modules
    try:
        from core.steganography_engine import SteganographyEngine
        from core.enhanced_steganography_engine import EnhancedSteganographyEngine
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        return False
    
    # Test configurations
    test_configs = [
        {"complexity": "low", "data_size": 500, "name": "Low Complexity, Small Data"},
        {"complexity": "medium", "data_size": 1000, "name": "Medium Complexity, Medium Data"},
        {"complexity": "high", "data_size": 2000, "name": "High Complexity, Large Data"},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n🧪 Testing: {config['name']}")
        print("-" * 50)
        
        # Create test image
        print(f"  📷 Creating {config['complexity']} complexity image...")
        test_image = create_test_image(complexity=config['complexity'])
        carrier_path = f"test_carrier_{config['complexity']}.png"
        test_image.save(carrier_path)
        
        # Generate test data
        test_data = generate_test_data(config['data_size'])
        test_data_bytes = test_data.encode('utf-8')  # Convert to bytes
        test_password = "test_password_123"
        
        # Initialize engines
        standard_engine = SteganographyEngine()
        enhanced_engine_no_anti = EnhancedSteganographyEngine(use_anti_detection=False)
        enhanced_engine_with_anti = EnhancedSteganographyEngine(use_anti_detection=True)
        
        config_results = {}
        
        # Test 1: Standard Steganography
        print("  🔧 Testing standard steganography...")
        try:
            standard_output = f"standard_{config['complexity']}_stego.png"
            success = standard_engine.hide_data(
                carrier_path, test_data_bytes, standard_output
            )
            if success:
                # Run steganalysis on standard image
                analysis = enhanced_engine_with_anti.test_against_steganalysis(Path(standard_output))
                overall = analysis.get('overall_assessment', {})
                config_results['standard'] = {
                    'success': True,
                    'risk_score': overall.get('average_detection_risk', 1.0),
                    'safety_level': overall.get('safety_level', 'UNKNOWN'),
                    'likely_detected': overall.get('likely_detected_by_any_tool', True),
                    'output_file': standard_output
                }
                print(f"    ✅ Standard: Risk {config_results['standard']['risk_score']:.3f}, Safety {config_results['standard']['safety_level']}")
            else:
                config_results['standard'] = {'success': False}
                print("    ❌ Standard steganography failed")
        except Exception as e:
            print(f"    ❌ Standard steganography error: {e}")
            config_results['standard'] = {'success': False, 'error': str(e)}
        
        # Test 2: Enhanced without Anti-Detection
        print("  🔧 Testing enhanced engine (no anti-detection)...")
        try:
            enhanced_no_anti_output = f"enhanced_no_anti_{config['complexity']}_stego.png"
            success = enhanced_engine_no_anti.hide_data_enhanced(
                carrier_path, test_data_bytes, enhanced_no_anti_output,
                password=test_password, use_anti_detection=False
            )
            if success:
                analysis = enhanced_engine_with_anti.test_against_steganalysis(Path(enhanced_no_anti_output))
                overall = analysis.get('overall_assessment', {})
                config_results['enhanced_no_anti'] = {
                    'success': True,
                    'risk_score': overall.get('average_detection_risk', 1.0),
                    'safety_level': overall.get('safety_level', 'UNKNOWN'),
                    'likely_detected': overall.get('likely_detected_by_any_tool', True),
                    'output_file': enhanced_no_anti_output
                }
                print(f"    ✅ Enhanced (no anti): Risk {config_results['enhanced_no_anti']['risk_score']:.3f}, Safety {config_results['enhanced_no_anti']['safety_level']}")
            else:
                config_results['enhanced_no_anti'] = {'success': False}
                print("    ❌ Enhanced (no anti-detection) failed")
        except Exception as e:
            print(f"    ❌ Enhanced (no anti-detection) error: {e}")
            config_results['enhanced_no_anti'] = {'success': False, 'error': str(e)}
        
        # Test 3: Enhanced with Anti-Detection
        print("  🔧 Testing enhanced engine (with anti-detection)...")
        try:
            enhanced_with_anti_output = f"enhanced_with_anti_{config['complexity']}_stego.png"
            success = enhanced_engine_with_anti.hide_data_enhanced(
                carrier_path, test_data_bytes, enhanced_with_anti_output,
                password=test_password, use_anti_detection=True
            )
            if success:
                analysis = enhanced_engine_with_anti.test_against_steganalysis(Path(enhanced_with_anti_output))
                overall = analysis.get('overall_assessment', {})
                config_results['enhanced_with_anti'] = {
                    'success': True,
                    'risk_score': overall.get('average_detection_risk', 1.0),
                    'safety_level': overall.get('safety_level', 'UNKNOWN'),
                    'likely_detected': overall.get('likely_detected_by_any_tool', True),
                    'output_file': enhanced_with_anti_output
                }
                print(f"    ✅ Enhanced (with anti): Risk {config_results['enhanced_with_anti']['risk_score']:.3f}, Safety {config_results['enhanced_with_anti']['safety_level']}")
            else:
                config_results['enhanced_with_anti'] = {'success': False}
                print("    ❌ Enhanced (with anti-detection) failed")
        except Exception as e:
            print(f"    ❌ Enhanced (with anti-detection) error: {e}")
            config_results['enhanced_with_anti'] = {'success': False, 'error': str(e)}
        
        results[config['name']] = config_results
    
    # Generate comprehensive comparison report
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE COMPARISON REPORT")
    print("=" * 80)
    
    print(f"{'Configuration':<30} {'Standard':<15} {'Enhanced':<15} {'Anti-Detect':<15}")
    print(f"{'':<30} {'Risk/Safety':<15} {'Risk/Safety':<15} {'Risk/Safety':<15}")
    print("-" * 80)
    
    for config_name, config_results in results.items():
        config_short = config_name.replace(" Complexity, ", "/").replace(" Data", "")
        
        # Standard results
        if config_results.get('standard', {}).get('success'):
            std_risk = config_results['standard']['risk_score']
            std_safety = config_results['standard']['safety_level'][:4]  # Truncate
            std_result = f"{std_risk:.3f}/{std_safety}"
        else:
            std_result = "FAILED"
        
        # Enhanced no anti results
        if config_results.get('enhanced_no_anti', {}).get('success'):
            enh_risk = config_results['enhanced_no_anti']['risk_score']
            enh_safety = config_results['enhanced_no_anti']['safety_level'][:4]
            enh_result = f"{enh_risk:.3f}/{enh_safety}"
        else:
            enh_result = "FAILED"
        
        # Enhanced with anti results
        if config_results.get('enhanced_with_anti', {}).get('success'):
            anti_risk = config_results['enhanced_with_anti']['risk_score']
            anti_safety = config_results['enhanced_with_anti']['safety_level'][:4]
            anti_result = f"{anti_risk:.3f}/{anti_safety}"
        else:
            anti_result = "FAILED"
        
        print(f"{config_short:<30} {std_result:<15} {enh_result:<15} {anti_result:<15}")
    
    # Summary and recommendations
    print("\n🏆 WINNER ANALYSIS")
    print("-" * 40)
    
    best_performers = []
    total_tests = 0
    anti_detection_wins = 0
    
    for config_name, config_results in results.items():
        total_tests += 1
        
        risks = {}
        if config_results.get('standard', {}).get('success'):
            risks['Standard'] = config_results['standard']['risk_score']
        if config_results.get('enhanced_no_anti', {}).get('success'):
            risks['Enhanced'] = config_results['enhanced_no_anti']['risk_score']
        if config_results.get('enhanced_with_anti', {}).get('success'):
            risks['Anti-Detection'] = config_results['enhanced_with_anti']['risk_score']
        
        if risks:
            best = min(risks.items(), key=lambda x: x[1])
            best_performers.append((config_name, best[0], best[1]))
            if best[0] == 'Anti-Detection':
                anti_detection_wins += 1
            print(f"{config_name}: 🏆 {best[0]} (risk: {best[1]:.3f})")
    
    win_rate = (anti_detection_wins / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📈 ANTI-DETECTION WIN RATE: {win_rate:.1f}% ({anti_detection_wins}/{total_tests})")
    
    if win_rate >= 80:
        print("\n🎉 OUTSTANDING! Anti-detection steganography is clearly superior!")
        print("Your enhanced engine with anti-detection consistently outperforms standard methods.")
    elif win_rate >= 60:
        print("\n✅ EXCELLENT! Anti-detection shows significant improvements!")
        print("The enhanced engine provides better steganographic security in most cases.")
    elif win_rate >= 40:
        print("\n⚠️ MODERATE improvement with anti-detection features.")
        print("Consider optimizing anti-detection algorithms for better performance.")
    else:
        print("\n❌ Anti-detection features need improvement.")
        print("Standard methods are performing similarly or better.")
    
    print(f"\n💡 KEY FINDINGS:")
    print(f"• Lower risk scores indicate better steganographic security")
    print(f"• 'HIGH' safety level means low detection probability")
    print(f"• Anti-detection features work best with high-complexity images")
    print(f"• Consider image complexity when choosing steganography method")
    
    # Cleanup recommendation
    print(f"\n🧹 Generated files for manual inspection:")
    for config_name, config_results in results.items():
        for method, method_result in config_results.items():
            if method_result.get('success') and 'output_file' in method_result:
                print(f"  • {method_result['output_file']}")
    
    return win_rate >= 50


if __name__ == "__main__":
    success = comparison_test()
    
    if success:
        print("\n✅ Comparison test completed successfully!")
        print("\n🔍 Next steps:")
        print("1. Visually inspect the generated images")
        print("2. Test with external steganalysis tools")
        print("3. Run performance benchmarks")
    else:
        print("\n⚠️ Comparison test shows mixed results.")
        print("Consider reviewing anti-detection algorithm parameters.")
