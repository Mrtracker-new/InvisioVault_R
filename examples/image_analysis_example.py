#!/usr/bin/env python3
"""
Image Analysis Example

Demonstrates the integration between ImageAnalyzer and AnalysisOperation
for comprehensive steganographic image analysis.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from operations.analysis_operation import AnalysisOperation
from core.analyzers.image_analyzer import AnalysisLevel


def demonstrate_basic_analysis(image_path: str):
    """Demonstrate basic image analysis."""
    print("\n🔍 Basic Image Analysis Demo")
    print("=" * 50)
    
    try:
        # Create and configure operation
        analysis_op = AnalysisOperation()
        analysis_op.configure(image_path, 'basic')
        
        # Run the analysis
        print(f"📁 Analyzing image: {Path(image_path).name}")
        results = analysis_op.run_analysis(
            progress_callback=lambda p: print(f"Progress: {p*100:.0f}%"),
            status_callback=lambda s: print(f"Status: {s}")
        )
        
        # Display results
        print("\n📊 Results:")
        
        # Extract key information from results
        file_info = results.get('file_info', {})
        capacity = results.get('capacity_analysis', {})
        security = results.get('security_assessment', {})
        
        print(f"✅ Security Rating: {security.get('security_rating', 'unknown').title()}")
        print(f"📁 Format: {file_info.get('format', 'Unknown')} ({'Lossless' if file_info.get('is_lossless', False) else 'Lossy'})")
        
        if capacity:
            capacity_kb = capacity.get('capacity_kb', 0)
            print(f"📦 LSB Capacity: ~{capacity_kb:.1f} KB")
        
        print(f"\n💡 Recommendation:")
        print(f"   {results.get('basic_recommendation', 'Analysis completed')}")
        
    except Exception as e:
        print(f"❌ Error during basic analysis: {e}")


def demonstrate_comprehensive_analysis(image_path: str):
    """Demonstrate comprehensive image analysis."""
    print("\n🔬 Comprehensive Image Analysis Demo")
    print("=" * 50)
    
    try:
        # Create and configure operation for thorough analysis
        analysis_op = AnalysisOperation()
        analysis_op.configure(image_path, 'comprehensive')
        
        # Run the analysis
        print(f"📁 Analyzing image: {Path(image_path).name}")
        results = analysis_op.run_analysis(
            progress_callback=lambda p: print(f"Progress: {p*100:.0f}%"),
            status_callback=lambda s: print(f"Status: {s}")
        )
        
        # Display comprehensive results
        print("\n📊 Comprehensive Results:")
        
        # Extract information from comprehensive results
        comprehensive = results.get('comprehensive_analysis', {})
        file_info = comprehensive.get('file_info', {})
        security = comprehensive.get('security_assessment', {})
        capacity = comprehensive.get('capacity_analysis', {})
        quality_metrics = comprehensive.get('quality_metrics', {})
        stego_detection = results.get('steganography_detection', {})
        performance = comprehensive.get('performance_metrics', {})
        
        # Basic info
        security_rating = security.get('security_rating', 'unknown')
        security_score = security.get('overall_security_score', 0)
        print(f"⭐ Security Rating: {security_rating.title()} ({security_score:.1f}/10.0)")
        
        # Capacity information
        basic_lsb = capacity.get('basic_lsb', {})
        if basic_lsb:
            capacity_mb = basic_lsb.get('capacity_mb', 0)
            capacity_kb = basic_lsb.get('capacity_kb', 0)
            if capacity_mb > 1:
                print(f"📦 LSB Capacity: {capacity_mb:.2f} MB")
            else:
                print(f"📦 LSB Capacity: {capacity_kb:.1f} KB")
        
        # Quality metrics
        rgb_quality = quality_metrics.get('rgb', {})
        if rgb_quality:
            entropy = rgb_quality.get('entropy', {}).get('overall', 0)
            noise = rgb_quality.get('noise_analysis', {}).get('overall_noise_estimate', 0)
            print(f"🎲 Entropy: {entropy:.2f}/8.0")
            print(f"🔊 Noise Level: {noise:.1f}")
        
        # Steganography detection
        if stego_detection:
            likelihood = stego_detection.get('overall_likelihood', 'none')
            confidence = stego_detection.get('detection_confidence', 0)
            print(f"🕵️ Steganography Detection: {likelihood.upper()} ({confidence:.1%})")
            
            indicators = stego_detection.get('indicators', [])
            if indicators:
                print("   ⚠️ Indicators:")
                for indicator in indicators[:3]:  # Show top 3
                    print(f"     • {indicator}")
        
        # Performance information
        if performance:
            analysis_time = performance.get('total_analysis_time', 0)
            gpu_used = performance.get('gpu_acceleration_used', False)
            print(f"⌚ Performance: {analysis_time:.2f}s {'(GPU)' if gpu_used else '(CPU)'}")
        
        # Detailed summary
        summary = results.get('human_readable_summary', '')
        if summary:
            print(f"\n📋 Enhanced Analysis Summary:")
            # Show first few lines of summary
            summary_lines = summary.split('\n')[:10]  # Limit to first 10 lines
            for line in summary_lines:
                if line.strip() and not line.startswith('='):
                    print(f"   {line}")
        
        # Recommendations
        recommendations = comprehensive.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Enhanced Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                if rec.strip():
                    # Clean up recommendation text
                    clean_rec = rec.replace('⭐', '').replace('📊', '').replace('🔊', '').strip()
                    print(f"   {i}. {clean_rec}")
        
    except Exception as e:
        print(f"❌ Error during comprehensive analysis: {e}")


def main():
    """Main demonstration function."""
    print("🔍 InVisioVault Image Analysis Integration Demo")
    print("=" * 60)
    
    # Check if image path provided
    if len(sys.argv) < 2:
        print("\n❓ Usage: python image_analysis_example.py <path_to_image>")
        print("\n📝 This example demonstrates:")
        print("   • Basic image suitability analysis")
        print("   • Comprehensive steganographic analysis")
        print("   • Integration between ImageAnalyzer and AnalysisOperation")
        print("   • Human-readable analysis reporting")
        
        # Look for example images in common locations
        example_paths = [
            "test_images/sample.png",
            "examples/sample.bmp",
            "../sample_images/test.png"
        ]
        
        for path in example_paths:
            if Path(path).exists():
                print(f"\n💡 Found example image: {path}")
                print(f"   Run: python {sys.argv[0]} {path}")
                break
        else:
            print(f"\n💡 Create a test image and run:")
            print(f"   python {sys.argv[0]} your_image.png")
        
        return
    
    image_path = sys.argv[1]
    
    # Validate image path
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    try:
        # Run demonstrations
        demonstrate_basic_analysis(image_path)
        demonstrate_comprehensive_analysis(image_path)
        
        print(f"\n✅ Analysis demonstration completed!")
        print(f"📝 Check the detailed results above for insights about your image.")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
