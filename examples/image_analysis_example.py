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
from core.image_analyzer import AnalysisLevel


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
        print(f"✅ Suitability: {analysis_op.get_suitability_rating()}")
        
        capacity = analysis_op.get_capacity_estimate()
        if capacity:
            print(f"📦 Capacity: ~{capacity.get('estimated_capacity_kb', 0):.1f} KB")
        
        print(f"\n💡 Recommendation:")
        print(f"   {analysis_op.get_human_readable_summary()}")
        
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
        
        # Basic info
        rating = analysis_op.get_suitability_rating()
        print(f"⭐ Overall Rating: {rating}")
        
        # Capacity information
        capacity = analysis_op.get_capacity_estimate()
        if capacity:
            capacity_mb = capacity.get('lsb_capacity_mb', 0)
            capacity_kb = capacity.get('lsb_capacity_kb', 0)
            if capacity_mb > 1:
                print(f"📦 LSB Capacity: {capacity_mb:.2f} MB")
            else:
                print(f"📦 LSB Capacity: {capacity_kb:.1f} KB")
        
        # Steganography detection
        has_stego = analysis_op.has_potential_steganography()
        print(f"🕵️ Potential Hidden Data: {'Yes' if has_stego else 'No'}")
        
        # Detailed summary
        print(f"\n📋 Analysis Summary:")
        summary_lines = analysis_op.get_human_readable_summary().split('\n')
        for line in summary_lines:
            if line.strip():
                print(f"   {line}")
        
        # Recommendations
        recommendations = analysis_op.get_recommendations()
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                if rec.strip():
                    print(f"   {i}. {rec}")
        
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
