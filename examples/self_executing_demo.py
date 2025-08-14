#!/usr/bin/env python3
"""
Self-Executing Image Demo
Demonstrates creating and using self-executing images with InVisioVault.

Author: Rolan (RNR)
Purpose: Educational demonstration of advanced steganography techniques
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.self_executing_engine import SelfExecutingEngine
from utils.logger import Logger


def create_sample_image():
    """Create a sample image for testing."""
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        width, height = 400, 300
        image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        sample_path = project_root / "examples" / "sample_carrier.png"
        image.save(sample_path)
        
        print(f"✅ Sample image created: {sample_path}")
        return str(sample_path)
        
    except Exception as e:
        print(f"❌ Failed to create sample image: {e}")
        return None


def create_sample_script():
    """Create a sample Python script for embedding."""
    script_content = '''#!/usr/bin/env python3
"""
Hello World Self-Executing Script
Embedded within an image using InVisioVault
"""

import sys
import platform
from datetime import datetime

print("🚀 Hello from Self-Executing Image!")
print("=" * 50)
print(f"📅 Execution Time: {datetime.now()}")
print(f"🐍 Python Version: {sys.version}")
print(f"💻 Platform: {platform.system()} {platform.release()}")
print(f"🏗️  Architecture: {platform.machine()}")

print("\\n🎯 This script was embedded in an image using steganography!")
print("🔒 The image looked completely normal but contained this executable code.")

print("\\n⚡ Key Features Demonstrated:")
print("  • Steganographic script embedding")
print("  • Encrypted payload storage")
print("  • Cross-platform execution")
print("  • Educational security research")

print("\\n🛡️  Security Note:")
print("This is for educational purposes only. Always be cautious")
print("with self-executing content from untrusted sources!")

input("\\n📝 Press Enter to exit...")
'''
    return script_content


def demo_script_embedding():
    """Demonstrate embedding and executing a script in an image."""
    print("🎯 Demo: Script-Executing Images")
    print("=" * 50)
    
    engine = SelfExecutingEngine()
    
    # Create sample image
    print("1. Creating sample carrier image...")
    image_path = create_sample_image()
    if not image_path:
        return False
    
    # Create script content
    print("2. Preparing Python script...")
    script_content = create_sample_script()
    
    # Output paths
    script_image_path = project_root / "examples" / "script_executing_image.png"
    
    print("3. Embedding script in image...")
    success = engine.create_script_executing_image(
        image_path=image_path,
        script_content=script_content,
        script_type='.py',
        output_path=str(script_image_path),
        password='demo_password_123',
        auto_execute=False  # Require manual execution for safety
    )
    
    if success:
        print(f"✅ Script-executing image created: {script_image_path}")
        
        # Analyze the created image
        print("4. Analyzing the created image...")
        result = engine.extract_and_execute(
            image_path=str(script_image_path),
            password='demo_password_123',
            execution_mode='safe'
        )
        
        if result.get('success'):
            print("✅ Analysis successful!")
            print(f"   Type: {result.get('type')}")
            print(f"   Script Type: {result.get('script_type')}")
            print(f"   Auto-Execute: {result.get('auto_execute')}")
            
            # Ask user if they want to execute
            response = input("\n🤔 Execute the embedded script? (y/n): ")
            if response.lower() == 'y':
                print("5. Executing embedded script...")
                exec_result = engine.extract_and_execute(
                    image_path=str(script_image_path),
                    password='demo_password_123',
                    execution_mode='auto'
                )
                
                if exec_result.get('success'):
                    print("✅ Script executed successfully!")
                    print("📤 Script Output:")
                    print("-" * 30)
                    print(exec_result.get('stdout', 'No output'))
                    if exec_result.get('stderr'):
                        print("⚠️  Errors:")
                        print(exec_result.get('stderr'))
                else:
                    print(f"❌ Script execution failed: {exec_result.get('error')}")
            else:
                print("⏭️  Script execution skipped")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    else:
        print("❌ Failed to create script-executing image")
    
    return success


def demo_polyglot_creation():
    """Demonstrate creating a polyglot executable (if possible)."""
    print("\n🚀 Demo: Polyglot Files")
    print("=" * 50)
    
    # This would require an actual executable file
    # For demo purposes, we'll show the concept
    print("ℹ️  Polyglot demo requires an existing executable file.")
    print("📝 Example usage:")
    print("   engine.create_polyglot_executable(")
    print("       image_path='carrier.png',")
    print("       executable_path='program.exe',")
    print("       output_path='polyglot.png',")
    print("       password='optional_password'")
    print("   )")
    print("🎯 Result: File that's both image AND executable!")


def demo_analysis_tools():
    """Demonstrate analysis capabilities."""
    print("\n🔍 Demo: Analysis Tools")
    print("=" * 50)
    
    engine = SelfExecutingEngine()
    
    # Check for existing script image from previous demo
    script_image_path = project_root / "examples" / "script_executing_image.png"
    
    if script_image_path.exists():
        print(f"📂 Analyzing existing image: {script_image_path}")
        
        # Analyze without password (should fail)
        print("1. Testing analysis without password...")
        result = engine.extract_and_execute(
            image_path=str(script_image_path),
            password=None,
            execution_mode='safe'
        )
        
        if not result.get('success'):
            print("❌ Analysis failed (as expected) - password required")
        
        # Analyze with correct password
        print("2. Testing analysis with correct password...")
        result = engine.extract_and_execute(
            image_path=str(script_image_path),
            password='demo_password_123',
            execution_mode='safe'
        )
        
        if result.get('success'):
            print("✅ Analysis successful with password!")
            print(f"   Content Type: {result.get('type')}")
            print(f"   Script Type: {result.get('script_type')}")
            print("🔒 Encryption working correctly!")
        else:
            print("❌ Analysis failed even with password")
    else:
        print("⚠️  No script image found. Run script embedding demo first.")


def main():
    """Main demo function."""
    print("🎉 InVisioVault Self-Executing Images Demo")
    print("=" * 60)
    print("📚 Educational demonstration of advanced steganography")
    print("🛡️  For research and learning purposes only!")
    print()
    
    try:
        # Demo 1: Script embedding
        success = demo_script_embedding()
        
        if success:
            # Demo 2: Analysis tools
            demo_analysis_tools()
        
        # Demo 3: Polyglot concept
        demo_polyglot_creation()
        
        print("\n🎓 Demo Complete!")
        print("💡 Key Takeaways:")
        print("  • Images can contain executable code")
        print("  • Encryption provides additional security")
        print("  • Analysis tools can detect embedded content")
        print("  • Educational value for security research")
        
        print("\n🚀 Next Steps:")
        print("  • Launch the GUI dialog for interactive creation")
        print("  • Experiment with different script types")
        print("  • Try the custom viewer application")
        print("  • Explore polyglot file techniques")
        
    except Exception as e:
        logger = Logger()
        logger.error(f"Demo error: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
