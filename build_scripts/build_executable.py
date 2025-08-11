#!/usr/bin/env python3
"""
InvisioVault Executable Builder
Professional build script for creating Windows executable

Author: Rolan (RNR)
Purpose: Automated, professional executable building with validation
Features:
- Automated build process
- Dependency checking
- File validation
- Build cleanup
- Professional output

Usage:
    python build_executable.py [options]
    
Options:
    --clean     Clean build directories before building
    --onedir    Create directory distribution (faster startup)
    --debug     Create debug version with console
    --verbose   Show detailed build output
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
import time

class InvisioVaultBuilder:
    def __init__(self):
        # Script is in build_scripts/, project root is parent directory
        self.project_root = Path(__file__).parent.parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.spec_file = self.project_root / "InvisioVault.spec"
        
    def print_banner(self):
        """Print professional build banner"""
        print("=" * 80)
        print("🚀 InvisioVault Executable Builder")
        print("   Advanced Steganography Suite - Professional Build System")
        print("   Author: Rolan (RNR) | Educational Project")
        print("=" * 80)
        print()
        
    def check_requirements(self):
        """Check if all required tools and files are available"""
        print("📋 Checking build requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
            
        # Check PyInstaller
        try:
            import PyInstaller
            print(f"✅ PyInstaller {PyInstaller.__version__} found")
        except ImportError:
            print("❌ PyInstaller not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
            
        # Check main.py
        if not (self.project_root / "main.py").exists():
            print("❌ main.py not found")
            return False
        print("✅ main.py found")
        
        # Check icon file
        icon_path = self.project_root / "assets" / "icons" / "InvisioVault.ico"
        if not icon_path.exists():
            print("⚠️ Icon file not found, will build without icon")
        else:
            print("✅ Application icon found")
            
        # Check spec file
        if self.spec_file.exists():
            print("✅ Spec file found")
        else:
            print("⚠️ Spec file not found, will use basic build")
            
        print("✅ All requirements satisfied")
        print()
        return True
        
    def clean_build(self):
        """Clean previous build artifacts"""
        print("🧹 Cleaning previous build artifacts...")
        
        directories_to_clean = [self.build_dir, self.dist_dir]
        for directory in directories_to_clean:
            if directory.exists():
                print(f"   Removing {directory}")
                shutil.rmtree(directory)
                
        # Clean PyInstaller cache
        pycache_dirs = list(self.project_root.rglob("__pycache__"))
        for pycache_dir in pycache_dirs:
            if pycache_dir.exists():
                shutil.rmtree(pycache_dir)
                
        print("✅ Cleanup completed")
        print()
        
    def build_executable(self, onedir=False, debug=False, verbose=False):
        """Build the executable using PyInstaller"""
        print("🔨 Building InvisioVault executable...")
        print(f"   Mode: {'Directory' if onedir else 'Single File'}")
        print(f"   Debug: {'Enabled' if debug else 'Disabled'}")
        print()
        
        # Build command
        cmd = [sys.executable, "-m", "PyInstaller"]
        
        if self.spec_file.exists():
            # Use spec file for advanced configuration
            cmd.extend([str(self.spec_file)])
            if onedir:
                print("⚠️ Directory mode requested but using spec file configuration")
        else:
            # Basic command line build
            cmd.extend([
                "--name=InvisioVault",
                "--windowed" if not debug else "--console",
                "--onefile" if not onedir else "--onedir",
                "--clean",
            ])
            
            # Add icon if available
            icon_path = self.project_root / "assets" / "icons" / "InvisioVault.ico"
            if icon_path.exists():
                cmd.append(f"--icon={icon_path}")
                
            # Add hidden imports for common issues
            hidden_imports = [
                "PySide6.QtCore",
                "PySide6.QtWidgets", 
                "PySide6.QtGui",
                "numpy",
                "PIL.Image",
                "cryptography.fernet"
            ]
            
            for import_name in hidden_imports:
                cmd.extend(["--hidden-import", import_name])
                
            cmd.append("main.py")
            
        # Add verbose flag
        if verbose:
            cmd.append("--log-level=DEBUG")
        else:
            cmd.append("--log-level=INFO")
            
        # Execute build
        print("⚡ Starting PyInstaller build process...")
        print(f"Command: {' '.join(cmd)}")
        print()
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, 
                                  capture_output=not verbose, text=True)
            
            build_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Build completed successfully in {build_time:.1f}s")
                return True
            else:
                print("❌ Build failed!")
                if not verbose and result.stderr:
                    print("Error output:")
                    print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Build failed with exception: {e}")
            return False
            
    def validate_executable(self):
        """Validate the built executable"""
        print("🔍 Validating built executable...")
        
        exe_path = self.dist_dir / "InvisioVault.exe"
        if not exe_path.exists():
            print("❌ Executable not found!")
            return False
            
        # Check file size (should be reasonable)
        file_size = exe_path.stat().st_size / (1024 * 1024)  # MB
        print(f"📊 Executable size: {file_size:.1f} MB")
        
        if file_size < 50:
            print("⚠️ Executable seems small, might be missing dependencies")
        elif file_size > 500:
            print("⚠️ Executable seems large, consider optimizing")
        else:
            print("✅ Executable size looks reasonable")
            
        # Try to get version info (Windows only)
        if sys.platform == "win32":
            try:
                import win32api  # type: ignore # Optional dependency for Windows version info
                info = win32api.GetFileVersionInfo(str(exe_path), "\\")
                print("✅ Version information embedded successfully")
            except ImportError:
                print("⚠️ win32api not available - install pywin32 for version info validation")
            except Exception as e:
                print(f"⚠️ Could not verify version information: {e}")
                
        print("✅ Executable validation completed")
        return True
        
    def show_results(self):
        """Show build results and instructions"""
        print()
        print("=" * 80)
        print("🎉 Build Process Completed!")
        print("=" * 80)
        
        exe_path = self.dist_dir / "InvisioVault.exe"
        if exe_path.exists():
            file_size = exe_path.stat().st_size / (1024 * 1024)
            print(f"📦 Executable: {exe_path}")
            print(f"📊 Size: {file_size:.1f} MB")
            print()
            print("🚀 Ready to use!")
            print("   • Double-click the executable to launch InvisioVault")
            print("   • No Python installation required on target systems")
            print("   • All dependencies are bundled")
            print()
            print("📤 Distribution:")
            print(f"   • Copy {exe_path.name} to any Windows computer")
            print("   • Requires no additional files or installation")
            print("   • Professional-grade executable with icon and version info")
        else:
            print("❌ Build failed - executable not created")
            
        print("=" * 80)

def main():
    """Main build function"""
    parser = argparse.ArgumentParser(
        description="Build InvisioVault executable",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clean", action="store_true", 
                       help="Clean build directories before building")
    parser.add_argument("--onedir", action="store_true",
                       help="Create directory distribution instead of single file")
    parser.add_argument("--debug", action="store_true",
                       help="Create debug version with console window")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed build output")
    
    args = parser.parse_args()
    
    # Create builder instance
    builder = InvisioVaultBuilder()
    
    # Show banner
    builder.print_banner()
    
    # Check requirements
    if not builder.check_requirements():
        print("❌ Requirements check failed")
        sys.exit(1)
        
    # Clean if requested
    if args.clean:
        builder.clean_build()
        
    # Build executable
    if not builder.build_executable(args.onedir, args.debug, args.verbose):
        print("❌ Build failed")
        sys.exit(1)
        
    # Validate result
    if not builder.validate_executable():
        print("⚠️ Validation warnings (executable may still work)")
        
    # Show results
    builder.show_results()

if __name__ == "__main__":
    main()
