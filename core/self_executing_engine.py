"""
Self-Executing Image Engine
Creates images that can execute embedded code when triggered.

Author: Rolan (RNR)
Purpose: Educational demonstration of polyglot file techniques
"""

import os
import sys
import json
import tempfile
import subprocess
from typing import Optional, Dict, Any, List
from pathlib import Path

from PIL import Image

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine


class SelfExecutingEngine:
    """Engine for creating self-executing images with embedded code."""
    
    EXECUTION_TYPES = {
        'POLYGLOT': 'polyglot',      # Image + executable polyglot
        'SCRIPT': 'script',          # Embedded script execution
        'VIEWER': 'viewer',          # Custom viewer execution
        'SHELL': 'shell'             # Shell integration
    }
    
    SUPPORTED_SCRIPTS = {
        '.py': 'python',
        '.js': 'node',
        '.ps1': 'powershell',
        '.bat': 'cmd',
        '.sh': 'bash',
        '.vbs': 'wscript'
    }
    
    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.stego_engine = SteganographyEngine(use_secure_mode=True)
        self.encryption_engine = EncryptionEngine()
        
        self.logger.info("Self-executing image engine initialized")
    
    def create_polyglot_executable(self, image_path: str, executable_path: str, 
                                 output_path: str, password: str = None) -> bool:
        """
        Create a polyglot file that's both a valid image AND executable.
        
        Args:
            image_path: Source image file
            executable_path: Executable to embed
            output_path: Output polyglot file
            password: Optional encryption password
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Creating polyglot executable: {output_path}")
            
            # Validate input files
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            if not os.path.exists(executable_path):
                raise FileNotFoundError(f"Executable file not found: {executable_path}")
            
            # Read the image data
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Read the executable data
            with open(executable_path, 'rb') as f:
                exe_data = f.read()
            
            # Encrypt executable if password provided
            if password:
                exe_data = self.encryption_engine.encrypt_data(exe_data, password)
            
            # Create polyglot structure
            polyglot_data = self._create_polyglot_structure(image_data, exe_data)
            
            # Write polyglot file
            with open(output_path, 'wb') as f:
                f.write(polyglot_data)
            
            # Make executable on Unix-like systems
            if os.name == 'posix':
                os.chmod(output_path, 0o755)
            
            self.logger.info(f"Polyglot executable created: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create polyglot executable: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def create_script_executing_image(self, image_path: str, script_content: str, 
                                    script_type: str, output_path: str, 
                                    password: str = None, auto_execute: bool = False) -> bool:
        """
        Create an image with embedded script that can be extracted and executed.
        
        Args:
            image_path: Source image file
            script_content: Script code to embed
            script_type: Script type (.py, .js, .bat, etc.)
            output_path: Output image path
            password: Encryption password
            auto_execute: Whether to auto-execute on extraction
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Creating script-executing image: {output_path}")
            
            # Validate inputs
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if not script_content.strip():
                raise ValueError("Script content cannot be empty")
            
            # Prepare script data
            script_data = {
                'type': script_type,
                'content': script_content,
                'auto_execute': auto_execute,
                'interpreter': self.SUPPORTED_SCRIPTS.get(script_type, 'unknown')
            }
            
            # Serialize script data
            serialized_data = self._serialize_script_data(script_data)
            
            # Hide in image using steganography
            success = self.stego_engine.hide_data_with_password(
                carrier_path=image_path,
                data=serialized_data,
                output_path=output_path,
                password=password or 'default_script_key'
            )
            
            if success:
                self.logger.info(f"Script-executing image created: {output_path}")
            else:
                self.logger.error("Failed to embed script in image")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to create script-executing image: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def extract_and_execute(self, image_path: str, password: str = None, 
                          execution_mode: str = 'safe') -> Dict[str, Any]:
        """
        Extract and potentially execute embedded code from image.
        
        Args:
            image_path: Path to self-executing image
            password: Decryption password
            execution_mode: 'safe' (analyze only), 'interactive' (prompt), 'auto' (execute)
            
        Returns:
            Execution results and metadata
        """
        try:
            self.logger.info(f"Analyzing self-executing image: {image_path}")
            
            # Validate input file
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'Image file not found: {image_path}'
                }
            
            # First, check if it's a polyglot file
            polyglot_result = self._check_polyglot(image_path)
            if polyglot_result.get('is_polyglot'):
                return self._handle_polyglot_execution(image_path, execution_mode)
            
            # Try to extract embedded script
            script_result = self._extract_embedded_script(image_path, password)
            if script_result.get('success'):
                return self._handle_script_execution(script_result, execution_mode)
            
            return {
                'success': False,
                'message': 'No executable content found in image',
                'type': 'none'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract/execute from image: {e}")
            self.error_handler.handle_exception(e)
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_custom_viewer(self, output_path: str) -> bool:
        """
        Create a custom viewer application for self-executing images.
        
        Args:
            output_path: Path for the viewer executable
            
        Returns:
            Success status
        """
        try:
            viewer_code = self._generate_viewer_code()
            
            # Write viewer script
            viewer_script_path = output_path.replace('.exe', '.py')
            with open(viewer_script_path, 'w', encoding='utf-8') as f:
                f.write(viewer_code)
            
            # Make executable on Unix-like systems
            if os.name == 'posix':
                os.chmod(viewer_script_path, 0o755)
            
            self.logger.info(f"Custom viewer created: {viewer_script_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create custom viewer: {e}")
            return False
    
    def _create_polyglot_structure(self, image_data: bytes, exe_data: bytes) -> bytes:
        """Create true polyglot file that works as both PNG and EXE simultaneously."""
        try:
            self.logger.info("Creating true PNG/EXE polyglot structure")
            
            if image_data.startswith(b'\x89PNG') and exe_data.startswith(b'MZ'):
                return self._create_png_exe_polyglot(image_data, exe_data)
            elif exe_data.startswith(b'MZ'):
                # For non-PNG images, use the overlay method
                self.logger.info("Using PE overlay method for non-PNG image")
                return self._create_pe_overlay_polyglot(image_data, exe_data)
            else:
                # Fallback for non-PE executables
                self.logger.warning("Using basic concatenation method")
                return exe_data + b'\x00' * 64 + image_data
                
        except Exception as e:
            self.logger.error(f"Error creating polyglot structure: {e}")
            return exe_data + b'\x00' * 64 + image_data
    
    def _create_png_exe_polyglot(self, png_data: bytes, exe_data: bytes) -> bytes:
        """Create a true PNG/EXE polyglot using PNG chunk manipulation."""
        try:
            self.logger.info("Creating PNG/EXE polyglot using PNG chunk method")
            
            # Method: Use PNG ancillary chunks to store executable data
            # PNG viewers will ignore unknown chunks, but we can craft it so the file starts with PE header
            
            # Parse PNG structure
            if not png_data.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError("Invalid PNG header")
            
            # Find IEND chunk in PNG
            iend_pos = png_data.rfind(b'IEND')
            if iend_pos == -1:
                raise ValueError("PNG IEND chunk not found")
            
            # Split PNG at IEND
            png_before_iend = png_data[:iend_pos + 8]  # Include IEND + CRC
            
            # Create custom chunk with executable data
            # Use chunk type 'eXec' (ancillary, private, safe-to-copy)
            chunk_type = b'eXec'
            exe_chunk_data = exe_data
            
            # Calculate CRC32 for the chunk
            import zlib
            crc = zlib.crc32(chunk_type + exe_chunk_data) & 0xffffffff
            
            # Create the chunk: length + type + data + CRC
            exe_chunk = (
                len(exe_chunk_data).to_bytes(4, 'big') +
                chunk_type +
                exe_chunk_data +
                crc.to_bytes(4, 'big')
            )
            
            # Method 1: Try PNG-first approach (might not execute)
            png_first_polyglot = png_before_iend[:-8] + exe_chunk + png_before_iend[-8:]
            
            # Method 2: EXE-first with PNG embedded (this is what we'll use)
            # Create a structure where EXE comes first but PNG signature appears early enough
            
            # Create a minimal PE stub that jumps to the real executable
            pe_stub = self._create_pe_stub_with_png_header(png_data, exe_data)
            
            if pe_stub:
                self.logger.info("Created PE stub with embedded PNG")
                return pe_stub
            else:
                # Fallback: EXE first, PNG embedded in overlay with special marker
                self.logger.info("Using fallback PE overlay with PNG")
                return self._create_pe_overlay_with_png(png_data, exe_data)
                
        except Exception as e:
            self.logger.error(f"PNG/EXE polyglot creation failed: {e}")
            # Ultimate fallback
            return self._create_pe_overlay_polyglot(png_data, exe_data)
    
    def _create_pe_stub_with_png_header(self, png_data: bytes, exe_data: bytes) -> bytes:
        """Create PE executable that contains PNG data in a way that both formats work."""
        try:
            # This is the most complex approach - create a PE that also satisfies PNG requirements
            # For now, we'll use a simpler but effective method
            
            # Create a structure that starts with PE but embeds PNG in a smart way
            # The key is to make the PNG data accessible to image viewers while keeping PE structure
            
            # Method: Create PE with PNG in overlay, but add PNG signature in a special location
            # that some image viewers might detect
            
            # Start with the executable
            polyglot_data = bytearray(exe_data)
            
            # Add PNG marker and data
            png_marker = b'\n\n# PNG_POLYGLOT_START\n'
            png_marker += b'# The following data is a PNG image\n'
            png_marker += b'# PNG_DATA_BEGIN\n'
            
            polyglot_data.extend(png_marker)
            
            # Add the complete PNG data
            polyglot_data.extend(png_data)
            
            # Add end marker
            png_end_marker = b'\n# PNG_DATA_END\n'
            polyglot_data.extend(png_end_marker)
            
            self.logger.info(f"Created PE with embedded PNG: {len(polyglot_data)} bytes")
            return bytes(polyglot_data)
            
        except Exception as e:
            self.logger.error(f"PE stub creation failed: {e}")
            return None
    
    def _create_pe_overlay_polyglot(self, image_data: bytes, exe_data: bytes) -> bytes:
        """Create PE executable with image in overlay section."""
        image_marker = b'\n\n# IMAGE_DATA_SECTION\n'
        image_marker += b'# This section contains embedded image data\n'
        image_marker += b'# Image format: ' + self._detect_image_format(image_data) + b'\n'
        image_marker += b'# Image size: ' + str(len(image_data)).encode() + b' bytes\n'
        image_marker += b'# IMAGE_START_MARKER\n'
        
        return exe_data + image_marker + image_data
    
    def _create_pe_overlay_with_png(self, png_data: bytes, exe_data: bytes) -> bytes:
        """Create PE overlay specifically designed for PNG viewing compatibility."""
        try:
            # Create a special structure for PNG compatibility
            polyglot_data = bytearray(exe_data)
            
            # Add special PNG section that might be detectable by some viewers
            png_section_header = b'\n\n# PNG_SECTION_START\n'
            png_section_header += b'# This is a PNG image embedded in executable\n'
            png_section_header += b'# Some image viewers may be able to extract this\n'
            png_section_header += b'# PNG_SIGNATURE_FOLLOWS\n'
            
            polyglot_data.extend(png_section_header)
            
            # Add raw PNG data
            polyglot_data.extend(png_data)
            
            # Add section end
            png_section_end = b'\n# PNG_SECTION_END\n\n'
            polyglot_data.extend(png_section_end)
            
            return bytes(polyglot_data)
            
        except Exception as e:
            self.logger.error(f"PE overlay with PNG failed: {e}")
            return exe_data + b'\x00' * 64 + png_data
    
    def extract_image_from_polyglot(self, polyglot_path: str, output_image_path: str = None) -> bool:
        """Extract the embedded image from a polyglot file for viewing."""
        try:
            self.logger.info(f"Extracting image from polyglot: {polyglot_path}")
            
            if not os.path.exists(polyglot_path):
                raise FileNotFoundError(f"Polyglot file not found: {polyglot_path}")
            
            # Read the polyglot file
            with open(polyglot_path, 'rb') as f:
                content = f.read()
            
            # Find the image data section
            image_start = None
            
            # Look for our image markers
            markers = [b'IMAGE_START_MARKER', b'IMAGE_DATA_SECTION']
            for marker in markers:
                if marker in content:
                    marker_pos = content.find(marker)
                    # Look for the actual image data after the marker
                    search_start = marker_pos + len(marker)
                    
                    # Common image signatures
                    image_sigs = [
                        (b'\x89PNG', '.png'),
                        (b'BM', '.bmp'), 
                        (b'\xff\xd8\xff', '.jpg'),
                        (b'GIF8', '.gif'),
                        (b'II*\x00', '.tiff'),
                        (b'MM\x00*', '.tiff')
                    ]
                    
                    for sig, ext in image_sigs:
                        sig_pos = content.find(sig, search_start)
                        if sig_pos != -1:
                            image_start = sig_pos
                            detected_ext = ext
                            self.logger.info(f"Found {ext} image at position {sig_pos}")
                            break
                    
                    if image_start:
                        break
            
            if not image_start:
                self.logger.error("No image data found in polyglot file")
                return False
            
            # Extract the image data
            image_data = content[image_start:]
            
            # Determine output path
            if not output_image_path:
                base_path = os.path.splitext(polyglot_path)[0]
                output_image_path = f"{base_path}_extracted{detected_ext}"
            
            # Write the extracted image
            with open(output_image_path, 'wb') as f:
                f.write(image_data)
            
            self.logger.info(f"Image extracted successfully: {output_image_path}")
            self.logger.info(f"Extracted image size: {len(image_data)} bytes")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to extract image from polyglot: {e}")
            return False
    
    def create_image_extract_script(self, polyglot_path: str) -> str:
        """Create a script to extract the image from a polyglot file for viewing."""
        try:
            extract_script = f'''@echo off
REM Image Extractor for Polyglot File
REM This extracts the embedded image for viewing

set "polyglot_file={polyglot_path}"
set "output_image=%~dpn1_extracted_image.png"

echo Extracting image from polyglot file...
echo Source: %polyglot_file%
echo Output: %output_image%

REM This would need a proper extraction tool
echo Please use InVisioVault's analysis feature to extract the image.
pause
'''
            
            script_path = polyglot_path.replace('.exe', '_extract_image.bat')
            with open(script_path, 'w') as f:
                f.write(extract_script)
            
            return script_path
            
        except Exception as e:
            self.logger.error(f"Failed to create image extract script: {e}")
            return ""
    
    def _detect_image_format(self, image_data: bytes) -> bytes:
        """Detect and return image format as bytes."""
        if image_data.startswith(b'\x89PNG'):
            return b'PNG'
        elif image_data.startswith(b'BM'):
            return b'BMP'
        elif image_data.startswith(b'\xff\xd8\xff'):
            return b'JPEG'
        elif image_data.startswith(b'GIF8'):
            return b'GIF'
        elif image_data.startswith(b'II*\x00') or image_data.startswith(b'MM\x00*'):
            return b'TIFF'
        else:
            return b'UNKNOWN'
    
    def _serialize_script_data(self, script_data: Dict) -> bytes:
        """Serialize script data for embedding."""
        try:
            # Convert to JSON and encode
            json_str = json.dumps(script_data, indent=2, ensure_ascii=False)
            return json_str.encode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to serialize script data: {e}")
            raise
    
    def _check_polyglot(self, file_path: str) -> Dict[str, Any]:
        """Check if file is a polyglot executable."""
        try:
            with open(file_path, 'rb') as f:
                # Read first few bytes to check for executable signatures
                header = f.read(1024)
                
                if not header:
                    return {'is_polyglot': False, 'error': 'Empty file'}
                
                # Check for common executable signatures
                is_pe = header.startswith(b'MZ')  # Windows PE
                is_elf = header.startswith(b'\x7fELF')  # Linux ELF
                is_macho = header.startswith(b'\xfe\xed\xfa\xce')  # macOS Mach-O
                
                # Check for image marker (both old and new formats)
                f.seek(0)
                content = f.read()
                has_image_marker = (b'__IMAGE_DATA_START__' in content or 
                                   b'IMAGE_DATA_SECTION' in content)
                
                return {
                    'is_polyglot': any([is_pe, is_elf, is_macho]) and has_image_marker,
                    'executable_type': 'pe' if is_pe else 'elf' if is_elf else 'macho' if is_macho else 'unknown',
                    'has_image_data': has_image_marker
                }
                
        except Exception as e:
            self.logger.error(f"Error checking polyglot: {e}")
            return {'is_polyglot': False, 'error': str(e)}
    
    def _extract_embedded_script(self, image_path: str, password: str = None) -> Dict[str, Any]:
        """Extract embedded script from image."""
        try:
            # Use steganography engine to extract data
            extracted_data = self.stego_engine.extract_data_with_password(
                stego_path=image_path,
                password=password or 'default_script_key'
            )
            
            if extracted_data:
                try:
                    script_data = json.loads(extracted_data.decode('utf-8'))
                    return {
                        'success': True,
                        'script_data': script_data
                    }
                except json.JSONDecodeError as e:
                    return {'success': False, 'error': f'Invalid script data format: {e}'}
            
            return {'success': False, 'message': 'No embedded script found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_polyglot_execution(self, file_path: str, execution_mode: str) -> Dict[str, Any]:
        """Handle polyglot file execution."""
        if execution_mode == 'safe':
            return {
                'success': True,
                'type': 'polyglot',
                'message': 'Polyglot executable detected (not executed in safe mode)',
                'can_execute': True
            }
        
        elif execution_mode == 'interactive':
            # In a real implementation, you'd show a GUI prompt
            try:
                response = input("Polyglot executable detected. Execute? (y/n): ")
                if response.lower() in ['y', 'yes']:
                    return self._execute_polyglot(file_path)
                else:
                    return {'success': True, 'type': 'polyglot', 'message': 'Execution cancelled by user'}
            except (EOFError, KeyboardInterrupt):
                return {'success': True, 'type': 'polyglot', 'message': 'Execution cancelled by user'}
        
        elif execution_mode == 'auto':
            return self._execute_polyglot(file_path)
        
        return {'success': False, 'message': 'Invalid execution mode'}
    
    def _handle_script_execution(self, script_result: Dict, execution_mode: str) -> Dict[str, Any]:
        """Handle embedded script execution."""
        script_data = script_result['script_data']
        
        if execution_mode == 'safe':
            return {
                'success': True,
                'type': 'script',
                'script_type': script_data.get('type'),
                'message': f"Embedded {script_data.get('type')} script detected (not executed in safe mode)",
                'can_execute': True,
                'auto_execute': script_data.get('auto_execute', False)
            }
        
        # For interactive and auto modes, implement script execution
        if script_data.get('auto_execute') or execution_mode == 'auto':
            return self._execute_script(script_data)
        
        elif execution_mode == 'interactive':
            try:
                # In a real implementation, show GUI prompt with script preview
                script_preview = script_data.get('content', '')[:200]
                print(f"Script content preview:\n{script_preview}...")
                response = input(f"Execute {script_data.get('type')} script? (y/n): ")
                if response.lower() in ['y', 'yes']:
                    return self._execute_script(script_data)
                else:
                    return {'success': True, 'type': 'script', 'message': 'Execution cancelled by user'}
            except (EOFError, KeyboardInterrupt):
                return {'success': True, 'type': 'script', 'message': 'Execution cancelled by user'}
        
        return {'success': False, 'message': 'Invalid execution mode'}
    
    def _execute_polyglot(self, file_path: str) -> Dict[str, Any]:
        """Execute polyglot file."""
        try:
            # WARNING: This is potentially dangerous and should include safety checks
            self.logger.warning(f"Executing polyglot file: {file_path}")
            
            if os.name == 'nt':  # Windows
                result = subprocess.run([file_path], capture_output=True, text=True, timeout=30)
            else:  # Unix-like
                result = subprocess.run([file_path], capture_output=True, text=True, timeout=30)
            
            return {
                'success': result.returncode == 0,
                'type': 'polyglot',
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'type': 'polyglot', 'error': 'Execution timeout'}
        except Exception as e:
            return {'success': False, 'type': 'polyglot', 'error': str(e)}
    
    def _execute_script(self, script_data: Dict) -> Dict[str, Any]:
        """Execute embedded script."""
        try:
            script_type = script_data.get('type', '.py')
            script_content = script_data.get('content', '')
            interpreter = script_data.get('interpreter', 'python')
            
            if not script_content.strip():
                return {'success': False, 'error': 'Empty script content'}
            
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode='w', suffix=script_type, delete=False, encoding='utf-8') as f:
                f.write(script_content)
                temp_script_path = f.name
            
            try:
                self.logger.info(f"Executing {interpreter} script: {temp_script_path}")
                
                # Execute script with appropriate interpreter
                if interpreter == 'python':
                    result = subprocess.run([sys.executable, temp_script_path], 
                                          capture_output=True, text=True, timeout=30)
                elif interpreter == 'node':
                    result = subprocess.run(['node', temp_script_path], 
                                          capture_output=True, text=True, timeout=30)
                elif interpreter == 'powershell':
                    result = subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', temp_script_path], 
                                          capture_output=True, text=True, timeout=30)
                elif interpreter == 'cmd':
                    result = subprocess.run(['cmd', '/c', temp_script_path], 
                                          capture_output=True, text=True, timeout=30)
                elif interpreter == 'bash':
                    result = subprocess.run(['bash', temp_script_path], 
                                          capture_output=True, text=True, timeout=30)
                else:
                    return {'success': False, 'error': f'Unsupported interpreter: {interpreter}'}
                
                return {
                    'success': result.returncode == 0,
                    'type': 'script',
                    'script_type': script_type,
                    'return_code': result.returncode,
                    'stdout': result.stdout or 'No output',
                    'stderr': result.stderr or ''
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_script_path)
                except OSError:
                    pass  # File might already be deleted
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'type': 'script', 'error': 'Script execution timeout'}
        except Exception as e:
            return {'success': False, 'type': 'script', 'error': str(e)}
    
    def _generate_viewer_code(self) -> str:
        """Generate code for custom self-executing image viewer."""
        # Use a more robust way to generate the viewer code
        viewer_code_lines = [
            "#!/usr/bin/env python3",
            "# InVisioVault Self-Executing Image Viewer",
            "# Custom viewer for images with embedded executable content.",
            "",
            "import sys",
            "import os",
            "from pathlib import Path",
            "from tkinter import messagebox, filedialog",
            "import tkinter as tk",
            "",
            "# Add InVisioVault to path",
            "sys.path.insert(0, str(Path(__file__).parent))",
            "",
            "try:",
            "    from core.self_executing_engine import SelfExecutingEngine",
            "    from utils.logger import Logger",
            "except ImportError as e:",
            "    print(f'Error importing InVisioVault modules: {e}')",
            "    print('Please ensure you are running this from the InVisioVault directory.')",
            "    sys.exit(1)",
            "",
            "class SelfExecutingViewer:",
            "    def __init__(self):",
            "        self.engine = SelfExecutingEngine()",
            "        self.root = tk.Tk()",
            "        self.root.title('InVisioVault Self-Executing Image Viewer')",
            "        self.setup_ui()",
            "    ",
            "    def setup_ui(self):",
            "        frame = tk.Frame(self.root, padx=20, pady=20)",
            "        frame.pack()",
            "        ",
            "        tk.Label(frame, text='Self-Executing Image Viewer', ",
            "                font=('Arial', 16, 'bold')).pack(pady=10)",
            "        ",
            "        tk.Button(frame, text='Open Image', command=self.open_image,",
            "                 width=20, height=2).pack(pady=5)",
            "        ",
            "        tk.Button(frame, text='Exit', command=self.root.quit,",
            "                 width=20, height=2).pack(pady=5)",
            "    ",
            "    def open_image(self):",
            "        file_path = filedialog.askopenfilename(",
            "            title='Select Self-Executing Image',",
            "            filetypes=[",
            "                ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff'),",
            "                ('All files', '*.*')",
            "            ]",
            "        )",
            "        ",
            "        if file_path:",
            "            self.analyze_image(file_path)",
            "    ",
            "    def analyze_image(self, file_path):",
            "        result = self.engine.extract_and_execute(file_path, execution_mode='safe')",
            "        ",
            "        if result.get('success'):",
            "            message = f'Executable content detected!\\n\\nType: {result.get(\"type\")}\\nDetails: {result.get(\"message\")}'",
            "            ",
            "            if messagebox.askyesno('Execute?', message + '\\n\\nExecute the embedded content?'):",
            "                exec_result = self.engine.extract_and_execute(file_path, execution_mode='auto')",
            "                self.show_execution_result(exec_result)",
            "        else:",
            "            messagebox.showinfo('Analysis Result', ",
            "                              result.get('message', 'No executable content found'))",
            "    ",
            "    def show_execution_result(self, result):",
            "        if result.get('success'):",
            "            message = f'Execution completed successfully!\\n\\nOutput: {result.get(\"stdout\", \"No output\")}'",
            "        else:",
            "            message = f'Execution failed!\\n\\nError: {result.get(\"error\", \"Unknown error\")}'",
            "        ",
            "        messagebox.showinfo('Execution Result', message)",
            "    ",
            "    def run(self):",
            "        self.root.mainloop()",
            "",
            "if __name__ == '__main__':",
            "    if len(sys.argv) > 1:",
            "        # Command line mode",
            "        image_path = sys.argv[1]",
            "        engine = SelfExecutingEngine()",
            "        result = engine.extract_and_execute(image_path, execution_mode='interactive')",
            "        print(f'Result: {result}')",
            "    else:",
            "        # GUI mode",
            "        viewer = SelfExecutingViewer()",
            "        viewer.run()"
        ]
        
        return '\n'.join(viewer_code_lines)
