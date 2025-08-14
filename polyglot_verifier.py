#!/usr/bin/env python3
"""
InVisioVault Polyglot Verifier

Part of InVisioVault Advanced Steganography Suite
Created by Rolan (RNR) for Educational Excellence

Comprehensive testing and verification tool for PNG/EXE polyglot files.
Validates both PNG and PE format compatibility, performs structure analysis,
and provides detailed diagnostics for polyglot troubleshooting.
"""
"""
Polyglot Verification Tool

This tool verifies that polyglot files work correctly as both PNG images
and Windows executables. It includes comprehensive testing and debugging.

Features:
- PNG format validation
- PE executable validation
- Header analysis and conflict detection
- Execution testing with safety measures
- Image viewer compatibility testing
- Detailed diagnostic reports
"""

import struct
import os
import sys
import subprocess
import tempfile
import zlib
from typing import Dict, List, Tuple, Optional
import time


class PolyglotVerifier:
    """Comprehensive polyglot file verification."""
    
    PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
    
    def __init__(self):
        self.results = {}
        self.debug = True
    
    def log(self, message: str):
        if self.debug:
            print(f"[VERIFY] {message}")
    
    def analyze_file_structure(self, file_path: str) -> Dict:
        """Analyze the internal structure of a polyglot file."""
        
        self.log(f"Analyzing file structure: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            analysis = {
                'file_size': len(data),
                'formats_detected': [],
                'png_analysis': {},
                'pe_analysis': {},
                'structure_issues': []
            }
            
            # Check for PNG format
            if data.startswith(self.PNG_SIGNATURE):
                analysis['formats_detected'].append('PNG')
                analysis['png_analysis'] = self.analyze_png_structure(data)
            
            # Check for PE format (can be at start or embedded)
            pe_locations = self.find_pe_signatures(data)
            if pe_locations:
                analysis['formats_detected'].append('PE')
                analysis['pe_analysis'] = self.analyze_pe_structure(data, pe_locations)
            
            # Check for common issues
            analysis['structure_issues'] = self.detect_structure_issues(data)
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_png_structure(self, data: bytes) -> Dict:
        """Analyze PNG structure for validity."""
        
        png_info = {
            'valid_signature': False,
            'chunks': [],
            'has_required_chunks': False,
            'image_info': {},
            'issues': []
        }
        
        try:
            # Check signature
            if data.startswith(self.PNG_SIGNATURE):
                png_info['valid_signature'] = True
                offset = 8  # After signature
                
                # Parse chunks
                while offset < len(data):
                    if offset + 8 > len(data):
                        break
                    
                    # Read chunk header
                    chunk_length = struct.unpack('>I', data[offset:offset+4])[0]
                    chunk_type = data[offset+4:offset+8]
                    
                    chunk_info = {
                        'type': chunk_type.decode('latin-1', errors='ignore'),
                        'length': chunk_length,
                        'offset': offset
                    }
                    
                    # Validate chunk
                    if offset + 8 + chunk_length + 4 > len(data):
                        png_info['issues'].append(f"Chunk {chunk_info['type']} extends beyond file")
                        break
                    
                    chunk_data = data[offset+8:offset+8+chunk_length]
                    expected_crc = struct.unpack('>I', data[offset+8+chunk_length:offset+8+chunk_length+4])[0]
                    actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xffffffff
                    
                    chunk_info['crc_valid'] = (expected_crc == actual_crc)
                    if not chunk_info['crc_valid']:
                        png_info['issues'].append(f"Chunk {chunk_info['type']} has invalid CRC")
                    
                    # Parse IHDR chunk
                    if chunk_type == b'IHDR' and chunk_length >= 13:
                        width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack('>IIBBBBB', chunk_data)
                        png_info['image_info'] = {
                            'width': width,
                            'height': height,
                            'bit_depth': bit_depth,
                            'color_type': color_type
                        }
                    
                    png_info['chunks'].append(chunk_info)
                    
                    # Move to next chunk
                    offset += 8 + chunk_length + 4
                    
                    # Check for IEND (end of PNG)
                    if chunk_type == b'IEND':
                        break
                
                # Check required chunks
                chunk_types = [chunk['type'] for chunk in png_info['chunks']]
                png_info['has_required_chunks'] = 'IHDR' in chunk_types and 'IEND' in chunk_types
                
                if not png_info['has_required_chunks']:
                    png_info['issues'].append("Missing required chunks (IHDR or IEND)")
                
        except Exception as e:
            png_info['issues'].append(f"PNG parsing error: {e}")
        
        return png_info
    
    def find_pe_signatures(self, data: bytes) -> List[int]:
        """Find all PE signatures in the file."""
        
        pe_locations = []
        
        # Look for MZ signatures
        offset = 0
        while True:
            mz_pos = data.find(b'MZ', offset)
            if mz_pos == -1:
                break
            
            # Check if this is a valid DOS header
            if mz_pos + 64 <= len(data):
                try:
                    pe_offset = struct.unpack('<I', data[mz_pos+60:mz_pos+64])[0]
                    if mz_pos + pe_offset + 4 <= len(data):
                        if data[mz_pos + pe_offset:mz_pos + pe_offset + 4] == b'PE\x00\x00':
                            pe_locations.append(mz_pos)
                except:
                    pass
            
            offset = mz_pos + 1
        
        return pe_locations
    
    def analyze_pe_structure(self, data: bytes, pe_locations: List[int]) -> Dict:
        """Analyze PE structure for each found PE signature."""
        
        pe_info = {
            'locations': pe_locations,
            'executables': [],
            'issues': []
        }
        
        for pe_start in pe_locations:
            try:
                # Parse DOS header
                if pe_start + 64 > len(data):
                    continue
                
                pe_offset = struct.unpack('<I', data[pe_start+60:pe_start+64])[0]
                absolute_pe_offset = pe_start + pe_offset
                
                if absolute_pe_offset + 24 > len(data):
                    pe_info['issues'].append(f"PE header at 0x{pe_start:x} extends beyond file")
                    continue
                
                # Parse PE header
                if data[absolute_pe_offset:absolute_pe_offset+4] != b'PE\x00\x00':
                    pe_info['issues'].append(f"Invalid PE signature at 0x{absolute_pe_offset:x}")
                    continue
                
                # Parse COFF header
                coff_header = data[absolute_pe_offset+4:absolute_pe_offset+24]
                machine, num_sections, timestamp, ptr_to_symbols, num_symbols, opt_header_size, characteristics = struct.unpack('<HHIIIHH', coff_header)
                
                exe_info = {
                    'dos_start': pe_start,
                    'pe_header_offset': absolute_pe_offset,
                    'machine': f"0x{machine:x}",
                    'num_sections': num_sections,
                    'opt_header_size': opt_header_size,
                    'characteristics': f"0x{characteristics:x}",
                    'is_executable': bool(characteristics & 0x0002),  # IMAGE_FILE_EXECUTABLE_IMAGE
                    'is_dll': bool(characteristics & 0x2000)         # IMAGE_FILE_DLL
                }
                
                # Parse optional header for more details
                opt_header_start = absolute_pe_offset + 24
                if opt_header_start + 2 <= len(data):
                    magic = struct.unpack('<H', data[opt_header_start:opt_header_start+2])[0]
                    exe_info['pe_format'] = 'PE32+' if magic == 0x20b else 'PE32'
                    exe_info['magic'] = f"0x{magic:x}"
                
                pe_info['executables'].append(exe_info)
                
            except Exception as e:
                pe_info['issues'].append(f"Error analyzing PE at 0x{pe_start:x}: {e}")
        
        return pe_info
    
    def detect_structure_issues(self, data: bytes) -> List[str]:
        """Detect common structural issues in polyglot files."""
        
        issues = []
        
        # Check for conflicting headers
        if data.startswith(self.PNG_SIGNATURE) and data.startswith(b'MZ'):
            issues.append("File starts with both PNG and PE signatures (impossible)")
        
        # Check file size
        if len(data) > 100 * 1024 * 1024:  # 100MB
            issues.append("File is very large - may cause compatibility issues")
        
        # Check for unusual patterns
        if b'\x00' * 1000 in data:
            issues.append("File contains large null byte sequences")
        
        return issues
    
    def test_png_compatibility(self, file_path: str) -> Dict:
        """Test PNG compatibility with various image viewers."""
        
        self.log("Testing PNG compatibility...")
        
        results = {
            'format_valid': False,
            'viewer_tests': {},
            'issues': []
        }
        
        try:
            # Basic format validation
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if data.startswith(self.PNG_SIGNATURE):
                results['format_valid'] = True
                
                # Try to create a temporary copy with .png extension
                temp_png = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                try:
                    temp_png.write(data)
                    temp_png.close()
                    
                    # Test with built-in Python PIL if available
                    try:
                        from PIL import Image
                        img = Image.open(temp_png.name)
                        results['viewer_tests']['PIL'] = {
                            'success': True,
                            'format': img.format,
                            'mode': img.mode,
                            'size': img.size
                        }
                    except ImportError:
                        results['viewer_tests']['PIL'] = {'success': False, 'error': 'PIL not available'}
                    except Exception as e:
                        results['viewer_tests']['PIL'] = {'success': False, 'error': str(e)}
                    
                finally:
                    try:
                        os.unlink(temp_png.name)
                    except:
                        pass
            else:
                results['issues'].append("File does not start with PNG signature")
        
        except Exception as e:
            results['issues'].append(f"PNG test error: {e}")
        
        return results
    
    def test_pe_execution(self, file_path: str) -> Dict:
        """Test PE execution capabilities safely."""
        
        self.log("Testing PE execution...")
        
        results = {
            'executable_found': False,
            'execution_test': {},
            'safety_checks': {},
            'issues': []
        }
        
        try:
            # First, analyze the file to see if it contains PE
            analysis = self.analyze_file_structure(file_path)
            
            if 'PE' in analysis.get('formats_detected', []):
                results['executable_found'] = True
                
                # Safety checks first
                results['safety_checks'] = {
                    'file_size': analysis['file_size'],
                    'safe_size': analysis['file_size'] < 10 * 1024 * 1024,  # 10MB limit
                    'contains_pe': True
                }
                
                # Only attempt execution if safety checks pass
                if results['safety_checks']['safe_size']:
                    try:
                        # Create temporary executable
                        temp_exe = tempfile.NamedTemporaryFile(suffix='.exe', delete=False)
                        try:
                            with open(file_path, 'rb') as f:
                                temp_exe.write(f.read())
                            temp_exe.close()
                            
                            # Attempt execution with timeout
                            self.log(f"Testing execution of {temp_exe.name}")
                            start_time = time.time()
                            
                            result = subprocess.run(
                                [temp_exe.name],
                                capture_output=True,
                                text=True,
                                timeout=5,  # 5 second timeout
                                cwd=tempfile.gettempdir()
                            )
                            
                            execution_time = time.time() - start_time
                            
                            results['execution_test'] = {
                                'success': result.returncode == 0,
                                'exit_code': result.returncode,
                                'stdout': result.stdout[:500] if result.stdout else '',  # Limit output
                                'stderr': result.stderr[:500] if result.stderr else '',
                                'execution_time': execution_time
                            }
                            
                        finally:
                            try:
                                os.unlink(temp_exe.name)
                            except:
                                pass
                                
                    except subprocess.TimeoutExpired:
                        results['execution_test'] = {
                            'success': False,
                            'error': 'Execution timeout (>5 seconds)'
                        }
                    except FileNotFoundError:
                        results['execution_test'] = {
                            'success': False,
                            'error': 'File not found or not executable'
                        }
                    except Exception as e:
                        results['execution_test'] = {
                            'success': False,
                            'error': str(e)
                        }
                else:
                    results['issues'].append("File too large for safe execution testing")
            else:
                results['issues'].append("No PE executable found in file")
        
        except Exception as e:
            results['issues'].append(f"PE test error: {e}")
        
        return results
    
    def generate_report(self, file_path: str) -> str:
        """Generate comprehensive verification report."""
        
        print(f"\nPolyglot Verification Report")
        print("=" * 60)
        print(f"File: {file_path}")
        
        if not os.path.exists(file_path):
            return "❌ File not found"
        
        # File basic info
        file_size = os.path.getsize(file_path)
        print(f"Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # Structural analysis
        print(f"\n📊 STRUCTURAL ANALYSIS")
        print("-" * 30)
        
        structure = self.analyze_file_structure(file_path)
        if 'error' in structure:
            print(f"❌ Analysis failed: {structure['error']}")
            return "Analysis failed"
        
        formats = structure.get('formats_detected', [])
        print(f"Detected formats: {', '.join(formats) if formats else 'None'}")
        
        if structure.get('structure_issues'):
            print(f"⚠️  Structure issues:")
            for issue in structure['structure_issues']:
                print(f"   - {issue}")
        
        # PNG Analysis
        if 'PNG' in formats:
            print(f"\n🖼️  PNG ANALYSIS")
            print("-" * 20)
            
            png_info = structure['png_analysis']
            print(f"Valid signature: {'✓' if png_info['valid_signature'] else '❌'}")
            print(f"Required chunks: {'✓' if png_info['has_required_chunks'] else '❌'}")
            print(f"Chunks found: {len(png_info['chunks'])}")
            
            if png_info.get('image_info'):
                img = png_info['image_info']
                print(f"Dimensions: {img['width']}x{img['height']}")
                print(f"Bit depth: {img['bit_depth']}, Color type: {img['color_type']}")
            
            if png_info.get('issues'):
                print(f"Issues:")
                for issue in png_info['issues']:
                    print(f"   - {issue}")
            
            # PNG compatibility test
            png_test = self.test_png_compatibility(file_path)
            print(f"\nPNG Compatibility:")
            print(f"Format valid: {'✓' if png_test['format_valid'] else '❌'}")
            
            for viewer, test_result in png_test['viewer_tests'].items():
                if test_result['success']:
                    print(f"{viewer}: ✓ (Format: {test_result.get('format', 'Unknown')})")
                else:
                    print(f"{viewer}: ❌ ({test_result.get('error', 'Unknown error')})")
        
        # PE Analysis
        if 'PE' in formats:
            print(f"\n⚙️  PE ANALYSIS")
            print("-" * 20)
            
            pe_info = structure['pe_analysis']
            print(f"PE locations found: {len(pe_info['locations'])}")
            
            for i, exe in enumerate(pe_info['executables']):
                print(f"\nExecutable #{i+1}:")
                print(f"  DOS start: 0x{exe['dos_start']:x}")
                print(f"  PE header: 0x{exe['pe_header_offset']:x}")
                print(f"  Machine: {exe['machine']}")
                print(f"  Format: {exe.get('pe_format', 'Unknown')}")
                print(f"  Sections: {exe['num_sections']}")
                print(f"  Executable: {'✓' if exe['is_executable'] else '❌'}")
                print(f"  DLL: {'✓' if exe['is_dll'] else '❌'}")
            
            if pe_info.get('issues'):
                print(f"Issues:")
                for issue in pe_info['issues']:
                    print(f"   - {issue}")
            
            # PE execution test
            pe_test = self.test_pe_execution(file_path)
            print(f"\nPE Execution Test:")
            print(f"Executable found: {'✓' if pe_test['executable_found'] else '❌'}")
            
            if pe_test.get('safety_checks'):
                safety = pe_test['safety_checks']
                print(f"Safe for testing: {'✓' if safety['safe_size'] else '❌'}")
            
            if pe_test.get('execution_test'):
                exec_result = pe_test['execution_test']
                if exec_result.get('success'):
                    print(f"Execution: ✓ (Exit code: {exec_result['exit_code']})")
                    if exec_result.get('stdout'):
                        print(f"Output: {exec_result['stdout'][:100]}...")
                else:
                    print(f"Execution: ❌ ({exec_result.get('error', 'Unknown error')})")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT")
        print("-" * 25)
        
        png_works = 'PNG' in formats and structure['png_analysis']['valid_signature']
        pe_works = 'PE' in formats and structure['pe_analysis']['executables']
        
        if png_works and pe_works:
            print("✅ SUCCESS: File works as both PNG and PE!")
            status = "PASS"
        elif png_works:
            print("⚠️  PARTIAL: File works as PNG but not as PE")
            status = "PNG_ONLY"
        elif pe_works:
            print("⚠️  PARTIAL: File works as PE but not as PNG")
            status = "PE_ONLY"
        else:
            print("❌ FAILURE: File doesn't work as either format")
            status = "FAIL"
        
        print(f"\nRecommendations:")
        if not png_works:
            print("- Fix PNG structure issues for image viewer compatibility")
        if not pe_works:
            print("- Fix PE structure issues for executable compatibility")
        if png_works and pe_works:
            print("- Polyglot is working correctly!")
            print("- Test with various image viewers and execution environments")
        
        return status


def main():
    """Main verification function."""
    
    if len(sys.argv) < 2:
        print("Polyglot Verification Tool")
        print("Usage:")
        print(f"  {sys.argv[0]} <polyglot_file>")
        print()
        print("This tool verifies that polyglot files work as both PNG and EXE.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    verifier = PolyglotVerifier()
    status = verifier.generate_report(file_path)
    
    # Set exit code based on results
    exit_codes = {
        'PASS': 0,
        'PNG_ONLY': 1,
        'PE_ONLY': 1,
        'FAIL': 2
    }
    
    sys.exit(exit_codes.get(status, 2))


if __name__ == "__main__":
    main()
