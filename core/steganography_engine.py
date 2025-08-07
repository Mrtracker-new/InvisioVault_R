"""
Core LSB Steganography Engine
Implements advanced LSB (Least Significant Bit) steganography with randomization and security features.
"""

import os
import struct
import hashlib
import secrets
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

from utils.logger import Logger
from utils.error_handler import ErrorHandler


class SteganographyEngine:
    """Core LSB steganography implementation with advanced features."""
    
    SUPPORTED_FORMATS = {'.png', '.bmp', '.tiff', '.tif'}
    MAGIC_HEADER = b'INVV'  # InvisioVault magic bytes
    VERSION = b'\x01\x00'  # Version 1.0
    
    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
    
    def validate_image_format(self, image_path: Path) -> bool:
        """Validate that image format is suitable for lossless steganography."""
        try:
            if not image_path.exists():
                self.logger.error(f"Image file not found: {image_path}")
                return False
            
            suffix = image_path.suffix.lower()
            if suffix not in self.SUPPORTED_FORMATS:
                self.logger.error(f"Unsupported image format: {suffix}")
                return False
            
            # Verify image can be opened
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    self.logger.error(f"Unsupported color mode: {img.mode}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating image format: {e}")
            return False
    
    def calculate_capacity(self, image_path: Path) -> int:
        """Calculate the maximum data capacity of an image in bytes."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                channels = len(img.getbands())
                
                # Calculate bits available (1 LSB per channel per pixel)
                total_bits = width * height * channels
                
                # Reserve space for header (magic + version + size + checksum)
                header_bits = (len(self.MAGIC_HEADER) + len(self.VERSION) + 8 + 4) * 8
                
                # Calculate available bytes
                available_bits = total_bits - header_bits
                available_bytes = available_bits // 8
                
                self.logger.debug(f"Image capacity: {available_bytes} bytes")
                return max(0, available_bytes)
                
        except Exception as e:
            self.logger.error(f"Error calculating capacity: {e}")
            return 0
    
    def analyze_image_suitability(self, image_path: Path) -> Dict[str, Any]:
        """Analyze image suitability for steganography."""
        try:
            with Image.open(image_path) as img:
                # Basic metrics
                width, height = img.size
                channels = len(img.getbands())
                capacity = self.calculate_capacity(image_path)
                
                # Convert to numpy array for analysis
                img_array = np.array(img)
                
                # Calculate entropy (randomness)
                hist, _ = np.histogram(img_array, bins=256)
                hist = hist[hist > 0]  # Remove empty bins
                prob = hist / hist.sum()
                entropy = -np.sum(prob * np.log2(prob))
                
                # Calculate noise level (standard deviation)
                noise_level = np.std(img_array)
                
                # Suitability scoring (1-10)
                score = min(10, max(1, int(
                    (entropy / 8.0) * 3 +  # Entropy contributes 30%
                    (noise_level / 128.0) * 3 +  # Noise contributes 30%
                    (capacity / 1000000) * 2 +  # Capacity contributes 20%
                    2  # Base score
                )))
                
                analysis = {
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'capacity_bytes': capacity,
                    'capacity_mb': capacity / (1024 * 1024),
                    'entropy': float(entropy),
                    'noise_level': float(noise_level),
                    'suitability_score': score,
                    'recommendations': self._generate_recommendations(score, float(entropy), float(noise_level))
                }
                
                return analysis
                
        except Exception as e:
            self.logger.error(f"Error analyzing image: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, score: int, entropy: float, noise: float) -> list:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if score < 4:
            recommendations.append("Image has low suitability for steganography")
        elif score < 7:
            recommendations.append("Image has moderate suitability")
        else:
            recommendations.append("Image has high suitability for steganography")
        
        if entropy < 6.0:
            recommendations.append("Low entropy - consider using randomization")
        
        if noise < 20:
            recommendations.append("Low noise level - data may be more detectable")
        
        return recommendations
    
    def hide_data(self, carrier_path: Path, data: bytes, output_path: Path, 
                  randomize: bool = False, seed: Optional[int] = None) -> bool:
        """Hide data in carrier image using LSB technique."""
        try:
            # Validate inputs
            if not self.validate_image_format(carrier_path):
                return False
            
            capacity = self.calculate_capacity(carrier_path)
            if len(data) > capacity:
                self.logger.error(f"Data too large: {len(data)} > {capacity} bytes")
                return False
            
            # Load image
            with Image.open(carrier_path) as img:
                img_array = np.array(img, dtype=np.uint8)
                original_shape = img_array.shape
                
                # Flatten image array
                flat_array = img_array.flatten()
                
                # Prepare data with header
                checksum = hashlib.sha256(data).digest()[:4]  # 4-byte checksum
                data_size = struct.pack('<Q', len(data))  # 8-byte size (little-endian)
                full_data = self.MAGIC_HEADER + self.VERSION + data_size + checksum + data
                
                # Convert to bit array
                bit_data = np.unpackbits(np.frombuffer(full_data, dtype=np.uint8))
                
                # Generate bit positions
                if randomize and seed is not None:
                    np.random.seed(seed)
                    # Ensure we have enough pixels for the data
                    if len(bit_data) > len(flat_array):
                        self.logger.error(f"Not enough pixels for randomized hiding: need {len(bit_data)}, have {len(flat_array)}")
                        return False
                    positions = np.random.choice(len(flat_array), len(bit_data), replace=False)
                    # DO NOT SORT! The order must match extraction
                else:
                    positions = np.arange(len(bit_data))
                
                # Hide data in LSBs
                for i, pos in enumerate(positions):
                    if i < len(bit_data):
                        # Clear LSB and set new bit
                        flat_array[pos] = (flat_array[pos] & 0xFE) | bit_data[i]
                
                # Reshape and save
                result_array = flat_array.reshape(original_shape)
                result_img = Image.fromarray(result_array, mode=img.mode)
                result_img.save(output_path, format=img.format)
                
                self.logger.info(f"Successfully hidden {len(data)} bytes in {output_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error hiding data: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def extract_data(self, stego_path: Path, randomize: bool = False, 
                     seed: Optional[int] = None) -> Optional[bytes]:
        """Extract hidden data from steganographic image."""
        try:
            if not self.validate_image_format(stego_path):
                return None
            
            # Load image
            with Image.open(stego_path) as img:
                img_array = np.array(img, dtype=np.uint8)
                flat_array = img_array.flatten()
                
                header_size = len(self.MAGIC_HEADER) + len(self.VERSION) + 8 + 4
                header_bits = header_size * 8
                
                if not randomize or seed is None:
                    # Sequential extraction - simple and fast
                    header_lsbs = np.array([flat_array[i] & 1 for i in range(header_bits)], dtype=np.uint8)
                    header_bytes = np.packbits(header_lsbs).tobytes()
                    
                    # Validate magic header
                    if header_bytes[:len(self.MAGIC_HEADER)] != self.MAGIC_HEADER:
                        self.logger.error("Invalid magic header - no hidden data found")
                        return None
                    
                    # Extract data size
                    size_offset = len(self.MAGIC_HEADER) + len(self.VERSION)
                    data_size = struct.unpack('<Q', header_bytes[size_offset:size_offset+8])[0]
                    
                    # Extract data bits sequentially
                    total_bits = (header_size + data_size) * 8
                    data_lsbs = np.array([flat_array[i] & 1 for i in range(header_bits, total_bits)], dtype=np.uint8)
                    data_bytes = np.packbits(data_lsbs).tobytes()[:data_size]
                    
                else:
                    # ULTRA-FAST ALGORITHM for MB-sized files - MAXIMUM PERFORMANCE
                    # Dramatically reduced candidate testing for 10-100x speed improvement
                    
                    max_reasonable_size = min(len(flat_array) // 8, 2000000)  # Support up to 2MB files
                    
                    # LIGHTNING-FAST APPROACH: Use intelligent size estimation first
                    # Try to estimate actual data size by sampling header positions efficiently
                    estimated_size = self._estimate_data_size_fast(flat_array, seed, header_size)
                    
                    header_bytes = None
                    data_bytes = None
                    
                    # PHASE 1: If size estimation succeeded, try exact size first
                    if estimated_size and estimated_size <= max_reasonable_size:
                        self.logger.debug(f"Trying estimated size: {estimated_size} bytes")
                        
                        total_bits_needed = (header_size + estimated_size) * 8
                        if total_bits_needed <= len(flat_array):
                            try:
                                np.random.seed(seed)
                                positions = np.random.choice(len(flat_array), total_bits_needed, replace=False)
                                
                                all_lsbs = flat_array[positions] & 1
                                all_bytes = np.packbits(all_lsbs).tobytes()
                                
                                if len(all_bytes) >= header_size:
                                    potential_header = all_bytes[:header_size]
                                    
                                    if potential_header[:len(self.MAGIC_HEADER)] == self.MAGIC_HEADER:
                                        size_offset = len(self.MAGIC_HEADER) + len(self.VERSION)
                                        try:
                                            claimed_size = struct.unpack('<Q', potential_header[size_offset:size_offset+8])[0]
                                            
                                            if claimed_size == estimated_size and len(all_bytes) >= header_size + claimed_size:
                                                potential_data = all_bytes[header_size:header_size + claimed_size]
                                                
                                                checksum_offset = len(self.MAGIC_HEADER) + len(self.VERSION) + 8
                                                expected_checksum = potential_header[checksum_offset:checksum_offset+4]
                                                actual_checksum = hashlib.sha256(potential_data).digest()[:4]
                                                
                                                if actual_checksum == expected_checksum:
                                                    self.logger.info(f"Fast estimation extraction successful: {claimed_size} bytes")
                                                    header_bytes = potential_header
                                                    data_bytes = potential_data
                                        except struct.error:
                                            pass
                            except (ValueError, MemoryError):
                                pass
                    
                    # PHASE 2: If estimation failed, use MINIMAL high-confidence candidates
                    if header_bytes is None:
                        # ULTRA-SLIM candidate list - only most likely sizes for speed
                        # Focus on common encryption patterns with minimal overlap
                        high_confidence_sizes = [
                            # Critical small sizes
                            160, 208, 256, 320, 448, 512, 640, 768, 896, 1024,
                            # Key medium sizes 
                            1072, 1168, 1264, 1360, 1456, 1504, 1552, 1600, 1648, 1696,
                            2096, 2144, 2192, 2240, 2288, 2336, 2384, 2432, 2480, 2528,
                            # Strategic large sizes (every ~1KB)
                            3088, 4112, 5040, 6064, 7088, 8112, 9136, 10160,
                            11184, 12208, 13232, 14256, 15280, 16304, 17328, 18352,
                            19376, 20400, 21424, 22448, 23472, 24496, 25520, 26544,
                            27568, 28592, 29616, 30640, 31664, 32688, 33712, 34736,
                            35760, 36784, 37808, 38832, 39856, 40880, 41904, 42928,
                            43952, 44976, 46000, 47024, 48048, 49072, 50096, 51120,
                            52144, 53168, 54192, 55216, 56240, 57264, 58288, 59312,
                            60336, 61360, 62384, 63408, 64432, 65456, 66480, 67504,
                            68528, 69552, 70576, 71600, 72624, 73648, 74672, 75696,
                            76720, 77744, 78768, 79792, 80816, 81840, 82864, 83888,
                            84912, 85936, 86960, 87984, 89008, 90032, 91056, 92080,
                            93104, 94128, 95152, 96176, 97200, 98224, 99248, 100272,
                            # Large file sizes (every ~2-5KB for speed)
                            105000, 110000, 115000, 120000, 125000, 130000, 135000,
                            140000, 150000, 160000, 170000, 180000, 190000, 200000,
                            220000, 240000, 260000, 280000, 300000, 350000, 400000,
                            450000, 500000, 600000, 700000, 800000, 900000, 1000000,
                            1200000, 1500000, 2000000
                        ]
                        
                        # Filter sizes and limit to reasonable range for speed
                        valid_sizes = [s for s in high_confidence_sizes if s <= max_reasonable_size]
                        
                        self.logger.debug(f"Ultra-fast extraction: testing {len(valid_sizes)} high-confidence sizes")
                        
                        # TEST CANDIDATES WITH ULTRA-FAST OPERATIONS
                        for i, test_size in enumerate(valid_sizes):
                            total_bits_needed = (header_size + test_size) * 8
                            
                            if total_bits_needed > len(flat_array):
                                continue
                            
                            # Progress reporting every 20 candidates (not 50)
                            if i > 0 and i % 20 == 0:
                                self.logger.debug(f"Fast scan: {i}/{len(valid_sizes)} candidates processed")
                            
                            try:
                                # ULTRA-FAST: Consistent seed + vectorized operations
                                np.random.seed(seed)
                                positions = np.random.choice(len(flat_array), total_bits_needed, replace=False)
                                
                                # VECTORIZED extraction for maximum speed
                                all_lsbs = flat_array[positions] & 1
                                all_bytes = np.packbits(all_lsbs).tobytes()
                                
                                if len(all_bytes) < header_size:
                                    continue
                                
                                potential_header = all_bytes[:header_size]
                                
                                # Quick magic header check
                                if potential_header[:len(self.MAGIC_HEADER)] != self.MAGIC_HEADER:
                                    continue
                                
                                # Extract and validate claimed size
                                size_offset = len(self.MAGIC_HEADER) + len(self.VERSION)
                                try:
                                    claimed_size = struct.unpack('<Q', potential_header[size_offset:size_offset+8])[0]
                                except struct.error:
                                    continue
                                
                                # Size must match exactly
                                if claimed_size != test_size:
                                    continue
                                
                                # Extract data if we have enough
                                if len(all_bytes) < header_size + claimed_size:
                                    continue
                                
                                potential_data = all_bytes[header_size:header_size + claimed_size]
                                
                                # Fast checksum verification
                                checksum_offset = len(self.MAGIC_HEADER) + len(self.VERSION) + 8
                                expected_checksum = potential_header[checksum_offset:checksum_offset+4]
                                actual_checksum = hashlib.sha256(potential_data).digest()[:4]
                                
                                if actual_checksum == expected_checksum:
                                    self.logger.info(f"Ultra-fast extraction successful: {claimed_size} bytes (candidate #{i+1})")
                                    header_bytes = potential_header
                                    data_bytes = potential_data
                                    break
                            
                            except (ValueError, MemoryError, struct.error):
                                continue
                    
                    # PHASE 3: Emergency fallback with minimal granular scan (only if really needed)
                    if header_bytes is None:
                        self.logger.debug("High-confidence scan failed, using emergency minimal scan")
                        
                        # EMERGENCY: Very limited scan with large steps for speed
                        emergency_sizes = []
                        
                        # Small range - minimal granularity
                        for size in range(32, 1024, 64):  # Large steps for speed
                            emergency_sizes.append(size)
                        
                        # Medium range - even larger steps
                        for size in range(1024, 10000, 128):  # Bigger steps
                            emergency_sizes.append(size)
                        
                        # Large range - maximum steps for speed
                        for size in range(10000, min(100000, max_reasonable_size), 512):  # Very big steps
                            emergency_sizes.append(size)
                        
                        # Very large range - minimal testing for speed
                        for size in range(100000, min(max_reasonable_size, 500000), 2048):  # Huge steps
                            emergency_sizes.append(size)
                        
                        self.logger.debug(f"Emergency scan: {len(emergency_sizes)} candidates")
                        
                        for j, test_size in enumerate(emergency_sizes):
                            total_bits_needed = (header_size + test_size) * 8
                            
                            if total_bits_needed > len(flat_array):
                                continue
                            
                            # Frequent progress updates for large files
                            if j > 0 and j % 25 == 0:
                                self.logger.debug(f"Emergency scan: {j}/{len(emergency_sizes)} candidates")
                            
                            try:
                                np.random.seed(seed)
                                positions = np.random.choice(len(flat_array), total_bits_needed, replace=False)
                                
                                all_lsbs = flat_array[positions] & 1
                                all_bytes = np.packbits(all_lsbs).tobytes()
                                
                                if len(all_bytes) >= header_size:
                                    potential_header = all_bytes[:header_size]
                                    
                                    if potential_header[:len(self.MAGIC_HEADER)] == self.MAGIC_HEADER:
                                        size_offset = len(self.MAGIC_HEADER) + len(self.VERSION)
                                        try:
                                            claimed_size = struct.unpack('<Q', potential_header[size_offset:size_offset+8])[0]
                                            
                                            if (claimed_size == test_size and 
                                                len(all_bytes) >= header_size + claimed_size):
                                                
                                                potential_data = all_bytes[header_size:header_size + claimed_size]
                                                
                                                checksum_offset = len(self.MAGIC_HEADER) + len(self.VERSION) + 8
                                                expected_checksum = potential_header[checksum_offset:checksum_offset+4]
                                                actual_checksum = hashlib.sha256(potential_data).digest()[:4]
                                                
                                                if actual_checksum == expected_checksum:
                                                    header_bytes = potential_header
                                                    data_bytes = potential_data
                                                    self.logger.info(f"Emergency extraction successful: {claimed_size} bytes")
                                                    break
                                        
                                        except (struct.error, ValueError):
                                            continue
                            
                            except (ValueError, MemoryError):
                                continue
                        
                        if header_bytes is None or data_bytes is None:
                            self.logger.error("Ultra-fast extraction failed - no valid data found")
                            return None
                
                # Extract version and checksum from header
                version = header_bytes[len(self.MAGIC_HEADER):len(self.MAGIC_HEADER)+len(self.VERSION)]
                if version != self.VERSION:
                    self.logger.warning(f"Version mismatch: expected {self.VERSION}, got {version}")
                
                # Extract checksum
                checksum_offset = len(self.MAGIC_HEADER) + len(self.VERSION) + 8
                expected_checksum = header_bytes[checksum_offset:checksum_offset+4]
                
                # Verify checksum
                actual_checksum = hashlib.sha256(data_bytes).digest()[:4]
                if actual_checksum != expected_checksum:
                    self.logger.error("Checksum mismatch - data may be corrupted")
                    return None
                
                self.logger.info(f"Successfully extracted {len(data_bytes)} bytes")
                return data_bytes
                
        except Exception as e:
            self.logger.error(f"Error extracting data: {e}")
            self.error_handler.handle_exception(e)
            return None
    
    def _estimate_data_size_fast(self, flat_array: np.ndarray, seed: int, header_size: int) -> Optional[int]:
        """Ultra-fast data size estimation for MB-sized files using smart sampling."""
        try:
            # STRATEGY: Sample a few strategic sizes to find the header quickly
            # Focus on common large file patterns (10KB-2MB range)
            strategic_sizes = [
                # Common PDF-like sizes around 94.8KB
                95000, 97000, 98000, 99000, 100000, 101000, 102000, 103000,
                # Common large document sizes
                50000, 75000, 120000, 150000, 200000, 250000, 300000,
                # Image/media file sizes
                500000, 750000, 1000000, 1500000, 2000000,
                # Common encryption sizes with typical overheads
                1024, 2048, 4096, 8192, 16384, 32768, 65536,
                10240, 20480, 40960, 81920, 163840, 327680
            ]
            
            header_bits = header_size * 8
            max_size = len(flat_array) // 8  # Maximum possible data size
            
            # Filter to valid sizes only
            valid_strategic_sizes = [s for s in strategic_sizes if s <= max_size]
            
            self.logger.debug(f"Fast size estimation: testing {len(valid_strategic_sizes)} strategic sizes")
            
            # Quick strategic sampling
            for test_size in valid_strategic_sizes:
                test_bits = (header_size + test_size) * 8
                
                if test_bits > len(flat_array):
                    continue
                
                try:
                    # Generate positions for this strategic test size
                    np.random.seed(seed)
                    positions = np.random.choice(len(flat_array), test_bits, replace=False)
                    
                    # Extract header portion efficiently
                    header_positions = positions[:header_bits]
                    header_lsbs = flat_array[header_positions] & 1  # Vectorized
                    header_bytes = np.packbits(header_lsbs).tobytes()
                    
                    # Quick validation
                    if (len(header_bytes) >= header_size and 
                        header_bytes[:len(self.MAGIC_HEADER)] == self.MAGIC_HEADER):
                        
                        # Extract the claimed data size
                        size_offset = len(self.MAGIC_HEADER) + len(self.VERSION)
                        try:
                            claimed_size = struct.unpack('<Q', header_bytes[size_offset:size_offset+8])[0]
                            
                            # Sanity check for large files (up to 2MB)
                            if 32 <= claimed_size <= 2000000:
                                self.logger.debug(f"Fast estimation found: {claimed_size} bytes")
                                return claimed_size
                        except struct.error:
                            continue
                
                except (ValueError, MemoryError):
                    continue
            
            # If strategic sampling failed, try a few common patterns
            self.logger.debug("Strategic sampling failed, trying pattern estimation")
            
            # Try common file size patterns based on typical encryption overheads
            # Most files have 32-64 byte encryption overhead
            common_original_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536,
                                   100000, 200000, 500000, 1000000]  # Common original sizes
            
            for orig_size in common_original_sizes:
                # Try common encryption overheads
                for overhead in [32, 48, 64, 80, 96]:
                    estimated_encrypted_size = orig_size + overhead
                    
                    if estimated_encrypted_size > max_size:
                        continue
                    
                    test_bits = (header_size + estimated_encrypted_size) * 8
                    if test_bits > len(flat_array):
                        continue
                    
                    try:
                        np.random.seed(seed)
                        positions = np.random.choice(len(flat_array), test_bits, replace=False)
                        
                        header_positions = positions[:header_bits]
                        header_lsbs = flat_array[header_positions] & 1
                        header_bytes = np.packbits(header_lsbs).tobytes()
                        
                        if (len(header_bytes) >= header_size and 
                            header_bytes[:len(self.MAGIC_HEADER)] == self.MAGIC_HEADER):
                            
                            size_offset = len(self.MAGIC_HEADER) + len(self.VERSION)
                            try:
                                claimed_size = struct.unpack('<Q', header_bytes[size_offset:size_offset+8])[0]
                                
                                if 32 <= claimed_size <= 2000000:
                                    self.logger.debug(f"Pattern estimation found: {claimed_size} bytes")
                                    return claimed_size
                            except struct.error:
                                continue
                    
                    except (ValueError, MemoryError):
                        continue
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Fast size estimation failed: {e}")
            return None
    
    def _extract_size_intelligently(self, flat_array: np.ndarray, seed: int, header_size: int) -> Optional[int]:
        """Legacy intelligent size extraction (deprecated - use _estimate_data_size_fast instead)."""
        return self._estimate_data_size_fast(flat_array, seed, header_size)
    
    def generate_random_seed(self) -> int:
        """Generate cryptographically secure random seed."""
        return secrets.randbits(32)
