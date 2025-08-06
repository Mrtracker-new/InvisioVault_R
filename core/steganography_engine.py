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
                header_bits = (len(self.MAGIC_HEADER) + len(self.VERSION) + 8 + 32) * 8
                
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
                    positions = np.random.choice(len(flat_array), len(bit_data), replace=False)
                    positions.sort()  # Sort for extraction
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
                
                # Extract header first to determine data size
                header_size = len(self.MAGIC_HEADER) + len(self.VERSION) + 8 + 4  # +8 for size, +4 for checksum
                header_bits = header_size * 8
                
                if randomize and seed is not None:
                    np.random.seed(seed)
                    header_positions = np.random.choice(len(flat_array), header_bits, replace=False)
                    header_positions.sort()
                else:
                    header_positions = np.arange(header_bits)
                
                # Extract header bits
                header_lsbs = np.array([flat_array[pos] & 1 for pos in header_positions], dtype=np.uint8)
                header_bytes = np.packbits(header_lsbs).tobytes()
                
                # Validate magic header
                if header_bytes[:len(self.MAGIC_HEADER)] != self.MAGIC_HEADER:
                    self.logger.error("Invalid magic header - no hidden data found")
                    return None
                
                # Extract version
                version = header_bytes[len(self.MAGIC_HEADER):len(self.MAGIC_HEADER)+len(self.VERSION)]
                if version != self.VERSION:
                    self.logger.warning(f"Version mismatch: expected {self.VERSION}, got {version}")
                
                # Extract data size
                size_offset = len(self.MAGIC_HEADER) + len(self.VERSION)
                data_size = struct.unpack('<Q', header_bytes[size_offset:size_offset+8])[0]
                
                # Extract checksum
                checksum_offset = size_offset + 8
                expected_checksum = header_bytes[checksum_offset:checksum_offset+4]
                
                # Extract actual data
                total_bits = (header_size + data_size) * 8
                
                if randomize and seed is not None:
                    np.random.seed(seed)
                    all_positions = np.random.choice(len(flat_array), total_bits, replace=False)
                    all_positions.sort()
                    data_positions = all_positions[header_bits:]
                else:
                    data_positions = np.arange(header_bits, total_bits)
                
                # Extract data bits
                data_lsbs = np.array([flat_array[pos] & 1 for pos in data_positions], dtype=np.uint8)
                data_bytes = np.packbits(data_lsbs).tobytes()[:data_size]
                
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
    
    def generate_random_seed(self) -> int:
        """Generate cryptographically secure random seed."""
        return secrets.randbits(32)
