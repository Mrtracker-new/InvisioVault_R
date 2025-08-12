"""
Secure Steganography Engine
Enhanced steganography engine designed to be undetectable by forensic tools like binwalk, 
hexdump, and other signature-based analysis tools.

Key Security Improvements:
1. No magic headers or signatures
2. Randomized data distribution across all LSBs
3. Noise injection to mask patterns
4. Password-derived entropy seeding
5. Indistinguishable from normal LSB noise
"""

import os
import struct
import hashlib
import secrets
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from PIL import Image
import zlib

from utils.logger import Logger
from utils.error_handler import ErrorHandler


class SecureSteganographyEngine:
    """Ultra-secure steganography engine with no detectable signatures."""
    
    SUPPORTED_FORMATS = {'.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
    
    def hide_data_secure(self, carrier_path, data: bytes, output_path, 
                        password: str, compression_level: int = 6) -> bool:
        """
        Hide data using ultra-secure steganography with no detectable patterns.
        
        Args:
            carrier_path: Path to carrier image
            data: Data to hide
            output_path: Output path
            password: Password for encryption and position derivation
            compression_level: Compression level (1-9)
        
        Returns:
            Success status
        """
        try:
            # Convert paths
            if isinstance(carrier_path, str):
                carrier_path = Path(carrier_path)
            if isinstance(output_path, str):
                output_path = Path(output_path)
            
            # Validate inputs
            if not self._validate_image(carrier_path):
                return False
            
            # Load and prepare image
            with Image.open(carrier_path) as img:
                # Convert to RGB to ensure consistent channel count
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                img_array = np.array(img, dtype=np.uint8)
                original_shape = img_array.shape
                
                # Calculate capacity
                total_pixels = img_array.size
                max_capacity = total_pixels // 8  # 1 bit per pixel
                
                # Prepare secure payload
                secure_payload = self._create_secure_payload(data, password, compression_level)
                
                if len(secure_payload) > max_capacity:
                    self.logger.error(f"Data too large: {len(secure_payload)} > {max_capacity} bytes")
                    return False
                
                # Generate secure positions
                positions = self._generate_secure_positions(
                    img_array, len(secure_payload) * 8, password
                )
                
                # Embed data securely
                modified_array = self._embed_data_securely(
                    img_array, secure_payload, positions, password
                )
                
                # Apply noise injection to mask patterns
                modified_array = self._inject_masking_noise(
                    modified_array, img_array, password
                )
                
                # Save result
                result_img = Image.fromarray(modified_array, mode=img.mode)
                result_img.save(output_path, format=img.format, optimize=True)
                
                self.logger.info(f"Secure hiding successful: {len(data)} bytes")
                return True
                
        except Exception as e:
            self.logger.error(f"Secure hiding failed: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def extract_data_secure(self, stego_path, password: str) -> Optional[bytes]:
        """
        Extract data using secure steganography methods.
        
        Args:
            stego_path: Path to steganographic image
            password: Password for extraction
        
        Returns:
            Extracted data or None
        """
        try:
            # Convert path
            if isinstance(stego_path, str):
                stego_path = Path(stego_path)
            
            if not self._validate_image(stego_path):
                return None
            
            # Load image
            with Image.open(stego_path) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                img_array = np.array(img, dtype=np.uint8)
                
                # Try different payload sizes (no magic header to detect)
                return self._secure_extraction_search(img_array, password)
                
        except Exception as e:
            self.logger.error(f"Secure extraction failed: {e}")
            return None
    
    def _create_secure_payload(self, data: bytes, password: str, compression_level: int) -> bytes:
        """Create secure payload with no detectable signatures."""
        
        # 1. Compress data
        compressed_data = zlib.compress(data, compression_level)
        
        # 2. Create password-derived key for encryption
        salt = self._derive_salt(password, b"payload_salt")
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, 32)
        
        # 3. Simple XOR encryption (undetectable pattern)
        encrypted_data = self._xor_encrypt(compressed_data, key)
        
        # 4. Create secure header (looks like random noise)
        # No magic bytes - use password-derived pseudo-random pattern
        header_key = self._derive_salt(password, b"header_key")[:16]
        
        # Size information embedded in pseudo-random pattern
        size_info = struct.pack('<Q', len(data))  # Original size
        compressed_size_info = struct.pack('<Q', len(compressed_data))  # Compressed size
        
        # Create checksum of original data
        checksum = hashlib.sha256(data).digest()[:4]
        
        # Combine header components (all look random)
        header_data = header_key + size_info + compressed_size_info + checksum
        
        # Encrypt header to look completely random
        header_encrypted = self._xor_encrypt(header_data, key[:len(header_data)])
        
        # Final payload: encrypted_header + encrypted_data
        payload = header_encrypted + encrypted_data
        
        return payload
    
    def _derive_salt(self, password: str, purpose: bytes) -> bytes:
        """Derive a salt from password and purpose."""
        return hashlib.pbkdf2_hmac('sha256', password.encode(), purpose, 10000, 16)
    
    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple XOR encryption that produces no detectable patterns."""
        if not key:
            return data
        
        # Extend key to data length
        extended_key = (key * ((len(data) // len(key)) + 1))[:len(data)]
        
        # XOR data with key
        result = bytes(a ^ b for a, b in zip(data, extended_key))
        return result
    
    def _generate_secure_positions(self, img_array: np.ndarray, bits_needed: int, 
                                 password: str) -> np.ndarray:
        """Generate secure, pseudo-random positions based on password."""
        
        total_pixels = img_array.size
        
        if bits_needed > total_pixels:
            raise ValueError(f"Need {bits_needed} bits but only have {total_pixels} pixels")
        
        # Create deterministic but unpredictable sequence
        seed_material = hashlib.sha256(
            password.encode() + b"position_seed" + str(img_array.shape).encode()
        ).digest()
        
        # Use first 8 bytes as seed
        seed = int.from_bytes(seed_material[:8], 'little') % (2**31)
        
        # Generate positions using numpy random with our seed
        rng = np.random.default_rng(seed)
        
        # Create flat pixel indices
        flat_indices = rng.choice(total_pixels, size=bits_needed, replace=False)
        
        # Convert to 3D coordinates
        height, width = img_array.shape[:2]
        channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
        
        # Convert flat indices to (y, x, channel) coordinates
        positions = []
        for idx in flat_indices:
            if len(img_array.shape) == 3:  # Color image
                pixel_idx = idx // channels
                channel = idx % channels
                y = pixel_idx // width
                x = pixel_idx % width
                positions.append((y, x, channel))
            else:  # Grayscale
                y = idx // width
                x = idx % width
                positions.append((y, x, 0))
        
        return np.array(positions)
    
    def _embed_data_securely(self, img_array: np.ndarray, payload: bytes, 
                           positions: np.ndarray, password: str) -> np.ndarray:
        """Embed data with additional security measures."""
        
        modified_array = img_array.copy()
        
        # Convert payload to bits
        payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        
        # Embed each bit
        for i, (y, x, c) in enumerate(positions[:len(payload_bits)]):
            if len(img_array.shape) == 3:
                original_value = modified_array[y, x, c]
            else:
                original_value = modified_array[y, x]
            
            # Set LSB to payload bit
            new_value = (original_value & 0xFE) | payload_bits[i]
            
            if len(img_array.shape) == 3:
                modified_array[y, x, c] = new_value
            else:
                modified_array[y, x] = new_value
        
        return modified_array
    
    def _inject_masking_noise(self, modified_array: np.ndarray, 
                            original_array: np.ndarray, password: str) -> np.ndarray:
        """Inject subtle noise to mask steganographic changes."""
        
        # Create password-derived noise seed
        noise_seed = int.from_bytes(
            hashlib.sha256(password.encode() + b"noise_seed").digest()[:8], 'little'
        ) % (2**31)
        
        rng = np.random.default_rng(noise_seed)
        
        # Find pixels that weren't modified
        if len(modified_array.shape) == 3:
            unchanged_mask = np.all(modified_array == original_array, axis=2)
            height, width, channels = modified_array.shape
        else:
            unchanged_mask = modified_array == original_array
            height, width = modified_array.shape
            channels = 1
        
        # Inject minimal noise into some unchanged pixels
        noise_probability = 0.01  # 1% of unchanged pixels get minimal noise
        noise_mask = rng.random((height, width)) < noise_probability
        final_noise_mask = unchanged_mask & noise_mask
        
        # Apply very subtle noise (±1 to LSBs occasionally)
        noise_indices = np.where(final_noise_mask)
        for i in range(len(noise_indices[0])):
            y, x = noise_indices[0][i], noise_indices[1][i]
            
            if len(modified_array.shape) == 3:
                for c in range(channels):
                    if rng.random() < 0.3:  # 30% chance per channel
                        current = modified_array[y, x, c]
                        # Flip LSB randomly
                        modified_array[y, x, c] = current ^ 1
            else:
                if rng.random() < 0.3:
                    current = modified_array[y, x]
                    modified_array[y, x] = current ^ 1
        
        return modified_array
    
    def _secure_extraction_search(self, img_array: np.ndarray, password: str) -> Optional[bytes]:
        """Search for hidden data using secure methods."""
        
        # Try different payload sizes
        # Start with common compressed file sizes
        candidates = [
            # Small files (few KB)
            50, 100, 200, 500, 1024, 2048, 4096, 8192,
            # Medium files (10s of KB)
            16384, 32768, 65536, 131072, 262144,
            # Large files (100s of KB to MB)
            524288, 1048576, 2097152, 5242880, 10485760
        ]
        
        max_possible_size = img_array.size // 8
        valid_candidates = [size for size in candidates if size <= max_possible_size]
        
        # Add some intermediate sizes
        for size in range(100, min(10000, max_possible_size), 100):
            if size not in valid_candidates:
                valid_candidates.append(size)
        
        valid_candidates.sort()
        
        for candidate_size in valid_candidates:
            try:
                result = self._try_extract_size(img_array, candidate_size, password)
                if result:
                    return result
            except Exception as e:
                self.logger.debug(f"Extraction attempt for size {candidate_size} failed: {e}")
                continue
        
        return None
    
    def _try_extract_size(self, img_array: np.ndarray, payload_size: int, password: str) -> Optional[bytes]:
        """Try to extract data assuming a specific payload size."""
        
        try:
            # Generate positions for this size
            positions = self._generate_secure_positions(img_array, payload_size * 8, password)
            
            # Extract bits
            extracted_bits = []
            for y, x, c in positions:
                if len(img_array.shape) == 3:
                    bit = img_array[y, x, c] & 1
                else:
                    bit = img_array[y, x] & 1
                extracted_bits.append(bit)
            
            # Convert to bytes
            extracted_bytes = np.packbits(extracted_bits).tobytes()[:payload_size]
            
            # Try to parse as secure payload
            return self._parse_secure_payload(extracted_bytes, password)
            
        except Exception:
            return None
    
    def _parse_secure_payload(self, payload: bytes, password: str) -> Optional[bytes]:
        """Parse secure payload and verify integrity."""
        
        try:
            # Derive decryption key
            salt = self._derive_salt(password, b"payload_salt")
            key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, 32)
            
            # Expected header size: 16 (header_key) + 8 (orig_size) + 8 (comp_size) + 4 (checksum) = 36 bytes
            if len(payload) < 36:
                return None
            
            # Decrypt header
            encrypted_header = payload[:36]
            header_data = self._xor_encrypt(encrypted_header, key[:36])
            
            # Parse header
            header_key = header_data[:16]
            orig_size = struct.unpack('<Q', header_data[16:24])[0]
            comp_size = struct.unpack('<Q', header_data[24:32])[0]
            expected_checksum = header_data[32:36]
            
            # Verify header key
            expected_header_key = self._derive_salt(password, b"header_key")[:16]
            if header_key != expected_header_key:
                return None
            
            # Check if compressed size makes sense
            if comp_size > len(payload) - 36 or comp_size == 0:
                return None
            
            # Extract encrypted compressed data
            encrypted_data = payload[36:36 + comp_size]
            
            # Decrypt compressed data
            compressed_data = self._xor_encrypt(encrypted_data, key)
            
            # Decompress
            try:
                original_data = zlib.decompress(compressed_data)
            except zlib.error:
                return None
            
            # Verify size matches
            if len(original_data) != orig_size:
                return None
            
            # Verify checksum
            actual_checksum = hashlib.sha256(original_data).digest()[:4]
            if actual_checksum != expected_checksum:
                return None
            
            self.logger.info(f"Secure extraction successful: {len(original_data)} bytes")
            return original_data
            
        except Exception:
            return None
    
    def _validate_image(self, image_path: Path) -> bool:
        """Validate image file."""
        try:
            if not image_path.exists():
                self.logger.error(f"Image file not found: {image_path}")
                return False
            
            suffix = image_path.suffix.lower()
            if suffix not in self.SUPPORTED_FORMATS:
                self.logger.error(f"Unsupported format: {suffix}")
                return False
            
            with Image.open(image_path) as img:
                # Verify image can be processed
                img.verify()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Image validation failed: {e}")
            return False
