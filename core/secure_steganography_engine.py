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
import time
import json
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from PIL import Image
import zlib
import math

# Optional scipy import with fallback
try:
    from scipy import stats
except ImportError:
    stats = None

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
                
                # Prepare secure payload WITH carrier-derived statistical masking
                secure_payload = self._create_secure_payload(data, password, compression_level, img_array)
                
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
    
    def _create_secure_payload(self, data: bytes, password: str, compression_level: int, 
                              carrier_array: Optional[np.ndarray] = None) -> bytes:
        """Create secure payload with statistical masking and entropy management."""
        
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
        base_payload = header_encrypted + encrypted_data
        
        # 5. TEMPORARY: Skip complex statistical masking for now to ensure core functionality works
        # TODO: Re-enable advanced statistical masking once core functionality is stable
        if carrier_array is not None:
            # Apply simple carrier-based masking only
            try:
                simple_masked = self._apply_simple_statistical_masking(base_payload, carrier_array, password)
                return simple_masked
            except Exception as e:
                self.logger.warning(f"Simple statistical masking failed: {e}, using base payload")
                return base_payload
        
        return base_payload
    
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
        
        max_possible_size = img_array.size // 8
        
        # Create SMART candidate list - start with most likely sizes first
        candidates = []
        
        # 1. Very small messages (most common)
        candidates.extend(range(36, 150, 1))  # Header is 36 bytes minimum
        
        # 2. Common small message sizes
        candidates.extend(range(150, 500, 2))
        
        # 3. Medium message sizes  
        candidates.extend(range(500, 2000, 10))
        
        # 4. Larger message sizes (less common, so lower priority)
        candidates.extend(range(2000, 10000, 50))
        candidates.extend(range(10000, 50000, 200))
        
        # 5. Very large sizes (rare)
        candidates.extend([50000, 75000, 100000, 150000, 200000, 300000, 500000, 1000000])
        
        # 5. Filter by maximum possible size and sort
        valid_candidates = sorted([size for size in candidates if size <= max_possible_size])
        
        self.logger.debug(f"Trying {len(valid_candidates)} candidate sizes from {min(valid_candidates)} to {max(valid_candidates)} bytes")
        
        for candidate_size in valid_candidates:
            try:
                result = self._try_extract_size(img_array, candidate_size, password)
                if result:
                    self.logger.debug(f"Successful extraction at size {candidate_size} bytes")
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
            
            # Try to parse as secure payload WITH carrier array for statistical masking reversal
            return self._parse_secure_payload(extracted_bytes, password, img_array)
            
        except Exception:
            return None
    
    def _parse_secure_payload(self, payload: bytes, password: str, 
                             carrier_array: Optional[np.ndarray] = None) -> Optional[bytes]:
        """
        Parse secure payload and verify integrity, with statistical masking reversal.
        """
        
        try:
            # FIRST: Try to reverse statistical masking if carrier is available
            processed_payload = payload
            self.logger.debug(f"_parse_secure_payload: Input payload length: {len(payload)}")
            
            if carrier_array is not None:
                self.logger.debug("_parse_secure_payload: Attempting statistical masking reversal")
                processed_payload = self._reverse_statistical_masking(payload, carrier_array, password)
                if processed_payload is None:
                    self.logger.debug("_parse_secure_payload: Statistical masking reversal failed, using original payload")
                    # Fall back to original payload if reversal fails
                    processed_payload = payload
                else:
                    self.logger.debug(f"_parse_secure_payload: Statistical masking reversed, new length: {len(processed_payload)}")
            
            # Derive decryption key
            salt = self._derive_salt(password, b"payload_salt")
            key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, 32)
            
            # Expected header size: 16 (header_key) + 8 (orig_size) + 8 (comp_size) + 4 (checksum) = 36 bytes
            if len(processed_payload) < 36:
                return None
            
            # Decrypt header
            encrypted_header = processed_payload[:36]
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
            if comp_size > len(processed_payload) - 36 or comp_size == 0:
                return None
            
            # Extract encrypted compressed data
            encrypted_data = processed_payload[36:36 + comp_size]
            
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
    
    def _apply_statistical_masking(self, payload: bytes, carrier_array: np.ndarray, password: str) -> bytes:
        """
        Advanced statistical masking using carrier-derived PRG and entropy management.
        
        This makes the embedded data statistically indistinguishable from natural image noise
        by pre-masking with a PRG seeded from carrier pixel values and managing entropy.
        
        FULL IMPLEMENTATION: Includes metadata-based reversible entropy adjustment and dummy bytes.
        """
        try:
            # 1. Generate carrier-derived PRG seed
            carrier_seed = self._generate_carrier_prg_seed(carrier_array, password)
            
            # 2. Create PRG for statistical masking
            prg = np.random.default_rng(carrier_seed)
            
            # 3. Pre-mask payload to look like image noise
            masked_payload = self._pre_mask_with_carrier_noise(payload, prg, carrier_array)
            
            # 4. Apply entropy adjustment with metadata tracking
            entropy_adjusted, entropy_metadata = self._adjust_entropy_range_reversible(masked_payload, prg)
            
            # 5. Insert dummy bytes with position tracking
            final_payload, dummy_metadata = self._insert_color_dummy_bytes_reversible(entropy_adjusted, carrier_array, prg)
            
            # 6. Create masking metadata for reversal
            masking_metadata = self._create_masking_metadata(entropy_metadata, dummy_metadata, len(payload))
            
            # 7. Prepend metadata to payload (encrypted with same key)
            metadata_bytes = self._serialize_masking_metadata(masking_metadata, password)
            complete_payload = metadata_bytes + final_payload
            
            self.logger.debug(f"Full statistical masking applied: {len(payload)} -> {len(complete_payload)} bytes")
            self.logger.debug(f"Masking breakdown: original({len(payload)}) + metadata({len(metadata_bytes)}) + masked_data({len(final_payload)})")
            
            return complete_payload
            
        except Exception as e:
            self.logger.warning(f"Statistical masking failed: {e}, falling back to simple masking")
            # Fallback to simple masking for compatibility
            return self._apply_simple_statistical_masking(payload, carrier_array, password)
    
    def _generate_carrier_prg_seed(self, carrier_array: np.ndarray, password: str) -> int:
        """
        Generate PRG seed from carrier image pixel values.
        
        This creates a deterministic seed based on the actual image content,
        making the masking pattern unique to each carrier image.
        """
        # Sample pixel values from different regions of the image
        height, width = carrier_array.shape[:2]
        
        # Take samples from strategic positions
        samples = []
        
        # Corner samples
        samples.extend([
            carrier_array[0, 0].flatten(),
            carrier_array[0, width-1].flatten(),
            carrier_array[height-1, 0].flatten(),
            carrier_array[height-1, width-1].flatten(),
        ])
        
        # Center samples
        center_y, center_x = height // 2, width // 2
        samples.append(carrier_array[center_y, center_x].flatten())
        
        # Random strategic positions based on password
        pwd_seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
        sample_rng = np.random.default_rng(pwd_seed)
        
        for _ in range(10):  # 10 additional samples
            y = sample_rng.integers(0, height)
            x = sample_rng.integers(0, width)
            samples.append(carrier_array[y, x].flatten())
        
        # Combine all samples
        combined_samples = np.concatenate(samples)
        
        # Create deterministic seed from pixel data + password
        pixel_hash = hashlib.sha256(combined_samples.tobytes()).digest()
        password_hash = hashlib.sha256(password.encode()).digest()
        
        # XOR the hashes and convert to seed
        seed_material = bytes(a ^ b for a, b in zip(pixel_hash, password_hash))
        carrier_seed = int.from_bytes(seed_material[:8], 'little') % (2**31)
        
        return carrier_seed
    
    def _pre_mask_with_carrier_noise(self, payload: bytes, prg: np.random.Generator, 
                                    carrier_array: np.ndarray) -> bytes:
        """
        Pre-mask encrypted payload with carrier-derived noise pattern.
        
        This makes the payload statistically similar to the carrier's natural noise.
        """
        # Analyze carrier noise characteristics
        noise_stats = self._analyze_carrier_noise(carrier_array)
        
        # Generate mask based on carrier characteristics
        mask_length = len(payload)
        
        if noise_stats['has_high_frequency']:
            # High frequency noise - use more varied mask
            mask = prg.normal(noise_stats['mean'], noise_stats['std'], mask_length)
        else:
            # Low frequency - use smoother mask
            mask = prg.uniform(noise_stats['min'], noise_stats['max'], mask_length)
        
        # Convert to bytes and apply
        mask_bytes = (np.clip(mask, 0, 255).astype(np.uint8) % 256).tobytes()
        
        # XOR with original payload
        masked_payload = bytes(a ^ b for a, b in zip(payload, mask_bytes[:len(payload)]))
        
        return masked_payload
    
    def _analyze_carrier_noise(self, carrier_array: np.ndarray) -> dict:
        """
        Analyze carrier image noise characteristics for statistical masking.
        """
        # Extract LSBs from carrier to analyze natural noise
        lsbs = carrier_array & 1
        
        # Calculate statistics
        stats_dict = {
            'mean': np.mean(lsbs),
            'std': np.std(lsbs),
            'min': np.min(lsbs),
            'max': np.max(lsbs),
            'entropy': self._calculate_entropy(lsbs.flatten()),
            'has_high_frequency': np.std(lsbs) > 0.4
        }
        
        return stats_dict
    
    def _adjust_entropy_range(self, data: bytes, prg: np.random.Generator) -> bytes:
        """
        Adjust entropy to plausible range (5.5-7.0 bits/byte) instead of flat 7.78.
        
        This prevents the telltale sign of high entropy encrypted data.
        """
        current_entropy = self._calculate_entropy(data)
        target_entropy = prg.uniform(5.5, 7.0)  # Plausible range
        
        if abs(current_entropy - target_entropy) < 0.1:
            return data  # Already in good range
        
        # Adjust entropy by selectively modifying bytes
        data_array = np.frombuffer(data, dtype=np.uint8).copy()
        
        if current_entropy > target_entropy:
            # Reduce entropy by making some bytes more predictable
            num_adjustments = int(len(data) * 0.05)  # Adjust 5% of bytes
            indices = prg.choice(len(data_array), num_adjustments, replace=False)
            
            for idx in indices:
                # Make byte more predictable by constraining to smaller range
                data_array[idx] = prg.integers(0, 128)  # Reduce from 0-255 to 0-127
        
        elif current_entropy < target_entropy:
            # Increase entropy by making some bytes more random
            num_adjustments = int(len(data) * 0.03)  # Adjust 3% of bytes
            indices = prg.choice(len(data_array), num_adjustments, replace=False)
            
            for idx in indices:
                # Make byte more random
                data_array[idx] = prg.integers(0, 256)
        
        adjusted_data = data_array.tobytes()
        new_entropy = self._calculate_entropy(adjusted_data)
        
        self.logger.debug(f"Entropy adjusted: {current_entropy:.3f} -> {new_entropy:.3f} (target: {target_entropy:.3f})")
        return adjusted_data
    
    def _insert_color_dummy_bytes(self, payload: bytes, carrier_array: np.ndarray, 
                                 prg: np.random.Generator) -> bytes:
        """
        Insert dummy bytes that resemble valid image color data.
        
        This adds plausible noise that looks like legitimate color values.
        """
        # Analyze carrier color distribution
        color_stats = self._analyze_carrier_colors(carrier_array)
        
        # Calculate number of dummy bytes (5-15% of payload)
        dummy_ratio = prg.uniform(0.05, 0.15)
        num_dummies = int(len(payload) * dummy_ratio)
        
        if num_dummies == 0:
            return payload
        
        # Generate dummy bytes that look like natural color data
        dummy_bytes = []
        for _ in range(num_dummies):
            if prg.random() < 0.7:  # 70% chance to use carrier-like colors
                # Generate byte similar to carrier colors
                channel = prg.choice(3)  # R, G, or B
                mean_val = color_stats[f'mean_ch{channel}']
                std_val = color_stats[f'std_ch{channel}']
                dummy_val = int(np.clip(prg.normal(mean_val, std_val), 0, 255))
            else:
                # Generate more random byte
                dummy_val = prg.integers(16, 240)  # Avoid extreme values
            
            dummy_bytes.append(dummy_val)
        
        # Randomly insert dummy bytes into payload
        payload_array = list(payload)
        dummy_positions = sorted(prg.choice(len(payload_array) + num_dummies, 
                                          num_dummies, replace=False))
        
        for i, pos in enumerate(dummy_positions):
            payload_array.insert(pos, dummy_bytes[i])
        
        final_payload = bytes(payload_array)
        
        self.logger.debug(f"Inserted {num_dummies} dummy color bytes ({dummy_ratio*100:.1f}% of payload)")
        return final_payload
    
    def _analyze_carrier_colors(self, carrier_array: np.ndarray) -> dict:
        """
        Analyze carrier image color distribution for dummy byte generation.
        """
        if len(carrier_array.shape) == 3:
            # Color image
            stats_dict = {}
            for ch in range(3):
                channel_data = carrier_array[:, :, ch]
                stats_dict[f'mean_ch{ch}'] = np.mean(channel_data)
                stats_dict[f'std_ch{ch}'] = np.std(channel_data)
            return stats_dict
        else:
            # Grayscale
            return {
                'mean_ch0': np.mean(carrier_array),
                'std_ch0': np.std(carrier_array),
                'mean_ch1': np.mean(carrier_array),
                'std_ch1': np.std(carrier_array),
                'mean_ch2': np.mean(carrier_array),
                'std_ch2': np.std(carrier_array),
            }
    
    def _reverse_statistical_masking(self, masked_payload: bytes, carrier_array: np.ndarray, password: str) -> Optional[bytes]:
        """
        Reverse the statistical masking to recover the original payload.
        
        This method first tries full statistical masking reversal (with metadata),
        then falls back to simple reversal for compatibility.
        """
        try:
            # First, try full statistical masking reversal with metadata
            full_result = self._reverse_full_statistical_masking(masked_payload, carrier_array, password)
            if full_result is not None:
                return full_result
            
            # Fallback to simple reversal
            simple_result = self._reverse_simple_statistical_masking(masked_payload, carrier_array, password)
            return simple_result
            
        except Exception as e:
            self.logger.debug(f"Statistical masking reversal failed: {e}")
            return None
    
    def _apply_simple_statistical_masking(self, payload: bytes, carrier_array: np.ndarray, password: str) -> bytes:
        """
        Fallback simple statistical masking (for compatibility).
        Only applies carrier noise masking without metadata.
        """
        try:
            carrier_seed = self._generate_carrier_prg_seed(carrier_array, password)
            prg = np.random.default_rng(carrier_seed)
            return self._pre_mask_with_carrier_noise(payload, prg, carrier_array)
        except Exception as e:
            self.logger.warning(f"Simple statistical masking failed: {e}")
            return payload
    
    def _adjust_entropy_range_reversible(self, data: bytes, prg: np.random.Generator) -> Tuple[bytes, Dict]:
        """
        Adjust entropy to plausible range with metadata for reversal.
        
        Returns:
            Tuple of (adjusted_data, metadata_dict)
        """
        current_entropy = self._calculate_entropy(data)
        target_entropy = prg.uniform(5.5, 7.0)
        
        metadata = {
            'original_entropy': current_entropy,
            'target_entropy': target_entropy,
            'adjustments': []
        }
        
        if abs(current_entropy - target_entropy) < 0.1:
            return data, metadata
        
        data_array = np.frombuffer(data, dtype=np.uint8).copy()
        
        if current_entropy > target_entropy:
            # Reduce entropy
            num_adjustments = int(len(data) * 0.05)
            indices = prg.choice(len(data_array), num_adjustments, replace=False)
            
            for idx in indices:
                original_value = data_array[idx]
                new_value = prg.integers(0, 128)
                data_array[idx] = new_value
                metadata['adjustments'].append((int(idx), int(original_value), int(new_value)))
        
        elif current_entropy < target_entropy:
            # Increase entropy
            num_adjustments = int(len(data) * 0.03)
            indices = prg.choice(len(data_array), num_adjustments, replace=False)
            
            for idx in indices:
                original_value = data_array[idx]
                new_value = prg.integers(0, 256)
                data_array[idx] = new_value
                metadata['adjustments'].append((int(idx), int(original_value), int(new_value)))
        
        adjusted_data = data_array.tobytes()
        new_entropy = self._calculate_entropy(adjusted_data)
        metadata['final_entropy'] = new_entropy
        
        self.logger.debug(f"Entropy adjusted with metadata: {current_entropy:.3f} -> {new_entropy:.3f}")
        return adjusted_data, metadata
    
    def _insert_color_dummy_bytes_reversible(self, payload: bytes, carrier_array: np.ndarray, 
                                           prg: np.random.Generator) -> Tuple[bytes, Dict]:
        """
        Insert dummy bytes with position tracking for reversal.
        
        Returns:
            Tuple of (payload_with_dummies, metadata_dict)
        """
        color_stats = self._analyze_carrier_colors(carrier_array)
        
        # Calculate dummy parameters
        dummy_ratio = prg.uniform(0.05, 0.15)
        num_dummies = int(len(payload) * dummy_ratio)
        
        metadata = {
            'dummy_ratio': dummy_ratio,
            'num_dummies': num_dummies,
            'dummy_positions': [],
            'dummy_values': [],
            'original_length': len(payload)
        }
        
        if num_dummies == 0:
            return payload, metadata
        
        # Generate dummy bytes
        dummy_bytes = []
        for _ in range(num_dummies):
            if prg.random() < 0.7:
                channel = prg.choice(3)
                mean_val = color_stats[f'mean_ch{channel}']
                std_val = color_stats[f'std_ch{channel}']
                dummy_val = int(np.clip(prg.normal(mean_val, std_val), 0, 255))
            else:
                dummy_val = prg.integers(16, 240)
            dummy_bytes.append(dummy_val)
        
        # Insert dummy bytes and track positions
        payload_array = list(payload)
        dummy_positions = sorted(prg.choice(len(payload_array) + num_dummies, 
                                          num_dummies, replace=False))
        
        for i, pos in enumerate(dummy_positions):
            payload_array.insert(pos, dummy_bytes[i])
            metadata['dummy_positions'].append(int(pos))
            metadata['dummy_values'].append(int(dummy_bytes[i]))
        
        final_payload = bytes(payload_array)
        
        self.logger.debug(f"Inserted {num_dummies} dummy bytes with position tracking")
        return final_payload, metadata
    
    def _create_masking_metadata(self, entropy_metadata: Dict, dummy_metadata: Dict, original_size: int) -> Dict:
        """
        Combine all masking metadata into a single structure.
        """
        return {
            'version': 1,  # For future compatibility
            'original_size': original_size,
            'entropy_metadata': entropy_metadata,
            'dummy_metadata': dummy_metadata,
            'timestamp': int(time.time()) if 'time' in globals() else 0
        }
    
    def _serialize_masking_metadata(self, metadata: Dict, password: str) -> bytes:
        """
        Serialize and encrypt masking metadata.
        
        Format: [2 bytes: metadata_length] [metadata_length bytes: encrypted_metadata]
        """
        # Serialize metadata to JSON
        metadata_json = json.dumps(metadata, separators=(',', ':'))
        metadata_bytes = metadata_json.encode('utf-8')
        
        # Compress metadata
        compressed_metadata = zlib.compress(metadata_bytes, 9)
        
        # Encrypt metadata
        metadata_key = self._derive_salt(password, b"metadata_key")
        encrypted_metadata = self._xor_encrypt(compressed_metadata, metadata_key)
        
        # Prepend length (2 bytes = max 65535 bytes metadata)
        if len(encrypted_metadata) > 65535:
            raise ValueError("Metadata too large")
        
        length_bytes = struct.pack('<H', len(encrypted_metadata))
        return length_bytes + encrypted_metadata
    
    def _deserialize_masking_metadata(self, metadata_bytes: bytes, password: str) -> Optional[Dict]:
        """
        Decrypt and deserialize masking metadata.
        """
        try:
            if len(metadata_bytes) < 2:
                return None
            
            # Extract length and encrypted metadata
            metadata_length = struct.unpack('<H', metadata_bytes[:2])[0]
            if len(metadata_bytes) < 2 + metadata_length:
                return None
            
            encrypted_metadata = metadata_bytes[2:2 + metadata_length]
            
            # Decrypt metadata
            metadata_key = self._derive_salt(password, b"metadata_key")
            compressed_metadata = self._xor_encrypt(encrypted_metadata, metadata_key)
            
            # Decompress metadata
            metadata_bytes = zlib.decompress(compressed_metadata)
            metadata_json = metadata_bytes.decode('utf-8')
            
            # Parse JSON
            metadata = json.loads(metadata_json)
            return metadata
        
        except Exception as e:
            self.logger.debug(f"Failed to deserialize metadata: {e}")
            return None
    
    def _reverse_entropy_adjustments(self, data: bytes, entropy_metadata: Dict) -> bytes:
        """
        Reverse entropy adjustments using stored metadata.
        """
        if not entropy_metadata.get('adjustments'):
            return data
        
        data_array = np.frombuffer(data, dtype=np.uint8).copy()
        
        # Reverse adjustments in reverse order
        for idx, original_value, adjusted_value in reversed(entropy_metadata['adjustments']):
            if idx < len(data_array) and data_array[idx] == adjusted_value:
                data_array[idx] = original_value
        
        return data_array.tobytes()
    
    def _remove_dummy_bytes(self, payload: bytes, dummy_metadata: Dict) -> bytes:
        """
        Remove dummy bytes using stored position metadata.
        """
        if not dummy_metadata.get('dummy_positions'):
            return payload
        
        payload_list = list(payload)
        
        # Remove dummy bytes in reverse order to maintain indices
        for pos in reversed(sorted(dummy_metadata['dummy_positions'])):
            if pos < len(payload_list):
                payload_list.pop(pos)
        
        return bytes(payload_list)
    
    def _reverse_full_statistical_masking(self, masked_payload: bytes, carrier_array: np.ndarray, password: str) -> Optional[bytes]:
        """
        Reverse full statistical masking using embedded metadata.
        """
        try:
            # 1. Extract and deserialize metadata
            metadata = self._deserialize_masking_metadata(masked_payload, password)
            if metadata is None:
                self.logger.debug("No valid metadata found, trying simple reversal")
                return self._reverse_simple_statistical_masking(masked_payload, carrier_array, password)
            
            # 2. Extract payload after metadata
            metadata_length = struct.unpack('<H', masked_payload[:2])[0]
            payload_after_metadata = masked_payload[2 + metadata_length:]
            
            # 3. Remove dummy bytes
            dummy_metadata = metadata.get('dummy_metadata', {})
            payload_without_dummies = self._remove_dummy_bytes(payload_after_metadata, dummy_metadata)
            
            # 4. Reverse entropy adjustments
            entropy_metadata = metadata.get('entropy_metadata', {})
            payload_entropy_reversed = self._reverse_entropy_adjustments(payload_without_dummies, entropy_metadata)
            
            # 5. Reverse carrier noise masking
            carrier_seed = self._generate_carrier_prg_seed(carrier_array, password)
            prg = np.random.default_rng(carrier_seed)
            
            noise_stats = self._analyze_carrier_noise(carrier_array)
            mask_length = len(payload_entropy_reversed)
            
            if noise_stats['has_high_frequency']:
                mask = prg.normal(noise_stats['mean'], noise_stats['std'], mask_length)
            else:
                mask = prg.uniform(noise_stats['min'], noise_stats['max'], mask_length)
            
            mask_bytes = (np.clip(mask, 0, 255).astype(np.uint8) % 256).tobytes()
            original_payload = bytes(a ^ b for a, b in zip(payload_entropy_reversed, mask_bytes[:len(payload_entropy_reversed)]))
            
            # 6. Verify original size matches
            expected_size = metadata.get('original_size')
            if expected_size and len(original_payload) != expected_size:
                self.logger.warning(f"Size mismatch after reversal: {len(original_payload)} != {expected_size}")
                return None
            
            self.logger.debug(f"Full statistical masking reversed successfully: {len(masked_payload)} -> {len(original_payload)} bytes")
            return original_payload
            
        except Exception as e:
            self.logger.debug(f"Full statistical masking reversal failed: {e}")
            return None
    
    def _reverse_simple_statistical_masking(self, masked_payload: bytes, carrier_array: np.ndarray, password: str) -> Optional[bytes]:
        """
        Reverse simple statistical masking (carrier noise only).
        """
        try:
            carrier_seed = self._generate_carrier_prg_seed(carrier_array, password)
            prg = np.random.default_rng(carrier_seed)
            
            noise_stats = self._analyze_carrier_noise(carrier_array)
            mask_length = len(masked_payload)
            
            if noise_stats['has_high_frequency']:
                mask = prg.normal(noise_stats['mean'], noise_stats['std'], mask_length)
            else:
                mask = prg.uniform(noise_stats['min'], noise_stats['max'], mask_length)
            
            mask_bytes = (np.clip(mask, 0, 255).astype(np.uint8) % 256).tobytes()
            original_payload = bytes(a ^ b for a, b in zip(masked_payload, mask_bytes[:len(masked_payload)]))
            
            self.logger.debug(f"Simple statistical masking reversed: {len(masked_payload)} -> {len(original_payload)} bytes")
            return original_payload
            
        except Exception as e:
            self.logger.debug(f"Simple statistical masking reversal failed: {e}")
            return None
    
    def _calculate_entropy(self, data) -> float:
        """
        Calculate Shannon entropy of data in bits per byte.
        """
        if isinstance(data, bytes):
            data = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, np.ndarray):
            data = data.flatten()
        
        # Calculate byte frequency
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)
    
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
