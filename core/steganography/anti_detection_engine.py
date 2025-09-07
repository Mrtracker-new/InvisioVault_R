"""
Anti-Detection Steganography Engine
Advanced steganography with techniques to evade common steganalysis tools like 
StegExpose, zsteg, StegSeek, and other detection methods.
"""

import os
import struct
import hashlib
import secrets
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import cv2

from utils.logger import Logger
from utils.error_handler import ErrorHandler


class AntiDetectionEngine:
    """Advanced steganography engine with anti-detection techniques."""
    
    # Use different magic header to avoid signature detection
    MAGIC_HEADER = b'JPEG'  # Disguise as JPEG comment
    VERSION = b'\x00\x10'   # Mimic JPEG version
    
    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        
        # Anti-detection parameters
        self.noise_level = 0.8  # Subtle noise to mask patterns
        self.histogram_preservation = True  # Preserve original histogram
        self.adaptive_capacity = True  # Adapt based on image content
        self.multi_channel_distribution = True  # Distribute across channels
        
    def enhanced_hide_data(self, carrier_path: Path, data: bytes, output_path: Path, 
                          password: Optional[str] = None, use_anti_detection: bool = True) -> bool:
        """
        Hide data using advanced anti-detection techniques.
        
        Args:
            carrier_path: Path to carrier image
            data: Data to hide
            output_path: Output path
            password: Password for additional randomization
            use_anti_detection: Enable anti-detection features
            
        Returns:
            Success status
        """
        try:
            if not self._validate_image(carrier_path):
                return False
                
            with Image.open(carrier_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                img_array = np.array(img, dtype=np.uint8)
                original_shape = img_array.shape
                
                if use_anti_detection:
                    # Apply pre-processing to reduce detectability
                    img_array = self._apply_anti_detection_preprocessing(img_array)
                
                # Prepare enhanced payload
                payload = self._create_enhanced_payload(data, password)
                
                # Calculate adaptive capacity
                capacity_map = self._calculate_adaptive_capacity(img_array)
                available_positions = self._get_secure_positions(
                    img_array, len(payload) * 8, password, capacity_map
                )
                
                if len(available_positions) < len(payload) * 8:
                    self.logger.error(f"Insufficient secure capacity: need {len(payload) * 8}, have {len(available_positions)}")
                    return False
                
                # Enhanced hiding with anti-detection
                modified_array = self._enhanced_lsb_hiding(
                    img_array, payload, available_positions, password
                )
                
                if use_anti_detection:
                    # Apply post-processing to further reduce detectability
                    modified_array = self._apply_anti_detection_postprocessing(
                        original_shape, modified_array, img_array
                    )
                
                # Save with optimized settings
                result_img = Image.fromarray(modified_array, 'RGB')
                self._save_with_anti_detection(result_img, output_path)
                
                self.logger.info(f"Enhanced hiding successful: {len(data)} bytes with anti-detection")
                return True
                
        except Exception as e:
            self.logger.error(f"Enhanced hiding failed: {e}")
            return False
    
    def enhanced_extract_data(self, stego_path: Path, password: Optional[str] = None) -> Optional[bytes]:
        """
        Extract data using enhanced anti-detection aware methods.
        
        Args:
            stego_path: Path to steganographic image
            password: Password for extraction
            
        Returns:
            Extracted data or None
        """
        try:
            if not self._validate_image(stego_path):
                return None
                
            with Image.open(stego_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                img_array = np.array(img, dtype=np.uint8)
                
                # Calculate capacity map (same as hiding)
                capacity_map = self._calculate_adaptive_capacity(img_array)
                
                # Try different payload sizes with anti-detection awareness
                return self._enhanced_extraction_search(img_array, password, capacity_map)
                
        except Exception as e:
            self.logger.error(f"Enhanced extraction failed: {e}")
            return None
    
    def _apply_anti_detection_preprocessing(self, img_array: np.ndarray) -> np.ndarray:
        """Apply preprocessing to reduce detectability."""
        
        # 1. Subtle noise injection to break up patterns
        noise = np.random.normal(0, 0.5, img_array.shape).astype(np.int8)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 2. Micro-blur to smooth transitions
        img_pil = Image.fromarray(img_array)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.1))
        img_array = np.array(img_pil)
        
        return img_array
    
    def _apply_anti_detection_postprocessing(self, original_shape: tuple, 
                                           modified_array: np.ndarray, 
                                           original_array: np.ndarray) -> np.ndarray:
        """Apply postprocessing to further reduce detectability."""
        
        # 1. Histogram matching to preserve original distribution
        if self.histogram_preservation:
            modified_array = self._match_histogram(modified_array, original_array)
        
        # 2. Selective smoothing of high-frequency artifacts
        modified_array = self._selective_smoothing(modified_array, original_array)
        
        # 3. Edge-aware noise reduction
        modified_array = self._edge_aware_filtering(modified_array)
        
        return modified_array
    
    def _calculate_adaptive_capacity(self, img_array: np.ndarray) -> np.ndarray:
        """Calculate adaptive capacity map based on image content."""
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate local complexity measures
        # 1. Gradient magnitude (edge strength)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 2. Local variance (texture complexity)
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        
        # 3. Laplacian (local sharpness)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        
        # Combine measures to create capacity map
        # Higher values = better hiding locations
        capacity_map = (
            0.4 * gradient_magnitude / gradient_magnitude.max() +
            0.3 * variance / variance.max() +
            0.3 * laplacian / laplacian.max()
        )
        
        # Normalize to 0-1 range
        capacity_map = (capacity_map - capacity_map.min()) / (capacity_map.max() - capacity_map.min())
        
        return capacity_map
    
    def _get_secure_positions(self, img_array: np.ndarray, bits_needed: int, 
                             password: Optional[str], capacity_map: np.ndarray) -> List[Tuple[int, int, int]]:
        """Get secure positions for hiding based on capacity map and password."""
        
        height, width, channels = img_array.shape
        
        # Create position candidates with their security scores
        candidates = []
        
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    # Skip positions in smooth areas (too detectable)
                    if capacity_map[y, x] < 0.3:
                        continue
                        
                    # Calculate position security score
                    security_score = capacity_map[y, x]
                    
                    # Boost score for positions in complex regions
                    if capacity_map[y, x] > 0.7:
                        security_score *= 1.5
                    
                    candidates.append(((y, x, c), security_score))
        
        # Sort by security score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select best positions using password-derived randomization
        if password:
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
        
        # Take more positions than needed, then randomize selection
        safe_positions = [pos for pos, score in candidates[:bits_needed * 2]]
        
        if len(safe_positions) < bits_needed:
            self.logger.warning("Insufficient secure positions, using all available")
            return safe_positions
        
        # Randomize selection from safe positions
        selected_indices = np.random.choice(len(safe_positions), bits_needed, replace=False)
        return [safe_positions[i] for i in selected_indices]
    
    def _create_enhanced_payload(self, data: bytes, password: Optional[str] = None) -> bytes:
        """Create enhanced payload with anti-detection features."""
        
        # Add decoy header that looks like JPEG comment
        decoy_comment = b"JPEG File Interchange Format"
        
        # Create actual header with obfuscation
        data_size = len(data)
        size_bytes = struct.pack('<Q', data_size)
        
        # Checksum with additional salt
        salt = secrets.token_bytes(8)
        data_hash = hashlib.sha256(salt + data).digest()[:4]
        
        # Assemble payload with decoy elements
        payload = (
            self.MAGIC_HEADER +     # Disguised magic
            self.VERSION +          # Disguised version
            decoy_comment[:16] +    # Decoy JPEG comment
            size_bytes +           # Data size
            salt +                 # Salt for checksum
            data_hash +            # Checksum
            data                   # Actual data
        )
        
        # Add padding to avoid size-based detection
        padding_size = secrets.randbelow(64) + 32  # 32-95 bytes of padding
        padding = secrets.token_bytes(padding_size)
        payload += padding
        
        return payload
    
    def _enhanced_lsb_hiding(self, img_array: np.ndarray, payload: bytes, 
                           positions: List[Tuple[int, int, int]], password: Optional[str]) -> np.ndarray:
        """Enhanced LSB hiding with anti-detection techniques."""
        
        modified_array = img_array.copy()
        payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        
        # Create randomized mapping if password provided
        if password and len(positions) >= len(payload_bits):
            seed = int(hashlib.sha256(f"{password}_mapping".encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            position_indices = np.random.choice(len(positions), len(payload_bits), replace=False)
            selected_positions = [positions[i] for i in position_indices]
        else:
            selected_positions = positions[:len(payload_bits)]
        
        # Hide bits with enhanced techniques
        for i, bit in enumerate(payload_bits):
            if i >= len(selected_positions):
                break
                
            y, x, c = selected_positions[i]
            original_value = modified_array[y, x, c]
            
            # Enhanced LSB modification with pattern breaking
            new_value = self._anti_detection_lsb_modify(original_value, bit, i, password)
            modified_array[y, x, c] = new_value
        
        return modified_array
    
    def _anti_detection_lsb_modify(self, original_value: int, bit: int, 
                                  position: int, password: Optional[str]) -> int:
        """Modify LSB with anti-detection techniques."""
        
        # Standard LSB modification
        new_value = (original_value & 0xFE) | bit
        
        # Add subtle randomization based on position and password
        if password:
            seed = hash(f"{password}_{position}") & 0x7FFFFFFF
            np.random.seed(seed % (2**31))
            
            # Occasionally modify the second LSB to break patterns
            if np.random.random() < 0.1 and abs(original_value - new_value) <= 1:
                # Flip second LSB randomly to add noise
                if np.random.random() < 0.5:
                    new_value ^= 0x02  # Flip second bit
                    
        # Ensure we stay in valid range
        return np.clip(new_value, 0, 255)
    
    def _match_histogram(self, modified: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Match histogram of modified image to original."""
        
        result = modified.copy()
        
        for channel in range(3):  # RGB channels
            # Calculate histograms
            orig_hist, bins = np.histogram(original[:,:,channel].flatten(), 256, (0, 256))
            mod_hist, _ = np.histogram(modified[:,:,channel].flatten(), 256, (0, 256))
            
            # Calculate cumulative distribution functions
            orig_cdf = orig_hist.cumsum()
            mod_cdf = mod_hist.cumsum()
            
            # Normalize
            orig_cdf = orig_cdf / orig_cdf.max()
            mod_cdf = mod_cdf / mod_cdf.max()
            
            # Create mapping
            mapping = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                # Find closest match in original CDF
                closest_idx = np.argmin(np.abs(mod_cdf[i] - orig_cdf))
                mapping[i] = closest_idx
            
            # Apply mapping
            result[:,:,channel] = mapping[modified[:,:,channel]]
        
        return result
    
    def _selective_smoothing(self, modified: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Apply selective smoothing to reduce artifacts."""
        
        # Calculate difference map
        diff = np.abs(modified.astype(np.float32) - original.astype(np.float32))
        diff_gray = np.mean(diff, axis=2)
        
        # Create mask for areas with significant changes
        threshold = np.percentile(diff_gray, 95)  # Top 5% of changes
        mask = diff_gray > threshold
        
        # Apply gentle blur only to changed areas
        if np.any(mask):
            blurred = cv2.GaussianBlur(modified, (3, 3), 0.5)
            
            # Blend original and blurred based on mask
            result = modified.copy()
            for c in range(3):
                result[:,:,c] = np.where(mask, 
                                       0.7 * modified[:,:,c] + 0.3 * blurred[:,:,c],
                                       modified[:,:,c])
        else:
            result = modified
        
        return result.astype(np.uint8)
    
    def _edge_aware_filtering(self, img_array: np.ndarray) -> np.ndarray:
        """Apply edge-aware filtering to reduce noise while preserving edges."""
        
        # Convert to float for processing
        img_float = img_array.astype(np.float32) / 255.0
        
        # Apply bilateral filter (edge-preserving)
        filtered = cv2.bilateralFilter(img_float, 5, 0.1, 0.1)
        
        # Convert back to uint8
        return (filtered * 255).astype(np.uint8)
    
    def _save_with_anti_detection(self, img: Image.Image, output_path: Path):
        """Save image with anti-detection optimization."""
        
        # Determine format based on extension
        ext = output_path.suffix.lower()
        
        if ext == '.png':
            # PNG: Use optimal compression without additional metadata
            img.save(output_path, 'PNG', optimize=True, compress_level=6)
        elif ext in ['.jpg', '.jpeg']:
            # JPEG: Use quality that preserves data but looks natural
            img.save(output_path, 'JPEG', quality=95, optimize=True)
        elif ext in ['.bmp']:
            # BMP: Direct save
            img.save(output_path, 'BMP')
        elif ext in ['.tiff', '.tif']:
            # TIFF: Use LZW compression
            img.save(output_path, 'TIFF', compression='lzw')
        else:
            # Default to PNG
            img.save(output_path, 'PNG', optimize=True)
    
    def _enhanced_extraction_search(self, img_array: np.ndarray, password: Optional[str], 
                                   capacity_map: np.ndarray) -> Optional[bytes]:
        """Enhanced extraction with anti-detection awareness."""
        
        # Try multiple size candidates
        candidates = [
            # Common encrypted file sizes
            160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
            1152, 1280, 1408, 1536, 1792, 2048, 2304, 2560, 3072, 4096,
            5120, 6144, 8192, 10240, 16384, 32768, 65536, 131072, 262144,
            524288, 1048576, 2097152, 5242880, 10485760
        ]
        
        for candidate_size in candidates:
            try:
                # Calculate payload size including headers and padding
                header_size = len(self.MAGIC_HEADER) + len(self.VERSION) + 16 + 8 + 8 + 4
                max_padding = 95
                total_size = header_size + candidate_size + max_padding
                
                # Get positions for this size
                positions = self._get_secure_positions(
                    img_array, total_size * 8, password, capacity_map
                )
                
                if len(positions) < total_size * 8:
                    continue
                
                # Extract bits
                extracted_bits = []
                for i in range(total_size * 8):
                    if i < len(positions):
                        y, x, c = positions[i]
                        bit = img_array[y, x, c] & 1
                        extracted_bits.append(bit)
                
                extracted_bytes = np.packbits(extracted_bits).tobytes()[:total_size]
                
                # Parse and validate
                result = self._parse_enhanced_payload(extracted_bytes, candidate_size)
                if result:
                    self.logger.info(f"Enhanced extraction successful: {len(result)} bytes")
                    return result
                    
            except Exception:
                continue
        
        self.logger.error("Enhanced extraction failed - no valid data found")
        return None
    
    def _parse_enhanced_payload(self, payload: bytes, expected_data_size: int) -> Optional[bytes]:
        """Parse enhanced payload with anti-detection features."""
        
        try:
            offset = 0
            
            # Check magic header
            magic = payload[offset:offset + len(self.MAGIC_HEADER)]
            if magic != self.MAGIC_HEADER:
                return None
            offset += len(self.MAGIC_HEADER)
            
            # Check version
            version = payload[offset:offset + len(self.VERSION)]
            offset += len(self.VERSION)
            
            # Skip decoy comment
            offset += 16
            
            # Extract data size
            data_size = struct.unpack('<Q', payload[offset:offset + 8])[0]
            offset += 8
            
            # Validate size matches expectation
            if data_size != expected_data_size:
                return None
            
            # Extract salt
            salt = payload[offset:offset + 8]
            offset += 8
            
            # Extract checksum
            expected_checksum = payload[offset:offset + 4]
            offset += 4
            
            # Extract data
            data = payload[offset:offset + data_size]
            
            # Verify checksum
            actual_checksum = hashlib.sha256(salt + data).digest()[:4]
            if actual_checksum != expected_checksum:
                return None
            
            return data
            
        except Exception:
            return None
    
    def _validate_image(self, image_path: Path) -> bool:
        """Validate image format and accessibility."""
        
        try:
            if not image_path.exists():
                self.logger.error(f"Image not found: {image_path}")
                return False
            
            with Image.open(image_path) as img:
                # Check if format is suitable
                if img.format not in ['PNG', 'BMP', 'TIFF']:
                    self.logger.warning(f"Format {img.format} may not be optimal for steganography")
                
                # Check image size
                width, height = img.size
                if width * height < 10000:  # Less than 100x100
                    self.logger.warning("Image may be too small for secure steganography")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Image validation failed: {e}")
            return False
    
    
    def analyze_detectability_risk(self, stego_path: Path) -> Dict[str, Any]:
        """Analyze the detectability risk of a steganographic image."""
        
        try:
            with Image.open(stego_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                # Calculate various detectability metrics
                metrics = {}
                
                # 1. LSB histogram analysis
                metrics['lsb_histogram_evenness'] = self._calculate_lsb_evenness(img_array)
                
                # 2. Chi-square test simulation
                metrics['chi_square_risk'] = self._simulate_chi_square_test(img_array)
                
                # 3. Histogram analysis
                metrics['histogram_anomalies'] = self._detect_histogram_anomalies(img_array)
                
                # 4. Noise pattern analysis  
                metrics['noise_pattern_risk'] = self._analyze_noise_patterns(img_array)
                
                # 5. Overall risk assessment
                risk_score = (
                    metrics['lsb_histogram_evenness'] * 0.3 +
                    metrics['chi_square_risk'] * 0.3 +
                    metrics['histogram_anomalies'] * 0.2 +
                    metrics['noise_pattern_risk'] * 0.2
                )
                
                metrics['overall_risk_score'] = risk_score
                metrics['risk_level'] = self._classify_risk_level(risk_score)
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Risk analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_lsb_evenness(self, img_array: np.ndarray) -> float:
        """Calculate LSB distribution evenness (lower = better)."""
        
        lsb_counts = {'0': 0, '1': 0}
        
        for channel in range(img_array.shape[2]):
            channel_data = img_array[:, :, channel].flatten()
            lsbs = channel_data & 1
            lsb_counts['0'] += np.sum(lsbs == 0)
            lsb_counts['1'] += np.sum(lsbs == 1)
        
        total = lsb_counts['0'] + lsb_counts['1']
        if total == 0:
            return 0.0
            
        # Calculate deviation from perfect 50/50 distribution
        expected = total / 2
        deviation = abs(lsb_counts['0'] - expected) / expected
        
        return min(deviation, 1.0)  # Normalize to 0-1
    
    def _simulate_chi_square_test(self, img_array: np.ndarray) -> float:
        """Simulate chi-square test for randomness (higher = more suspicious)."""
        
        # Sample a portion of the image to avoid excessive computation
        sample_size = min(10000, img_array.size // 3)
        flat_img = img_array.flatten()
        
        # Random sampling
        indices = np.random.choice(len(flat_img), sample_size, replace=False)
        sample = flat_img[indices]
        
        # Calculate chi-square statistic for LSBs
        lsbs = sample & 1
        observed_0 = np.sum(lsbs == 0)
        observed_1 = np.sum(lsbs == 1)
        
        expected = sample_size / 2
        chi_square = ((observed_0 - expected)**2 + (observed_1 - expected)**2) / expected
        
        # Normalize to 0-1 (higher values indicate more suspicious patterns)
        return min(chi_square / 100.0, 1.0)
    
    def _detect_histogram_anomalies(self, img_array: np.ndarray) -> float:
        """Detect histogram anomalies that might indicate steganography."""
        
        anomaly_score = 0.0
        
        for channel in range(img_array.shape[2]):
            hist, _ = np.histogram(img_array[:, :, channel], bins=256, range=(0, 256))
            
            # Look for unusual patterns in histogram
            # 1. Check for artificial peaks/valleys
            smoothed = np.convolve(hist, np.ones(5)/5, mode='same')
            differences = np.abs(hist - smoothed)
            anomaly_score += float(np.mean(differences)) / 10000.0  # Normalize
        
        return float(min(anomaly_score, 1.0))
    
    def _analyze_noise_patterns(self, img_array: np.ndarray) -> float:
        """Analyze noise patterns for steganographic artifacts."""
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate high-frequency noise
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_variance = float(np.var(np.asarray(laplacian, dtype=np.float64)))
        
        # Normalize to 0-1 scale (lower values indicate more artificial patterns)
        normalized_noise = min(noise_variance / 1000.0, 1.0)
        
        # Return inverted score (higher = more suspicious)
        return 1.0 - normalized_noise
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify overall risk level."""
        
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.6:
            return "MEDIUM"
        else:
            return "HIGH"
