"""
Enhanced Audio Steganography Engine
Implements advanced audio steganography with improved reliability, error recovery,
and robust data extraction capabilities.
"""

import os
import sys
import tempfile
import struct
import hashlib
import secrets
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    from pydub import AudioSegment
    from scipy import signal
    from scipy.fftpack import dct, idct
    import soundfile as sf
except ImportError as e:
    print(f"Warning: Audio dependencies not fully installed: {e}")
    print("Please install: pip install librosa pydub scipy soundfile")

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from core.encryption_engine import EncryptionEngine, SecurityLevel
from core.multimedia_analyzer import MultimediaAnalyzer


class ExtractionStrategy(Enum):
    """Extraction strategies for different scenarios."""
    STANDARD = "standard"
    ERROR_CORRECTION = "error_correction"
    REDUNDANT = "redundant"
    PARTIAL_RECOVERY = "partial_recovery"
    BRUTE_FORCE = "brute_force"


@dataclass
class AudioMetadata:
    """Metadata for audio steganography operations."""
    technique: str
    sample_rate: int
    channels: int
    duration: float
    format: str
    compression_resistant: bool
    error_correction_enabled: bool
    redundancy_level: int
    checksum: str
    timestamp: str


@dataclass
class ExtractionResult:
    """Result of data extraction attempt."""
    success: bool
    data: Optional[bytes]
    metadata: Optional[AudioMetadata]
    error_message: Optional[str]
    confidence_score: float
    recovery_method: Optional[str]


class AudioSteganographyEngine:
    """Enhanced audio steganography with improved reliability."""
    
    # Enhanced magic headers for different techniques
    MAGIC_HEADERS = {
        'lsb': b'INVV_LSB',
        'spread_spectrum': b'INVV_SSP',
        'phase_coding': b'INVV_PHC',
        'dct': b'INVV_DCT',
        'dwt': b'INVV_DWT',
        'echo': b'INVV_ECH'
    }
    
    VERSION = b'\x02\x00'  # Version 2.0
    
    # Reed-Solomon error correction parameters
    RS_SYMBOLS = 32  # Number of error correction symbols
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MAXIMUM):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.security_level = security_level
        self.encryption_engine = EncryptionEngine(security_level)
        self.analyzer = MultimediaAnalyzer()
        
        # Enhanced parameters
        self.sample_skip = 4  # Use every 4th sample
        self.max_duration = 1800  # 30 minutes max
        self.redundancy_factor = 3  # Triple redundancy for critical data
        self.error_correction_enabled = True
        self.adaptive_embedding = True
        
        # Cache for password verification
        self._password_cache = {}
        
        self.logger.info(f"Enhanced audio steganography engine initialized with {security_level.value} security")
    
    def hide_data_with_redundancy(self, audio_path: Path, data: bytes, output_path: Path,
                                  password: str, technique: str = 'lsb', 
                                  redundancy_level: int = 3,
                                  error_correction: bool = True) -> bool:
        """
        Hide data with redundancy and error correction for improved reliability.
        
        Args:
            audio_path: Path to carrier audio file
            data: Data to hide
            output_path: Output audio path
            password: Password for encryption
            technique: Steganography technique
            redundancy_level: Number of redundant copies (1-5)
            error_correction: Enable error correction codes
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Starting enhanced audio steganography with {redundancy_level}x redundancy")
            
            # Validate inputs
            if not self._validate_audio_file(audio_path):
                return False
            
            # Analyze capacity with redundancy
            analysis = self.analyzer.analyze_audio_file(audio_path)
            effective_capacity = analysis['capacity_bytes'] // redundancy_level
            
            if len(data) > effective_capacity:
                raise ValueError(f"Data too large with {redundancy_level}x redundancy: {len(data)} > {effective_capacity}")
            
            # Add error correction if enabled
            if error_correction:
                data = self._add_error_correction(data)
            
            # Prepare metadata
            metadata = self._create_metadata(technique, audio_path, redundancy_level, error_correction)
            
            # Encrypt and prepare data with metadata
            prepared_data = self._prepare_data_with_metadata(data, password, metadata)
            
            # Load audio
            audio_data, sample_rate = self._load_audio_safely(audio_path)
            
            # Embed data with redundancy
            modified_audio = self._embed_with_redundancy(
                audio_data, prepared_data, password, technique, 
                redundancy_level, sample_rate
            )
            
            if modified_audio is None:
                raise Exception("Failed to embed data")
            
            # Save with format preservation
            return self._save_audio_with_verification(
                modified_audio, sample_rate, output_path, audio_path
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced hiding failed: {e}")
            return False
    
    def extract_data_with_recovery(self, audio_path: Path, password: str,
                                   technique: str = 'auto',
                                   max_attempts: int = 5) -> Optional[bytes]:
        """
        Extract data with multiple recovery strategies.
        
        Args:
            audio_path: Path to audio file
            password: Password for decryption
            technique: Technique to use ('auto' for automatic detection)
            max_attempts: Maximum extraction attempts
            
        Returns:
            Extracted data or None
        """
        try:
            self.logger.info(f"Starting enhanced extraction with recovery (max attempts: {max_attempts})")
            
            # Validate audio file
            if not self._validate_audio_file(audio_path):
                return None
            
            # Try password verification cache
            if self._verify_password_cache(audio_path, password):
                self.logger.info("Password verified from cache")
            
            # Load audio
            audio_data, sample_rate = self._load_audio_safely(audio_path)
            
            # Determine techniques to try
            techniques_to_try = self._determine_techniques(technique)
            
            # Try each technique with different strategies
            for tech in techniques_to_try:
                self.logger.info(f"Trying technique: {tech}")
                
                # Try standard extraction
                result = self._try_extraction(
                    audio_data, password, tech, sample_rate,
                    ExtractionStrategy.STANDARD
                )
                
                if result.success:
                    self._update_password_cache(audio_path, password, True)
                    return result.data
                
                # Try with error correction
                if max_attempts > 1:
                    result = self._try_extraction(
                        audio_data, password, tech, sample_rate,
                        ExtractionStrategy.ERROR_CORRECTION
                    )
                    
                    if result.success:
                        self._update_password_cache(audio_path, password, True)
                        return result.data
                
                # Try redundant extraction
                if max_attempts > 2:
                    result = self._try_extraction(
                        audio_data, password, tech, sample_rate,
                        ExtractionStrategy.REDUNDANT
                    )
                    
                    if result.success:
                        self._update_password_cache(audio_path, password, True)
                        return result.data
                
                # Try partial recovery
                if max_attempts > 3:
                    result = self._try_extraction(
                        audio_data, password, tech, sample_rate,
                        ExtractionStrategy.PARTIAL_RECOVERY
                    )
                    
                    if result.success and result.confidence_score > 0.7:
                        self.logger.warning(f"Partial recovery succeeded with {result.confidence_score:.1%} confidence")
                        return result.data
            
            # If all else fails, try brute force recovery
            if max_attempts > 4:
                self.logger.warning("Attempting brute force recovery...")
                result = self._brute_force_recovery(audio_data, password, sample_rate)
                if result.success:
                    return result.data
            
            self.logger.error("All extraction attempts failed")
            self._log_extraction_failure_details(audio_path, techniques_to_try)
            return None
            
        except Exception as e:
            self.logger.error(f"Enhanced extraction failed: {e}")
            return None
    
    def _validate_audio_file(self, audio_path: Path) -> bool:
        """Validate audio file thoroughly."""
        try:
            if not audio_path.exists():
                self.logger.error(f"Audio file not found: {audio_path}")
                return False
            
            if not self.analyzer.is_audio_file(audio_path):
                self.logger.error(f"Invalid audio format: {audio_path.suffix}")
                return False
            
            # Check file integrity
            try:
                info = sf.info(str(audio_path))
                if info.frames == 0:
                    self.logger.error("Audio file is empty")
                    return False
            except Exception as e:
                self.logger.warning(f"Could not verify with soundfile, trying librosa: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio validation failed: {e}")
            return False
    
    def _load_audio_safely(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio with multiple fallback methods."""
        try:
            # Try soundfile first (preserves quality better)
            try:
                audio_data, sample_rate = sf.read(str(audio_path), always_2d=True)
                audio_data = audio_data.T  # Transpose to match librosa format
                self.logger.debug(f"Loaded with soundfile: {audio_data.shape} @ {sample_rate}Hz")
                return audio_data, sample_rate
            except Exception as e:
                self.logger.debug(f"Soundfile failed, trying librosa: {e}")
            
            # Fallback to librosa
            audio_data, sample_rate = librosa.load(
                str(audio_path),
                sr=None,
                mono=False,
                duration=self.max_duration
            )
            
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            
            self.logger.debug(f"Loaded with librosa: {audio_data.shape} @ {sample_rate}Hz")
            return audio_data, int(sample_rate)
            
        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}")
            raise
    
    def _add_error_correction(self, data: bytes) -> bytes:
        """Add Reed-Solomon error correction codes."""
        try:
            # Simple parity-based error correction for now
            # In production, use proper Reed-Solomon library
            data_array = np.frombuffer(data, dtype=np.uint8)
            
            # Add simple checksum blocks
            block_size = 128
            corrected_data = bytearray()
            
            for i in range(0, len(data_array), block_size):
                block = data_array[i:i+block_size]
                corrected_data.extend(block)
                
                # Add parity bytes
                parity = np.bitwise_xor.reduce(block)
                corrected_data.append(parity)
            
            return bytes(corrected_data)
            
        except Exception as e:
            self.logger.warning(f"Error correction failed, using original data: {e}")
            return data
    
    def _create_metadata(self, technique: str, audio_path: Path, 
                        redundancy_level: int, error_correction: bool) -> AudioMetadata:
        """Create comprehensive metadata for the operation."""
        try:
            info = sf.info(str(audio_path))
            
            return AudioMetadata(
                technique=technique,
                sample_rate=info.samplerate,
                channels=info.channels,
                duration=info.duration,
                format=audio_path.suffix.lower(),
                compression_resistant=technique in ['spread_spectrum', 'phase_coding'],
                error_correction_enabled=error_correction,
                redundancy_level=redundancy_level,
                checksum=hashlib.sha256(str(audio_path).encode()).hexdigest()[:16],
                timestamp=str(int(os.path.getmtime(str(audio_path))))
            )
        except:
            # Fallback metadata
            return AudioMetadata(
                technique=technique,
                sample_rate=44100,
                channels=2,
                duration=0.0,
                format=audio_path.suffix.lower(),
                compression_resistant=False,
                error_correction_enabled=error_correction,
                redundancy_level=redundancy_level,
                checksum="",
                timestamp=""
            )
    
    def _prepare_data_with_metadata(self, data: bytes, password: str, 
                                   metadata: AudioMetadata) -> bytes:
        """Prepare data with metadata and encryption."""
        try:
            # Serialize metadata
            metadata_json = json.dumps({
                'technique': metadata.technique,
                'redundancy': metadata.redundancy_level,
                'error_correction': metadata.error_correction_enabled,
                'checksum': metadata.checksum
            }).encode('utf-8')
            
            metadata_size = len(metadata_json)
            
            # Encrypt data
            encrypted_data = self.encryption_engine.encrypt_with_metadata(data, password)
            
            # Create enhanced header
            magic = self.MAGIC_HEADERS.get(metadata.technique, b'INVV_AUD')
            data_size = len(encrypted_data)
            checksum = hashlib.sha256(encrypted_data).digest()[:16]
            
            # Pack header: magic + version + metadata_size + data_size + checksum + metadata + data
            header = (
                magic +
                self.VERSION +
                struct.pack('<H', metadata_size) +  # 2 bytes for metadata size
                struct.pack('<Q', data_size) +      # 8 bytes for data size
                checksum +                           # 16 bytes for checksum
                metadata_json
            )
            
            return header + encrypted_data
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    def _embed_with_redundancy(self, audio_data: np.ndarray, data: bytes,
                              password: str, technique: str,
                              redundancy_level: int, sample_rate: int) -> Optional[np.ndarray]:
        """Embed data with redundancy across the audio."""
        try:
            modified_audio = audio_data.copy()
            
            # Divide audio into segments for redundant embedding
            channels, samples = audio_data.shape
            segment_size = samples // redundancy_level
            
            for i in range(redundancy_level):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size if i < redundancy_level - 1 else samples
                
                # Create a unique seed for each redundant copy
                segment_seed = hashlib.sha256(
                    f"{password}_{i}".encode()
                ).hexdigest()[:16]
                
                # Embed in segment
                segment_audio = modified_audio[:, start_idx:end_idx]
                
                if technique == 'lsb':
                    segment_result = self._embed_lsb_enhanced(
                        segment_audio, data, segment_seed, sample_rate
                    )
                elif technique == 'spread_spectrum':
                    segment_result = self._embed_spread_spectrum_enhanced(
                        segment_audio, data, segment_seed, sample_rate
                    )
                elif technique == 'phase_coding':
                    segment_result = self._embed_phase_enhanced(
                        segment_audio, data, segment_seed, sample_rate
                    )
                else:
                    segment_result = self._embed_lsb_enhanced(
                        segment_audio, data, segment_seed, sample_rate
                    )
                
                if segment_result is not None:
                    modified_audio[:, start_idx:end_idx] = segment_result
                    self.logger.debug(f"Embedded redundant copy {i+1}/{redundancy_level}")
            
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"Redundant embedding failed: {e}")
            return None
    
    def _embed_lsb_enhanced(self, audio_data: np.ndarray, data: bytes,
                           seed: str, sample_rate: int) -> Optional[np.ndarray]:
        """Enhanced LSB embedding with better distribution."""
        try:
            # Generate deterministic random sequence
            seed_int = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed_int)
            
            self.logger.debug(f"Embedding with seed: {seed[:8]}... (int: {seed_int})")
            
            # Debug: Show what we're embedding
            self.logger.debug(f"Embedding {len(data)} bytes of data")
            self.logger.debug(f"First 20 bytes of data to embed: {data[:20].hex() if len(data) >= 20 else data.hex()}")
            
            # Convert data to bits
            data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            
            # Prepare audio
            channels, samples = audio_data.shape
            modified_audio = audio_data.copy()
            
            # Convert to integer representation
            audio_int = (modified_audio * 32767).astype(np.int16)
            
            # Calculate embedding positions with better distribution
            total_samples = channels * samples
            usable_samples = total_samples // self.sample_skip
            
            if len(data_bits) > usable_samples:
                self.logger.warning(f"Data too large for segment: {len(data_bits)} > {usable_samples}")
                return None
            
            # Generate non-overlapping positions
            all_positions = np.arange(0, total_samples, self.sample_skip)
            selected_positions = rng.choice(
                all_positions, 
                size=min(len(data_bits), len(all_positions)), 
                replace=False
            )
            
            # Embed bits
            flat_audio = audio_int.flatten()
            for i, pos in enumerate(selected_positions):
                if i < len(data_bits):
                    # Enhanced LSB embedding with minimal distortion
                    flat_audio[pos] = (flat_audio[pos] & 0xFFFE) | data_bits[i]
            
            # Reshape and convert back
            modified_int = flat_audio.reshape(channels, samples)
            modified_audio = modified_int.astype(np.float32) / 32767.0
            
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"Enhanced LSB embedding failed: {e}")
            return None
    
    def _embed_spread_spectrum_enhanced(self, audio_data: np.ndarray, data: bytes,
                                       seed: str, sample_rate: int) -> Optional[np.ndarray]:
        """Enhanced spread spectrum with adaptive power."""
        try:
            seed_int = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed_int)
            
            data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            channels, samples = audio_data.shape
            modified_audio = audio_data.copy()
            
            # Adaptive parameters based on audio characteristics
            chip_rate = min(1000, samples // (len(data_bits) * 2))
            
            # Calculate adaptive amplitude based on audio power
            audio_power = float(np.mean(np.abs(audio_data)))
            amplitude = min(0.002, audio_power * 0.01)  # Adaptive amplitude
            
            for channel in range(channels):
                for bit_idx, bit in enumerate(data_bits):
                    start_sample = bit_idx * chip_rate
                    end_sample = start_sample + chip_rate
                    
                    if end_sample > samples:
                        break
                    
                    # Generate spreading sequence
                    spread_seq = rng.uniform(-1, 1, min(chip_rate, samples - start_sample))
                    
                    # Modulate with adaptive power
                    if bit == 1:
                        signal_to_add = spread_seq * amplitude
                    else:
                        signal_to_add = -spread_seq * amplitude
                    
                    # Add to audio
                    modified_audio[channel, start_sample:start_sample + len(spread_seq)] += signal_to_add
            
            # Normalize
            max_val = np.max(np.abs(modified_audio))
            if max_val > 1.0:
                modified_audio = modified_audio / max_val
            
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"Enhanced spread spectrum failed: {e}")
            return None
    
    def _embed_phase_enhanced(self, audio_data: np.ndarray, data: bytes,
                             seed: str, sample_rate: int) -> Optional[np.ndarray]:
        """Enhanced phase coding with better preservation."""
        try:
            data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            channels, samples = audio_data.shape
            modified_audio = audio_data.copy()
            
            # Adaptive segment length
            segment_length = 2048  # Larger segments for better quality
            num_segments = samples // segment_length
            
            if len(data_bits) > num_segments * channels:
                segment_length = 1024  # Fall back to smaller segments
                num_segments = samples // segment_length
            
            # Phase shift parameters
            phase_shift = np.pi / 6  # Smaller shift for less distortion
            
            bit_idx = 0
            for channel in range(channels):
                for seg_idx in range(num_segments):
                    if bit_idx >= len(data_bits):
                        break
                    
                    start = seg_idx * segment_length
                    end = min(start + segment_length, samples)
                    segment = modified_audio[channel, start:end]
                    
                    # Apply windowing to reduce artifacts
                    window = np.hanning(len(segment))
                    windowed_segment = segment * window
                    
                    # FFT
                    fft_segment = np.fft.fft(windowed_segment)
                    
                    # Modify phase based on bit
                    if data_bits[bit_idx] == 1:
                        phase = np.angle(fft_segment)
                        magnitude = np.abs(fft_segment)
                        modified_fft = magnitude * np.exp(1j * (phase + phase_shift))
                    else:
                        phase = np.angle(fft_segment)
                        magnitude = np.abs(fft_segment)
                        modified_fft = magnitude * np.exp(1j * (phase - phase_shift))
                    
                    # Inverse FFT
                    modified_segment = np.real(np.fft.ifft(modified_fft))
                    
                    # Apply inverse window and overlap-add
                    modified_audio[channel, start:end] = modified_segment
                    
                    bit_idx += 1
            
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"Enhanced phase coding failed: {e}")
            return None
    
    def _try_extraction(self, audio_data: np.ndarray, password: str,
                       technique: str, sample_rate: int,
                       strategy: ExtractionStrategy) -> ExtractionResult:
        """Try extraction with specific strategy."""
        try:
            self.logger.debug(f"Trying {technique} with {strategy.value} strategy")
            
            if strategy == ExtractionStrategy.STANDARD:
                # Standard extraction
                data = self._extract_standard(audio_data, password, technique, sample_rate)
                if data:
                    return ExtractionResult(
                        success=True,
                        data=data,
                        metadata=None,
                        error_message=None,
                        confidence_score=1.0,
                        recovery_method=strategy.value
                    )
            
            elif strategy == ExtractionStrategy.ERROR_CORRECTION:
                # Try with error correction
                data = self._extract_with_error_correction(
                    audio_data, password, technique, sample_rate
                )
                if data:
                    return ExtractionResult(
                        success=True,
                        data=data,
                        metadata=None,
                        error_message=None,
                        confidence_score=0.95,
                        recovery_method=strategy.value
                    )
            
            elif strategy == ExtractionStrategy.REDUNDANT:
                # Try redundant extraction
                data = self._extract_redundant(
                    audio_data, password, technique, sample_rate
                )
                if data:
                    return ExtractionResult(
                        success=True,
                        data=data,
                        metadata=None,
                        error_message=None,
                        confidence_score=0.9,
                        recovery_method=strategy.value
                    )
            
            elif strategy == ExtractionStrategy.PARTIAL_RECOVERY:
                # Try partial recovery
                data, confidence = self._extract_partial(
                    audio_data, password, technique, sample_rate
                )
                if data:
                    return ExtractionResult(
                        success=True,
                        data=data,
                        metadata=None,
                        error_message="Partial recovery",
                        confidence_score=confidence,
                        recovery_method=strategy.value
                    )
            
            return ExtractionResult(
                success=False,
                data=None,
                metadata=None,
                error_message=f"Extraction failed with {strategy.value}",
                confidence_score=0.0,
                recovery_method=None
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                data=None,
                metadata=None,
                error_message=str(e),
                confidence_score=0.0,
                recovery_method=None
            )
    
    def _extract_standard(self, audio_data: np.ndarray, password: str,
                         technique: str, sample_rate: int) -> Optional[bytes]:
        """Standard extraction method."""
        try:
            # IMPORTANT: When redundancy_level=1 is used during embedding,
            # the data is ALWAYS embedded with seed f"{password}_0" 
            # So we should try that first for LSB technique
            
            if technique == 'lsb':
                # Try with segment 0 seed first (for redundancy_level=1 data)
                segment_seed = hashlib.sha256(f"{password}_0".encode()).hexdigest()[:16]
                self.logger.debug(f"Trying extraction with segment_0 seed: {segment_seed[:8]}...")
                encrypted_data = self._extract_lsb_with_seed(audio_data, segment_seed, sample_rate)
                
                if not encrypted_data:
                    # Fallback to direct password seed extraction (for non-redundant data)
                    self.logger.debug("Segment 0 extraction failed, trying direct password seed")
                    password_seed = hashlib.sha256(password.encode()).hexdigest()[:16]
                    self.logger.debug(f"Trying extraction with password seed: {password_seed[:8]}...")
                    encrypted_data = self._extract_lsb_with_seed(audio_data, password_seed, sample_rate)
            elif technique == 'spread_spectrum':
                # Try segment 0 seed first for spread spectrum too
                segment_seed = hashlib.sha256(f"{password}_0".encode()).hexdigest()[:16]
                encrypted_data = self._extract_spread_spectrum_with_seed(audio_data, segment_seed, sample_rate)
                if not encrypted_data:
                    encrypted_data = self._extract_spread_spectrum_enhanced(audio_data, password, sample_rate)
            elif technique == 'phase_coding':
                # Try segment 0 seed first for phase coding too  
                segment_seed = hashlib.sha256(f"{password}_0".encode()).hexdigest()[:16]
                encrypted_data = self._extract_phase_with_seed(audio_data, segment_seed, sample_rate)
                if not encrypted_data:
                    encrypted_data = self._extract_phase_enhanced(audio_data, password, sample_rate)
            else:
                return None
            
            if encrypted_data:
                return self._decrypt_and_verify(encrypted_data, password)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Standard extraction failed: {e}")
            return None
    
    def _extract_with_error_correction(self, audio_data: np.ndarray, password: str,
                                      technique: str, sample_rate: int) -> Optional[bytes]:
        """Extract with error correction."""
        try:
            # Extract raw data
            raw_data = self._extract_standard(audio_data, password, technique, sample_rate)
            
            if not raw_data:
                return None
            
            # Apply error correction
            corrected_data = self._apply_error_correction(raw_data)
            
            return corrected_data
            
        except Exception as e:
            self.logger.debug(f"Error correction extraction failed: {e}")
            return None
    
    def _extract_redundant(self, audio_data: np.ndarray, password: str,
                          technique: str, sample_rate: int) -> Optional[bytes]:
        """Extract from redundant copies and vote."""
        try:
            channels, samples = audio_data.shape
            
            # Use same redundancy level as embedding (3 by default)
            redundancy_level = 3
            segment_size = samples // redundancy_level
            extracted_copies = []
            
            for i in range(redundancy_level):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size if i < redundancy_level - 1 else samples
                
                # Create unique seed for segment - same as embedding
                segment_seed = hashlib.sha256(
                    f"{password}_{i}".encode()
                ).hexdigest()[:16]
                
                # Extract from segment
                segment_audio = audio_data[:, start_idx:end_idx]
                
                if technique == 'lsb':
                    segment_data = self._extract_lsb_with_seed(
                        segment_audio, segment_seed, sample_rate
                    )
                else:
                    segment_data = None
                
                if segment_data:
                    extracted_copies.append(segment_data)
                    self.logger.debug(f"Extracted copy {i+1}/{redundancy_level}")
            
            # Vote on the most common result
            if extracted_copies:
                # Find the most common extraction
                from collections import Counter
                data_counter = Counter(extracted_copies)
                most_common = data_counter.most_common(1)[0]
                
                if most_common[1] >= 2:  # At least 2 copies match
                    return self._decrypt_and_verify(most_common[0], password)
                elif len(extracted_copies) == 1:
                    # If only one copy found, try it anyway
                    self.logger.warning("Only one redundant copy recovered, attempting decryption")
                    return self._decrypt_and_verify(extracted_copies[0], password)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Redundant extraction failed: {e}")
            return None
    
    def _extract_partial(self, audio_data: np.ndarray, password: str,
                        technique: str, sample_rate: int) -> Tuple[Optional[bytes], float]:
        """Attempt partial recovery of data."""
        try:
            # Try to extract whatever we can
            partial_data = self._extract_with_tolerance(
                audio_data, password, technique, sample_rate
            )
            
            if partial_data:
                # Calculate confidence based on recovered amount
                confidence = min(len(partial_data) / 1024, 1.0)  # Rough estimate
                return partial_data, confidence
            
            return None, 0.0
            
        except Exception as e:
            self.logger.debug(f"Partial extraction failed: {e}")
            return None, 0.0
    
    def _extract_lsb_enhanced(self, audio_data: np.ndarray, password: str,
                             sample_rate: int) -> Optional[bytes]:
        """Enhanced LSB extraction."""
        try:
            # Generate same random sequence
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            
            # Convert to integer
            audio_int = (audio_data * 32767).astype(np.int16)
            channels, samples = audio_int.shape
            
            # Calculate positions
            total_samples = channels * samples
            all_positions = np.arange(0, total_samples, self.sample_skip)
            
            # We need to know how many bits to extract
            # First, try to extract the header and initial data
            min_header_bits = 250 * 8  # Extract 250 bytes initially (2000 bits)
            
            if len(all_positions) < min_header_bits:
                return None
            
            # Generate positions for header
            header_positions = rng.choice(
                all_positions,
                size=min(min_header_bits, len(all_positions)),  # Match embedding exactly
                replace=False
            )
            
            # Extract bits
            flat_audio = audio_int.flatten()
            extracted_bits = []
            
            for pos in header_positions:
                if pos < len(flat_audio):
                    extracted_bits.append(flat_audio[pos] & 1)
            
            # Convert to bytes
            bit_array = np.array(extracted_bits, dtype=np.uint8)
            padding_needed = (8 - (len(bit_array) % 8)) % 8
            if padding_needed > 0:
                bit_array = np.append(bit_array, np.zeros(padding_needed, dtype=np.uint8))
            
            all_bytes = np.packbits(bit_array)
            
            # Find magic header
            for magic_key, magic_value in self.MAGIC_HEADERS.items():
                for offset in range(min(100, len(all_bytes) - 50)):
                    if offset + len(magic_value) > len(all_bytes):
                        continue
                    
                    if bytes(all_bytes[offset:offset + len(magic_value)]) == magic_value:
                        # Found header, parse it
                        return self._parse_enhanced_header(all_bytes, offset, rng, all_positions, flat_audio, seed)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Enhanced LSB extraction failed: {e}")
            return None
    
    def _extract_lsb_with_seed(self, audio_data: np.ndarray, seed: str,
                               sample_rate: int) -> Optional[bytes]:
        """Extract LSB with specific seed."""
        try:
            # Use the seed directly instead of password for RNG
            seed_int = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed_int)
            
            # Convert to integer
            audio_int = (audio_data * 32767).astype(np.int16)
            channels, samples = audio_int.shape
            
            # Calculate positions
            total_samples = channels * samples
            all_positions = np.arange(0, total_samples, self.sample_skip)
            
            # We need enough data for header extraction first
            # Need to extract more initially to ensure we get the full header + data
            min_header_bits = 250 * 8  # Extract 250 bytes initially (2000 bits)
            
            if len(all_positions) < min_header_bits:
                return None
            
            # Generate positions for initial extraction
            header_positions = rng.choice(
                all_positions,
                size=min(min_header_bits, len(all_positions)),  # Don't multiply by 2
                replace=False
            )
            
            # Extract initial bits
            flat_audio = audio_int.flatten()
            extracted_bits = []
            
            for pos in header_positions:
                if pos < len(flat_audio):
                    extracted_bits.append(flat_audio[pos] & 1)
            
            # Convert to bytes
            bit_array = np.array(extracted_bits, dtype=np.uint8)
            padding_needed = (8 - (len(bit_array) % 8)) % 8
            if padding_needed > 0:
                bit_array = np.append(bit_array, np.zeros(padding_needed, dtype=np.uint8))
            
            all_bytes = np.packbits(bit_array)
            
            # Look for magic header and parse properly
            self.logger.debug(f"Looking for magic headers in {len(all_bytes)} bytes")
            for magic_key, magic_value in self.MAGIC_HEADERS.items():
                for offset in range(min(100, len(all_bytes) - 50)):
                    if offset + len(magic_value) > len(all_bytes):
                        continue
                    
                    if bytes(all_bytes[offset:offset + len(magic_value)]) == magic_value:
                        # Found header, parse it properly like in enhanced extraction
                        self.logger.debug(f"Found {magic_key} header at offset {offset}")
                        return self._parse_enhanced_header(all_bytes, offset, rng, all_positions, flat_audio, seed_int)
            
            # Debug: Show first few bytes to understand what we're seeing
            if len(all_bytes) > 0:
                self.logger.debug(f"No magic header found. First 50 bytes: {bytes(all_bytes[:50]).hex() if len(all_bytes) >= 50 else bytes(all_bytes).hex()}")
            else:
                self.logger.debug("No magic header found. No bytes extracted.")
            return None
            
        except Exception as e:
            self.logger.debug(f"Seeded LSB extraction failed: {e}")
            return None
    
    def _extract_spread_spectrum_enhanced(self, audio_data: np.ndarray, password: str,
                                         sample_rate: int) -> Optional[bytes]:
        """Enhanced spread spectrum extraction."""
        try:
            # For now, return None to fall back to other methods
            # Full implementation would require complete spread spectrum extraction
            self.logger.debug("Spread spectrum extraction not fully implemented yet")
            return None
        except Exception as e:
            self.logger.error(f"Spread spectrum extraction failed: {e}")
            return None
    
    def _extract_spread_spectrum_with_seed(self, audio_data: np.ndarray, seed: str,
                                          sample_rate: int) -> Optional[bytes]:
        """Extract spread spectrum with specific seed."""
        try:
            # For now, return None to fall back to other methods
            self.logger.debug("Seeded spread spectrum extraction not fully implemented yet")
            return None
        except Exception as e:
            self.logger.debug(f"Seeded spread spectrum extraction failed: {e}")
            return None
    
    def _extract_phase_enhanced(self, audio_data: np.ndarray, password: str,
                               sample_rate: int) -> Optional[bytes]:
        """Enhanced phase coding extraction."""
        try:
            # For now, return None to fall back to other methods
            # Full implementation would require complete phase coding extraction
            self.logger.debug("Phase coding extraction not fully implemented yet")
            return None
        except Exception as e:
            self.logger.error(f"Phase coding extraction failed: {e}")
            return None
    
    def _extract_phase_with_seed(self, audio_data: np.ndarray, seed: str,
                                sample_rate: int) -> Optional[bytes]:
        """Extract phase coding with specific seed."""
        try:
            # For now, return None to fall back to other methods
            self.logger.debug("Seeded phase coding extraction not fully implemented yet")
            return None
        except Exception as e:
            self.logger.debug(f"Seeded phase coding extraction failed: {e}")
            return None
    
    def _extract_with_tolerance(self, audio_data: np.ndarray, password: str,
                               technique: str, sample_rate: int) -> Optional[bytes]:
        """Extract with error tolerance."""
        try:
            # Try extraction with relaxed parameters
            if technique == 'lsb':
                return self._extract_lsb_enhanced(audio_data, password, sample_rate)
            else:
                return None
        except Exception as e:
            self.logger.debug(f"Tolerant extraction failed: {e}")
            return None
    
    def _parse_enhanced_header(self, all_bytes: np.ndarray, offset: int,
                              rng: np.random.RandomState, all_positions: np.ndarray,
                              flat_audio: np.ndarray, original_seed: Optional[int] = None) -> Optional[bytes]:
        """Parse enhanced header and extract data."""
        try:
            # Parse enhanced header structure
            header_start = offset
            magic_len = 8
            version_len = 2
            metadata_size_len = 2
            data_size_len = 8
            checksum_len = 16
            
            # Read header components
            version = bytes(all_bytes[header_start + magic_len:header_start + magic_len + version_len])
            metadata_size = struct.unpack('<H', bytes(all_bytes[header_start + magic_len + version_len:header_start + magic_len + version_len + metadata_size_len]))[0]
            data_size = struct.unpack('<Q', bytes(all_bytes[header_start + magic_len + version_len + metadata_size_len:header_start + magic_len + version_len + metadata_size_len + data_size_len]))[0]
            checksum = bytes(all_bytes[header_start + magic_len + version_len + metadata_size_len + data_size_len:header_start + magic_len + version_len + metadata_size_len + data_size_len + checksum_len])
            
            # Validate sizes
            if data_size <= 0 or data_size > 100 * 1024 * 1024:
                return None
            
            # Calculate total size needed
            total_header_size = magic_len + version_len + metadata_size_len + data_size_len + checksum_len + metadata_size
            total_size = total_header_size + data_size
            
            # Extract remaining data if needed
            total_bits_needed = total_size * 8
            
            if total_bits_needed > len(all_positions):
                self.logger.warning(f"Not enough positions for full extraction")
                return None
            
            # CRITICAL FIX: We need to recreate the RNG from scratch to get ALL positions
            # The RNG state has already been used for the initial header extraction
            # We need to start fresh to get the complete sequence
            if original_seed is not None:
                fresh_rng = np.random.RandomState(original_seed)
            else:
                # Fallback: try to extract from rng state (may not work correctly)
                rng_state = rng.get_state()
                fresh_rng = np.random.RandomState(int(rng_state[1][0]))
            
            # Generate all positions needed from the beginning
            selected_positions = fresh_rng.choice(
                all_positions,
                size=min(total_bits_needed, len(all_positions)),
                replace=False
            )
            
            # Extract all bits
            all_extracted_bits = []
            for pos in selected_positions:
                if pos < len(flat_audio):
                    all_extracted_bits.append(flat_audio[pos] & 1)
            
            # Convert to bytes
            bit_array = np.array(all_extracted_bits, dtype=np.uint8)
            padding_needed = (8 - (len(bit_array) % 8)) % 8
            if padding_needed > 0:
                bit_array = np.append(bit_array, np.zeros(padding_needed, dtype=np.uint8))
            
            complete_bytes = np.packbits(bit_array)
            
            # Extract encrypted data
            data_start = header_start + total_header_size
            data_end = data_start + data_size
            
            if data_end > len(complete_bytes):
                return None
            
            encrypted_data = bytes(complete_bytes[data_start:data_end])
            
            # Verify checksum
            actual_checksum = hashlib.sha256(encrypted_data).digest()[:16]
            if actual_checksum != checksum:
                self.logger.warning("Checksum mismatch in enhanced extraction")
                # Continue anyway for recovery attempt
            
            return encrypted_data
            
        except Exception as e:
            self.logger.error(f"Header parsing failed: {e}")
            return None
    
    def _decrypt_and_verify(self, encrypted_data: bytes, password: str) -> Optional[bytes]:
        """Decrypt and verify extracted data."""
        try:
            # Decrypt the data
            original_data = self.encryption_engine.decrypt_with_metadata(encrypted_data, password)
            
            # Remove error correction if present
            if self.error_correction_enabled:
                original_data = self._remove_error_correction(original_data)
            
            return original_data
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return None
    
    def _apply_error_correction(self, data: bytes) -> bytes:
        """Apply error correction to recovered data."""
        try:
            # Simple parity-based correction
            data_array = np.frombuffer(data, dtype=np.uint8)
            corrected = bytearray()
            
            block_size = 129  # 128 data + 1 parity
            for i in range(0, len(data_array), block_size):
                block = data_array[i:i+block_size]
                if len(block) < block_size:
                    corrected.extend(block[:-1] if len(block) > 1 else block)
                else:
                    # Verify and correct using parity
                    data_part = block[:-1]
                    parity = block[-1]
                    calculated_parity = np.bitwise_xor.reduce(data_part)
                    
                    if parity == calculated_parity:
                        corrected.extend(data_part)
                    else:
                        # Attempt correction
                        self.logger.debug("Parity mismatch, attempting correction")
                        corrected.extend(data_part)
            
            return bytes(corrected)
            
        except Exception as e:
            self.logger.warning(f"Error correction failed: {e}")
            return data
    
    def _remove_error_correction(self, data: bytes) -> bytes:
        """Remove error correction codes from data."""
        try:
            # Remove parity bytes
            data_array = np.frombuffer(data, dtype=np.uint8)
            cleaned = bytearray()
            
            block_size = 129  # 128 data + 1 parity
            for i in range(0, len(data_array), block_size):
                block = data_array[i:i+block_size]
                if len(block) <= block_size:
                    # Remove parity byte
                    cleaned.extend(block[:-1] if len(block) > 1 else block)
            
            return bytes(cleaned)
            
        except Exception as e:
            self.logger.warning(f"Error correction removal failed: {e}")
            return data
    
    def _save_audio_with_verification(self, audio_data: np.ndarray, sample_rate: int,
                                     output_path: Path, original_path: Path) -> bool:
        """Save audio with format preservation and verification."""
        try:
            # Determine output format
            output_format = output_path.suffix.lower().lstrip('.')
            original_format = original_path.suffix.lower().lstrip('.')
            
            # Warn about format changes
            if output_format != original_format:
                self.logger.warning(
                    f"Output format ({output_format}) differs from input ({original_format}). "
                    "This may affect extraction reliability."
                )
            
            # Check for lossy formats
            lossy_formats = {'.mp3', '.aac', '.ogg', '.m4a', '.wma'}
            lossless_formats = {'.wav', '.flac', '.aiff', '.au'}
            
            if output_path.suffix.lower() in lossy_formats:
                self.logger.warning(
                    f"WARNING: Saving to lossy format {output_format.upper()}. "
                    "This will likely corrupt LSB steganography data. "
                    "Use lossless formats (WAV, FLAC) for reliable extraction."
                )
            
            # Try soundfile first for better quality
            try:
                # Transpose for soundfile format
                audio_to_save = audio_data.T if audio_data.ndim == 2 else audio_data
                
                # Determine subtype based on format
                if output_format == 'wav':
                    subtype = 'PCM_16'
                elif output_format == 'flac':
                    subtype = 'PCM_16'
                else:
                    subtype = None
                
                sf.write(
                    str(output_path),
                    audio_to_save,
                    sample_rate,
                    subtype=subtype
                )
                
                self.logger.info(f"Audio saved with soundfile: {output_path.name}")
                
            except Exception as e:
                self.logger.debug(f"Soundfile save failed, trying pydub: {e}")
                
                # Fall back to pydub
                if audio_data.ndim == 2:
                    channels, samples = audio_data.shape
                    if channels == 1:
                        audio_int = (audio_data[0] * 32767).astype(np.int16)
                        audio_segment = AudioSegment(
                            audio_int.tobytes(),
                            frame_rate=sample_rate,
                            sample_width=2,
                            channels=1
                        )
                    else:
                        audio_left = (audio_data[0] * 32767).astype(np.int16)
                        audio_right = (audio_data[1] * 32767).astype(np.int16) if channels > 1 else audio_left
                        stereo_data = np.empty((audio_left.size + audio_right.size,), dtype=np.int16)
                        stereo_data[0::2] = audio_left
                        stereo_data[1::2] = audio_right
                        audio_segment = AudioSegment(
                            stereo_data.tobytes(),
                            frame_rate=sample_rate,
                            sample_width=2,
                            channels=2
                        )
                else:
                    audio_int = (audio_data * 32767).astype(np.int16)
                    audio_segment = AudioSegment(
                        audio_int.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=2,
                        channels=1
                    )
                
                # Set quality parameters
                export_params = {}
                if output_format in ['wav', 'flac']:
                    if output_format == 'flac':
                        export_params['parameters'] = ["-compression_level", "8"]
                else:
                    export_params['bitrate'] = "320k"
                
                audio_segment.export(str(output_path), format=output_format, **export_params)
                self.logger.info(f"Audio saved with pydub: {output_path.name}")
            
            # Verify file was created
            if output_path.exists() and output_path.stat().st_size > 0:
                self.logger.info(f"Audio file verified: {output_path.stat().st_size} bytes")
                return True
            else:
                self.logger.error("Output file not created or empty")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            return False
    
    def _determine_techniques(self, technique: str) -> List[str]:
        """Determine which techniques to try."""
        if technique == 'auto':
            # Try all techniques in order of reliability
            return ['lsb', 'spread_spectrum', 'phase_coding']
        else:
            return [technique]
    
    def _brute_force_recovery(self, audio_data: np.ndarray, password: str,
                             sample_rate: int) -> ExtractionResult:
        """Last resort brute force recovery."""
        try:
            self.logger.warning("Attempting brute force recovery - this may take time...")
            
            # Try different combinations of parameters
            techniques = ['lsb', 'spread_spectrum', 'phase_coding']
            sample_skips = [2, 3, 4, 5, 6, 8]
            
            for tech in techniques:
                for skip in sample_skips:
                    self.sample_skip = skip
                    result = self._extract_standard(audio_data, password, tech, sample_rate)
                    if result:
                        self.logger.info(f"Brute force succeeded with {tech} and skip={skip}")
                        return ExtractionResult(
                            success=True,
                            data=result,
                            metadata=None,
                            error_message="Recovered via brute force",
                            confidence_score=0.6,
                            recovery_method="brute_force"
                        )
            
            return ExtractionResult(
                success=False,
                data=None,
                metadata=None,
                error_message="Brute force recovery failed",
                confidence_score=0.0,
                recovery_method=None
            )
            
        except Exception as e:
            self.logger.error(f"Brute force recovery failed: {e}")
            return ExtractionResult(
                success=False,
                data=None,
                metadata=None,
                error_message=str(e),
                confidence_score=0.0,
                recovery_method=None
            )
    
    def _verify_password_cache(self, audio_path: Path, password: str) -> bool:
        """Check password cache for faster verification."""
        cache_key = f"{audio_path}:{hashlib.sha256(password.encode()).hexdigest()[:16]}"
        return self._password_cache.get(cache_key, False)
    
    def _update_password_cache(self, audio_path: Path, password: str, valid: bool):
        """Update password cache."""
        cache_key = f"{audio_path}:{hashlib.sha256(password.encode()).hexdigest()[:16]}"
        self._password_cache[cache_key] = valid
        
        # Limit cache size
        if len(self._password_cache) > 100:
            # Remove oldest entries
            keys_to_remove = list(self._password_cache.keys())[:50]
            for key in keys_to_remove:
                del self._password_cache[key]
    
    def _log_extraction_failure_details(self, audio_path: Path, techniques_tried: List[str]):
        """Log detailed failure information for debugging."""
        self.logger.error("\n" + "="*60)
        self.logger.error("EXTRACTION FAILURE DETAILS")
        self.logger.error("="*60)
        
        self.logger.error(f"File: {audio_path}")
        self.logger.error(f"Format: {audio_path.suffix}")
        self.logger.error(f"Techniques tried: {', '.join(techniques_tried)}")
        
        # Check if file is lossy
        lossy_formats = {'.mp3', '.aac', '.ogg', '.m4a', '.wma'}
        if audio_path.suffix.lower() in lossy_formats:
            self.logger.error("\n⚠️  LOSSY FORMAT DETECTED")
            self.logger.error("The audio file is in a lossy format which often corrupts LSB data.")
            self.logger.error("Recommendations:")
            self.logger.error("1. Use the original lossless file (WAV/FLAC) if available")
            self.logger.error("2. Try spread_spectrum or phase_coding techniques (more resilient)")
            self.logger.error("3. Ensure the file wasn't converted after hiding data")
        
        self.logger.error("\nTROUBLESHOOTING STEPS:")
        self.logger.error("1. Verify the password is correct (case-sensitive)")
        self.logger.error("2. Confirm this file contains hidden data")
        self.logger.error("3. Check if the audio file was modified after hiding")
        self.logger.error("4. Try different extraction techniques")
        self.logger.error("5. Use the same settings that were used for hiding")
        
        self.logger.error("="*60 + "\n")
