"""
Audio Steganography Engine
Implements advanced audio steganography techniques for hiding data in audio files.
"""

import os
import sys
import tempfile
import struct
import hashlib
import secrets
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import warnings

import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    from pydub import AudioSegment
    from scipy import signal
except ImportError as e:
    print(f"Warning: Audio dependencies not fully installed: {e}")
    print("Please install: pip install librosa pydub scipy")

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from core.encryption_engine import EncryptionEngine, SecurityLevel
from core.multimedia_analyzer import MultimediaAnalyzer


class AudioSteganographyEngine:
    """Advanced audio steganography implementation using multiple techniques."""
    
    MAGIC_HEADER = b'INVV_AUD'  # InvisioVault Audio magic bytes
    VERSION = b'\x01\x00'  # Version 1.0
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MAXIMUM):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.security_level = security_level
        self.encryption_engine = EncryptionEngine(security_level)
        self.analyzer = MultimediaAnalyzer()
        
        # Audio processing parameters
        self.sample_skip = 4  # Use every 4th sample to avoid detection
        self.max_duration = 1800  # Limit processing to 30 minutes
        
        self.logger.info(f"Audio steganography engine initialized with {security_level.value} security")
    
    def hide_data_in_audio(self, audio_path: Path, data: bytes, output_path: Path, 
                          password: str, technique: str = 'lsb', quality: str = 'high') -> bool:
        """
        Hide data in audio file using specified steganography technique.
        
        Args:
            audio_path: Path to carrier audio file
            data: Data to hide (will be encrypted)
            output_path: Output audio path
            password: Password for encryption and randomization
            technique: Steganography technique ('lsb', 'spread_spectrum', 'phase_coding')
            quality: Output quality ('high', 'medium', 'low')
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Starting audio steganography: {audio_path.name}")
            
            # Validate input
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if not self.analyzer.is_audio_file(audio_path):
                raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
            
            # Analyze audio capacity
            analysis = self.analyzer.analyze_audio_file(audio_path)
            if 'error' in analysis:
                raise Exception(f"Audio analysis failed: {analysis['error']}")
            
            capacity = analysis['capacity_bytes']
            if len(data) > capacity:
                raise ValueError(f"Data too large: {len(data)} bytes exceeds capacity {capacity} bytes")
            
            # Encrypt and prepare data
            encrypted_data = self._prepare_data_for_hiding(data, password)
            
            # Load audio file
            audio_data, sample_rate = self._load_audio_file(audio_path)
            
            # Apply steganography technique
            if technique == 'lsb':
                modified_audio = self._hide_data_lsb(audio_data, encrypted_data, password, sample_rate)
            elif technique == 'spread_spectrum':
                modified_audio = self._hide_data_spread_spectrum(audio_data, encrypted_data, password, sample_rate)
            elif technique == 'phase_coding':
                modified_audio = self._hide_data_phase_coding(audio_data, encrypted_data, password, sample_rate)
            else:
                raise ValueError(f"Unsupported technique: {technique}")
            
            if modified_audio is None:
                raise Exception("Failed to hide data in audio")
            
            # Save modified audio
            success = self._save_audio_file(modified_audio, sample_rate, output_path, quality)
            
            if success:
                self.logger.info(f"Audio steganography completed successfully: {output_path.name}")
                return True
            else:
                raise Exception("Failed to save modified audio")
            
        except Exception as e:
            self.logger.error(f"Audio steganography failed: {e}")
            return False
    
    def extract_data_from_audio(self, audio_path: Path, password: str, 
                               technique: str = 'lsb') -> Optional[bytes]:
        """
        Extract hidden data from audio file.
        
        Args:
            audio_path: Path to audio file containing hidden data
            password: Password for decryption
            technique: Steganography technique used for hiding
            
        Returns:
            Extracted data or None if failed
        """
        try:
            self.logger.info(f"Starting audio data extraction: {audio_path.name}")
            
            # Validate input
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if not self.analyzer.is_audio_file(audio_path):
                raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
            
            # Load audio file
            audio_data, sample_rate = self._load_audio_file(audio_path)
            
            # Extract data using appropriate technique
            if technique == 'lsb':
                encrypted_data = self._extract_data_lsb(audio_data, password, sample_rate)
            elif technique == 'spread_spectrum':
                encrypted_data = self._extract_data_spread_spectrum(audio_data, password, sample_rate)
            elif technique == 'phase_coding':
                encrypted_data = self._extract_data_phase_coding(audio_data, password, sample_rate)
            else:
                raise ValueError(f"Unsupported technique: {technique}")
            
            if not encrypted_data:
                raise Exception("No hidden data found in audio")
            
            # Decrypt and verify data
            original_data = self._extract_and_decrypt_data(encrypted_data, password)
            
            self.logger.info(f"Audio data extraction completed: {len(original_data)} bytes")
            return original_data
            
        except Exception as e:
            self.logger.error(f"Audio data extraction failed: {e}")
            return None
    
    def _prepare_data_for_hiding(self, data: bytes, password: str) -> bytes:
        """Encrypt and prepare data with header for hiding."""
        try:
            # Encrypt the data
            encrypted_data = self.encryption_engine.encrypt_with_metadata(data, password)
            
            # Create header: magic + version + size + checksum
            data_size = len(encrypted_data)
            checksum = hashlib.md5(encrypted_data).digest()
            
            header = (
                self.MAGIC_HEADER +
                self.VERSION +
                struct.pack('<Q', data_size) +  # 8 bytes for size
                checksum  # 16 bytes for MD5
            )
            
            # Combine header + encrypted data
            prepared_data = header + encrypted_data
            
            self.logger.debug(f"Data prepared for hiding: {len(prepared_data)} bytes total")
            return prepared_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            raise
    
    def _load_audio_file(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate."""
        try:
            # Load with librosa for consistent format
            audio_data, sample_rate = librosa.load(
                str(audio_path), 
                sr=None,  # Keep original sample rate
                mono=False,  # Keep stereo if present
                duration=self.max_duration  # Limit duration
            )
            
            # Ensure audio is in correct format
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)  # Make it (1, samples) for mono
            
            self.logger.debug(f"Loaded audio: {audio_data.shape} samples at {sample_rate}Hz")
            return audio_data, sample_rate
            
        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}")
            raise
    
    def _hide_data_lsb(self, audio_data: np.ndarray, data: bytes, 
                       password: str, sample_rate: int) -> Optional[np.ndarray]:
        """Hide data using LSB (Least Significant Bit) technique."""
        try:
            # Generate deterministic random sequence from password
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            
            # Convert data to bit array
            data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            total_bits = len(data_bits)
            
            # Prepare audio data for modification
            channels, samples = audio_data.shape
            modified_audio = audio_data.copy()
            
            # Convert to integer representation (16-bit)
            audio_int = (modified_audio * 32767).astype(np.int16)
            
            # Calculate available positions (use every nth sample)
            total_samples = channels * samples
            usable_samples = total_samples // self.sample_skip
            
            if total_bits > usable_samples:
                raise ValueError(f"Not enough samples for data: {total_bits} > {usable_samples}")
            
            # Generate random positions
            all_positions = np.arange(0, total_samples, self.sample_skip)
            selected_positions = rng.choice(all_positions, size=total_bits, replace=False)
            
            # Hide bits in LSBs
            flat_audio = audio_int.flatten()
            for i, pos in enumerate(selected_positions):
                bit_to_hide = data_bits[i]
                flat_audio[pos] = (flat_audio[pos] & 0xFFFE) | bit_to_hide
            
            # Reshape and convert back to float
            modified_int = flat_audio.reshape(channels, samples)
            modified_audio = modified_int.astype(np.float32) / 32767.0
            
            self.logger.info("Data successfully hidden using LSB technique")
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"LSB hiding failed: {e}")
            return None
    
    def _hide_data_spread_spectrum(self, audio_data: np.ndarray, data: bytes,
                                  password: str, sample_rate: int) -> Optional[np.ndarray]:
        """Hide data using spread spectrum technique."""
        try:
            # Generate spreading sequence from password
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            
            # Convert data to bit array
            data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            total_bits = len(data_bits)
            
            channels, samples = audio_data.shape
            modified_audio = audio_data.copy()
            
            # Parameters for spread spectrum
            chip_rate = 1000  # Chips per bit
            amplitude = 0.001  # Low amplitude to maintain quality
            
            # Check if we have enough space
            required_samples = total_bits * chip_rate
            if required_samples > samples:
                raise ValueError(f"Audio too short for spread spectrum: {required_samples} > {samples}")
            
            for channel in range(channels):
                for bit_idx, bit in enumerate(data_bits):
                    start_sample = bit_idx * chip_rate
                    end_sample = start_sample + chip_rate
                    
                    if end_sample > samples:
                        break
                    
                    # Generate spreading sequence for this bit
                    spread_seq = rng.uniform(-1, 1, chip_rate)
                    
                    # Modulate bit with spreading sequence
                    if bit == 1:
                        signal_to_add = spread_seq * amplitude
                    else:
                        signal_to_add = -spread_seq * amplitude
                    
                    # Add to audio (spread spectrum)
                    modified_audio[channel, start_sample:end_sample] += signal_to_add
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(modified_audio))
            if max_val > 1.0:
                modified_audio = modified_audio / max_val
            
            self.logger.info("Data successfully hidden using spread spectrum technique")
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"Spread spectrum hiding failed: {e}")
            return None
    
    def _hide_data_phase_coding(self, audio_data: np.ndarray, data: bytes,
                               password: str, sample_rate: int) -> Optional[np.ndarray]:
        """Hide data using phase coding technique."""
        try:
            # Convert data to bit array
            data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            total_bits = len(data_bits)
            
            channels, samples = audio_data.shape
            modified_audio = audio_data.copy()
            
            # Parameters for phase coding
            segment_length = 1024  # Samples per segment
            phase_shift = np.pi / 4  # 45 degree phase shift
            
            # Calculate number of segments
            num_segments = samples // segment_length
            
            if total_bits > num_segments * channels:
                raise ValueError(f"Not enough segments for phase coding: {total_bits} > {num_segments * channels}")
            
            # Process each channel
            for channel in range(channels):
                bit_idx = 0
                
                for seg_idx in range(num_segments):
                    if bit_idx >= total_bits:
                        break
                    
                    start = seg_idx * segment_length
                    end = start + segment_length
                    
                    # Get segment
                    segment = modified_audio[channel, start:end]
                    
                    # Apply FFT
                    fft_segment = np.fft.fft(segment)
                    
                    # Modify phase based on bit
                    if data_bits[bit_idx] == 1:
                        # Add phase shift
                        phase = np.angle(fft_segment)
                        magnitude = np.abs(fft_segment)
                        modified_fft = magnitude * np.exp(1j * (phase + phase_shift))
                    else:
                        # Subtract phase shift
                        phase = np.angle(fft_segment)
                        magnitude = np.abs(fft_segment)
                        modified_fft = magnitude * np.exp(1j * (phase - phase_shift))
                    
                    # Apply inverse FFT
                    modified_segment = np.real(np.fft.ifft(modified_fft))
                    modified_audio[channel, start:end] = modified_segment
                    
                    bit_idx += 1
            
            self.logger.info("Data successfully hidden using phase coding technique")
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"Phase coding hiding failed: {e}")
            return None
    
    def _extract_data_lsb(self, audio_data: np.ndarray, password: str, 
                          sample_rate: int) -> Optional[bytes]:
        """Extract data using LSB technique."""
        try:
            # Generate same random sequence used for hiding
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            
            # Convert to integer representation (same as hiding)
            audio_int = (audio_data * 32767).astype(np.int16)
            channels, samples = audio_int.shape
            
            # Calculate available positions (same logic as hiding)
            total_samples = channels * samples
            all_positions = np.arange(0, total_samples, self.sample_skip)
            flat_audio = audio_int.flatten()
            
            # We need to try different sizes since we don't know the data size yet
            # Start with a reasonable estimate and work up
            header_size = 34  # bytes
            header_bits = header_size * 8  # 272 bits
            
            if len(all_positions) < header_bits:
                return None
            
            # Try increasing sizes to find the right data
            for attempt_multiplier in [2, 5, 10, 20, 50]:
                try:
                    # Reset RNG for each attempt
                    rng = np.random.RandomState(seed)
                    
                    # Estimate total bits needed (header + some data)
                    estimated_bits = min(header_bits * attempt_multiplier, len(all_positions))
                    
                    # Generate positions for this attempt
                    selected_positions = rng.choice(all_positions, size=estimated_bits, replace=False)
                    
                    # Extract header bits
                    header_bits_data = []
                    for i in range(header_bits):
                        pos = selected_positions[i]
                        header_bits_data.append(flat_audio[pos] & 1)
                    
                    # Convert header bits to bytes
                    header_bit_array = np.array(header_bits_data, dtype=np.uint8)
                    header_bytes = np.packbits(header_bit_array)
                    
                    # Parse header
                    magic = bytes(header_bytes[:8])
                    if magic != self.MAGIC_HEADER:
                        continue  # Try next size
                    
                    version = bytes(header_bytes[8:10])
                    data_size = struct.unpack('<Q', bytes(header_bytes[10:18]))[0]
                    checksum = bytes(header_bytes[18:34])
                    
                    # Calculate actual total bits needed
                    data_bits_needed = data_size * 8
                    total_bits_needed = header_bits + data_bits_needed
                    
                    if total_bits_needed > len(all_positions):
                        continue  # Not enough space, try next size
                    
                    if total_bits_needed > estimated_bits:
                        # We need more bits, reset and get the correct amount
                        rng = np.random.RandomState(seed)
                        selected_positions = rng.choice(all_positions, size=total_bits_needed, replace=False)
                    
                    # Extract all bits
                    extracted_bits = []
                    for i in range(total_bits_needed):
                        pos = selected_positions[i]
                        extracted_bits.append(flat_audio[pos] & 1)
                    
                    # Convert to bytes
                    bit_array = np.array(extracted_bits, dtype=np.uint8)
                    all_bytes = np.packbits(bit_array)
                    
                    # Get encrypted data (skip header)
                    encrypted_data = bytes(all_bytes[header_size:header_size + data_size])
                    
                    # Verify checksum
                    actual_checksum = hashlib.md5(encrypted_data).digest()
                    if actual_checksum != checksum:
                        self.logger.debug(f"Checksum mismatch for attempt {attempt_multiplier}")
                        continue  # Try next size
                    
                    self.logger.info(f"Successfully extracted data on attempt {attempt_multiplier}")
                    return encrypted_data
                    
                except Exception as e:
                    self.logger.debug(f"Attempt {attempt_multiplier} failed: {e}")
                    continue
            
            # If all attempts failed
            return None
            
        except Exception as e:
            self.logger.error(f"LSB extraction failed: {e}")
            return None
    
    def _extract_data_spread_spectrum(self, audio_data: np.ndarray, password: str,
                                     sample_rate: int) -> Optional[bytes]:
        """Extract data using spread spectrum technique."""
        try:
            # This is a simplified version - full implementation would require
            # correlation analysis and synchronization
            self.logger.warning("Spread spectrum extraction not fully implemented")
            return None
            
        except Exception as e:
            self.logger.error(f"Spread spectrum extraction failed: {e}")
            return None
    
    def _extract_data_phase_coding(self, audio_data: np.ndarray, password: str,
                                  sample_rate: int) -> Optional[bytes]:
        """Extract data using phase coding technique."""
        try:
            # This is a simplified version - full implementation would require
            # phase analysis and synchronization
            self.logger.warning("Phase coding extraction not fully implemented")
            return None
            
        except Exception as e:
            self.logger.error(f"Phase coding extraction failed: {e}")
            return None
    
    def _save_audio_file(self, audio_data: np.ndarray, sample_rate: int, 
                        output_path: Path, quality: str) -> bool:
        """Save modified audio to file."""
        try:
            # Convert to the right format for pydub
            if audio_data.ndim == 2:
                # Stereo or multi-channel
                channels, samples = audio_data.shape
                
                if channels == 1:
                    # Mono
                    audio_int = (audio_data[0] * 32767).astype(np.int16)
                    audio_segment = AudioSegment(
                        audio_int.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=2,  # 16-bit
                        channels=1
                    )
                else:
                    # Stereo (take first 2 channels)
                    audio_left = (audio_data[0] * 32767).astype(np.int16)
                    audio_right = (audio_data[1] * 32767).astype(np.int16) if channels > 1 else audio_left
                    
                    # Interleave stereo channels
                    stereo_data = np.empty((audio_left.size + audio_right.size,), dtype=np.int16)
                    stereo_data[0::2] = audio_left
                    stereo_data[1::2] = audio_right
                    
                    audio_segment = AudioSegment(
                        stereo_data.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=2,  # 16-bit
                        channels=2
                    )
            else:
                # 1D array (mono)
                audio_int = (audio_data * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    audio_int.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,  # 16-bit
                    channels=1
                )
            
            # Set export parameters based on quality
            if quality == 'high':
                bitrate = "320k"
            elif quality == 'medium':
                bitrate = "192k"
            else:  # low
                bitrate = "128k"
            
            # Determine format from file extension
            file_format = output_path.suffix.lower().lstrip('.')
            if file_format == 'mp3':
                audio_segment.export(str(output_path), format="mp3", bitrate=bitrate)
            elif file_format == 'wav':
                audio_segment.export(str(output_path), format="wav")
            elif file_format == 'flac':
                audio_segment.export(str(output_path), format="flac")
            else:
                # Default to wav for unknown formats
                audio_segment.export(str(output_path), format="wav")
            
            return output_path.exists()
            
        except Exception as e:
            self.logger.error(f"Failed to save audio file: {e}")
            return False
    
    def _extract_and_decrypt_data(self, encrypted_data: bytes, password: str) -> bytes:
        """Decrypt extracted data."""
        try:
            # Decrypt the data
            original_data = self.encryption_engine.decrypt_with_metadata(encrypted_data, password)
            
            self.logger.debug(f"Data decrypted successfully: {len(original_data)} bytes")
            return original_data
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def calculate_capacity(self, audio_path: Path) -> int:
        """Calculate audio steganography capacity in bytes."""
        try:
            analysis = self.analyzer.analyze_audio_file(audio_path)
            return analysis.get('capacity_bytes', 0)
        except Exception:
            return 0
    
    def validate_audio_format(self, audio_path: Path) -> bool:
        """Validate if audio format is supported."""
        return self.analyzer.is_audio_file(audio_path)
