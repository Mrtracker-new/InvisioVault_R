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
            self.logger.info(f"Using technique: {technique.upper()}")
            
            # Validate input
            if not audio_path.exists():
                self._log_extraction_error("File not found", {
                    'file_path': str(audio_path),
                    'technique': technique,
                    'error_type': 'file_not_found'
                })
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if not self.analyzer.is_audio_file(audio_path):
                self._log_extraction_error("Unsupported file format", {
                    'file_path': str(audio_path),
                    'file_extension': audio_path.suffix,
                    'technique': technique,
                    'error_type': 'unsupported_format'
                })
                raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
            
            # Load and analyze audio file
            audio_data, sample_rate = self._load_audio_file(audio_path)
            file_info = {
                'sample_rate': sample_rate,
                'channels': audio_data.shape[0],
                'samples': audio_data.shape[1],
                'duration': audio_data.shape[1] / sample_rate,
                'format': audio_path.suffix.lower()
            }
            self.logger.info(f"Audio file info: {file_info}")
            
            # Warn about lossy format extraction
            lossy_formats = ['.mp3', '.aac', '.ogg', '.m4a', '.wma']
            if audio_path.suffix.lower() in lossy_formats:
                self.logger.warning(
                    f"WARNING: Attempting to extract from lossy format {audio_path.suffix.upper()}. "
                    f"LSB steganography data is often destroyed by lossy compression. "
                    f"If extraction fails, try:\n"
                    f"1. Use spread_spectrum or phase_coding techniques (more resilient to compression)\n"
                    f"2. Use the original lossless file (WAV/FLAC) if available\n"
                    f"3. Verify the file wasn't converted to {audio_path.suffix.upper()} after hiding data"
                )
            
            # Extract data using appropriate technique
            if technique == 'lsb':
                encrypted_data = self._extract_data_lsb(audio_data, password, sample_rate)
            elif technique == 'spread_spectrum':
                encrypted_data = self._extract_data_spread_spectrum(audio_data, password, sample_rate)
            elif technique == 'phase_coding':
                encrypted_data = self._extract_data_phase_coding(audio_data, password, sample_rate)
            else:
                self._log_extraction_error("Unsupported technique", {
                    'technique': technique,
                    'supported_techniques': ['lsb', 'spread_spectrum', 'phase_coding'],
                    'error_type': 'unsupported_technique'
                })
                raise ValueError(f"Unsupported technique: {technique}")
            
            if not encrypted_data:
                self._log_extraction_error("No hidden data found", {
                    'file_info': file_info,
                    'technique': technique,
                    'error_type': 'no_data_found',
                    'possible_causes': [
                        'Wrong extraction technique (try lsb, spread_spectrum, or phase_coding)',
                        'Incorrect password',
                        'File doesn\'t contain hidden data',
                        'Audio file was modified after data hiding (compression, format change)',
                        'Audio file is corrupted or truncated'
                    ]
                })
                raise Exception("No hidden data found in audio")
            
            # Decrypt and verify data
            try:
                original_data = self._extract_and_decrypt_data(encrypted_data, password)
                self.logger.info(f"Audio data extraction completed successfully: {len(original_data)} bytes")
                return original_data
            except Exception as decrypt_error:
                self._log_extraction_error("Decryption failed", {
                    'file_info': file_info,
                    'technique': technique,
                    'encrypted_data_size': len(encrypted_data),
                    'error_type': 'decryption_failed',
                    'decrypt_error': str(decrypt_error),
                    'possible_causes': [
                        'Incorrect password',
                        'Data corruption during extraction',
                        'Wrong extraction technique used',
                        'Audio file was modified after hiding'
                    ]
                })
                raise
            
        except Exception as e:
            if "extraction failed" not in str(e).lower():
                self._log_extraction_error("General extraction error", {
                    'file_path': str(audio_path) if audio_path else 'unknown',
                    'technique': technique,
                    'error': str(e),
                    'error_type': 'general_error'
                })
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
            self.logger.info(f"LSB hiding - using seed: {seed}")
            rng = np.random.RandomState(seed)
            
            # Convert data to bit array
            data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            total_bits = len(data_bits)
            self.logger.info(f"Hiding {len(data)} bytes ({total_bits} bits)")
            
            # Log first few bytes of data for debugging
            first_bytes = data[:20] if len(data) >= 20 else data
            self.logger.debug(f"First 20 bytes of data to hide: {first_bytes.hex()}")
            
            # Prepare audio data for modification
            channels, samples = audio_data.shape
            self.logger.info(f"Audio shape for hiding: {channels} channels, {samples} samples")
            modified_audio = audio_data.copy()
            
            # Convert to integer representation (16-bit)
            audio_int = (modified_audio * 32767).astype(np.int16)
            
            # Calculate available positions (use every nth sample)
            total_samples = channels * samples
            usable_samples = total_samples // self.sample_skip
            self.logger.info(f"Total samples: {total_samples}, Usable samples: {usable_samples}")
            
            if total_bits > usable_samples:
                raise ValueError(f"Not enough samples for data: {total_bits} > {usable_samples}")
            
            # Generate random positions
            all_positions = np.arange(0, total_samples, self.sample_skip)
            selected_positions = rng.choice(all_positions, size=total_bits, replace=False)
            self.logger.debug(f"Generated {len(selected_positions)} positions for hiding")
            
            # Hide bits in LSBs
            flat_audio = audio_int.flatten()
            bits_hidden = 0
            for i, pos in enumerate(selected_positions):
                bit_to_hide = data_bits[i]
                original_value = flat_audio[pos]
                flat_audio[pos] = (flat_audio[pos] & 0xFFFE) | bit_to_hide
                bits_hidden += 1
                
                # Log first few bit modifications for debugging
                if i < 10:
                    self.logger.debug(f"Position {pos}: {original_value} -> {flat_audio[pos]} (bit: {bit_to_hide})")
            
            self.logger.info(f"Successfully hid {bits_hidden} bits in audio")
            
            # Reshape and convert back to float
            modified_int = flat_audio.reshape(channels, samples)
            modified_audio = modified_int.astype(np.float32) / 32767.0
            
            self.logger.info("Data successfully hidden using LSB technique")
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"LSB hiding failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
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
        """Extract data using LSB technique - exactly mirrors hiding process."""
        try:
            self.logger.info(f"Starting LSB extraction from audio file")
            
            # Generate same random sequence used for hiding
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            self.logger.info(f"Using seed: {seed}")
            
            # Ensure audio data is in the right format
            if audio_data.ndim == 1:
                # Convert mono to 2D format
                audio_data = audio_data.reshape(1, -1)
            
            # Convert to integer representation (same as hiding)
            audio_int = (audio_data * 32767).astype(np.int16)
            channels, samples = audio_int.shape
            self.logger.info(f"Audio shape: {channels} channels, {samples} samples")
            
            # Calculate available positions (same logic as hiding)
            total_samples = channels * samples
            all_positions = np.arange(0, total_samples, self.sample_skip)
            flat_audio = audio_int.flatten()
            self.logger.info(f"Total samples: {total_samples}, Available positions: {len(all_positions)}")
            
            # Check minimum requirements for header (34 bytes = 272 bits)
            min_header_bits = 34 * 8  # 272 bits
            if len(all_positions) < min_header_bits:
                self.logger.error(f"Not enough positions for header: {len(all_positions)} < {min_header_bits}")
                return None
            
            # Try multiple extraction strategies for robustness
            extraction_strategies = [
                int(len(all_positions) * 0.9),   # 90% of available positions
                int(len(all_positions) * 0.8),   # 80% of available positions
                int(len(all_positions) * 0.7),   # 70% of available positions
                len(all_positions)                # All available positions
            ]
            
            for strategy_idx, max_extract_bits in enumerate(extraction_strategies):
                try:
                    self.logger.info(f"Trying extraction strategy {strategy_idx + 1}: {max_extract_bits} bits")
                    
                    if max_extract_bits < min_header_bits:
                        continue
                    
                    # Generate positions using same algorithm as hiding
                    rng = np.random.RandomState(seed)
                    
                    # Ensure we don't try to extract more positions than available
                    actual_extract_bits = min(max_extract_bits, len(all_positions))
                    selected_positions = rng.choice(all_positions, size=actual_extract_bits, replace=False)
                    
                    # Extract all bits
                    extracted_bits = []
                    for pos in selected_positions:
                        if pos < len(flat_audio):  # Bounds check
                            extracted_bits.append(flat_audio[pos] & 1)
                        else:
                            self.logger.warning(f"Position {pos} out of bounds for audio length {len(flat_audio)}")
                    
                    if len(extracted_bits) < min_header_bits:
                        self.logger.warning(f"Not enough bits extracted: {len(extracted_bits)} < {min_header_bits}")
                        continue
                    
                    # Convert to bytes with proper padding
                    bit_array = np.array(extracted_bits, dtype=np.uint8)
                    
                    # Ensure bit array length is multiple of 8 for proper byte packing
                    padding_needed = (8 - (len(bit_array) % 8)) % 8
                    if padding_needed > 0:
                        bit_array = np.append(bit_array, np.zeros(padding_needed, dtype=np.uint8))
                    
                    all_bytes = np.packbits(bit_array)
                    
                    self.logger.info(f"Extracted {len(all_bytes)} bytes, searching for header...")
                    
                    # Look for magic header in the extracted data
                    header_found = False
                    data_start_offset = 0
                    
                    # Try different starting offsets in case there's alignment issues
                    search_range = min(200, len(all_bytes) - 34)  # Increased search range
                    for offset in range(0, max(1, search_range)):
                        if offset + 34 > len(all_bytes):
                            break
                            
                        potential_magic = bytes(all_bytes[offset:offset + 8])
                        
                        if potential_magic == self.MAGIC_HEADER:
                            self.logger.info(f"Found magic header at offset {offset}")
                            header_found = True
                            data_start_offset = offset
                            break
                    
                    if not header_found:
                        self.logger.debug(f"Strategy {strategy_idx + 1}: Magic header not found")
                        # Log first few bytes for debugging on last strategy
                        if strategy_idx == len(extraction_strategies) - 1:
                            first_bytes = bytes(all_bytes[:50]) if len(all_bytes) >= 50 else bytes(all_bytes)
                            self.logger.debug(f"First 50 bytes: {first_bytes.hex()}")
                        continue
                    
                    # Parse header starting from found offset
                    header_start = data_start_offset
                    try:
                        magic = bytes(all_bytes[header_start:header_start + 8])
                        version = bytes(all_bytes[header_start + 8:header_start + 10])
                        
                        # Safely unpack data size
                        if header_start + 18 > len(all_bytes):
                            self.logger.warning(f"Header incomplete at offset {header_start}")
                            continue
                            
                        data_size_bytes = bytes(all_bytes[header_start + 10:header_start + 18])
                        if len(data_size_bytes) != 8:
                            self.logger.warning(f"Invalid data size bytes: {len(data_size_bytes)} != 8")
                            continue
                            
                        data_size = struct.unpack('<Q', data_size_bytes)[0]
                        
                        # Validate data size is reasonable
                        if data_size <= 0 or data_size > 100 * 1024 * 1024:  # Max 100MB
                            self.logger.warning(f"Suspicious data size: {data_size} bytes")
                            continue
                        
                        if header_start + 34 > len(all_bytes):
                            self.logger.warning(f"Checksum incomplete at offset {header_start}")
                            continue
                            
                        checksum = bytes(all_bytes[header_start + 18:header_start + 34])
                        
                        self.logger.info(f"Header parsed - version: {version.hex()}, data size: {data_size} bytes")
                        
                        # Extract the actual encrypted data
                        data_start = header_start + 34
                        data_end = data_start + data_size
                        
                        if data_end > len(all_bytes):
                            self.logger.warning(
                                f"Not enough data extracted for payload: need {data_end}, have {len(all_bytes)}. "
                                f"May need to extract more bits."
                            )
                            continue
                        
                        encrypted_data = bytes(all_bytes[data_start:data_end])
                        
                        # Verify checksum
                        actual_checksum = hashlib.md5(encrypted_data).digest()
                        if actual_checksum != checksum:
                            self.logger.warning(
                                f"Checksum mismatch: expected {checksum.hex()}, got {actual_checksum.hex()}"
                            )
                            continue
                        
                        self.logger.info(
                            f"Successfully extracted and verified {len(encrypted_data)} bytes "
                            f"of encrypted data using strategy {strategy_idx + 1}"
                        )
                        return encrypted_data
                        
                    except (struct.error, ValueError) as parse_error:
                        self.logger.warning(f"Header parsing error at offset {header_start}: {parse_error}")
                        continue
                        
                except Exception as strategy_error:
                    self.logger.warning(f"Strategy {strategy_idx + 1} failed: {strategy_error}")
                    continue
            
            # If all strategies failed
            self.logger.error(
                "All extraction strategies failed. Possible reasons:\n"
                "- Wrong password\n"
                "- File doesn't contain hidden data\n"
                "- Audio format was changed after hiding (e.g., compressed)\n"
                "- Different extraction technique needed"
            )
            return None
            
        except Exception as e:
            self.logger.error(f"LSB extraction failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _extract_data_spread_spectrum(self, audio_data: np.ndarray, password: str,
                                     sample_rate: int) -> Optional[bytes]:
        """Extract data using spread spectrum technique."""
        try:
            self.logger.info("Starting spread spectrum extraction")
            
            # Generate same spreading sequence used for hiding
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            
            channels, samples = audio_data.shape
            
            # Parameters must match hiding process exactly
            chip_rate = 1000  # Chips per bit
            amplitude = 0.001  # Low amplitude to maintain quality
            
            # Estimate maximum possible data length
            max_bits = samples // chip_rate
            
            # Start with header size (34 bytes = 272 bits) and expand as needed
            min_header_bits = 34 * 8
            
            # Try to extract data by correlating with known spreading sequences
            extracted_bits = []
            
            for bit_idx in range(max_bits):
                start_sample = bit_idx * chip_rate
                end_sample = start_sample + chip_rate
                
                if end_sample > samples:
                    break
                
                # Extract segment from first channel (averaging if multi-channel)
                if channels == 1:
                    segment = audio_data[0, start_sample:end_sample]
                else:
                    # Average across channels
                    segment = np.mean(audio_data[:, start_sample:end_sample], axis=0)
                
                # Generate the same spreading sequence used during hiding
                spread_seq = rng.uniform(-1, 1, chip_rate)
                
                # Correlate with spreading sequence to extract bit
                correlation = np.correlate(segment, spread_seq, mode='valid')[0]
                
                # Determine bit based on correlation sign
                if correlation > 0:
                    extracted_bits.append(1)
                else:
                    extracted_bits.append(0)
                
                # Stop early if we have enough for analysis
                if len(extracted_bits) >= min_header_bits + 100:  # Header + some data
                    break
            
            if len(extracted_bits) < min_header_bits:
                self.logger.error(f"Not enough bits extracted: {len(extracted_bits)} < {min_header_bits}")
                return None
            
            # Convert bits to bytes
            bit_array = np.array(extracted_bits, dtype=np.uint8)
            
            # Pad to byte boundary
            padding_needed = (8 - (len(bit_array) % 8)) % 8
            if padding_needed > 0:
                bit_array = np.append(bit_array, np.zeros(padding_needed, dtype=np.uint8))
            
            all_bytes = np.packbits(bit_array)
            
            # Look for magic header
            for offset in range(min(100, len(all_bytes) - 34)):
                if offset + 34 > len(all_bytes):
                    break
                    
                potential_magic = bytes(all_bytes[offset:offset + 8])
                
                if potential_magic == self.MAGIC_HEADER:
                    self.logger.info(f"Found magic header at offset {offset}")
                    
                    # Parse header
                    try:
                        version = bytes(all_bytes[offset + 8:offset + 10])
                        data_size_bytes = bytes(all_bytes[offset + 10:offset + 18])
                        data_size = struct.unpack('<Q', data_size_bytes)[0]
                        checksum = bytes(all_bytes[offset + 18:offset + 34])
                        
                        # Validate data size
                        if data_size <= 0 or data_size > 100 * 1024 * 1024:
                            continue
                        
                        # Extract encrypted data
                        data_start = offset + 34
                        data_end = data_start + data_size
                        
                        if data_end > len(all_bytes):
                            # Need to extract more bits
                            additional_bits_needed = (data_end - len(all_bytes)) * 8
                            self.logger.info(f"Need {additional_bits_needed} more bits, extracting...")
                            
                            # Continue extraction from where we left off
                            current_bit_idx = len(extracted_bits)
                            while len(extracted_bits) < (data_end * 8) and current_bit_idx < max_bits:
                                start_sample = current_bit_idx * chip_rate
                                end_sample = start_sample + chip_rate
                                
                                if end_sample > samples:
                                    break
                                
                                if channels == 1:
                                    segment = audio_data[0, start_sample:end_sample]
                                else:
                                    segment = np.mean(audio_data[:, start_sample:end_sample], axis=0)
                                
                                spread_seq = rng.uniform(-1, 1, chip_rate)
                                correlation = np.correlate(segment, spread_seq, mode='valid')[0]
                                
                                if correlation > 0:
                                    extracted_bits.append(1)
                                else:
                                    extracted_bits.append(0)
                                
                                current_bit_idx += 1
                            
                            # Repack bits to bytes
                            bit_array = np.array(extracted_bits, dtype=np.uint8)
                            padding_needed = (8 - (len(bit_array) % 8)) % 8
                            if padding_needed > 0:
                                bit_array = np.append(bit_array, np.zeros(padding_needed, dtype=np.uint8))
                            all_bytes = np.packbits(bit_array)
                            
                            if data_end > len(all_bytes):
                                self.logger.warning(f"Still not enough data after additional extraction")
                                continue
                        
                        encrypted_data = bytes(all_bytes[data_start:data_end])
                        
                        # Verify checksum
                        actual_checksum = hashlib.md5(encrypted_data).digest()
                        if actual_checksum != checksum:
                            self.logger.warning("Checksum mismatch")
                            continue
                        
                        self.logger.info(f"Successfully extracted {len(encrypted_data)} bytes using spread spectrum")
                        return encrypted_data
                        
                    except (struct.error, ValueError) as e:
                        self.logger.warning(f"Header parsing error: {e}")
                        continue
            
            self.logger.error("No valid data found using spread spectrum extraction")
            return None
            
        except Exception as e:
            self.logger.error(f"Spread spectrum extraction failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _extract_data_phase_coding(self, audio_data: np.ndarray, password: str,
                                  sample_rate: int) -> Optional[bytes]:
        """Extract data using phase coding technique."""
        try:
            self.logger.info("Starting phase coding extraction")
            
            channels, samples = audio_data.shape
            
            # Parameters must match hiding process exactly
            segment_length = 1024  # Samples per segment
            phase_shift = np.pi / 4  # 45 degree phase shift
            
            # Calculate number of segments
            num_segments = samples // segment_length
            
            # Start with header size (34 bytes = 272 bits) and expand as needed
            min_header_bits = 34 * 8
            
            # Maximum bits we can extract
            max_bits = num_segments * channels
            
            if max_bits < min_header_bits:
                self.logger.error(f"Not enough segments for header: {max_bits} < {min_header_bits}")
                return None
            
            # Extract bits by analyzing phase differences
            extracted_bits = []
            
            # Process each channel
            for channel in range(channels):
                for seg_idx in range(num_segments):
                    if len(extracted_bits) >= max_bits:
                        break
                        
                    start = seg_idx * segment_length
                    end = start + segment_length
                    
                    # Get segment
                    segment = audio_data[channel, start:end]
                    
                    # Apply FFT
                    fft_segment = np.fft.fft(segment)
                    
                    # Calculate average phase in relevant frequency range
                    # Use middle frequencies to avoid DC and high-frequency noise
                    freq_start = len(fft_segment) // 4
                    freq_end = 3 * len(fft_segment) // 4
                    
                    phases = np.angle(fft_segment[freq_start:freq_end])
                    avg_phase = np.mean(phases)
                    
                    # Determine bit based on phase deviation
                    # During hiding, we added/subtracted phase_shift
                    # Check if the average phase suggests a positive or negative shift
                    if np.abs(avg_phase) > phase_shift / 2:
                        if avg_phase > 0:
                            extracted_bits.append(1)
                        else:
                            extracted_bits.append(0)
                    else:
                        # Phase is near zero, could be either - use a different approach
                        # Look at phase variance to determine bit
                        phase_variance = np.var(phases)
                        if phase_variance > np.pi / 8:  # Higher variance suggests modification
                            extracted_bits.append(1)
                        else:
                            extracted_bits.append(0)
                    
                    # Stop early if we have enough for analysis
                    if len(extracted_bits) >= min_header_bits + 100:  # Header + some data
                        break
                
                if len(extracted_bits) >= min_header_bits + 100:
                    break
            
            if len(extracted_bits) < min_header_bits:
                self.logger.error(f"Not enough bits extracted: {len(extracted_bits)} < {min_header_bits}")
                return None
            
            # Convert bits to bytes
            bit_array = np.array(extracted_bits, dtype=np.uint8)
            
            # Pad to byte boundary
            padding_needed = (8 - (len(bit_array) % 8)) % 8
            if padding_needed > 0:
                bit_array = np.append(bit_array, np.zeros(padding_needed, dtype=np.uint8))
            
            all_bytes = np.packbits(bit_array)
            
            # Look for magic header
            for offset in range(min(100, len(all_bytes) - 34)):
                if offset + 34 > len(all_bytes):
                    break
                    
                potential_magic = bytes(all_bytes[offset:offset + 8])
                
                if potential_magic == self.MAGIC_HEADER:
                    self.logger.info(f"Found magic header at offset {offset}")
                    
                    # Parse header
                    try:
                        version = bytes(all_bytes[offset + 8:offset + 10])
                        data_size_bytes = bytes(all_bytes[offset + 10:offset + 18])
                        data_size = struct.unpack('<Q', data_size_bytes)[0]
                        checksum = bytes(all_bytes[offset + 18:offset + 34])
                        
                        # Validate data size
                        if data_size <= 0 or data_size > 100 * 1024 * 1024:
                            continue
                        
                        # Extract encrypted data
                        data_start = offset + 34
                        data_end = data_start + data_size
                        
                        if data_end > len(all_bytes):
                            # Need to extract more bits
                            additional_bits_needed = (data_end - len(all_bytes)) * 8
                            self.logger.info(f"Need {additional_bits_needed} more bits, extracting...")
                            
                            # Continue extraction from where we left off
                            current_seg_idx = len(extracted_bits)
                            current_channel = 0
                            
                            while len(extracted_bits) < (data_end * 8) and current_seg_idx < max_bits:
                                # Calculate which channel and segment we're in
                                actual_seg_idx = current_seg_idx % num_segments
                                current_channel = current_seg_idx // num_segments
                                
                                if current_channel >= channels:
                                    break
                                
                                start = actual_seg_idx * segment_length
                                end = start + segment_length
                                
                                if end > samples:
                                    break
                                
                                segment = audio_data[current_channel, start:end]
                                fft_segment = np.fft.fft(segment)
                                
                                freq_start = len(fft_segment) // 4
                                freq_end = 3 * len(fft_segment) // 4
                                phases = np.angle(fft_segment[freq_start:freq_end])
                                avg_phase = np.mean(phases)
                                
                                if np.abs(avg_phase) > phase_shift / 2:
                                    if avg_phase > 0:
                                        extracted_bits.append(1)
                                    else:
                                        extracted_bits.append(0)
                                else:
                                    phase_variance = np.var(phases)
                                    if phase_variance > np.pi / 8:
                                        extracted_bits.append(1)
                                    else:
                                        extracted_bits.append(0)
                                
                                current_seg_idx += 1
                            
                            # Repack bits to bytes
                            bit_array = np.array(extracted_bits, dtype=np.uint8)
                            padding_needed = (8 - (len(bit_array) % 8)) % 8
                            if padding_needed > 0:
                                bit_array = np.append(bit_array, np.zeros(padding_needed, dtype=np.uint8))
                            all_bytes = np.packbits(bit_array)
                            
                            if data_end > len(all_bytes):
                                self.logger.warning(f"Still not enough data after additional extraction")
                                continue
                        
                        encrypted_data = bytes(all_bytes[data_start:data_end])
                        
                        # Verify checksum
                        actual_checksum = hashlib.md5(encrypted_data).digest()
                        if actual_checksum != checksum:
                            self.logger.warning("Checksum mismatch")
                            continue
                        
                        self.logger.info(f"Successfully extracted {len(encrypted_data)} bytes using phase coding")
                        return encrypted_data
                        
                    except (struct.error, ValueError) as e:
                        self.logger.warning(f"Header parsing error: {e}")
                        continue
            
            self.logger.error("No valid data found using phase coding extraction")
            return None
            
        except Exception as e:
            self.logger.error(f"Phase coding extraction failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
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
            
            # Get the requested output format
            output_format = output_path.suffix.lower().lstrip('.')
            
            # Define lossless and lossy formats
            lossless_formats = {'.wav', '.flac', '.aiff', '.au'}
            lossy_formats = {'.mp3', '.aac', '.ogg', '.m4a', '.wma'}
            
            # Warn about format choice for steganography
            if output_path.suffix.lower() not in lossless_formats:
                if output_path.suffix.lower() in lossy_formats:
                    self.logger.warning(
                        f"WARNING: Saving to lossy format {output_format.upper()}. "
                        "LSB steganography data may be corrupted or lost during compression. "
                        "For best results, use lossless formats like WAV or FLAC."
                    )
                else:
                    # Unknown format, default to WAV for safety
                    self.logger.warning(
                        f"Unknown audio format '{output_format}'. "
                        "Defaulting to WAV format to preserve steganography data."
                    )
                    output_path = output_path.with_suffix('.wav')
                    output_format = 'wav'
            else:
                self.logger.info(f"Using lossless format {output_format.upper()} - good choice for steganography!")
            
            # Set export parameters based on format and quality
            export_params = {}
            
            if output_format in ['wav', 'flac', 'aiff', 'au']:
                # Lossless formats - no compression parameters needed
                if output_format == 'flac':
                    # FLAC specific parameters
                    if quality == 'high':
                        export_params['parameters'] = ["-compression_level", "8"]
                    elif quality == 'medium':
                        export_params['parameters'] = ["-compression_level", "5"]
                    else:  # low (faster compression)
                        export_params['parameters'] = ["-compression_level", "0"]
            else:
                # Lossy formats - set bitrate
                if quality == 'high':
                    export_params['bitrate'] = "320k"
                elif quality == 'medium':
                    export_params['bitrate'] = "192k"
                else:  # low
                    export_params['bitrate'] = "128k"
            
            # Export in the requested format
            audio_segment.export(str(output_path), format=output_format, **export_params)
            
            # Verify file was created
            if output_path.exists():
                self.logger.info(f"Audio successfully saved as {output_format.upper()}: {output_path.name}")
                return True
            else:
                self.logger.error(f"Failed to create output file: {output_path}")
                return False
            
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
    
    def _log_extraction_error(self, error_message: str, error_details: Dict[str, Any]) -> None:
        """Log detailed extraction error information for debugging."""
        self.logger.error(f"AUDIO EXTRACTION ERROR: {error_message}")
        
        # Log error details in a structured format
        for key, value in error_details.items():
            if isinstance(value, list):
                self.logger.error(f"  {key}:")
                for item in value:
                    self.logger.error(f"    - {item}")
            else:
                self.logger.error(f"  {key}: {value}")
        
        # Add troubleshooting suggestions based on error type
        error_type = error_details.get('error_type', 'unknown')
        
        if error_type == 'no_data_found':
            self.logger.error("\nTROUBLESHOOTING SUGGESTIONS:")
            self.logger.error("1. Try different extraction techniques: lsb, spread_spectrum, phase_coding")
            self.logger.error("2. Verify the password is correct")
            self.logger.error("3. Ensure this audio file contains hidden data")
            self.logger.error("4. Check if the audio file was modified after hiding data")
            self.logger.error("5. Use the same audio format that was used for hiding")
            
        elif error_type == 'decryption_failed':
            self.logger.error("\nTROUBLESHOOTING SUGGESTIONS:")
            self.logger.error("1. Double-check the password")
            self.logger.error("2. Verify you're using the same extraction technique as was used for hiding")
            self.logger.error("3. Ensure the audio file hasn't been compressed or converted")
            
        elif error_type == 'unsupported_format':
            self.logger.error("\nTROUBLESHOOTING SUGGESTIONS:")
            self.logger.error("1. Convert the audio file to a supported format (WAV, FLAC, MP3, etc.)")
            self.logger.error("2. Use a lossless format (WAV, FLAC) for best steganography compatibility")
            
        elif error_type == 'unsupported_technique':
            technique = error_details.get('technique', 'unknown')
            supported = error_details.get('supported_techniques', [])
            self.logger.error("\nTROUBLESHOOTING SUGGESTIONS:")
            self.logger.error(f"1. Technique '{technique}' is not supported")
            self.logger.error(f"2. Use one of these supported techniques: {', '.join(supported)}")
            
        self.logger.error("\n" + "="*50)
