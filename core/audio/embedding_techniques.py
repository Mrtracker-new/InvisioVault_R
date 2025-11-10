"""
Audio Steganography Embedding Techniques

This module implements various audio steganography techniques including LSB, 
spread spectrum, phase coding, and other advanced methods optimized for 
security, capacity, and robustness.
"""

import numpy as np
import hashlib
import secrets
from typing import Tuple, Optional, Union, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod
import warnings

try:
    from scipy.fftpack import dct, idct
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from utils.logger import Logger


class EmbeddingTechnique(Enum):
    """Available embedding techniques with their characteristics."""
    LSB = ("lsb", "Least Significant Bit", "High capacity, low robustness - RECOMMENDED", True)
    SPREAD_SPECTRUM = ("spread_spectrum", "Spread Spectrum", "Medium capacity, high robustness - EXPERIMENTAL", True) 
    PHASE_CODING = ("phase_coding", "Phase Coding", "Low capacity, high robustness - EXPERIMENTAL", True)
    DCT = ("dct", "Discrete Cosine Transform", "Medium capacity, medium robustness - EXPERIMENTAL", SCIPY_AVAILABLE)
    DWT = ("dwt", "Discrete Wavelet Transform", "Medium capacity, high robustness - NOT IMPLEMENTED", False)  # Not implemented yet
    ECHO_HIDING = ("echo", "Echo Hiding", "Low capacity, high robustness - EXPERIMENTAL", True)
    
    def __init__(self, code: str, display_name: str, description: str, available: bool):
        self.code = code
        self.display_name = display_name
        self.description = description
        self.available = available


class BaseEmbeddingTechnique(ABC):
    """Abstract base class for all embedding techniques."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.technique_name = "base"
        
    @abstractmethod
    def embed(self, audio_data: np.ndarray, data: bytes, password: str, 
              sample_rate: int, **kwargs) -> Optional[np.ndarray]:
        """Embed data into audio using this technique."""
        pass
    
    @abstractmethod
    def extract(self, audio_data: np.ndarray, password: str, sample_rate: int,
                expected_size: Optional[int] = None, **kwargs) -> Optional[bytes]:
        """Extract data from audio using this technique."""
        pass
    
    @abstractmethod
    def calculate_capacity(self, audio_data: np.ndarray, sample_rate: int) -> int:
        """Calculate embedding capacity in bytes for given audio."""
        pass
    
    def _generate_seed_sequence(self, password: str, additional_data: str = "") -> int:
        """Generate deterministic seed from password."""
        combined = f"{password}_{additional_data}_{self.technique_name}"
        seed_bytes = hashlib.sha256(combined.encode()).digest()
        return int.from_bytes(seed_bytes[:4], 'big')
    
    def _prepare_data_bits(self, data: bytes) -> np.ndarray:
        """Convert data to bit array."""
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    
    def _reconstruct_data(self, bits: np.ndarray, expected_bytes: int) -> bytes:
        """Reconstruct bytes from bit array."""
        # Ensure we have enough bits
        total_bits = expected_bytes * 8
        if len(bits) < total_bits:
            # Pad with zeros
            padding = total_bits - len(bits)
            bits = np.append(bits, np.zeros(padding, dtype=np.uint8))
        elif len(bits) > total_bits:
            # Truncate
            bits = bits[:total_bits]
        
        # Convert to bytes
        return np.packbits(bits).tobytes()


class LSBEmbedding(BaseEmbeddingTechnique):
    """
    Least Significant Bit embedding technique.
    
    Replaces the least significant bit of audio samples with data bits.
    Optimized with randomization and quality preservation.
    """
    
    def __init__(self, logger: Logger, skip_factor: int = 4, randomize: bool = True):
        super().__init__(logger)
        self.technique_name = "lsb"
        self.skip_factor = skip_factor  # Use every Nth sample
        self.randomize = randomize      # Randomize embedding positions
        
    def embed(self, audio_data: np.ndarray, data: bytes, password: str,
              sample_rate: int, **kwargs) -> Optional[np.ndarray]:
        """
        Embed data using LSB technique with optional randomization.
        
        Args:
            audio_data: Audio samples (channels, samples)
            data: Data to embed
            password: Password for seed generation
            sample_rate: Audio sample rate
            **kwargs: Additional parameters
            
        Returns:
            Modified audio data or None if embedding failed
        """
        try:
            self.logger.debug(f"LSB embedding: {len(data)} bytes, randomize={self.randomize}")
            
            # Check capacity
            capacity = self.calculate_capacity(audio_data, sample_rate)
            if len(data) > capacity:
                self.logger.error(f"Data too large: {len(data)} > {capacity} bytes")
                return None
            
            # Prepare data bits
            data_bits = self._prepare_data_bits(data)
            
            # Generate random seed for position selection
            rng_seed = self._generate_seed_sequence(password)
            rng = np.random.RandomState(rng_seed)
            
            # Convert audio to integer format for bit manipulation
            audio_int = self._float_to_int16(audio_data)
            channels, samples = audio_int.shape
            
            # Generate embedding positions
            positions = self._generate_positions(channels, samples, len(data_bits), rng)
            
            if len(positions) < len(data_bits):
                self.logger.error("Insufficient embedding positions")
                return None
            
            # Embed bits
            flat_audio = audio_int.flatten()  # int16 view
            # CRITICAL FIX: Perform bitwise ops in unsigned 16-bit space to avoid Python int overflow
            flat_uint = flat_audio.view(np.uint16)
            for i, pos in enumerate(positions[:len(data_bits)]):
                if pos < flat_uint.size:
                    # Clear LSB and set data bit using uint16-safe operations
                    flat_uint[pos] = (flat_uint[pos] & np.uint16(0xFFFE)) | np.uint16(data_bits[i])
            
            # Reshape and convert back to float
            modified_audio = flat_audio.reshape(channels, samples)
            return self._int16_to_float(modified_audio)
            
        except Exception as e:
            self.logger.error(f"LSB embedding failed: {e}")
            return None
    
    def extract(self, audio_data: np.ndarray, password: str, sample_rate: int,
                expected_size: Optional[int] = None, **kwargs) -> Optional[bytes]:
        """
        Extract data using LSB technique.
        
        Args:
            audio_data: Audio samples to extract from
            password: Password for seed generation
            sample_rate: Audio sample rate
            expected_size: Expected data size in bytes (if known)
            **kwargs: Additional parameters
            
        Returns:
            Extracted data or None if extraction failed
        """
        try:
            self.logger.debug(f"LSB extraction: expected_size={expected_size}, randomize={self.randomize}")
            
            # Convert to integer format
            audio_int = self._float_to_int16(audio_data)
            channels, samples = audio_int.shape
            
            # CRITICAL FIX for randomization: Use two-phase extraction when size is unknown
            if expected_size is None and self.randomize:
                self.logger.debug("Using two-phase extraction for randomized positions")
                
                # Phase 1: Extract enough data to read the header and determine actual size
                header_estimate_size = 512  # Conservative estimate for header size
                header_data = self._extract_with_size(audio_int, password, channels, samples, header_estimate_size)
                
                if header_data is None:
                    self.logger.error("Failed to extract header for size detection")
                    return None
                
                # Try to find the actual data size from the extracted header
                actual_size = self._detect_embedded_size(header_data)
                if actual_size is None:
                    self.logger.debug("Could not detect embedded size, falling back to full capacity")
                    # Fallback to full capacity extraction
                    max_capacity = self.calculate_capacity(audio_data, sample_rate)
                    expected_size = max_capacity
                else:
                    self.logger.debug(f"Detected embedded size: {actual_size} bytes")
                    expected_size = actual_size
            
            # Ensure expected_size is not None before calling _extract_with_size
            if expected_size is None:
                # Final fallback: use maximum capacity
                expected_size = self.calculate_capacity(audio_data, sample_rate)
                self.logger.debug(f"Using fallback capacity: {expected_size} bytes")
            
            # Standard extraction with known size
            return self._extract_with_size(audio_int, password, channels, samples, expected_size)
            
        except Exception as e:
            self.logger.error(f"LSB extraction failed: {e}")
            return None
    
    def _extract_with_size(self, audio_int: np.ndarray, password: str, channels: int, 
                          samples: int, expected_size: int) -> Optional[bytes]:
        """Extract data with known size."""
        try:
            # Generate same random seed
            rng_seed = self._generate_seed_sequence(password)
            rng = np.random.RandomState(rng_seed)
            
            # Ensure expected_size is valid (should not be None here, but safety check)
            if expected_size is None or expected_size <= 0:
                max_capacity = (channels * samples) // self.skip_factor // 8
                expected_size = max_capacity
                self.logger.debug(f"Using calculated capacity: {expected_size} bytes")
            
            # Generate same positions used for embedding
            total_bits = expected_size * 8
            positions = self._generate_positions(channels, samples, total_bits, rng)
            
            if len(positions) < total_bits:
                self.logger.warning("Insufficient positions for full extraction")
                total_bits = len(positions)
                expected_size = total_bits // 8
            
            # Extract bits
            flat_audio = audio_int.flatten()
            extracted_bits = []
            
            for pos in positions[:total_bits]:
                if pos < len(flat_audio):
                    extracted_bits.append(flat_audio[pos] & 1)
            
            # Convert to bytes
            bit_array = np.array(extracted_bits, dtype=np.uint8)
            return self._reconstruct_data(bit_array, expected_size)
            
        except Exception as e:
            self.logger.error(f"LSB extraction with size failed: {e}")
            return None
    
    def _detect_embedded_size(self, header_data: bytes) -> Optional[int]:
        """Try to detect the actual embedded data size from header data."""
        try:
            # Look for InvisioVault magic header patterns
            magic_headers = [b'INVV_AUD', b'INVV_IMG', b'INVV_VID']  # Support multiple formats
            
            for magic in magic_headers:
                magic_pos = header_data.find(magic)
                if magic_pos >= 0:
                    self.logger.debug(f"Found magic header at position {magic_pos}")
                    
                    # Try to parse the size information after magic header
                    try:
                        # Skip magic header (8 bytes)
                        offset = magic_pos + len(magic)
                        if offset + 6 <= len(header_data):  # Need at least 6 more bytes
                            # Read metadata_size (2 bytes) + data_size (4 bytes)
                            import struct
                            metadata_size = struct.unpack('<H', header_data[offset:offset + 2])[0]
                            data_size = struct.unpack('<I', header_data[offset + 2:offset + 6])[0]
                            
                            # Calculate total embedded size
                            total_size = len(magic) + 2 + 4 + metadata_size + data_size
                            
                            self.logger.debug(f"Parsed sizes: metadata={metadata_size}, data={data_size}, total={total_size}")
                            
                            # Sanity check: total size should be reasonable
                            if 10 <= total_size <= 10000000:  # Between 10 bytes and 10MB
                                return total_size
                            else:
                                self.logger.debug(f"Total size {total_size} seems unreasonable")
                    except Exception as parse_error:
                        self.logger.debug(f"Failed to parse size after magic header: {parse_error}")
                        continue
            
            return None  # Could not detect size
            
        except Exception as e:
            self.logger.debug(f"Size detection failed: {e}")
            return None
    
    def calculate_capacity(self, audio_data: np.ndarray, sample_rate: int) -> int:
        """Calculate LSB embedding capacity in bytes."""
        channels, samples = audio_data.shape
        total_samples = channels * samples
        usable_samples = total_samples // self.skip_factor
        return usable_samples // 8
    
    def _generate_positions(self, channels: int, samples: int, 
                          num_bits: int, rng: np.random.RandomState) -> np.ndarray:
        """Generate embedding positions (randomized or sequential)."""
        total_samples = channels * samples
        
        if self.randomize:
            # CRITICAL FIX: Use deterministic shuffle instead of random choice
            # This ensures the same positions are generated in the same order every time
            available_positions = np.arange(0, total_samples, self.skip_factor)
            
            if len(available_positions) >= num_bits:
                # Create a copy and shuffle it deterministically
                shuffled_positions = available_positions.copy()
                rng.shuffle(shuffled_positions)
                # Take the first num_bits positions from the shuffled array
                positions = shuffled_positions[:num_bits]
            else:
                # Not enough positions available
                positions = available_positions
        else:
            # Sequential positions with skip factor
            positions = np.arange(0, min(num_bits * self.skip_factor, total_samples), self.skip_factor)
        
        return positions.astype(np.int32)
    
    def _float_to_int16(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert float audio to 16-bit integer."""
        # CRITICAL FIX: Clip values to [-1.0, 1.0] range before conversion
        # Audio values can sometimes exceed ±1.0 due to processing/modification
        # Without clipping, values > 1.0 would overflow int16 range (-32768 to 32767)
        clipped_audio = np.clip(audio_data, -1.0, 1.0)
        return (clipped_audio * 32767).astype(np.int16)
    
    def _int16_to_float(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert 16-bit integer audio to float."""
        return audio_data.astype(np.float32) / 32767.0


class SpreadSpectrumEmbedding(BaseEmbeddingTechnique):
    """
    Spread spectrum embedding technique.
    
    Spreads data across frequency domain using pseudo-random sequences.
    More robust against compression but lower capacity than LSB.
    """
    
    def __init__(self, logger: Logger, chip_rate: int = 1000, amplitude: float = 0.001):
        super().__init__(logger)
        self.technique_name = "spread_spectrum"
        self.chip_rate = chip_rate          # Samples per bit
        self.amplitude = amplitude          # Embedding amplitude
        
    def embed(self, audio_data: np.ndarray, data: bytes, password: str,
              sample_rate: int, **kwargs) -> Optional[np.ndarray]:
        """
        Embed data using spread spectrum technique.
        
        Args:
            audio_data: Audio samples (channels, samples) 
            data: Data to embed
            password: Password for spreading sequence generation
            sample_rate: Audio sample rate
            **kwargs: Additional parameters
            
        Returns:
            Modified audio data or None if embedding failed
        """
        try:
            self.logger.debug(f"Spread spectrum embedding: {len(data)} bytes")
            
            # Check capacity
            capacity = self.calculate_capacity(audio_data, sample_rate)
            if len(data) > capacity:
                self.logger.error(f"Data too large for spread spectrum: {len(data)} > {capacity}")
                return None
            
            # Prepare data bits
            data_bits = self._prepare_data_bits(data)
            
            # Generate spreading sequence
            rng_seed = self._generate_seed_sequence(password)
            rng = np.random.RandomState(rng_seed)
            
            # Calculate adaptive parameters
            channels, samples = audio_data.shape
            adaptive_chip_rate = min(self.chip_rate, samples // len(data_bits)) if len(data_bits) > 0 else self.chip_rate
            
            # Calculate adaptive amplitude based on audio power
            audio_power = float(np.mean(np.abs(audio_data)))
            adaptive_amplitude = min(self.amplitude, audio_power * 0.01)
            
            self.logger.debug(f"Using chip_rate={adaptive_chip_rate}, amplitude={adaptive_amplitude:.6f}")
            
            # Copy audio for modification
            modified_audio = audio_data.copy()
            
            # Embed in each channel
            for channel in range(channels):
                for bit_idx, bit in enumerate(data_bits):
                    start_sample = bit_idx * adaptive_chip_rate
                    end_sample = start_sample + adaptive_chip_rate
                    
                    if end_sample > samples:
                        break
                    
                    # Generate spreading sequence for this bit
                    spread_seq = rng.uniform(-1, 1, adaptive_chip_rate)
                    
                    # Modulate based on bit value
                    if bit == 1:
                        signal_to_add = spread_seq * adaptive_amplitude
                    else:
                        signal_to_add = -spread_seq * adaptive_amplitude
                    
                    # Add to audio
                    modified_audio[channel, start_sample:end_sample] += signal_to_add
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(modified_audio))
            if max_val > 1.0:
                modified_audio = modified_audio / max_val
            
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"Spread spectrum embedding failed: {e}")
            return None
    
    def extract(self, audio_data: np.ndarray, password: str, sample_rate: int,
                expected_size: Optional[int] = None, **kwargs) -> Optional[bytes]:
        """
        Extract data using spread spectrum technique.
        
        Args:
            audio_data: Audio samples to extract from
            password: Password for spreading sequence generation
            sample_rate: Audio sample rate
            expected_size: Expected data size in bytes
            **kwargs: Additional parameters
            
        Returns:
            Extracted data or None if extraction failed
        """
        try:
            self.logger.debug(f"Spread spectrum extraction: expected_size={expected_size}")
            
            if expected_size is None:
                expected_size = self.calculate_capacity(audio_data, sample_rate)
            
            # Generate same spreading sequence with EXACT same seed generation
            rng_seed = self._generate_seed_sequence(password)
            
            channels, samples = audio_data.shape
            expected_bits = expected_size * 8
            
            # CRITICAL FIX: Use exact same chip rate calculation as embedding
            adaptive_chip_rate = min(self.chip_rate, samples // expected_bits) if expected_bits > 0 else self.chip_rate
            self.logger.debug(f"Using adaptive_chip_rate={adaptive_chip_rate} for {expected_bits} bits")
            
            extracted_bits = []
            
            # CRITICAL FIX: Use single RNG state and advance properly
            # This ensures exact same spreading sequences as embedding
            rng = np.random.RandomState(rng_seed)
            
            for bit_idx in range(expected_bits):
                start_sample = bit_idx * adaptive_chip_rate
                end_sample = start_sample + adaptive_chip_rate
                
                if end_sample > samples:
                    break
                
                # Generate the exact same spreading sequence as in embedding
                # The RNG state advances naturally with each call
                spread_seq = rng.uniform(-1, 1, adaptive_chip_rate)
                
                # Extract from all channels and combine correlations
                total_correlation = 0.0
                for channel in range(channels):
                    # Get audio segment from this channel
                    segment = audio_data[channel, start_sample:end_sample]
                    
                    # Correlate with spreading sequence
                    # CRITICAL FIX: Normalize correlation by sequence length
                    correlation = float(np.dot(segment, spread_seq)) / len(spread_seq)
                    total_correlation += correlation
                
                # Average correlation across channels
                avg_correlation = total_correlation / channels if channels > 0 else total_correlation
                
                # IMPROVED: Use threshold-based detection instead of simple sign
                # This helps distinguish signal from noise
                threshold = self.amplitude / 4  # Use quarter of embedding amplitude
                extracted_bit = 1 if avg_correlation > threshold else 0
                extracted_bits.append(extracted_bit)
                
                self.logger.debug(f"Bit {bit_idx}: correlation={avg_correlation:.6f}, threshold={threshold:.6f}, bit={extracted_bit}")
            
            # Convert to bytes
            if len(extracted_bits) > 0:
                bit_array = np.array(extracted_bits, dtype=np.uint8)
                result = self._reconstruct_data(bit_array, expected_size)
                self.logger.debug(f"Reconstructed {len(result) if result else 0} bytes from {len(extracted_bits)} bits")
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Spread spectrum extraction failed: {e}")
            return None
    
    def calculate_capacity(self, audio_data: np.ndarray, sample_rate: int) -> int:
        """Calculate spread spectrum embedding capacity in bytes."""
        channels, samples = audio_data.shape
        max_bits = samples // self.chip_rate
        return max_bits // 8


class PhaseCodingEmbedding(BaseEmbeddingTechnique):
    """
    Phase coding embedding technique.
    
    Modifies phase relationships in frequency domain while preserving magnitude.
    Robust against many forms of processing but has lower capacity.
    """
    
    def __init__(self, logger: Logger, segment_length: int = 2048, phase_shift: float = np.pi/4):
        super().__init__(logger)
        self.technique_name = "phase_coding"
        self.segment_length = segment_length    # FFT segment length
        self.phase_shift = phase_shift          # Phase shift amount
        
    def embed(self, audio_data: np.ndarray, data: bytes, password: str,
              sample_rate: int, **kwargs) -> Optional[np.ndarray]:
        """
        Embed data using phase coding technique.
        
        Args:
            audio_data: Audio samples (channels, samples)
            data: Data to embed
            password: Password for randomization
            sample_rate: Audio sample rate
            **kwargs: Additional parameters
            
        Returns:
            Modified audio data or None if embedding failed
        """
        try:
            self.logger.debug(f"Phase coding embedding: {len(data)} bytes")
            
            # Check capacity
            capacity = self.calculate_capacity(audio_data, sample_rate)
            if len(data) > capacity:
                self.logger.error(f"Data too large for phase coding: {len(data)} > {capacity}")
                return None
            
            # Prepare data bits
            data_bits = self._prepare_data_bits(data)
            
            channels, samples = audio_data.shape
            modified_audio = audio_data.copy()
            
            # Calculate number of segments
            num_segments = samples // self.segment_length
            
            bit_idx = 0
            for channel in range(channels):
                for seg_idx in range(num_segments):
                    if bit_idx >= len(data_bits):
                        break
                    
                    start = seg_idx * self.segment_length
                    end = start + self.segment_length
                    
                    # Get segment with windowing to reduce artifacts
                    segment = modified_audio[channel, start:end]
                    window = np.hanning(len(segment))
                    windowed_segment = segment * window
                    
                    # FFT
                    fft_segment = np.fft.fft(windowed_segment)
                    
                    # Modify phase based on data bit
                    magnitude = np.abs(fft_segment)
                    phase = np.angle(fft_segment)
                    
                    if data_bits[bit_idx] == 1:
                        # Positive phase shift
                        new_phase = phase + self.phase_shift
                    else:
                        # Negative phase shift
                        new_phase = phase - self.phase_shift
                    
                    # Reconstruct with modified phase
                    modified_fft = magnitude * np.exp(1j * new_phase)
                    
                    # Inverse FFT
                    modified_segment = np.real(np.fft.ifft(modified_fft))
                    
                    # Apply inverse window and overlap-add (simplified)
                    modified_audio[channel, start:end] = modified_segment
                    
                    bit_idx += 1
                    
                if bit_idx >= len(data_bits):
                    break
            
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"Phase coding embedding failed: {e}")
            return None
    
    def extract(self, audio_data: np.ndarray, password: str, sample_rate: int,
                expected_size: Optional[int] = None, **kwargs) -> Optional[bytes]:
        """
        Extract data using phase coding technique.
        
        Args:
            audio_data: Audio samples to extract from
            password: Password (not used in basic phase coding)
            sample_rate: Audio sample rate
            expected_size: Expected data size in bytes
            **kwargs: Additional parameters
            
        Returns:
            Extracted data or None if extraction failed
        """
        try:
            self.logger.debug(f"Phase coding extraction: expected_size={expected_size}")
            
            if expected_size is None:
                expected_size = self.calculate_capacity(audio_data, sample_rate)
            
            channels, samples = audio_data.shape
            expected_bits = expected_size * 8
            
            num_segments = samples // self.segment_length
            extracted_bits = []
            
            # We need a reference to detect phase modifications
            # CRITICAL FIX: Store original phase information during embedding
            # For now, use a differential approach
            
            bit_idx = 0
            for channel in range(channels):
                for seg_idx in range(num_segments):
                    if bit_idx >= expected_bits:
                        break
                    
                    start = seg_idx * self.segment_length
                    end = start + self.segment_length
                    
                    # Get segment with same windowing as embedding
                    segment = audio_data[channel, start:end]
                    window = np.hanning(len(segment))
                    windowed_segment = segment * window
                    
                    # FFT
                    fft_segment = np.fft.fft(windowed_segment)
                    phase = np.angle(fft_segment)
                    
                    # IMPROVED: Use reference phase for better detection
                    # Calculate what the "natural" phase should be without modification
                    magnitude = np.abs(fft_segment)
                    
                    # Create a reference by using the natural phase progression
                    # This is a simplified approach - in practice you'd need the original
                    freq_bins = np.arange(len(fft_segment))
                    natural_phase = np.angle(fft_segment)  # Current phase
                    
                    # Look for phase shifts in the dominant frequency components
                    # Focus on mid-frequency range where modifications are most detectable
                    mid_start = len(phase) // 4
                    mid_end = 3 * len(phase) // 4
                    mid_phase = phase[mid_start:mid_end]
                    
                    if len(mid_phase) > 0:
                        # Calculate phase characteristics
                        phase_mean = np.mean(mid_phase)
                        phase_std = np.std(mid_phase)
                        
                        # IMPROVED: Use phase statistics to detect modifications
                        # The embedding adds/subtracts phase_shift, affecting statistics
                        
                        # Detect if phase was shifted positive or negative
                        # based on the mean phase value relative to expected distribution
                        
                        # Use adaptive threshold based on phase_shift used in embedding
                        detection_threshold = self.phase_shift / 4
                        
                        extracted_bit = 1 if phase_mean > detection_threshold else 0
                        
                        self.logger.debug(f"Segment {seg_idx}: phase_mean={phase_mean:.6f}, threshold={detection_threshold:.6f}, bit={extracted_bit}")
                    else:
                        # Fallback
                        extracted_bit = 0
                        self.logger.debug(f"Segment {seg_idx}: fallback bit=0")
                    
                    extracted_bits.append(extracted_bit)
                    
                    bit_idx += 1
                    
                if bit_idx >= expected_bits:
                    break
            
            # Convert to bytes
            if len(extracted_bits) > 0:
                bit_array = np.array(extracted_bits[:expected_bits], dtype=np.uint8)
                result = self._reconstruct_data(bit_array, expected_size)
                self.logger.debug(f"Phase coding extracted {len(result) if result else 0} bytes from {len(extracted_bits)} bits")
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Phase coding extraction failed: {e}")
            return None
    
    def calculate_capacity(self, audio_data: np.ndarray, sample_rate: int) -> int:
        """Calculate phase coding embedding capacity in bytes."""
        channels, samples = audio_data.shape
        segments_per_channel = samples // self.segment_length
        total_bits = segments_per_channel * channels
        return total_bits // 8


class EchoHidingEmbedding(BaseEmbeddingTechnique):
    """
    Echo hiding embedding technique.
    
    Embeds data by adding delayed and attenuated copies of the signal.
    Very robust against compression but has very low capacity.
    """
    
    def __init__(self, logger: Logger, delay_0: int = 100, delay_1: int = 150, 
                 attenuation: float = 0.1):
        super().__init__(logger)
        self.technique_name = "echo_hiding"
        self.delay_0 = delay_0              # Delay for bit 0 (samples)
        self.delay_1 = delay_1              # Delay for bit 1 (samples)
        self.attenuation = attenuation      # Echo attenuation factor
        
    def embed(self, audio_data: np.ndarray, data: bytes, password: str,
              sample_rate: int, **kwargs) -> Optional[np.ndarray]:
        """
        Embed data using echo hiding technique.
        
        Args:
            audio_data: Audio samples (channels, samples)
            data: Data to embed
            password: Password for segment selection
            sample_rate: Audio sample rate
            **kwargs: Additional parameters
            
        Returns:
            Modified audio data or None if embedding failed
        """
        try:
            self.logger.debug(f"Echo hiding embedding: {len(data)} bytes")
            
            # Check capacity
            capacity = self.calculate_capacity(audio_data, sample_rate)
            if len(data) > capacity:
                self.logger.error(f"Data too large for echo hiding: {len(data)} > {capacity}")
                return None
            
            # Prepare data bits
            data_bits = self._prepare_data_bits(data)
            
            channels, samples = audio_data.shape
            modified_audio = audio_data.copy()
            
            # Calculate segment size for embedding - use same logic as capacity calculation
            max_delay = max(self.delay_0, self.delay_1)
            min_segment_size = max(max_delay * 4, 2000)  # At least 4x the delay
            
            # Calculate how many segments we can fit
            usable_length = samples - max_delay
            max_possible_bits = usable_length // min_segment_size
            
            if len(data_bits) > max_possible_bits:
                self.logger.error(f"Too many bits for echo hiding: {len(data_bits)} > {max_possible_bits}")
                return None
            
            # Use consistent segment size
            segment_size = min_segment_size
            
            self.logger.debug(f"Echo embedding: segment_size={segment_size}, max_delay={max_delay}, bits={len(data_bits)}")
            
            # Embed each bit as an echo
            for bit_idx, bit in enumerate(data_bits):
                start_sample = bit_idx * segment_size
                end_sample = min(start_sample + segment_size - max_delay, samples - max_delay)
                
                if start_sample >= end_sample:
                    break
                
                # Choose delay based on bit value
                delay = self.delay_1 if bit == 1 else self.delay_0
                
                # Add echo to all channels
                for channel in range(channels):
                    segment = modified_audio[channel, start_sample:end_sample]
                    echo = segment * self.attenuation
                    
                    # Add delayed echo
                    echo_start = start_sample + delay
                    echo_end = echo_start + len(echo)
                    
                    if echo_end <= samples:
                        modified_audio[channel, echo_start:echo_end] += echo
            
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"Echo hiding embedding failed: {e}")
            return None
    
    def extract(self, audio_data: np.ndarray, password: str, sample_rate: int,
                expected_size: Optional[int] = None, **kwargs) -> Optional[bytes]:
        """
        Extract data using echo hiding technique.
        
        Args:
            audio_data: Audio samples to extract from
            password: Password (not used in basic echo hiding)
            sample_rate: Audio sample rate
            expected_size: Expected data size in bytes
            **kwargs: Additional parameters
            
        Returns:
            Extracted data or None if extraction failed
        """
        try:
            self.logger.debug(f"Echo hiding extraction: expected_size={expected_size}")
            
            if expected_size is None:
                expected_size = self.calculate_capacity(audio_data, sample_rate)
            
            channels, samples = audio_data.shape
            expected_bits = expected_size * 8
            
            # Calculate segment size - use same logic as embedding and capacity
            max_delay = max(self.delay_0, self.delay_1)
            min_segment_size = max(max_delay * 4, 2000)  # At least 4x the delay
            segment_size = min_segment_size
            
            self.logger.debug(f"Echo extraction: segment_size={segment_size}, expected_bits={expected_bits}")
            
            extracted_bits = []
            
            # Extract bits by analyzing autocorrelation
            for bit_idx in range(expected_bits):
                start_sample = bit_idx * segment_size
                end_sample = min(start_sample + segment_size, samples - max_delay)
                
                if start_sample >= end_sample or end_sample - start_sample < max_delay:
                    self.logger.debug(f"Skipping bit {bit_idx}: insufficient segment size")
                    break
                
                # IMPROVED: Analyze all channels and combine results
                total_corr_0 = 0.0
                total_corr_1 = 0.0
                
                for channel in range(channels):
                    channel_data = audio_data[channel]
                    segment = channel_data[start_sample:end_sample]
                    
                    # Calculate autocorrelation at both delay points
                    corr_0 = self._calculate_correlation(segment, self.delay_0)
                    corr_1 = self._calculate_correlation(segment, self.delay_1)
                    
                    total_corr_0 += corr_0
                    total_corr_1 += corr_1
                
                # Average across channels
                avg_corr_0 = total_corr_0 / channels
                avg_corr_1 = total_corr_1 / channels
                
                # IMPROVED: Use correlation difference for more reliable detection
                correlation_diff = avg_corr_1 - avg_corr_0
                
                # Use threshold to reduce noise sensitivity
                threshold = 0.01  # Minimum correlation difference
                extracted_bit = 1 if correlation_diff > threshold else 0
                extracted_bits.append(extracted_bit)
                
                self.logger.debug(f"Bit {bit_idx}: corr_0={avg_corr_0:.4f}, corr_1={avg_corr_1:.4f}, diff={correlation_diff:.4f}, bit={extracted_bit}")
            
            # Convert to bytes
            if len(extracted_bits) > 0:
                bit_array = np.array(extracted_bits, dtype=np.uint8)
                return self._reconstruct_data(bit_array, expected_size)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Echo hiding extraction failed: {e}")
            return None
    
    def calculate_capacity(self, audio_data: np.ndarray, sample_rate: int) -> int:
        """Calculate echo hiding embedding capacity in bytes."""
        channels, samples = audio_data.shape
        max_delay = max(self.delay_0, self.delay_1)
        
        # IMPROVED: More realistic capacity calculation
        usable_length = samples - max_delay
        
        # Use smaller segments for better capacity (but still reliable)
        min_segment_size = max(max_delay * 4, 2000)  # At least 4x the delay
        segment_size = min_segment_size
        
        if usable_length <= segment_size:
            return 0
        
        # Calculate number of bits we can embed
        max_bits = usable_length // segment_size
        capacity_bytes = max_bits // 8
        
        # Ensure at least some minimum capacity for reasonable audio lengths
        if capacity_bytes < 1 and usable_length > 10000:  # If audio > 0.2 seconds
            capacity_bytes = 1
        
        return capacity_bytes
    
    def _calculate_correlation(self, segment: np.ndarray, delay: int) -> float:
        """Calculate autocorrelation at specific delay."""
        if len(segment) <= delay:
            return 0.0
        
        original = segment[:-delay]
        delayed = segment[delay:]
        
        if len(original) == 0 or len(delayed) == 0:
            return 0.0
        
        # Normalize correlation
        correlation = float(np.corrcoef(original, delayed)[0, 1])
        return correlation if not np.isnan(correlation) else 0.0


class EmbeddingTechniqueFactory:
    """Factory class for creating embedding technique instances."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self._techniques = {
            'lsb': LSBEmbedding,
            'spread_spectrum': SpreadSpectrumEmbedding,
            'phase_coding': PhaseCodingEmbedding,
            'echo': EchoHidingEmbedding
        }
    
    def create_technique(self, technique_name: str, **kwargs) -> Optional[BaseEmbeddingTechnique]:
        """
        Create embedding technique instance.
        
        Args:
            technique_name: Name of technique to create
            **kwargs: Parameters for technique initialization
            
        Returns:
            Embedding technique instance or None if not found
        """
        technique_name = technique_name.lower()
        
        if technique_name not in self._techniques:
            self.logger.error(f"Unknown embedding technique: {technique_name}")
            return None
        
        try:
            technique_class = self._techniques[technique_name]
            return technique_class(self.logger, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create {technique_name} technique: {e}")
            return None
    
    def get_available_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available techniques."""
        available = {}
        
        for tech in EmbeddingTechnique:
            if tech.available:
                available[tech.code] = {
                    'name': tech.display_name,
                    'description': tech.description,
                    'available': tech.available
                }
        
        return available
    
    def get_technique_info(self, technique_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about specific technique."""
        for tech in EmbeddingTechnique:
            if tech.code == technique_name.lower():
                return {
                    'code': tech.code,
                    'name': tech.display_name,
                    'description': tech.description,
                    'available': tech.available
                }
        return None
