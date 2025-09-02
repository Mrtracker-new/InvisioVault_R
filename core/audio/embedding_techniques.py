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
    LSB = ("lsb", "Least Significant Bit", "High capacity, low robustness", True)
    SPREAD_SPECTRUM = ("spread_spectrum", "Spread Spectrum", "Medium capacity, high robustness", True) 
    PHASE_CODING = ("phase_coding", "Phase Coding", "Low capacity, high robustness", True)
    DCT = ("dct", "Discrete Cosine Transform", "Medium capacity, medium robustness", SCIPY_AVAILABLE)
    DWT = ("dwt", "Discrete Wavelet Transform", "Medium capacity, high robustness", False)  # Not implemented yet
    ECHO_HIDING = ("echo", "Echo Hiding", "Low capacity, high robustness", True)
    
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
            flat_audio = audio_int.flatten()
            for i, pos in enumerate(positions[:len(data_bits)]):
                if pos < len(flat_audio):
                    # Clear LSB and set data bit
                    flat_audio[pos] = (flat_audio[pos] & 0xFFFE) | data_bits[i]
            
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
            self.logger.debug(f"LSB extraction: expected_size={expected_size}")
            
            # Generate same random seed
            rng_seed = self._generate_seed_sequence(password)
            rng = np.random.RandomState(rng_seed)
            
            # Convert to integer format
            audio_int = self._float_to_int16(audio_data)
            channels, samples = audio_int.shape
            
            # Calculate maximum possible extraction size if not specified
            if expected_size is None:
                max_capacity = self.calculate_capacity(audio_data, sample_rate)
                expected_size = max_capacity
            
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
            self.logger.error(f"LSB extraction failed: {e}")
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
            # Generate random positions with skip factor
            available_positions = np.arange(0, total_samples, self.skip_factor)
            if len(available_positions) >= num_bits:
                # CRITICAL FIX: Sort positions to ensure deterministic order
                # This ensures extraction uses same positions in same order as embedding
                positions = rng.choice(available_positions, size=num_bits, replace=False)
                positions = np.sort(positions)  # Sort for consistent ordering
            else:
                positions = available_positions
        else:
            # Sequential positions with skip factor
            positions = np.arange(0, min(num_bits * self.skip_factor, total_samples), self.skip_factor)
        
        return positions.astype(np.int32)
    
    def _float_to_int16(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert float audio to 16-bit integer."""
        return (audio_data * 32767).astype(np.int16)
    
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
            
            # Generate same spreading sequence
            rng_seed = self._generate_seed_sequence(password)
            rng = np.random.RandomState(rng_seed)
            
            channels, samples = audio_data.shape
            expected_bits = expected_size * 8
            
            # Calculate chip rate used during embedding
            adaptive_chip_rate = min(self.chip_rate, samples // expected_bits) if expected_bits > 0 else self.chip_rate
            
            extracted_bits = []
            
            # Extract from first channel (could combine multiple channels for better reliability)
            channel_data = audio_data[0] if channels > 0 else audio_data
            
            for bit_idx in range(expected_bits):
                start_sample = bit_idx * adaptive_chip_rate
                end_sample = start_sample + adaptive_chip_rate
                
                if end_sample > len(channel_data):
                    break
                
                # Get audio segment
                segment = channel_data[start_sample:end_sample]
                
                # Generate same spreading sequence
                spread_seq = rng.uniform(-1, 1, adaptive_chip_rate)
                
                # Correlate with spreading sequence
                correlation = float(np.dot(segment, spread_seq))
                
                # Decide bit based on correlation sign
                extracted_bits.append(1 if correlation > 0 else 0)
            
            # Convert to bytes
            if len(extracted_bits) > 0:
                bit_array = np.array(extracted_bits, dtype=np.uint8)
                return self._reconstruct_data(bit_array, expected_size)
            
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
            
            bit_idx = 0
            for channel in range(channels):
                for seg_idx in range(num_segments):
                    if bit_idx >= expected_bits:
                        break
                    
                    start = seg_idx * self.segment_length
                    end = start + self.segment_length
                    
                    # Get segment
                    segment = audio_data[channel, start:end]
                    window = np.hanning(len(segment))
                    windowed_segment = segment * window
                    
                    # FFT
                    fft_segment = np.fft.fft(windowed_segment)
                    phase = np.angle(fft_segment)
                    
                    # Analyze phase characteristics to determine bit
                    # This is a simplified extraction - real implementation would need
                    # reference phase or more sophisticated detection
                    avg_phase = np.mean(phase)
                    extracted_bits.append(1 if avg_phase > 0 else 0)
                    
                    bit_idx += 1
                    
                if bit_idx >= expected_bits:
                    break
            
            # Convert to bytes
            if len(extracted_bits) > 0:
                bit_array = np.array(extracted_bits[:expected_bits], dtype=np.uint8)
                return self._reconstruct_data(bit_array, expected_size)
            
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
            
            # Calculate segment size for embedding
            segment_size = samples // len(data_bits) if len(data_bits) > 0 else samples
            max_delay = max(self.delay_0, self.delay_1)
            
            if segment_size <= max_delay:
                self.logger.error("Segments too small for echo hiding")
                return None
            
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
            
            # Calculate segment size
            segment_size = samples // expected_bits if expected_bits > 0 else samples
            max_delay = max(self.delay_0, self.delay_1)
            
            extracted_bits = []
            
            # Extract bits by analyzing autocorrelation
            for bit_idx in range(expected_bits):
                start_sample = bit_idx * segment_size
                end_sample = min(start_sample + segment_size - max_delay, samples - max_delay)
                
                if start_sample >= end_sample:
                    break
                
                # Analyze first channel (could combine multiple channels)
                channel_data = audio_data[0] if channels > 0 else audio_data
                segment = channel_data[start_sample:end_sample]
                
                # Calculate autocorrelation at both delay points
                corr_0 = self._calculate_correlation(segment, self.delay_0)
                corr_1 = self._calculate_correlation(segment, self.delay_1)
                
                # Determine bit based on stronger correlation
                extracted_bits.append(1 if corr_1 > corr_0 else 0)
            
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
        
        # Very conservative capacity calculation
        usable_length = samples - max_delay
        segment_size = 8000  # Minimum segment size for reliable echo detection
        
        if usable_length <= segment_size:
            return 0
        
        max_bits = usable_length // segment_size
        return max(1, max_bits // 8)  # Very low capacity
    
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
