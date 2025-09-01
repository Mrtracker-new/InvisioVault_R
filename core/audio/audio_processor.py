"""
Core Audio Processing Utilities for Steganography Operations

This module provides robust audio file handling, format conversion, and validation
capabilities optimized for steganographic operations.
"""

import os
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Suppress audio library warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
    from scipy import signal
    AUDIO_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Audio dependencies not available: {e}")
    print("Please install: pip install librosa soundfile pydub scipy")
    AUDIO_LIBS_AVAILABLE = False

from utils.logger import Logger
from utils.error_handler import ErrorHandler


class AudioFormat(Enum):
    """Supported audio formats with their characteristics."""
    WAV = ("wav", True, "Uncompressed PCM audio - Best for steganography")
    FLAC = ("flac", True, "Lossless compression - Excellent for steganography") 
    AIFF = ("aiff", True, "Apple lossless format - Good for steganography")
    AU = ("au", True, "Unix audio format - Good for steganography")
    MP3 = ("mp3", False, "Lossy compression - NOT recommended for steganography")
    AAC = ("aac", False, "Lossy compression - NOT recommended for steganography")
    OGG = ("ogg", False, "Lossy compression - NOT recommended for steganography")
    M4A = ("m4a", False, "Usually lossy - NOT recommended for steganography")
    
    def __init__(self, extension: str, lossless: bool, description: str):
        self.extension = extension
        self.lossless = lossless
        self.description = description
        

@dataclass
class AudioInfo:
    """Audio file information and characteristics."""
    path: Path
    format: str
    sample_rate: int
    channels: int
    frames: int
    duration: float
    bit_depth: int
    is_lossless: bool
    capacity_bits: int
    capacity_bytes: int
    file_size: int
    
    @property
    def is_suitable_for_steganography(self) -> bool:
        """Check if audio format is suitable for steganography."""
        return self.is_lossless and self.frames > 0 and self.duration > 0
        
    @property
    def format_warning(self) -> Optional[str]:
        """Get warning message for unsuitable formats."""
        if not self.is_lossless:
            return f"WARNING: {self.format.upper()} is a lossy format that may corrupt hidden data"
        return None


class AudioProcessor:
    """
    High-performance audio processing utility for steganography operations.
    
    Provides robust audio loading, format conversion, validation, and capacity analysis
    with multiple fallback mechanisms for maximum compatibility.
    """
    
    # Supported lossless formats for steganography
    LOSSLESS_FORMATS = {'.wav', '.flac', '.aiff', '.au'}
    LOSSY_FORMATS = {'.mp3', '.aac', '.ogg', '.m4a', '.wma'}
    
    # Quality parameters for different operations
    DEFAULT_SAMPLE_RATE = 44100
    MAX_DURATION = 3600  # 1 hour max
    MIN_CAPACITY_BYTES = 1024  # Minimum 1KB capacity
    
    def __init__(self):
        """Initialize audio processor with logging and error handling."""
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        
        if not AUDIO_LIBS_AVAILABLE:
            raise ImportError("Required audio libraries not available")
            
        self.logger.info("Audio processor initialized")
    
    def analyze_audio_file(self, audio_path: Path) -> AudioInfo:
        """
        Comprehensive audio file analysis for steganography suitability.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioInfo object with detailed characteristics
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If file format is unsupported
        """
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Get basic file info
            file_size = audio_path.stat().st_size
            format_ext = audio_path.suffix.lower()
            
            # Determine format characteristics
            format_info = self._get_format_info(format_ext)
            
            # Try multiple methods to get audio info
            audio_info = self._extract_audio_info(audio_path)
            
            # Calculate steganography capacity
            capacity_bits, capacity_bytes = self._calculate_capacity(
                audio_info['frames'], 
                audio_info['channels'],
                format_info['lossless']
            )
            
            info = AudioInfo(
                path=audio_path,
                format=format_ext.lstrip('.'),
                sample_rate=audio_info['sample_rate'],
                channels=audio_info['channels'],
                frames=audio_info['frames'],
                duration=audio_info['duration'],
                bit_depth=audio_info.get('bit_depth', 16),
                is_lossless=format_info['lossless'],
                capacity_bits=capacity_bits,
                capacity_bytes=capacity_bytes,
                file_size=file_size
            )
            
            self.logger.debug(f"Audio analysis complete: {info.format} | "
                            f"{info.duration:.1f}s | {info.capacity_bytes} bytes capacity")
            
            return info
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            raise ValueError(f"Could not analyze audio file: {e}")
    
    def load_audio(self, audio_path: Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Robust audio loading with multiple fallback mechanisms.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (None to keep original)
            
        Returns:
            Tuple of (audio_data, sample_rate)
            - audio_data: Shape (channels, samples) or (samples,) for mono
            - sample_rate: Sample rate in Hz
            
        Raises:
            ValueError: If audio cannot be loaded
        """
        try:
            self.logger.debug(f"Loading audio: {audio_path.name}")
            
            # Validate file first
            if not self._validate_audio_file(audio_path):
                raise ValueError(f"Invalid audio file: {audio_path}")
            
            # Try soundfile first (best quality preservation)
            try:
                audio_data, sample_rate = self._load_with_soundfile(audio_path, target_sr)
                self.logger.debug(f"Loaded with soundfile: {audio_data.shape} @ {sample_rate}Hz")
                return audio_data, sample_rate
            except Exception as e:
                self.logger.debug(f"Soundfile loading failed: {e}")
            
            # Fallback to librosa
            try:
                audio_data, sample_rate = self._load_with_librosa(audio_path, target_sr)
                self.logger.debug(f"Loaded with librosa: {audio_data.shape} @ {sample_rate}Hz")
                return audio_data, sample_rate
            except Exception as e:
                self.logger.debug(f"Librosa loading failed: {e}")
            
            # Final fallback to pydub
            try:
                audio_data, sample_rate = self._load_with_pydub(audio_path, target_sr)
                self.logger.debug(f"Loaded with pydub: {audio_data.shape} @ {sample_rate}Hz")
                return audio_data, sample_rate
            except Exception as e:
                self.logger.debug(f"Pydub loading failed: {e}")
            
            raise ValueError("All audio loading methods failed")
            
        except Exception as e:
            self.logger.error(f"Audio loading failed: {e}")
            raise
    
    def save_audio(self, audio_data: np.ndarray, sample_rate: int, 
                   output_path: Path, preserve_format: bool = True) -> bool:
        """
        Save audio with format optimization and quality preservation.
        
        Args:
            audio_data: Audio data array (channels, samples) or (samples,)
            sample_rate: Sample rate in Hz
            output_path: Output file path
            preserve_format: Whether to preserve original format characteristics
            
        Returns:
            Success status
        """
        try:
            self.logger.debug(f"Saving audio: {output_path.name}")
            
            # Validate output format
            format_warnings = self._check_output_format(output_path)
            for warning in format_warnings:
                self.logger.warning(warning)
            
            # Normalize audio data
            audio_data = self._normalize_audio(audio_data)
            
            # Try soundfile first for lossless formats
            if output_path.suffix.lower() in self.LOSSLESS_FORMATS:
                try:
                    success = self._save_with_soundfile(audio_data, sample_rate, output_path)
                    if success:
                        return True
                except Exception as e:
                    self.logger.debug(f"Soundfile saving failed: {e}")
            
            # Fallback to pydub for other formats
            try:
                success = self._save_with_pydub(audio_data, sample_rate, output_path)
                if success:
                    return True
            except Exception as e:
                self.logger.debug(f"Pydub saving failed: {e}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Audio saving failed: {e}")
            return False
    
    def convert_format(self, input_path: Path, output_path: Path, 
                      target_format: str = 'wav') -> bool:
        """
        Convert audio between formats with quality preservation.
        
        Args:
            input_path: Input audio file
            output_path: Output audio file  
            target_format: Target format (wav, flac, etc.)
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Converting {input_path.name} to {target_format}")
            
            # Load audio
            audio_data, sample_rate = self.load_audio(input_path)
            
            # Ensure output path has correct extension
            if not output_path.suffix.lower().endswith(target_format):
                output_path = output_path.with_suffix(f'.{target_format}')
            
            # Save with new format
            success = self.save_audio(audio_data, sample_rate, output_path)
            
            if success:
                self.logger.info(f"Format conversion successful: {output_path.name}")
            else:
                self.logger.error("Format conversion failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Format conversion failed: {e}")
            return False
    
    def validate_for_steganography(self, audio_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate audio file suitability for steganography operations.
        
        Args:
            audio_path: Audio file to validate
            
        Returns:
            Tuple of (is_suitable, list_of_warnings)
        """
        try:
            warnings = []
            
            # Analyze file
            info = self.analyze_audio_file(audio_path)
            
            # Check format suitability
            if not info.is_lossless:
                warnings.append(f"Lossy format ({info.format.upper()}) may corrupt hidden data")
            
            # Check capacity
            if info.capacity_bytes < self.MIN_CAPACITY_BYTES:
                warnings.append(f"Insufficient capacity: {info.capacity_bytes} < {self.MIN_CAPACITY_BYTES} bytes")
            
            # Check duration
            if info.duration > self.MAX_DURATION:
                warnings.append(f"File too long: {info.duration:.1f}s > {self.MAX_DURATION}s")
            
            # Check sample rate
            if info.sample_rate < 8000:
                warnings.append(f"Low sample rate: {info.sample_rate}Hz may not be suitable")
            
            # Check channels
            if info.channels > 2:
                warnings.append(f"Multichannel audio ({info.channels} channels) may have issues")
            
            is_suitable = len(warnings) == 0 or (len(warnings) == 1 and "Lossy format" in warnings[0])
            
            return is_suitable, warnings
            
        except Exception as e:
            return False, [f"Validation failed: {e}"]
    
    def get_capacity_info(self, audio_path: Path, technique: str = 'lsb') -> Dict[str, Any]:
        """
        Calculate detailed capacity information for different techniques.
        
        Args:
            audio_path: Audio file to analyze
            technique: Steganography technique (lsb, spread_spectrum, phase_coding)
            
        Returns:
            Dictionary with capacity information
        """
        try:
            info = self.analyze_audio_file(audio_path)
            
            # Base capacity calculation
            if technique.lower() == 'lsb':
                # LSB: 1 bit per sample (with skip factor)
                skip_factor = 4  # Use every 4th sample for quality
                usable_samples = (info.frames * info.channels) // skip_factor
                capacity_bits = usable_samples
            elif technique.lower() == 'spread_spectrum':
                # Spread spectrum: Lower capacity but more robust
                chip_rate = 100  # bits per second
                capacity_bits = int(info.duration * chip_rate)
            elif technique.lower() == 'phase_coding':
                # Phase coding: Segment-based capacity
                segment_length = 1024
                segments_per_channel = info.frames // segment_length
                capacity_bits = segments_per_channel * info.channels
            else:
                capacity_bits = info.capacity_bits
            
            capacity_bytes = capacity_bits // 8
            
            # Account for overhead (headers, encryption, error correction)
            overhead_factor = 0.85  # Reserve 15% for overhead
            effective_capacity = int(capacity_bytes * overhead_factor)
            
            return {
                'technique': technique,
                'total_bits': capacity_bits,
                'total_bytes': capacity_bytes,
                'effective_bytes': effective_capacity,
                'overhead_bytes': capacity_bytes - effective_capacity,
                'efficiency': overhead_factor,
                'recommended_max_bytes': effective_capacity - 1024,  # Leave buffer
                'duration': info.duration,
                'sample_rate': info.sample_rate,
                'channels': info.channels,
                'is_suitable': effective_capacity >= self.MIN_CAPACITY_BYTES
            }
            
        except Exception as e:
            self.logger.error(f"Capacity calculation failed: {e}")
            return {}
    
    def _validate_audio_file(self, audio_path: Path) -> bool:
        """Validate audio file existence and basic properties."""
        try:
            if not audio_path.exists():
                return False
            
            if audio_path.stat().st_size == 0:
                return False
            
            extension = audio_path.suffix.lower()
            supported_formats = self.LOSSLESS_FORMATS | self.LOSSY_FORMATS
            
            if extension not in supported_formats:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _get_format_info(self, extension: str) -> Dict[str, Any]:
        """Get format information for given extension."""
        for fmt in AudioFormat:
            if extension.lstrip('.').lower() == fmt.extension:
                return {
                    'extension': fmt.extension,
                    'lossless': fmt.lossless,
                    'description': fmt.description
                }
        
        # Unknown format - assume lossy for safety
        return {
            'extension': extension.lstrip('.'),
            'lossless': False,
            'description': 'Unknown format - may not be suitable for steganography'
        }
    
    def _extract_audio_info(self, audio_path: Path) -> Dict[str, Any]:
        """Extract audio information using multiple methods."""
        # Try soundfile first
        try:
            info = sf.info(str(audio_path))
            return {
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames,
                'duration': info.duration,
                'bit_depth': getattr(info, 'subtype_info', {}).get('bits', 16)
            }
        except Exception:
            pass
        
        # Fallback to librosa
        try:
            duration = librosa.get_duration(path=str(audio_path))
            y, sr = librosa.load(str(audio_path), sr=None, mono=False, duration=10)  # Load first 10s
            
            if y.ndim == 1:
                channels = 1
                frames = len(y)
            else:
                channels = y.shape[0] if y.shape[0] <= y.shape[1] else y.shape[1]
                frames = y.shape[1] if y.shape[0] <= y.shape[1] else y.shape[0]
            
            # Estimate total frames
            total_frames = int(frames * (duration / 10)) if duration > 10 else frames
            
            return {
                'sample_rate': sr,
                'channels': channels,
                'frames': total_frames,
                'duration': duration,
                'bit_depth': 16  # Assume 16-bit
            }
        except Exception:
            pass
        
        # Final fallback - minimal info
        return {
            'sample_rate': self.DEFAULT_SAMPLE_RATE,
            'channels': 2,
            'frames': 0,
            'duration': 0.0,
            'bit_depth': 16
        }
    
    def _calculate_capacity(self, frames: int, channels: int, is_lossless: bool) -> Tuple[int, int]:
        """Calculate steganography capacity."""
        if not is_lossless:
            # Lossy formats have reduced capacity due to compression artifacts
            capacity_bits = int((frames * channels) * 0.1)  # Very conservative
        else:
            # Lossless formats can use LSB reliably
            skip_factor = 4  # Use every 4th sample for quality
            capacity_bits = (frames * channels) // skip_factor
        
        capacity_bytes = capacity_bits // 8
        return capacity_bits, capacity_bytes
    
    def _load_with_soundfile(self, audio_path: Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load audio using soundfile library."""
        audio_data, sample_rate = sf.read(str(audio_path), always_2d=False)
        
        # Handle channel dimension
        if audio_data.ndim == 2:
            audio_data = audio_data.T  # (samples, channels) -> (channels, samples)
        else:
            audio_data = audio_data.reshape(1, -1)  # Mono -> (1, samples)
        
        # Resample if needed
        if target_sr and target_sr != sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        
        return audio_data.astype(np.float32), int(sample_rate)
    
    def _load_with_librosa(self, audio_path: Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load audio using librosa library."""
        audio_data, sample_rate = librosa.load(
            str(audio_path), 
            sr=target_sr, 
            mono=False,
            duration=self.MAX_DURATION
        )
        
        # Ensure proper shape
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        
        return audio_data.astype(np.float32), int(sample_rate)
    
    def _load_with_pydub(self, audio_path: Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load audio using pydub library."""
        audio_segment = AudioSegment.from_file(str(audio_path))
        
        # Convert to desired sample rate
        if target_sr:
            audio_segment = audio_segment.set_frame_rate(target_sr)
        
        # Convert to numpy array
        samples = audio_segment.get_array_of_samples()
        audio_data = np.array(samples, dtype=np.float32)
        
        # Normalize to [-1, 1]
        if audio_segment.sample_width == 2:  # 16-bit
            audio_data = audio_data / 32768.0
        elif audio_segment.sample_width == 3:  # 24-bit
            audio_data = audio_data / 8388608.0
        elif audio_segment.sample_width == 4:  # 32-bit
            audio_data = audio_data / 2147483648.0
        
        # Handle stereo
        if audio_segment.channels == 2:
            audio_data = audio_data.reshape(-1, 2).T
        else:
            audio_data = audio_data.reshape(1, -1)
        
        return audio_data, audio_segment.frame_rate
    
    def _save_with_soundfile(self, audio_data: np.ndarray, sample_rate: int, output_path: Path) -> bool:
        """Save audio using soundfile library."""
        try:
            # Prepare data for soundfile (samples, channels)
            if audio_data.ndim == 2:
                save_data = audio_data.T
            else:
                save_data = audio_data
            
            # Determine subtype based on format
            # Use 32-bit for better precision preservation for steganography
            format_ext = output_path.suffix.lower().lstrip('.')
            if format_ext == 'wav':
                subtype = 'PCM_32'
            elif format_ext == 'flac':
                subtype = 'PCM_24'  # FLAC supports up to 24-bit
            else:
                subtype = None
            
            sf.write(
                str(output_path),
                save_data,
                sample_rate,
                subtype=subtype
            )
            
            return output_path.exists() and output_path.stat().st_size > 0
            
        except Exception as e:
            self.logger.debug(f"Soundfile save error: {e}")
            return False
    
    def _save_with_pydub(self, audio_data: np.ndarray, sample_rate: int, output_path: Path) -> bool:
        """Save audio using pydub library."""
        try:
            # Convert to 16-bit integers
            if audio_data.ndim == 2:
                channels, samples = audio_data.shape
            else:
                channels, samples = 1, len(audio_data)
                audio_data = audio_data.reshape(1, -1)
            
            # Convert to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            if channels == 1:
                # Mono
                audio_segment = AudioSegment(
                    audio_int16[0].tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1
                )
            else:
                # Stereo - interleave channels
                stereo_data = np.empty(samples * 2, dtype=np.int16)
                stereo_data[0::2] = audio_int16[0]
                stereo_data[1::2] = audio_int16[1] if channels > 1 else audio_int16[0]
                
                audio_segment = AudioSegment(
                    stereo_data.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=2
                )
            
            # Export with format-specific parameters
            format_ext = output_path.suffix.lower().lstrip('.')
            export_params = {}
            
            if format_ext in ['wav', 'flac']:
                if format_ext == 'flac':
                    export_params['parameters'] = ['-compression_level', '8']
            else:
                export_params['bitrate'] = '320k'
            
            audio_segment.export(str(output_path), format=format_ext, **export_params)
            
            return output_path.exists() and output_path.stat().st_size > 0
            
        except Exception as e:
            self.logger.debug(f"Pydub save error: {e}")
            return False
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to prevent clipping."""
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        return audio_data
    
    def _check_output_format(self, output_path: Path) -> List[str]:
        """Check output format and return warnings."""
        warnings = []
        format_ext = output_path.suffix.lower()
        
        if format_ext in self.LOSSY_FORMATS:
            warnings.append(
                f"WARNING: Saving to lossy format {format_ext.upper()}. "
                "This may corrupt LSB steganography data. "
                "Use lossless formats (WAV, FLAC) for reliable extraction."
            )
        
        return warnings
