"""
Advanced Audio Steganography Engine

This is the main audio steganography engine that combines all techniques,
security features, and recovery mechanisms into a unified, production-ready system.
"""

import os
import struct
import hashlib
import json
import secrets
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from core.encryption_engine import EncryptionEngine, SecurityLevel
from core.audio.audio_processor import AudioProcessor, AudioInfo
from core.audio.embedding_techniques import (
    EmbeddingTechniqueFactory, 
    BaseEmbeddingTechnique,
    EmbeddingTechnique
)


class StegoMode(Enum):
    """Available steganography modes with different security/performance tradeoffs."""
    FAST = ("fast", "Fast embedding with basic security", 1, False, False)
    BALANCED = ("balanced", "Good security-performance balance", 2, True, False)
    SECURE = ("secure", "High security with redundancy", 3, True, True)
    MAXIMUM = ("maximum", "Maximum security and recovery", 5, True, True)
    
    def __init__(self, code: str, description: str, redundancy: int, 
                 error_correction: bool, anti_detection: bool):
        self.code = code
        self.description = description
        self.redundancy = redundancy
        self.error_correction = error_correction
        self.anti_detection = anti_detection


@dataclass
class EmbeddingConfig:
    """Configuration for audio steganography operations."""
    technique: str = 'lsb'
    mode: str = 'balanced'
    password: str = ''
    redundancy_level: int = 2
    error_correction: bool = True
    anti_detection: bool = False
    randomize_positions: bool = True
    custom_seed: Optional[int] = None
    quality_optimization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EmbeddingResult:
    """Result of steganography operation."""
    success: bool
    message: str
    technique_used: Optional[str] = None
    mode_used: Optional[str] = None
    capacity_used: Optional[int] = None
    capacity_total: Optional[int] = None
    redundancy_level: Optional[int] = None
    anti_detection_score: Optional[float] = None
    processing_time: Optional[float] = None
    warnings: Optional[List[str]] = None
    
    @property
    def capacity_utilization(self) -> Optional[float]:
        """Calculate capacity utilization percentage."""
        if self.capacity_used and self.capacity_total:
            return (self.capacity_used / self.capacity_total) * 100
        return None


@dataclass
class ExtractionResult:
    """Result of data extraction operation."""
    success: bool
    data: Optional[bytes] = None
    message: str = ""
    technique_detected: Optional[str] = None
    confidence_score: Optional[float] = None
    recovery_method: Optional[str] = None
    attempts_made: int = 0
    processing_time: Optional[float] = None
    warnings: Optional[List[str]] = None


class AudioSteganographyEngine:
    """
    Advanced Audio Steganography Engine
    
    Provides a comprehensive solution for hiding and extracting data in audio files
    with multiple techniques, security levels, error recovery, and anti-detection features.
    """
    
    # Protocol version for compatibility
    PROTOCOL_VERSION = "3.0"
    
    # Magic header for identifying steganographic data
    MAGIC_HEADER = b'INVV_AUD'
    
    # Maximum file size to prevent memory issues (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        """
        Initialize the audio steganography engine.
        
        Args:
            security_level: Security level for encryption
        """
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.security_level = security_level
        
        # Initialize components
        self.encryption_engine = EncryptionEngine(security_level)
        self.audio_processor = AudioProcessor()
        self.technique_factory = EmbeddingTechniqueFactory(self.logger)
        
        # Cache for repeated operations
        self._audio_cache = {}
        self._capacity_cache = {}
        
        self.logger.info(f"Audio steganography engine initialized (v{self.PROTOCOL_VERSION}, {security_level.value} security)")
    
    def hide_data(self, audio_path: Path, data: Union[bytes, str, Path], 
                  output_path: Path, config: EmbeddingConfig) -> EmbeddingResult:
        """
        Hide data in audio file using specified configuration.
        
        Args:
            audio_path: Path to carrier audio file
            data: Data to hide (bytes, string, or file path)
            output_path: Path for output audio file
            config: Embedding configuration
            
        Returns:
            EmbeddingResult with operation status and details
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting audio steganography: {audio_path.name} -> {output_path.name}")
            
            # Validate inputs
            validation_result = self._validate_inputs(audio_path, output_path, config)
            if not validation_result.success:
                return validation_result
            
            # Prepare data
            data_bytes = self._prepare_data(data)
            if data_bytes is None:
                return EmbeddingResult(False, "Failed to prepare data for embedding")
            
            # Analyze carrier audio
            audio_info = self._analyze_carrier(audio_path)
            if audio_info is None:
                return EmbeddingResult(False, "Failed to analyze carrier audio file")
            
            # Check capacity
            capacity_check = self._check_capacity(audio_info, data_bytes, config)
            if not capacity_check.success:
                return capacity_check
            
            # Apply mode-specific configuration
            config = self._apply_mode_config(config)
            
            # Load audio
            audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
            
            # Create embedding technique with appropriate parameters
            technique_params = {}
            if config.technique == 'lsb':
                technique_params['randomize'] = config.randomize_positions
            
            technique = self.technique_factory.create_technique(
                config.technique,
                **technique_params
            )
            
            if technique is None:
                return EmbeddingResult(False, f"Failed to create {config.technique} technique")
            
            # Prepare data with metadata and encryption
            prepared_data = self._prepare_data_for_embedding(data_bytes, config, audio_info)
            
            # Perform embedding with redundancy
            result_audio = self._embed_with_redundancy(
                audio_data, prepared_data, config, technique, sample_rate
            )
            
            if result_audio is None:
                return EmbeddingResult(False, "Embedding failed")
            
            # Apply anti-detection if enabled
            if config.anti_detection:
                result_audio = self._apply_anti_detection(result_audio, sample_rate)
            
            # Save result
            save_success = self.audio_processor.save_audio(
                result_audio, sample_rate, output_path
            )
            
            if not save_success:
                return EmbeddingResult(False, "Failed to save output audio")
            
            # Calculate results
            processing_time = time.time() - start_time
            capacity_used = len(prepared_data)
            capacity_total = technique.calculate_capacity(audio_data, sample_rate)
            
            # Generate anti-detection score
            anti_detection_score = None
            if config.anti_detection:
                anti_detection_score = self._calculate_detection_risk(result_audio, audio_data)
            
            # Check for warnings
            warnings = self._generate_warnings(audio_path, output_path, config)
            
            result = EmbeddingResult(
                success=True,
                message=f"Successfully embedded {len(data_bytes)} bytes using {config.technique}",
                technique_used=config.technique,
                mode_used=config.mode,
                capacity_used=capacity_used,
                capacity_total=capacity_total,
                redundancy_level=config.redundancy_level,
                anti_detection_score=anti_detection_score,
                processing_time=processing_time,
                warnings=warnings
            )
            
            self.logger.info(f"Embedding completed: {result.capacity_utilization:.1f}% capacity used")
            return result
            
        except Exception as e:
            error_msg = f"Audio steganography failed: {e}"
            self.logger.error(error_msg)
            return EmbeddingResult(False, error_msg)
    
    def extract_data(self, audio_path: Path, config: EmbeddingConfig,
                    expected_size: Optional[int] = None, 
                    max_attempts: int = 5) -> ExtractionResult:
        """
        Extract data from audio file with advanced recovery.
        
        Args:
            audio_path: Path to audio file containing hidden data
            config: Extraction configuration (mainly password)
            expected_size: Expected size in bytes (if known)
            max_attempts: Maximum extraction attempts with different strategies
            
        Returns:
            ExtractionResult with extracted data and status
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting audio extraction: {audio_path.name}")
            
            # Validate input
            if not audio_path.exists():
                return ExtractionResult(False, message=f"Audio file not found: {audio_path}")
            
            # Analyze audio
            is_suitable, warnings = self.audio_processor.validate_for_steganography(audio_path)
            if not is_suitable:
                self.logger.warning("Audio may not be suitable for extraction")
            
            # Load audio
            audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
            
            # Determine techniques to try
            techniques_to_try = self._determine_extraction_techniques(config.technique)
            
            # Try extraction with different strategies
            for attempt, technique_name in enumerate(techniques_to_try, 1):
                self.logger.debug(f"Extraction attempt {attempt}: {technique_name}")
                
                # Create technique with appropriate parameters
                technique_params = {}
                if technique_name == 'lsb':
                    technique_params['randomize'] = config.randomize_positions
                    self.logger.debug(f"Setting LSB randomize={config.randomize_positions} for extraction")
                
                technique = self.technique_factory.create_technique(technique_name, **technique_params)
                if technique is None:
                    continue
                
                # Try different extraction strategies
                for strategy in self._get_extraction_strategies(max_attempts):
                    try:
                        result = self._attempt_extraction(
                            audio_data, config, technique, sample_rate,
                            expected_size, strategy, attempt
                        )
                        
                        if result.success:
                            processing_time = time.time() - start_time
                            result.processing_time = processing_time
                            result.technique_detected = technique_name
                            result.attempts_made = attempt
                            result.warnings = warnings
                            
                            self.logger.info(f"Extraction successful: {len(result.data) if result.data else 0} bytes recovered")
                            return result
                            
                    except Exception as e:
                        self.logger.debug(f"Strategy {strategy} failed: {e}")
                        continue
            
            # All attempts failed
            processing_time = time.time() - start_time
            error_msg = self._generate_extraction_error_message(techniques_to_try, max_attempts)
            
            return ExtractionResult(
                success=False,
                message=error_msg,
                attempts_made=max_attempts * len(techniques_to_try),
                processing_time=processing_time,
                warnings=warnings
            )
            
        except Exception as e:
            error_msg = f"Audio extraction failed: {e}"
            self.logger.error(error_msg)
            return ExtractionResult(False, message=error_msg)
    
    def analyze_capacity(self, audio_path: Path, technique: str = 'lsb') -> Dict[str, Any]:
        """
        Analyze audio file capacity for steganography.
        
        Args:
            audio_path: Path to audio file
            technique: Embedding technique to analyze
            
        Returns:
            Dictionary with capacity information
        """
        try:
            # Use cached result if available
            cache_key = f"{audio_path}_{technique}"
            if cache_key in self._capacity_cache:
                return self._capacity_cache[cache_key]
            
            # Get detailed capacity info
            capacity_info = self.audio_processor.get_capacity_info(audio_path, technique)
            
            # Add additional analysis
            audio_info = self.audio_processor.analyze_audio_file(audio_path)
            
            # Check format suitability
            format_score = 1.0 if audio_info.is_lossless else 0.3
            
            # Calculate quality metrics
            quality_metrics = {
                'format_suitability': format_score,
                'duration_rating': min(audio_info.duration / 30.0, 1.0),  # 30s = full rating
                'sample_rate_rating': min(audio_info.sample_rate / 44100.0, 1.0),
                'channels_rating': 1.0 if audio_info.channels <= 2 else 0.8
            }
            
            overall_quality = np.mean(list(quality_metrics.values()))
            
            # Combine results
            result = {
                **capacity_info,
                'file_info': {
                    'path': str(audio_path),
                    'format': audio_info.format,
                    'duration': audio_info.duration,
                    'sample_rate': audio_info.sample_rate,
                    'channels': audio_info.channels,
                    'file_size': audio_info.file_size,
                    'is_lossless': audio_info.is_lossless
                },
                'quality_metrics': quality_metrics,
                'overall_suitability': overall_quality,
                'recommendations': self._generate_capacity_recommendations(audio_info, capacity_info)
            }
            
            # Cache result
            self._capacity_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Capacity analysis failed: {e}")
            return {'error': str(e)}
    
    def get_available_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available embedding techniques."""
        return self.technique_factory.get_available_techniques()
    
    def get_available_modes(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available steganography modes."""
        modes = {}
        for mode in StegoMode:
            modes[mode.code] = {
                'name': mode.code.title(),
                'description': mode.description,
                'redundancy': mode.redundancy,
                'error_correction': mode.error_correction,
                'anti_detection': mode.anti_detection
            }
        return modes
    
    def create_config(self, **kwargs) -> EmbeddingConfig:
        """
        Create embedding configuration with sensible defaults.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            EmbeddingConfig instance
        """
        return EmbeddingConfig(**kwargs)
    
    def _validate_inputs(self, audio_path: Path, output_path: Path, 
                        config: EmbeddingConfig) -> EmbeddingResult:
        """Validate input parameters."""
        try:
            # Check audio file
            if not audio_path.exists():
                return EmbeddingResult(False, f"Carrier audio file not found: {audio_path}")
            
            if audio_path.stat().st_size > self.MAX_FILE_SIZE:
                return EmbeddingResult(False, f"Audio file too large: {audio_path.stat().st_size} > {self.MAX_FILE_SIZE}")
            
            # Check password
            if not config.password or len(config.password) < 4:
                return EmbeddingResult(False, "Password must be at least 4 characters long")
            
            # Check technique
            if config.technique not in self.technique_factory._techniques:
                return EmbeddingResult(False, f"Unknown technique: {config.technique}")
            
            # Check output path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            return EmbeddingResult(True, "Validation passed")
            
        except Exception as e:
            return EmbeddingResult(False, f"Validation failed: {e}")
    
    def _prepare_data(self, data: Union[bytes, str, Path]) -> Optional[bytes]:
        """Prepare data for embedding."""
        try:
            if isinstance(data, bytes):
                return data
            elif isinstance(data, str):
                return data.encode('utf-8')
            elif isinstance(data, Path):
                if not data.exists():
                    self.logger.error(f"Data file not found: {data}")
                    return None
                return data.read_bytes()
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return None
    
    def _analyze_carrier(self, audio_path: Path) -> Optional[AudioInfo]:
        """Analyze carrier audio file."""
        try:
            return self.audio_processor.analyze_audio_file(audio_path)
        except Exception as e:
            self.logger.error(f"Carrier analysis failed: {e}")
            return None
    
    def _check_capacity(self, audio_info: AudioInfo, data: bytes, 
                       config: EmbeddingConfig) -> EmbeddingResult:
        """Check if audio has sufficient capacity."""
        try:
            # Calculate required capacity with overhead
            data_size = len(data)
            header_overhead = 256  # Estimated header size
            encryption_overhead = len(data) * 0.1  # 10% encryption overhead
            redundancy_overhead = data_size * (config.redundancy_level - 1)
            error_correction_overhead = data_size * 0.1 if config.error_correction else 0
            
            total_required = int(data_size + header_overhead + encryption_overhead + 
                               redundancy_overhead + error_correction_overhead)
            
            if total_required > audio_info.capacity_bytes:
                return EmbeddingResult(
                    False, 
                    f"Data too large: {total_required} bytes required > {audio_info.capacity_bytes} available"
                )
            
            return EmbeddingResult(True, "Capacity check passed")
            
        except Exception as e:
            return EmbeddingResult(False, f"Capacity check failed: {e}")
    
    def _apply_mode_config(self, config: EmbeddingConfig) -> EmbeddingConfig:
        """Apply mode-specific configuration only if not explicitly set by user."""
        # CRITICAL FIX: Don't overwrite explicitly set user parameters
        # Only apply mode defaults if user hasn't explicitly set the values
        
        for mode in StegoMode:
            if mode.code == config.mode:
                # Store original user values to check if they were explicitly set
                default_config = EmbeddingConfig()
                
                # Only apply mode settings if user hasn't changed from defaults
                # This ensures multimedia dialog explicit settings are preserved
                if config.redundancy_level == default_config.redundancy_level:
                    config.redundancy_level = mode.redundancy
                    self.logger.debug(f"Applied mode redundancy_level: {mode.redundancy}")
                else:
                    self.logger.debug(f"Preserving user redundancy_level: {config.redundancy_level}")
                    
                if config.error_correction == default_config.error_correction:
                    config.error_correction = mode.error_correction
                    self.logger.debug(f"Applied mode error_correction: {mode.error_correction}")
                else:
                    self.logger.debug(f"Preserving user error_correction: {config.error_correction}")
                    
                if config.anti_detection == default_config.anti_detection:
                    config.anti_detection = mode.anti_detection
                    self.logger.debug(f"Applied mode anti_detection: {mode.anti_detection}")
                else:
                    self.logger.debug(f"Preserving user anti_detection: {config.anti_detection}")
                break
                
        return config
    
    def _prepare_data_for_embedding(self, data: bytes, config: EmbeddingConfig, 
                                   audio_info: AudioInfo) -> bytes:
        """Prepare data with metadata and encryption."""
        try:
            # Create metadata
            metadata = {
                'version': self.PROTOCOL_VERSION,
                'technique': config.technique,
                'redundancy': config.redundancy_level,
                'error_correction': config.error_correction,
                'original_size': len(data),
                'timestamp': int(os.path.getmtime(audio_info.path)),
                'checksum': hashlib.sha256(data).hexdigest()
            }
            
            # Serialize metadata
            metadata_json = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
            metadata_size = len(metadata_json)
            
            # Encrypt data
            encrypted_data = self.encryption_engine.encrypt_with_metadata(data, config.password)
            
            # Create header: magic + version + metadata_size + data_size + metadata + encrypted_data
            header = (
                self.MAGIC_HEADER +                         # 8 bytes
                struct.pack('<H', metadata_size) +          # 2 bytes
                struct.pack('<I', len(encrypted_data)) +    # 4 bytes
                metadata_json +                             # Variable
                encrypted_data                              # Variable
            )
            
            return header
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    def _embed_with_redundancy(self, audio_data: np.ndarray, data: bytes, 
                              config: EmbeddingConfig, technique: BaseEmbeddingTechnique,
                              sample_rate: int) -> Optional[np.ndarray]:
        """Embed data with redundancy across the audio."""
        try:
            channels, samples = audio_data.shape
            modified_audio = audio_data.copy()
            
            # For redundancy level 1, just embed normally
            if config.redundancy_level == 1:
                result_audio = technique.embed(modified_audio, data, config.password, sample_rate)
                if result_audio is not None:
                    self.logger.debug(f"Embedded single copy with original password")
                    return result_audio
                else:
                    self.logger.error("Failed to embed single copy")
                    return None
            
            # For multiple redundancy, embed the same data multiple times across the audio
            # CRITICAL FIX: All copies should use the SAME password so they can be extracted
            # with the original password. Redundancy is for reliability, not security.
            segment_size = samples // config.redundancy_level
            
            for copy_idx in range(config.redundancy_level):
                start_idx = copy_idx * segment_size
                end_idx = start_idx + segment_size if copy_idx < config.redundancy_level - 1 else samples
                
                # CRITICAL FIX: Use the original password for all copies
                # This ensures that standard extraction (which uses the original password)
                # can find and extract the magic header from any of the copies
                password_to_use = config.password
                
                # Embed in segment
                segment_audio = modified_audio[:, start_idx:end_idx]
                
                # Update technique with appropriate password
                result_segment = technique.embed(
                    segment_audio, data, password_to_use, sample_rate
                )
                
                if result_segment is not None:
                    modified_audio[:, start_idx:end_idx] = result_segment
                    self.logger.debug(f"Embedded redundant copy {copy_idx + 1}/{config.redundancy_level} with original password")
                else:
                    self.logger.warning(f"Failed to embed copy {copy_idx + 1}")
            
            return modified_audio
            
        except Exception as e:
            self.logger.error(f"Redundant embedding failed: {e}")
            return None
    
    def _apply_anti_detection(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply anti-detection measures."""
        try:
            # Add subtle noise to mask statistical signatures
            noise_power = np.mean(np.abs(audio_data)) * 0.001  # Very subtle
            noise = np.random.normal(0, noise_power, audio_data.shape)
            
            # Apply gentle filtering to smooth statistical artifacts
            from scipy import signal as scipy_signal
            b, a = scipy_signal.butter(8, 0.99, 'low')
            
            channels, samples = audio_data.shape
            processed_audio = audio_data + noise
            
            for channel in range(channels):
                processed_audio[channel] = scipy_signal.filtfilt(b, a, processed_audio[channel])
            
            return processed_audio
            
        except Exception as e:
            self.logger.warning(f"Anti-detection processing failed: {e}")
            return audio_data
    
    def _calculate_detection_risk(self, modified_audio: np.ndarray, 
                                 original_audio: np.ndarray) -> float:
        """Calculate detection risk score (0.0 = undetectable, 1.0 = easily detected)."""
        try:
            # Calculate statistical differences
            diff = modified_audio - original_audio
            
            # Various detection metrics
            mse = float(np.mean(diff ** 2))
            max_diff = float(np.max(np.abs(diff)))
            std_ratio = float(np.std(modified_audio) / np.std(original_audio))
            
            # Combine metrics into risk score
            risk_score = min(1.0, (mse * 1000) + (max_diff * 10) + abs(1.0 - std_ratio))
            
            return risk_score
            
        except Exception:
            return 0.5  # Unknown risk
    
    def _generate_warnings(self, audio_path: Path, output_path: Path, 
                          config: EmbeddingConfig) -> List[str]:
        """Generate warnings for the embedding operation."""
        warnings = []
        
        # Check file formats
        input_format = audio_path.suffix.lower()
        output_format = output_path.suffix.lower()
        
        lossy_formats = {'.mp3', '.aac', '.ogg', '.m4a'}
        
        if input_format in lossy_formats:
            warnings.append(f"Input format ({input_format}) is lossy and may affect quality")
        
        if output_format in lossy_formats:
            warnings.append(f"Output format ({output_format}) is lossy and may corrupt hidden data")
        
        if input_format != output_format:
            warnings.append("Format conversion may affect data integrity")
        
        # Check configuration
        if config.redundancy_level == 1:
            warnings.append("Low redundancy level - data may be vulnerable to corruption")
        
        if not config.error_correction:
            warnings.append("Error correction disabled - recovery may be difficult")
        
        return warnings
    
    def _determine_extraction_techniques(self, requested_technique: str) -> List[str]:
        """Determine which techniques to try for extraction."""
        if requested_technique == 'auto':
            # Try all available techniques in order of reliability
            return ['lsb', 'spread_spectrum', 'phase_coding', 'echo']
        else:
            # Try requested technique first, then fallbacks
            techniques = [requested_technique]
            if requested_technique != 'lsb':
                techniques.append('lsb')  # LSB as fallback
            return techniques
    
    def _try_different_redundancy_levels(self, audio_data: np.ndarray, config: EmbeddingConfig,
                                        technique: BaseEmbeddingTechnique, sample_rate: int) -> Optional[bytes]:
        """Try extraction with different redundancy levels to handle config mismatches."""
        try:
            # Try different redundancy levels in order of common usage
            redundancy_levels = [config.redundancy_level, 2, 1, 3, 5]  # Try original first, then common values
            
            for redundancy in redundancy_levels:
                self.logger.debug(f"Trying redundancy level {redundancy}")
                
                # Create temporary config with this redundancy level
                temp_config = EmbeddingConfig(
                    technique=config.technique,
                    mode=config.mode,
                    password=config.password,
                    redundancy_level=redundancy,
                    error_correction=config.error_correction,
                    anti_detection=config.anti_detection,
                    randomize_positions=config.randomize_positions,
                    custom_seed=config.custom_seed,
                    quality_optimization=config.quality_optimization
                )
                
                # Try standard extraction with this redundancy level
                extracted_data = self._extract_with_redundancy(audio_data, temp_config, technique, sample_rate)
                
                if extracted_data:
                    # Try to decrypt and verify
                    decrypted_data = self._decrypt_and_verify(extracted_data, config.password)
                    if decrypted_data:
                        self.logger.info(f"✅ Success with redundancy level {redundancy}")
                        return extracted_data
                    else:
                        self.logger.debug(f"Extraction with redundancy {redundancy} failed decryption")
                else:
                    self.logger.debug(f"No data extracted with redundancy level {redundancy}")
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Redundancy level fallback failed: {e}")
            return None
    
    def _get_extraction_strategies(self, max_attempts: int) -> List[str]:
        """Get extraction strategies to try."""
        strategies = ['standard']
        if max_attempts > 1:
            strategies.append('redundancy_fallback')  # Try different redundancy levels
        if max_attempts > 2:
            strategies.append('error_correction')
        if max_attempts > 3:
            strategies.append('redundant')
        if max_attempts > 4:
            strategies.append('partial')
        if max_attempts > 5:
            strategies.append('brute_force')
        return strategies
    
    def _attempt_extraction(self, audio_data: np.ndarray, config: EmbeddingConfig,
                           technique: BaseEmbeddingTechnique, sample_rate: int,
                           expected_size: Optional[int], strategy: str, 
                           attempt: int) -> ExtractionResult:
        """Attempt extraction with specific strategy."""
        try:
            self.logger.debug(f"Extraction attempt {attempt} using {strategy} strategy")
            self.logger.debug(f"Audio shape: {audio_data.shape}, sample_rate: {sample_rate}")
            self.logger.debug(f"Config - redundancy: {config.redundancy_level}, technique: {config.technique}")
            
            # Try different approaches based on strategy
            if strategy == 'standard':
                # CRITICAL FIX: Handle redundancy in standard extraction
                # If redundancy_level > 1, try extracting from the first segment
                if config.redundancy_level > 1:
                    self.logger.debug(f"Redundancy detected ({config.redundancy_level}), trying first segment extraction")
                    channels, samples = audio_data.shape
                    segment_size = samples // config.redundancy_level
                    first_segment = audio_data[:, :segment_size]
                    
                    # Try to extract header from first segment
                    header_size = 8 + 2 + 4  # magic + metadata_size + data_size
                    self.logger.debug(f"Attempting to extract header from first segment ({header_size} bytes)...")
                    header_data = technique.extract(first_segment, config.password, sample_rate, header_size)
                    
                    self.logger.debug(f"Header extraction result: {len(header_data) if header_data else 0} bytes")
                    if header_data and len(header_data) >= header_size:
                        self.logger.debug(f"Header bytes: {header_data[:20].hex() if header_data else 'None'}...")
                        if header_data.startswith(self.MAGIC_HEADER):
                            self.logger.debug("✅ Magic header found in first segment!")
                            # Parse the header to get total size needed
                            try:
                                offset = len(self.MAGIC_HEADER)
                                metadata_size = struct.unpack('<H', header_data[offset:offset + 2])[0]
                                offset += 2
                                data_size = struct.unpack('<I', header_data[offset:offset + 4])[0]
                                
                                self.logger.debug(f"Parsed header: metadata_size={metadata_size}, data_size={data_size}")
                                
                                # Calculate total size needed
                                total_size = len(self.MAGIC_HEADER) + 2 + 4 + metadata_size + data_size
                                self.logger.debug(f"Total size needed: {total_size} bytes")
                                
                                # Extract the exact amount needed from first segment
                                extracted_data = technique.extract(first_segment, config.password, sample_rate, total_size)
                                self.logger.debug(f"Full extraction from first segment: {len(extracted_data) if extracted_data else 0} bytes")
                            except Exception as parse_error:
                                self.logger.debug(f"Header parsing failed: {parse_error}, falling back to full segment extraction")
                                # Fallback to full segment extraction if parsing fails
                                extracted_data = technique.extract(first_segment, config.password, sample_rate)
                        else:
                            self.logger.debug(f"❌ Magic header not found in first segment. Expected: {self.MAGIC_HEADER.hex()}, got: {header_data[:8].hex() if len(header_data) >= 8 else 'too short'}")
                            # Try full segment extraction
                            extracted_data = technique.extract(first_segment, config.password, sample_rate)
                    else:
                        self.logger.debug("❌ Header extraction from first segment failed or too short")
                        # Try full segment extraction
                        extracted_data = technique.extract(first_segment, config.password, sample_rate)
                else:
                    # No redundancy, extract from entire audio
                    # First, try to extract just the header to determine the actual size needed
                    header_size = 8 + 2 + 4  # magic + metadata_size + data_size
                    self.logger.debug(f"Attempting to extract header ({header_size} bytes)...")
                    header_data = technique.extract(audio_data, config.password, sample_rate, header_size)
                    
                    self.logger.debug(f"Header extraction result: {len(header_data) if header_data else 0} bytes")
                    if header_data and len(header_data) >= header_size:
                        self.logger.debug(f"Header bytes: {header_data[:20].hex() if header_data else 'None'}...")
                        if header_data.startswith(self.MAGIC_HEADER):
                            self.logger.debug("✅ Magic header found!")
                            # Parse the header to get total size needed
                            try:
                                offset = len(self.MAGIC_HEADER)
                                metadata_size = struct.unpack('<H', header_data[offset:offset + 2])[0]
                                offset += 2
                                data_size = struct.unpack('<I', header_data[offset:offset + 4])[0]
                                
                                self.logger.debug(f"Parsed header: metadata_size={metadata_size}, data_size={data_size}")
                                
                                # Calculate total size needed
                                total_size = len(self.MAGIC_HEADER) + 2 + 4 + metadata_size + data_size
                                self.logger.debug(f"Total size needed: {total_size} bytes")
                                
                                # Extract the exact amount needed
                                extracted_data = technique.extract(audio_data, config.password, sample_rate, total_size)
                                self.logger.debug(f"Full extraction result: {len(extracted_data) if extracted_data else 0} bytes")
                            except Exception as parse_error:
                                self.logger.debug(f"Header parsing failed: {parse_error}, falling back to full extraction")
                                # Fallback to full extraction if parsing fails
                                extracted_data = technique.extract(audio_data, config.password, sample_rate)
                        else:
                            self.logger.debug(f"❌ Magic header not found. Expected: {self.MAGIC_HEADER.hex()}, got: {header_data[:8].hex() if len(header_data) >= 8 else 'too short'}")
                            # Fallback to full extraction if header parsing fails
                            extracted_data = technique.extract(audio_data, config.password, sample_rate)
                    else:
                        self.logger.debug("❌ Header extraction failed or too short")
                        # Fallback to full extraction if header parsing fails
                        extracted_data = technique.extract(audio_data, config.password, sample_rate)
            elif strategy == 'redundancy_fallback':
                self.logger.debug("Using redundancy fallback strategy - trying different redundancy levels")
                extracted_data = self._try_different_redundancy_levels(audio_data, config, technique, sample_rate)
            elif strategy == 'redundant':
                self.logger.debug("Using redundant extraction strategy")
                extracted_data = self._extract_with_redundancy(audio_data, config, technique, sample_rate)
            elif strategy == 'error_correction':
                self.logger.debug("Using error correction extraction strategy")
                extracted_data = self._extract_with_error_correction(audio_data, config, technique, sample_rate)
            else:
                self.logger.debug(f"Using {strategy} extraction strategy")
                extracted_data = technique.extract(audio_data, config.password, sample_rate, expected_size)
            
            self.logger.debug(f"Strategy {strategy} extracted {len(extracted_data) if extracted_data else 0} bytes")
            
            if extracted_data:
                self.logger.debug(f"Attempting to decrypt and verify...")
                # Try to decrypt and verify
                original_data = self._decrypt_and_verify(extracted_data, config.password)
                if original_data:
                    self.logger.debug(f"✅ Decryption successful: {len(original_data)} bytes")
                    return ExtractionResult(
                        success=True,
                        data=original_data,
                        message=f"Successfully extracted using {strategy} strategy",
                        recovery_method=strategy,
                        confidence_score=1.0
                    )
                else:
                    self.logger.debug(f"❌ Decryption/verification failed")
            else:
                self.logger.debug(f"❌ No data extracted by {strategy} strategy")
            
            return ExtractionResult(False, message=f"{strategy} strategy failed")
            
        except Exception as e:
            self.logger.error(f"Exception in {strategy} strategy: {e}")
            return ExtractionResult(False, message=f"{strategy} strategy error: {e}")
    
    def _extract_with_redundancy(self, audio_data: np.ndarray, config: EmbeddingConfig,
                                technique: BaseEmbeddingTechnique, sample_rate: int) -> Optional[bytes]:
        """Extract using redundant copies with voting."""
        try:
            channels, samples = audio_data.shape
            redundancy_level = config.redundancy_level
            
            # For single redundancy, extract normally
            if redundancy_level == 1:
                return technique.extract(audio_data, config.password, sample_rate)
            
            # For multiple redundancy, extract from each segment
            segment_size = samples // redundancy_level
            extracted_copies = []
            
            for copy_idx in range(redundancy_level):
                start_idx = copy_idx * segment_size
                end_idx = start_idx + segment_size if copy_idx < redundancy_level - 1 else samples
                
                # CRITICAL FIX: Use original password for all copies
                # This matches the embedding logic where all copies use the same password
                password_to_use = config.password
                
                # Extract from segment
                segment_audio = audio_data[:, start_idx:end_idx]
                copy_data = technique.extract(segment_audio, password_to_use, sample_rate)
                
                if copy_data:
                    extracted_copies.append(copy_data)
                    self.logger.debug(f"Extracted redundant copy {copy_idx + 1}/{redundancy_level} with original password")
                else:
                    self.logger.debug(f"Failed to extract copy {copy_idx + 1}")
            
            # Vote on most common result
            if extracted_copies:
                from collections import Counter
                counter = Counter(extracted_copies)
                most_common = counter.most_common(1)[0]
                
                # Log voting results for debugging
                self.logger.debug(f"Redundancy voting: {len(extracted_copies)} copies extracted, most common has {most_common[1]} votes")
                
                if most_common[1] >= 2 or len(extracted_copies) == 1:
                    return most_common[0]
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Redundant extraction failed: {e}")
            return None
    
    def _extract_with_error_correction(self, audio_data: np.ndarray, config: EmbeddingConfig,
                                     technique: BaseEmbeddingTechnique, sample_rate: int) -> Optional[bytes]:
        """Extract with error correction."""
        try:
            # Standard extraction first
            extracted_data = technique.extract(audio_data, config.password, sample_rate)
            
            if extracted_data and config.error_correction:
                # Apply error correction (simplified implementation)
                return self._apply_error_correction(extracted_data)
            
            return extracted_data
            
        except Exception as e:
            self.logger.debug(f"Error correction extraction failed: {e}")
            return None
    
    def _apply_error_correction(self, data: bytes) -> bytes:
        """Apply simple error correction."""
        try:
            # This is a placeholder - real implementation would use Reed-Solomon or similar
            return data
        except Exception:
            return data
    
    def _decrypt_and_verify(self, encrypted_data: bytes, password: str) -> Optional[bytes]:
        """Decrypt extracted data and verify integrity."""
        try:
            self.logger.debug(f"Starting decryption/verification for {len(encrypted_data)} bytes")
            
            # Parse header
            if len(encrypted_data) < 14:  # Minimum header size
                self.logger.debug(f"❌ Data too short for header: {len(encrypted_data)} < 14 bytes")
                return None
            
            if not encrypted_data.startswith(self.MAGIC_HEADER):
                self.logger.debug(f"❌ Magic header mismatch. Expected: {self.MAGIC_HEADER.hex()}, got: {encrypted_data[:8].hex()}")
                return None
            
            self.logger.debug("✅ Magic header verified")
            
            offset = len(self.MAGIC_HEADER)
            metadata_size = struct.unpack('<H', encrypted_data[offset:offset + 2])[0]
            offset += 2
            data_size = struct.unpack('<I', encrypted_data[offset:offset + 4])[0]
            offset += 4
            
            self.logger.debug(f"Header parsed: metadata_size={metadata_size}, data_size={data_size}")
            
            # Validate sizes
            expected_total = len(self.MAGIC_HEADER) + 2 + 4 + metadata_size + data_size
            if len(encrypted_data) < expected_total:
                self.logger.debug(f"❌ Data length mismatch. Expected: {expected_total}, got: {len(encrypted_data)}")
                return None
            
            # Extract metadata and encrypted data
            metadata_json = encrypted_data[offset:offset + metadata_size]
            offset += metadata_size
            encrypted_payload = encrypted_data[offset:offset + data_size]
            
            self.logger.debug(f"Extracted metadata ({len(metadata_json)} bytes) and payload ({len(encrypted_payload)} bytes)")
            
            # Parse metadata
            try:
                metadata = json.loads(metadata_json.decode('utf-8'))
                self.logger.debug(f"Metadata parsed: {metadata}")
            except Exception as meta_error:
                self.logger.debug(f"❌ Metadata parsing failed: {meta_error}")
                return None
            
            # Decrypt payload
            self.logger.debug("Attempting decryption...")
            try:
                decrypted_data = self.encryption_engine.decrypt_with_metadata(encrypted_payload, password)
                self.logger.debug(f"✅ Decryption successful: {len(decrypted_data)} bytes")
            except Exception as decrypt_error:
                self.logger.debug(f"❌ Decryption failed: {decrypt_error}")
                return None
            
            # Verify checksum
            if 'checksum' in metadata:
                expected_checksum = metadata['checksum']
                actual_checksum = hashlib.sha256(decrypted_data).hexdigest()
                self.logger.debug(f"Checksum verification: expected={expected_checksum[:16]}..., actual={actual_checksum[:16]}...")
                if expected_checksum != actual_checksum:
                    self.logger.warning("❌ Checksum mismatch - data may be corrupted")
                    return None
                else:
                    self.logger.debug("✅ Checksum verified")
            else:
                self.logger.debug("No checksum found in metadata")
            
            return decrypted_data
            
        except Exception as e:
            self.logger.debug(f"❌ Decryption/verification failed with exception: {e}")
            return None
    
    def _generate_extraction_error_message(self, techniques_tried: List[str], 
                                         max_attempts: int) -> str:
        """Generate detailed error message for failed extraction."""
        return (
            f"All extraction attempts failed. Tried {len(techniques_tried)} techniques "
            f"with {max_attempts} strategies each. Possible causes:\n"
            f"• Incorrect password\n"
            f"• File doesn't contain hidden data\n"
            f"• Audio format was changed after embedding\n"
            f"• Data corrupted by lossy compression\n"
            f"• Different technique was used for embedding"
        )
    
    def _generate_capacity_recommendations(self, audio_info: AudioInfo, 
                                         capacity_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations for capacity optimization."""
        recommendations = []
        
        if not audio_info.is_lossless:
            recommendations.append("Convert to lossless format (WAV/FLAC) for better reliability")
        
        if capacity_info.get('effective_bytes', 0) < 1024:
            recommendations.append("Audio file too small - use longer audio for more capacity")
        
        if audio_info.sample_rate < 22050:
            recommendations.append("Higher sample rate audio would provide more capacity")
        
        if audio_info.channels == 1:
            recommendations.append("Stereo audio would provide approximately 2x more capacity")
        
        return recommendations


def create_audio_steganography_engine(security_level: SecurityLevel = SecurityLevel.STANDARD) -> AudioSteganographyEngine:
    """Factory function to create audio steganography engine."""
    return AudioSteganographyEngine(security_level)
