"""
Multimedia Analyzer
Analyzes video and audio files for steganography capacity and suitability.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Lazy loading flags for heavy dependencies
_cv2_loaded = False
_librosa_loaded = False
_pydub_loaded = False
_ffmpeg_loaded = False

# Global references for lazy-loaded modules
_cv2 = None
_librosa = None
_AudioSegment = None
_ffmpeg = None

from utils.logger import Logger
from utils.error_handler import ErrorHandler


def _load_cv2():
    """Lazy load OpenCV library."""
    global _cv2, _cv2_loaded
    if not _cv2_loaded:
        try:
            import cv2
            _cv2 = cv2
            _cv2_loaded = True
        except ImportError as e:
            raise ImportError(
                f"OpenCV not available: {e}. "
                "Please install: pip install opencv-python"
            )
    return _cv2


def _load_librosa():
    """Lazy load librosa library."""
    global _librosa, _librosa_loaded
    if not _librosa_loaded:
        try:
            import librosa
            _librosa = librosa
            _librosa_loaded = True
        except ImportError as e:
            raise ImportError(
                f"Librosa not available: {e}. "
                "Please install: pip install librosa"
            )
    return _librosa


def _load_pydub():
    """Lazy load pydub library."""
    global _AudioSegment, _pydub_loaded
    if not _pydub_loaded:
        try:
            from pydub import AudioSegment
            _AudioSegment = AudioSegment
            _pydub_loaded = True
        except ImportError as e:
            raise ImportError(
                f"Pydub not available: {e}. "
                "Please install: pip install pydub"
            )
    return _AudioSegment


def _load_ffmpeg():
    """Lazy load ffmpeg library."""
    global _ffmpeg, _ffmpeg_loaded
    if not _ffmpeg_loaded:
        try:
            # Try to import ffmpeg-python package
            import ffmpeg as ffmpeg_module
            _ffmpeg = ffmpeg_module
            _ffmpeg_loaded = True
        except ImportError as e:
            # Set to None if not available - this is optional for future use
            _ffmpeg = None
            _ffmpeg_loaded = True
            # Log warning instead of raising error since it's not currently used
            import warnings
            warnings.warn(
                f"FFmpeg not available: {e}. "
                "Install with: pip install ffmpeg-python (optional for future features)",
                ImportWarning
            )
    return _ffmpeg


class MultimediaAnalyzer:
    """Analyzes multimedia files for steganography capacity and suitability."""
    
    # Supported formats
    VIDEO_FORMATS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    AUDIO_FORMATS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    
    # Singleton instance
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Implement singleton pattern for performance."""
        if cls._instance is None:
            cls._instance = super(MultimediaAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once to avoid repeated overhead
        if not self._initialized:
            self.logger = Logger()
            self.error_handler = ErrorHandler()
            MultimediaAnalyzer._initialized = True
    
    def analyze_video_file(self, video_path: Path) -> Dict[str, Any]:
        """
        Analyze video file for steganography capacity and properties.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Validate format
            if video_path.suffix.lower() not in self.VIDEO_FORMATS:
                raise ValueError(f"Unsupported video format: {video_path.suffix}")
            
            # Lazy load OpenCV for video analysis
            cv2 = _load_cv2()
            assert cv2 is not None, "OpenCV module is required for video analysis"
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise Exception("Failed to open video file")
            
            # Get video properties
            assert cv2 is not None, "OpenCV module is required for video properties"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Calculate capacity
            # Use 1 LSB per color channel per pixel for selected frames
            pixels_per_frame = width * height
            channels = 3  # RGB
            bits_per_frame = pixels_per_frame * channels  # 1 LSB per channel
            
            # Use every 10th frame to avoid detection (configurable)
            usable_frames = frame_count // 10
            total_capacity_bits = usable_frames * bits_per_frame
            total_capacity_bytes = total_capacity_bits // 8
            
            # Reserve space for metadata (1KB)
            metadata_overhead = 1024
            usable_capacity = max(0, total_capacity_bytes - metadata_overhead)
            
            # Analyze sample frames for quality metrics
            quality_metrics = self._analyze_video_quality(cap, frame_count)
            
            cap.release()
            
            # Get file size
            file_size = video_path.stat().st_size
            
            # Calculate suitability score (1-10)
            suitability_score = self._calculate_video_suitability(
                quality_metrics, duration, file_size, usable_capacity
            )
            
            analysis = {
                'file_path': str(video_path),
                'file_size': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'format': video_path.suffix.lower(),
                'width': width,
                'height': height,
                'frame_count': frame_count,
                'fps': fps,
                'duration_seconds': duration,
                'duration_formatted': self._format_duration(duration),
                'capacity_bytes': usable_capacity,
                'capacity_mb': usable_capacity / (1024 * 1024),
                'usable_frames': usable_frames,
                'quality_metrics': quality_metrics,
                'suitability_score': suitability_score,
                'recommendations': self._generate_video_recommendations(
                    suitability_score, quality_metrics, duration
                )
            }
            
            self.logger.info(f"Video analysis completed: {video_path.name}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {e}")
            return {'error': str(e), 'file_path': str(video_path)}
    
    def analyze_audio_file(self, audio_path: Path) -> Dict[str, Any]:
        """
        Analyze audio file for steganography capacity and properties.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Validate format
            if audio_path.suffix.lower() not in self.AUDIO_FORMATS:
                raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
            
            # Lazy load pydub for audio analysis
            AudioSegment = _load_pydub()
            assert AudioSegment is not None, "Pydub AudioSegment is required for audio analysis"
            audio = AudioSegment.from_file(str(audio_path))
            
            # Get basic properties
            duration_ms = len(audio)
            duration_seconds = duration_ms / 1000.0
            sample_rate = audio.frame_rate
            channels = audio.channels
            frame_width = audio.frame_width  # bytes per sample
            
            # Calculate total samples
            total_samples = int(sample_rate * duration_seconds * channels)
            
            # Calculate capacity using LSB method
            # Use 1 LSB per sample, but only use a portion to avoid detection
            usable_samples = total_samples // 4  # Use 25% of samples
            capacity_bits = usable_samples
            capacity_bytes = capacity_bits // 8
            
            # Reserve space for metadata
            metadata_overhead = 512  # 512 bytes
            usable_capacity = max(0, capacity_bytes - metadata_overhead)
            
            # Analyze audio quality with librosa
            quality_metrics = self._analyze_audio_quality(audio_path, sample_rate)
            
            # Get file size
            file_size = audio_path.stat().st_size
            
            # Calculate suitability score
            suitability_score = self._calculate_audio_suitability(
                quality_metrics, duration_seconds, file_size, usable_capacity
            )
            
            analysis = {
                'file_path': str(audio_path),
                'file_size': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'format': audio_path.suffix.lower(),
                'duration_seconds': duration_seconds,
                'duration_formatted': self._format_duration(duration_seconds),
                'sample_rate': sample_rate,
                'channels': channels,
                'frame_width': frame_width,
                'total_samples': total_samples,
                'capacity_bytes': usable_capacity,
                'capacity_mb': usable_capacity / (1024 * 1024),
                'usable_samples': usable_samples,
                'quality_metrics': quality_metrics,
                'suitability_score': suitability_score,
                'recommendations': self._generate_audio_recommendations(
                    suitability_score, quality_metrics, duration_seconds
                )
            }
            
            self.logger.info(f"Audio analysis completed: {audio_path.name}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return {'error': str(e), 'file_path': str(audio_path)}
    
    def _analyze_video_quality(self, cap, frame_count: int) -> Dict[str, float]:
        """Analyze video quality metrics from sample frames."""
        try:
            # Get cv2 reference from lazy loading
            cv2 = _load_cv2()
            assert cv2 is not None, "OpenCV module is required for video quality analysis"
            
            # Sample frames at different points
            sample_points = [0.1, 0.3, 0.5, 0.7, 0.9]
            brightness_values = []
            contrast_values = []
            noise_values = []
            
            for point in sample_points:
                frame_idx = int(frame_count * point)
                assert cv2 is not None, "OpenCV module is required for frame positioning"
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                ret, frame = cap.read()
                if ret:
                    # Convert to grayscale for analysis
                    assert cv2 is not None, "OpenCV module is required for color conversion"
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate metrics
                    brightness = np.mean(gray)
                    contrast = np.std(gray)
                    
                    # Estimate noise using Laplacian variance
                    assert cv2 is not None, "OpenCV module is required for Laplacian calculation"
                    noise = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    brightness_values.append(brightness)
                    contrast_values.append(contrast)
                    noise_values.append(noise)
            
            return {
                'avg_brightness': float(np.mean(brightness_values)),
                'avg_contrast': float(np.mean(contrast_values)),
                'avg_noise': float(np.mean(noise_values)),
                'brightness_std': float(np.std(brightness_values)),
                'contrast_std': float(np.std(contrast_values))
            }
            
        except Exception as e:
            self.logger.warning(f"Video quality analysis failed: {e}")
            return {
                'avg_brightness': 128.0,
                'avg_contrast': 64.0,
                'avg_noise': 100.0,
                'brightness_std': 10.0,
                'contrast_std': 10.0
            }
    
    def _analyze_audio_quality(self, audio_path: Path, sample_rate: int) -> Dict[str, float]:
        """Analyze audio quality metrics."""
        try:
            # Get librosa reference from lazy loading
            librosa = _load_librosa()
            assert librosa is not None, "Librosa module is required for audio quality analysis"
            
            # Load audio with librosa for analysis
            y, sr = librosa.load(str(audio_path), sr=sample_rate, duration=30.0)  # Analyze first 30 seconds
            
            # Calculate spectral features
            assert librosa is not None, "Librosa module is required for spectral features"
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # Calculate zero crossing rate
            assert librosa is not None, "Librosa module is required for zero crossing rate"
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Calculate RMS energy
            assert librosa is not None, "Librosa module is required for RMS calculation"
            rms = np.mean(librosa.feature.rms(y=y))
            
            # Calculate dynamic range
            dynamic_range = np.max(np.abs(y)) - np.min(np.abs(y))
            
            return {
                'spectral_centroid': float(spectral_centroid),
                'spectral_bandwidth': float(spectral_bandwidth),
                'spectral_rolloff': float(spectral_rolloff),
                'zero_crossing_rate': float(zcr),
                'rms_energy': float(rms),
                'dynamic_range': float(dynamic_range)
            }
            
        except Exception as e:
            self.logger.warning(f"Audio quality analysis failed: {e}")
            return {
                'spectral_centroid': 2000.0,
                'spectral_bandwidth': 1500.0,
                'spectral_rolloff': 4000.0,
                'zero_crossing_rate': 0.1,
                'rms_energy': 0.1,
                'dynamic_range': 0.8
            }
    
    def _calculate_video_suitability(self, quality_metrics: Dict, duration: float, 
                                   file_size: int, capacity: int) -> int:
        """Calculate video suitability score (1-10)."""
        score = 5  # Base score
        
        # Duration factor (longer videos are better)
        if duration > 300:  # 5+ minutes
            score += 2
        elif duration > 60:  # 1+ minute
            score += 1
        elif duration < 10:  # Very short
            score -= 1
        
        # Quality factors
        noise = quality_metrics.get('avg_noise', 100)
        contrast = quality_metrics.get('avg_contrast', 64)
        
        if noise > 200:  # High noise is good for hiding
            score += 1
        if contrast > 80:  # High contrast is good
            score += 1
        
        # Capacity factor
        if capacity > 50 * 1024 * 1024:  # 50MB+
            score += 1
        elif capacity < 1024 * 1024:  # Less than 1MB
            score -= 1
        
        return max(1, min(10, score))
    
    def _calculate_audio_suitability(self, quality_metrics: Dict, duration: float,
                                   file_size: int, capacity: int) -> int:
        """Calculate audio suitability score (1-10)."""
        score = 5  # Base score
        
        # Duration factor
        if duration > 180:  # 3+ minutes
            score += 2
        elif duration > 60:  # 1+ minute
            score += 1
        elif duration < 30:  # Very short
            score -= 1
        
        # Quality factors
        dynamic_range = quality_metrics.get('dynamic_range', 0.8)
        spectral_bandwidth = quality_metrics.get('spectral_bandwidth', 1500)
        
        if dynamic_range > 0.6:  # Good dynamic range
            score += 1
        if spectral_bandwidth > 2000:  # Rich frequency content
            score += 1
        
        # Capacity factor
        if capacity > 10 * 1024 * 1024:  # 10MB+
            score += 1
        elif capacity < 512 * 1024:  # Less than 512KB
            score -= 1
        
        return max(1, min(10, score))
    
    def _generate_video_recommendations(self, score: int, quality_metrics: Dict, 
                                      duration: float) -> list:
        """Generate recommendations for video steganography."""
        recommendations = []
        
        if score < 4:
            recommendations.append("Video has low suitability for steganography")
        elif score < 7:
            recommendations.append("Video has moderate suitability")
        else:
            recommendations.append("Video has high suitability for steganography")
        
        if duration < 60:
            recommendations.append("Longer videos provide better capacity and security")
        
        noise = quality_metrics.get('avg_noise', 100)
        if noise < 50:
            recommendations.append("Low noise levels may make hidden data more detectable")
        
        return recommendations
    
    def _generate_audio_recommendations(self, score: int, quality_metrics: Dict,
                                      duration: float) -> list:
        """Generate recommendations for audio steganography."""
        recommendations = []
        
        if score < 4:
            recommendations.append("Audio has low suitability for steganography")
        elif score < 7:
            recommendations.append("Audio has moderate suitability")
        else:
            recommendations.append("Audio has high suitability for steganography")
        
        if duration < 60:
            recommendations.append("Longer audio files provide better capacity")
        
        dynamic_range = quality_metrics.get('dynamic_range', 0.8)
        if dynamic_range < 0.3:
            recommendations.append("Low dynamic range may limit hiding capacity")
        
        return recommendations
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def is_video_file(self, file_path: Path) -> bool:
        """Check if file is a supported video format."""
        return file_path.suffix.lower() in self.VIDEO_FORMATS
    
    def is_audio_file(self, file_path: Path) -> bool:
        """Check if file is a supported audio format."""
        return file_path.suffix.lower() in self.AUDIO_FORMATS
    
    def is_multimedia_file(self, file_path: Path) -> bool:
        """Check if file is a supported multimedia format."""
        return self.is_video_file(file_path) or self.is_audio_file(file_path)
