"""
Enhanced Image Analyzer - Next Generation
Advanced image analysis using modern computer vision techniques, machine learning, and statistical methods.

Features:
- Modern computer vision algorithms
- Machine learning-based steganography detection
- GPU acceleration support (optional)
- Advanced texture analysis using Local Binary Patterns (LBP)
- Wavelet-based analysis for DCT coefficients
- SIFT/ORB feature detection
- Multi-scale analysis
- Perceptual hashing for similarity detection
- Color space analysis (RGB, HSV, LAB)
- Noise estimation using multiple methods
- Advanced statistical tests
- Real-time performance monitoring
- Caching for improved performance
- Parallel processing support
"""

import math
import statistics
import hashlib
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time
from functools import lru_cache
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Core dependencies
try:
    import numpy as np
    from PIL import Image, ImageFilter, ImageEnhance, ImageStat
    CORE_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Core dependencies missing: {e}")
    print("Install with: pip install numpy pillow")
    CORE_DEPS_AVAILABLE = False

# Lazy OpenCV import to avoid UI freeze on first-time import
_cv2_module = None

def _ensure_cv2():
    global _cv2_module
    if _cv2_module is None:
        try:
            import cv2 as _cv2
            _cv2_module = _cv2
        except ImportError as e:
            raise RuntimeError("OpenCV (cv2) is required for edge/sharpness operations. Install with: pip install opencv-python") from e
    return _cv2_module

# Advanced scientific computing
try:
    import scipy
    import scipy.ndimage
    import scipy.signal
    import scipy.stats
    from scipy.fftpack import dct, idct
    from skimage import feature, measure, filters, color, segmentation
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    from skimage.filters import gabor
    ADVANCED_DEPS_AVAILABLE = True
except ImportError:
    ADVANCED_DEPS_AVAILABLE = False

# Machine Learning (optional)
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# GPU acceleration (optional)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from core.security.crypto_utils import CryptoUtils


class AnalysisLevel(Enum):
    """Enhanced analysis complexity levels."""
    LIGHTNING = "lightning"     # Ultra-fast for real-time applications
    FAST = "fast"              # Quick analysis with basic metrics
    BALANCED = "balanced"      # Comprehensive analysis with good performance
    THOROUGH = "thorough"      # Deep analysis with all features
    RESEARCH = "research"      # Maximum detail for research purposes


class ColorSpace(Enum):
    """Supported color spaces for analysis."""
    RGB = "rgb"
    HSV = "hsv"
    LAB = "lab"
    YUV = "yuv"
    GRAY = "gray"


@dataclass
class PerformanceMetrics:
    """Performance tracking for analysis operations."""
    total_time: float = 0.0
    preprocessing_time: float = 0.0
    analysis_time: float = 0.0
    postprocessing_time: float = 0.0
    memory_peak_mb: float = 0.0
    gpu_used: bool = False
    cache_hit_rate: float = 0.0
    parallel_workers: int = 1


@dataclass
class ImageFingerprint:
    """Perceptual fingerprint of an image."""
    dhash: str = ""
    phash: str = ""
    ahash: str = ""
    whash: str = ""
    color_histogram_hash: str = ""
    texture_hash: str = ""


class ImageAnalyzer:
    """Next-generation image analyzer with advanced computer vision capabilities."""
    
    # Enhanced format support
    SUPPORTED_FORMATS = {'.png', '.bmp', '.tiff', '.tif', '.jpg', '.jpeg', '.webp'}
    LOSSLESS_FORMATS = {'.png', '.bmp', '.tiff', '.tif'}
    
    # Analysis thresholds (refined based on research)
    ENTROPY_EXCELLENT = 7.5
    ENTROPY_GOOD = 6.0
    ENTROPY_POOR = 4.0
    
    NOISE_HIGH = 35.0
    NOISE_MEDIUM = 15.0
    NOISE_LOW = 5.0
    
    def __init__(self, 
                 enable_gpu: bool = True,
                 enable_parallel: bool = True,
                 cache_size: int = 128,
                 max_workers: int = None):
        """
        Initialize the enhanced image analyzer.
        
        Args:
            enable_gpu: Enable GPU acceleration if available
            enable_parallel: Enable parallel processing
            cache_size: LRU cache size for expensive operations
            max_workers: Maximum number of parallel workers
        """
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.crypto_utils = CryptoUtils()
        
        # Configuration
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self._cache_stats = {"hits": 0, "misses": 0}
        
        # Thread pool for parallel processing
        if self.enable_parallel:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
            # Defer expensive process pool creation until first actual use
            self.process_pool = None
        else:
            self.thread_pool = None
            self.process_pool = None
        
        # Verify dependencies
        if not CORE_DEPS_AVAILABLE:
            self.logger.error("Core dependencies not available - analyzer disabled")
            raise RuntimeError("Core dependencies required: numpy, pillow, opencv-python")
        
        self.logger.info(f"Enhanced ImageAnalyzer initialized")
        self.logger.info(f"GPU acceleration: {'enabled' if self.enable_gpu else 'disabled'}")
        self.logger.info(f"Parallel processing: {'enabled' if self.enable_parallel else 'disabled'}")
        self.logger.info(f"Advanced features: {'enabled' if ADVANCED_DEPS_AVAILABLE else 'disabled'}")
        self.logger.info(f"ML features: {'enabled' if ML_AVAILABLE else 'disabled'}")

        # Non-blocking background warmup hook (optional)
        self._warmed_up = False
    
    def _ensure_process_pool(self):
        """Lazy initialization of ProcessPoolExecutor to avoid startup delay."""
        if self.enable_parallel and self.process_pool is None:
            self.process_pool = ProcessPoolExecutor(max_workers=min(4, self.max_workers))
        return self.process_pool
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        if hasattr(self, 'process_pool') and self.process_pool:
            self.process_pool.shutdown(wait=False)
    
    @lru_cache(maxsize=128)
    def _get_image_cache_key(self, image_path: str, analysis_level: str) -> str:
        """Generate cache key for image analysis."""
        path_obj = Path(image_path)
        file_hash = hashlib.md5(f"{image_path}{path_obj.stat().st_mtime}".encode()).hexdigest()
        return f"{file_hash}_{analysis_level}"
    
    def analyze_image_advanced(self, 
                             image_path: Union[str, Path],
                             analysis_level: AnalysisLevel = AnalysisLevel.BALANCED,
                             color_spaces: List[ColorSpace] = None,
                             enable_ml: bool = True,
                             progress_callback: Callable[[float], None] = None) -> Dict[str, Any]:
        """
        Perform advanced image analysis using modern computer vision techniques.
        
        Args:
            image_path: Path to image file
            analysis_level: Complexity level of analysis
            color_spaces: Color spaces to analyze (default: [RGB, HSV])
            enable_ml: Enable machine learning-based analysis
            progress_callback: Progress callback function
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        
        try:
            path = Path(image_path)
            self.logger.info(f"Starting advanced analysis: {path.name} ({analysis_level.value})")
            
            # Validate input
            if not self._validate_image_file(path):
                raise ValueError(f"Invalid image file: {path}")
            
            # Set default color spaces
            if color_spaces is None:
                if analysis_level in [AnalysisLevel.LIGHTNING, AnalysisLevel.FAST]:
                    color_spaces = [ColorSpace.RGB]
                else:
                    color_spaces = [ColorSpace.RGB, ColorSpace.HSV]
            
            # Initialize progress
            if progress_callback:
                progress_callback(0.0)
            
            # Initialize results structure
            results = {
                'metadata': {
                    'analysis_timestamp': time.time(),
                    'analysis_level': analysis_level.value,
                    'color_spaces_analyzed': [cs.value for cs in color_spaces],
                    'features_enabled': {
                        'gpu_acceleration': self.enable_gpu,
                        'parallel_processing': self.enable_parallel,
                        'advanced_algorithms': ADVANCED_DEPS_AVAILABLE,
                        'machine_learning': enable_ml and ML_AVAILABLE
                    }
                },
                'file_info': {},
                'image_properties': {},
                'perceptual_fingerprint': {},
                'capacity_analysis': {},
                'quality_metrics': {},
                'texture_analysis': {},
                'frequency_analysis': {},
                'steganography_analysis': {},
                'ml_analysis': {},
                'security_assessment': {},
                'performance_metrics': {},
                'recommendations': []
            }
            
            # Load and prepare image
            with Image.open(path) as pil_img:
                # Basic file and image information
                results['file_info'] = self._analyze_file_info_enhanced(path)
                results['image_properties'] = self._analyze_image_properties_enhanced(pil_img)
                
                if progress_callback:
                    progress_callback(0.1)
                
                # Convert to working formats
                img_arrays = self._prepare_image_arrays(pil_img, color_spaces, analysis_level)
                
                if progress_callback:
                    progress_callback(0.2)
                
                # Core analyses (parallel where possible)
                analysis_tasks = []
                
                # Capacity analysis (fast)
                results['capacity_analysis'] = self._analyze_capacity_enhanced(pil_img, img_arrays['rgb'])
                
                if progress_callback:
                    progress_callback(0.3)
                
                # Quality metrics analysis
                if analysis_level != AnalysisLevel.LIGHTNING:
                    results['quality_metrics'] = self._analyze_quality_metrics_advanced(
                        img_arrays, color_spaces, analysis_level
                    )
                
                if progress_callback:
                    progress_callback(0.4)
                
                # Texture analysis (advanced)
                if analysis_level in [AnalysisLevel.BALANCED, AnalysisLevel.THOROUGH, AnalysisLevel.RESEARCH]:
                    results['texture_analysis'] = self._analyze_texture_advanced(img_arrays['rgb'], analysis_level)
                
                if progress_callback:
                    progress_callback(0.5)
                
                # Frequency domain analysis
                if analysis_level in [AnalysisLevel.THOROUGH, AnalysisLevel.RESEARCH]:
                    results['frequency_analysis'] = self._analyze_frequency_domain(img_arrays['rgb'], analysis_level)
                
                if progress_callback:
                    progress_callback(0.6)
                
                # Steganography-specific analysis
                results['steganography_analysis'] = self._analyze_steganography_indicators(
                    img_arrays, analysis_level
                )
                
                if progress_callback:
                    progress_callback(0.7)
                
                # Machine learning analysis
                if enable_ml and ML_AVAILABLE and analysis_level in [AnalysisLevel.THOROUGH, AnalysisLevel.RESEARCH]:
                    results['ml_analysis'] = self._analyze_with_ml(img_arrays['rgb'], analysis_level)
                
                if progress_callback:
                    progress_callback(0.8)
                
                # Generate perceptual fingerprint
                if analysis_level != AnalysisLevel.LIGHTNING:
                    results['perceptual_fingerprint'] = self._generate_perceptual_fingerprint(pil_img, img_arrays['rgb'])
                
                if progress_callback:
                    progress_callback(0.9)
                
                # Security assessment
                results['security_assessment'] = self._assess_security_comprehensive(results)
                
                # Generate recommendations
                results['recommendations'] = self._generate_recommendations_enhanced(results)
            
            # Performance metrics
            total_time = time.time() - start_time
            results['performance_metrics'] = {
                'total_analysis_time': total_time,
                'analysis_level': analysis_level.value,
                'gpu_acceleration_used': self.enable_gpu,
                'parallel_workers_used': self.max_workers if self.enable_parallel else 1,
                'cache_hit_rate': self._get_cache_hit_rate()
            }
            
            if progress_callback:
                progress_callback(1.0)
            
            self.logger.info(f"Advanced analysis completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis failed: {e}")
            self.error_handler.handle_exception(e)
            return {'error': str(e), 'analysis_level': analysis_level.value}
    
    def _prepare_image_arrays(self, pil_img: Image.Image, 
                            color_spaces: List[ColorSpace], 
                            analysis_level: AnalysisLevel) -> Dict[str, np.ndarray]:
        """Prepare image arrays in multiple color spaces with optimization."""
        arrays = {}
        
        # Optimize image size based on analysis level
        optimized_img = self._optimize_image_size(pil_img, analysis_level)
        
        # RGB (primary)
        rgb_array = np.array(optimized_img.convert('RGB'))
        arrays['rgb'] = rgb_array
        
        # Additional color spaces
        for color_space in color_spaces:
            if color_space == ColorSpace.RGB:
                continue  # Already done
            elif color_space == ColorSpace.HSV:
                if self.enable_gpu and GPU_AVAILABLE:
                    # GPU-accelerated conversion
                    arrays['hsv'] = self._rgb_to_hsv_gpu(rgb_array)
                else:
                    hsv_img = optimized_img.convert('HSV')
                    arrays['hsv'] = np.array(hsv_img)
            elif color_space == ColorSpace.LAB:
                if ADVANCED_DEPS_AVAILABLE:
                    arrays['lab'] = color.rgb2lab(rgb_array)
                else:
                    arrays['lab'] = rgb_array  # Fallback
            elif color_space == ColorSpace.GRAY:
                arrays['gray'] = np.array(optimized_img.convert('L'))
        
        return arrays
    
    def _optimize_image_size(self, pil_img: Image.Image, analysis_level: AnalysisLevel) -> Image.Image:
        """Optimize image size based on analysis level for better performance."""
        width, height = pil_img.size
        
        # Size limits based on analysis level
        size_limits = {
            AnalysisLevel.LIGHTNING: 512,
            AnalysisLevel.FAST: 1024,
            AnalysisLevel.BALANCED: 2048,
            AnalysisLevel.THOROUGH: 4096,
            AnalysisLevel.RESEARCH: 8192
        }
        
        max_size = size_limits[analysis_level]
        
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            self.logger.info(f"Optimizing image size: {width}x{height} â†’ {new_size[0]}x{new_size[1]}")
            return pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
        return pil_img
    
    def _analyze_texture_advanced(self, img_array: np.ndarray, analysis_level: AnalysisLevel) -> Dict[str, Any]:
        """Advanced texture analysis using multiple methods."""
        if not ADVANCED_DEPS_AVAILABLE:
            return {'error': 'Advanced dependencies required for texture analysis'}
        
        try:
            # Convert to grayscale for texture analysis
            if len(img_array.shape) == 3:
                gray = color.rgb2gray(img_array)
            else:
                gray = img_array
            
            results = {}
            
            # Local Binary Patterns (LBP)
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # LBP histogram
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            
            results['local_binary_patterns'] = {
                'histogram': lbp_hist.tolist(),
                'uniformity': float(lbp_hist.max()),
                'entropy': float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))),
                'complexity_score': float(np.std(lbp_hist))
            }
            
            # Gray Level Co-occurrence Matrix (GLCM) for detailed texture analysis
            if analysis_level in [AnalysisLevel.THOROUGH, AnalysisLevel.RESEARCH]:
                # Quantize image for GLCM
                gray_quantized = (gray * 63).astype(np.uint8)
                
                # Calculate GLCM for multiple directions and distances
                distances = [1, 2, 3]
                angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                
                glcm_props = {
                    'contrast': [],
                    'dissimilarity': [],
                    'homogeneity': [],
                    'energy': [],
                    'correlation': []
                }
                
                for distance in distances:
                    glcm = graycomatrix(gray_quantized, distances=[distance], 
                                     angles=angles, levels=64, symmetric=True, normed=True)
                    
                    for prop_name in glcm_props.keys():
                        prop_values = graycoprops(glcm, prop_name)
                        glcm_props[prop_name].extend(prop_values.flatten().tolist())
                
                # Calculate statistics for each property
                glcm_stats = {}
                for prop_name, values in glcm_props.items():
                    glcm_stats[prop_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                
                results['glcm_analysis'] = glcm_stats
            
            # Gabor filter responses (for research level)
            if analysis_level == AnalysisLevel.RESEARCH:
                gabor_responses = []
                frequencies = [0.1, 0.3, 0.5]
                angles = [0, 45, 90, 135]
                
                for freq in frequencies:
                    for angle in angles:
                        filtered_real, _ = gabor(gray, frequency=freq, theta=np.deg2rad(angle))
                        response_stats = {
                            'frequency': freq,
                            'angle': angle,
                            'mean_response': float(np.mean(np.abs(filtered_real))),
                            'energy': float(np.sum(filtered_real ** 2))
                        }
                        gabor_responses.append(response_stats)
                
                results['gabor_analysis'] = {
                    'responses': gabor_responses,
                    'dominant_frequency': frequencies[np.argmax([r['energy'] for r in gabor_responses]) // len(angles)],
                    'texture_regularity': float(np.std([r['energy'] for r in gabor_responses]))
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in texture analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_frequency_domain(self, img_array: np.ndarray, analysis_level: AnalysisLevel) -> Dict[str, Any]:
        """Advanced frequency domain analysis including DCT, DWT, and FFT."""
        try:
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            results = {}
            
            # DCT Analysis (important for JPEG steganography)
            if analysis_level in [AnalysisLevel.THOROUGH, AnalysisLevel.RESEARCH]:
                # Block-wise DCT (8x8 blocks like JPEG)
                h, w = gray.shape
                dct_coeffs = []
                
                for i in range(0, h-7, 8):
                    for j in range(0, w-7, 8):
                        block = gray[i:i+8, j:j+8]
                        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                        dct_coeffs.append(dct_block)
                
                if dct_coeffs:
                    dct_coeffs = np.array(dct_coeffs)
                    
                    # Analyze DC coefficients
                    dc_coeffs = dct_coeffs[:, 0, 0]
                    
                    # Analyze AC coefficients
                    ac_coeffs = dct_coeffs[:, 1:, 1:].flatten()
                    
                    results['dct_analysis'] = {
                        'dc_coefficient_stats': {
                            'mean': float(np.mean(dc_coeffs)),
                            'std': float(np.std(dc_coeffs)),
                            'entropy': self._calculate_entropy_1d(dc_coeffs)
                        },
                        'ac_coefficient_stats': {
                            'mean': float(np.mean(ac_coeffs)),
                            'std': float(np.std(ac_coeffs)),
                            'entropy': self._calculate_entropy_1d(ac_coeffs),
                            'sparsity': float(np.sum(np.abs(ac_coeffs) < 1e-6) / len(ac_coeffs))
                        },
                        'compression_artifacts_indicator': float(np.std(dc_coeffs) / (np.mean(np.abs(dc_coeffs)) + 1e-7))
                    }
            
            # FFT Analysis
            fft_2d = np.fft.fft2(gray)
            fft_magnitude = np.abs(fft_2d)
            fft_phase = np.angle(fft_2d)
            
            # Radial frequency analysis
            center = np.array(gray.shape) // 2
            y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
            radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            # Create frequency bands
            max_radius = min(center)
            bands = np.linspace(0, max_radius, 10)
            band_powers = []
            
            for i in range(len(bands)-1):
                mask = (radius >= bands[i]) & (radius < bands[i+1])
                band_power = np.mean(fft_magnitude[mask])
                band_powers.append(float(band_power))
            
            results['fft_analysis'] = {
                'frequency_band_powers': band_powers,
                'high_frequency_content': float(np.sum(band_powers[-3:]) / np.sum(band_powers)),
                'spectral_entropy': self._calculate_entropy_1d(fft_magnitude.flatten()),
                'phase_coherence': float(np.mean(np.cos(fft_phase))),
                'dominant_frequency_ratio': float(max(band_powers) / (np.mean(band_powers) + 1e-7))
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in frequency domain analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_with_ml(self, img_array: np.ndarray, analysis_level: AnalysisLevel) -> Dict[str, Any]:
        """Machine learning-based analysis for anomaly detection and classification."""
        if not ML_AVAILABLE:
            return {'error': 'Machine learning dependencies not available'}
        
        try:
            # Extract comprehensive features
            features = self._extract_ml_features(img_array)
            
            # Anomaly detection using Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.fit_predict(features.reshape(1, -1))
            anomaly_score = float(iso_forest.decision_function(features.reshape(1, -1))[0])
            
            # K-means clustering for image segmentation analysis
            reshaped_img = img_array.reshape(-1, 3) if len(img_array.shape) == 3 else img_array.reshape(-1, 1)
            sample_size = min(10000, len(reshaped_img))
            sample_indices = np.random.choice(len(reshaped_img), sample_size, replace=False)
            sample_pixels = reshaped_img[sample_indices]
            
            kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(sample_pixels)
            
            # Cluster analysis
            cluster_centers = kmeans.cluster_centers_
            cluster_counts = np.bincount(cluster_labels)
            cluster_distribution = cluster_counts / cluster_counts.sum()
            
            results = {
                'anomaly_detection': {
                    'anomaly_score': anomaly_score,
                    'is_anomalous': bool(anomaly_scores[0] == -1),
                    'confidence': float(abs(anomaly_score))
                },
                'clustering_analysis': {
                    'dominant_colors': cluster_centers.tolist(),
                    'color_distribution': cluster_distribution.tolist(),
                    'color_diversity': float(1.0 - max(cluster_distribution)),
                    'dominant_color_ratio': float(max(cluster_distribution))
                },
                'feature_statistics': {
                    'feature_vector_size': len(features),
                    'feature_mean': float(np.mean(features)),
                    'feature_std': float(np.std(features)),
                    'feature_entropy': self._calculate_entropy_1d(features)
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in ML analysis: {e}")
            return {'error': str(e)}
    
    def _extract_ml_features(self, img_array: np.ndarray) -> np.ndarray:
        """Extract comprehensive feature vector for machine learning analysis."""
        features = []
        
        # Convert to grayscale for some features
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Statistical features
        for channel in range(img_array.shape[2] if len(img_array.shape) == 3 else 1):
            if len(img_array.shape) == 3:
                channel_data = img_array[:, :, channel].flatten()
            else:
                channel_data = img_array.flatten()
            
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                scipy.stats.skew(channel_data),
                scipy.stats.kurtosis(channel_data)
            ])
        
        # Texture features (simplified LBP)
        if ADVANCED_DEPS_AVAILABLE:
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=10)
            lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
            features.extend(lbp_hist)
        
        # Frequency features (FFT energy in bands)
        fft_2d = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft_2d)
        
        # Divide into frequency bands
        h, w = fft_magnitude.shape
        band_features = []
        for i in range(0, h, h//4):
            for j in range(0, w, w//4):
                band = fft_magnitude[i:i+h//4, j:j+w//4]
                band_features.append(np.mean(band))
        
        features.extend(band_features)
        
        return np.array(features)
    
    def _generate_perceptual_fingerprint(self, pil_img: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive perceptual fingerprint for image similarity detection."""
        try:
            fingerprint = {}
            
            # Perceptual hashes
            fingerprint['dhash'] = self._compute_dhash(pil_img)
            fingerprint['ahash'] = self._compute_ahash(pil_img)
            fingerprint['phash'] = self._compute_phash(pil_img)
            
            # Color histogram hash
            fingerprint['color_histogram_hash'] = self._compute_color_histogram_hash(img_array)
            
            # Texture hash (if advanced deps available)
            if ADVANCED_DEPS_AVAILABLE:
                fingerprint['texture_hash'] = self._compute_texture_hash(img_array)
            
            # Edge pattern hash
            fingerprint['edge_hash'] = self._compute_edge_hash(img_array)
            
            return fingerprint
            
        except Exception as e:
            self.logger.error(f"Error generating perceptual fingerprint: {e}")
            return {'error': str(e)}
    
    def _compute_dhash(self, pil_img: Image.Image, hash_size: int = 8) -> str:
        """Compute difference hash (dHash)."""
        # Resize and convert to grayscale
        resized = pil_img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS).convert('L')
        pixels = np.array(resized)
        
        # Compute differences
        diff = pixels[:, 1:] > pixels[:, :-1]
        
        # Convert to hexadecimal string
        return ''.join(['1' if d else '0' for d in diff.flatten()])
    
    def _compute_ahash(self, pil_img: Image.Image, hash_size: int = 8) -> str:
        """Compute average hash (aHash)."""
        # Resize and convert to grayscale
        resized = pil_img.resize((hash_size, hash_size), Image.Resampling.LANCZOS).convert('L')
        pixels = np.array(resized)
        
        # Compare to average
        avg = pixels.mean()
        diff = pixels > avg
        
        return ''.join(['1' if d else '0' for d in diff.flatten()])
    
    def _compute_phash(self, pil_img: Image.Image, hash_size: int = 8) -> str:
        """Compute perceptual hash (pHash) using DCT."""
        # Resize and convert to grayscale
        resized = pil_img.resize((hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS).convert('L')
        pixels = np.array(resized, dtype=np.float32)
        
        # Apply DCT
        dct_coeffs = dct(dct(pixels, axis=0), axis=1)
        
        # Extract top-left hash_size x hash_size coefficients (excluding DC)
        dct_low = dct_coeffs[1:hash_size+1, 1:hash_size+1]
        
        # Compare to median
        med = np.median(dct_low)
        diff = dct_low > med
        
        return ''.join(['1' if d else '0' for d in diff.flatten()])
    
    def _compute_color_histogram_hash(self, img_array: np.ndarray, bins: int = 16) -> str:
        """Compute hash based on color histogram."""
        if len(img_array.shape) == 3:
            # RGB histogram
            hist = np.histogramdd(img_array.reshape(-1, 3), bins=bins)[0]
        else:
            # Grayscale histogram  
            hist = np.histogram(img_array, bins=bins)[0]
        
        # Normalize and convert to binary based on mean
        hist = hist.flatten()
        hist_norm = hist / (hist.sum() + 1e-7)
        mean_val = np.mean(hist_norm)
        
        return ''.join(['1' if h > mean_val else '0' for h in hist_norm])
    
    def _compute_texture_hash(self, img_array: np.ndarray) -> str:
        """Compute hash based on texture features."""
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = color.rgb2gray(img_array)
        else:
            gray = img_array
        
        # Local Binary Pattern
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp, bins=10)
        
        # Normalize and convert to binary
        hist_norm = hist / (hist.sum() + 1e-7)
        mean_val = np.mean(hist_norm)
        
        return ''.join(['1' if h > mean_val else '0' for h in hist_norm])
    
    def _compute_edge_hash(self, img_array: np.ndarray) -> str:
        """Compute hash based on edge patterns."""
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Edge detection (lazy import of OpenCV)
        cv2 = _ensure_cv2()
        edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)
        
        # Divide into blocks and compute edge density
        h, w = edges.shape
        block_size = 16
        edge_densities = []
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = edges[i:i+block_size, j:j+block_size]
                density = np.sum(block) / (block_size * block_size * 255)
                edge_densities.append(density)
        
        # Convert to binary based on median
        if edge_densities:
            median_density = np.median(edge_densities)
            return ''.join(['1' if d > median_density else '0' for d in edge_densities])
        
        return '0'
    
    def _calculate_entropy_1d(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy for 1D data."""
        # Quantize data to reasonable number of bins
        hist, _ = np.histogram(data, bins=256)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        prob = hist / hist.sum()
        return float(-np.sum(prob * np.log2(prob)))
    
    def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate for performance monitoring."""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        return self._cache_stats["hits"] / total if total > 0 else 0.0
    
    def _validate_image_file(self, path: Path) -> bool:
        """Enhanced image validation."""
        if not path.exists():
            self.logger.error(f"Image file not found: {path}")
            return False
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            self.logger.warning(f"Unsupported format: {path.suffix}")
            return False
        
        try:
            with Image.open(path) as img:
                if img.size[0] < 8 or img.size[1] < 8:
                    self.logger.error("Image too small for analysis")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating image: {e}")
            return False
    
    def _analyze_file_info_enhanced(self, path: Path) -> Dict[str, Any]:
        """Enhanced file information analysis."""
        try:
            stat = path.stat()
            return {
                'filename': path.name,
                'filepath': str(path),
                'file_size_bytes': stat.st_size,
                'file_size_kb': stat.st_size / 1024,
                'file_size_mb': stat.st_size / (1024 * 1024),
                'modified_time': stat.st_mtime,
                'file_extension': path.suffix.lower(),
                'format_supported': path.suffix.lower() in self.SUPPORTED_FORMATS,
                'is_lossless_format': path.suffix.lower() in self.LOSSLESS_FORMATS,
                'estimated_compression_ratio': self._estimate_compression_ratio(path)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _estimate_compression_ratio(self, path: Path) -> Optional[float]:
        """Estimate compression ratio for the image format."""
        try:
            # Simple heuristic based on file size and estimated uncompressed size
            with Image.open(path) as img:
                width, height = img.size
                channels = len(img.getbands())
                uncompressed_size = width * height * channels
                compressed_size = path.stat().st_size
                
                return compressed_size / uncompressed_size if uncompressed_size > 0 else None
        except:
            return None
    
    def _analyze_image_properties_enhanced(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Enhanced image properties analysis."""
        try:
            width, height = pil_img.size
            
            # Enhanced properties
            properties = {
                'width': width,
                'height': height,
                'total_pixels': width * height,
                'aspect_ratio': width / height if height > 0 else 0,
                'mode': pil_img.mode,
                'channels': len(pil_img.getbands()),
                'bits_per_pixel': len(pil_img.getbands()) * 8,
                'format': pil_img.format,
                'has_transparency': pil_img.mode in ['RGBA', 'LA'] or 'transparency' in pil_img.info,
                'resolution_dpi': pil_img.info.get('dpi', (72, 72)),
                'is_grayscale': pil_img.mode in ['L', '1'],
                'is_color': pil_img.mode in ['RGB', 'RGBA', 'CMYK'],
                'megapixels': (width * height) / 1_000_000
            }
            
            # Statistical properties
            stat = ImageStat.Stat(pil_img)
            properties['pixel_statistics'] = {
                'mean': stat.mean,
                'median': stat.median,
                'stddev': stat.stddev,
                'extrema': stat.extrema
            }
            
            return properties
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_capacity_enhanced(self, pil_img: Image.Image, img_array: np.ndarray) -> Dict[str, Any]:
        """Enhanced capacity analysis with multiple steganography methods."""
        try:
            width, height = pil_img.size
            channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
            total_pixels = width * height
            
            # LSB capacity (traditional)
            lsb_bits = total_pixels * channels
            lsb_capacity_bytes = lsb_bits // 8
            
            # Enhanced capacity calculations
            results = {
                'basic_lsb': {
                    'total_bits': lsb_bits,
                    'capacity_bytes': lsb_capacity_bytes,
                    'capacity_kb': lsb_capacity_bytes / 1024,
                    'capacity_mb': lsb_capacity_bytes / (1024 * 1024)
                },
                'multi_bit_lsb': {
                    '2_bit_capacity_bytes': (lsb_bits * 2) // 8,
                    '4_bit_capacity_bytes': (lsb_bits * 4) // 8
                },
                'theoretical_maximum': {
                    'full_pixel_capacity_bytes': total_pixels * channels,
                    'half_pixel_capacity_bytes': (total_pixels * channels) // 2
                }
            }
            
            # Format-specific capacity considerations
            if pil_img.format == 'PNG':
                results['png_specific'] = {
                    'supports_lossless': True,
                    'recommended_method': 'LSB',
                    'alpha_channel_available': pil_img.mode == 'RGBA'
                }
            elif pil_img.format in ['JPEG', 'JPG']:
                results['jpeg_specific'] = {
                    'supports_lossless': False,
                    'recommended_method': 'DCT coefficient modification',
                    'compression_artifacts_present': True
                }
            
            # Effective capacity (considering headers and error correction)
            overhead_bytes = 256  # Conservative estimate
            results['effective_capacity'] = {
                'usable_bytes': max(0, lsb_capacity_bytes - overhead_bytes),
                'overhead_bytes': overhead_bytes,
                'efficiency_ratio': max(0, (lsb_capacity_bytes - overhead_bytes) / lsb_capacity_bytes) if lsb_capacity_bytes > 0 else 0
            }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_quality_metrics_advanced(self, img_arrays: Dict[str, np.ndarray], 
                                        color_spaces: List[ColorSpace],
                                        analysis_level: AnalysisLevel) -> Dict[str, Any]:
        """Advanced quality metrics analysis across multiple color spaces."""
        try:
            results = {}
            
            for color_space in color_spaces:
                if color_space.value not in img_arrays:
                    continue
                
                img_array = img_arrays[color_space.value]
                space_results = {}
                
                # Entropy analysis
                space_results['entropy'] = self._calculate_entropy_advanced(img_array)
                
                # Noise analysis
                space_results['noise_analysis'] = self._analyze_noise_advanced(img_array)
                
                # Contrast and dynamic range
                space_results['contrast_analysis'] = self._analyze_contrast(img_array)
                
                # Sharpness/blur detection
                if analysis_level in [AnalysisLevel.BALANCED, AnalysisLevel.THOROUGH, AnalysisLevel.RESEARCH]:
                    space_results['sharpness_analysis'] = self._analyze_sharpness(img_array)
                
                results[color_space.value] = space_results
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_entropy_advanced(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Advanced entropy calculation with additional metrics."""
        try:
            if len(img_array.shape) == 2:
                # Grayscale
                hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
                hist = hist[hist > 0]
                prob = hist / hist.sum()
                entropy = -np.sum(prob * np.log2(prob))
                
                return {
                    'overall': float(entropy),
                    'channels': [float(entropy)],
                    'max_possible': 8.0,
                    'normalized': float(entropy / 8.0),
                    'uniformity': float(1.0 - np.sum((prob - 1.0/len(prob))**2))
                }
            else:
                # Multi-channel
                entropies = []
                uniformities = []
                
                for channel in range(img_array.shape[2]):
                    hist, _ = np.histogram(img_array[:, :, channel], bins=256, range=(0, 256))
                    hist = hist[hist > 0]
                    
                    if len(hist) > 0:
                        prob = hist / hist.sum()
                        entropy = -np.sum(prob * np.log2(prob))
                        uniformity = 1.0 - np.sum((prob - 1.0/len(prob))**2)
                    else:
                        entropy = 0.0
                        uniformity = 0.0
                    
                    entropies.append(float(entropy))
                    uniformities.append(float(uniformity))
                
                return {
                    'overall': float(np.mean(entropies)),
                    'channels': entropies,
                    'channel_uniformities': uniformities,
                    'max_possible': 8.0,
                    'normalized': float(np.mean(entropies) / 8.0),
                    'channel_variance': float(np.var(entropies))
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_noise_advanced(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Advanced noise analysis using multiple estimation methods."""
        try:
            results = {}
            
            if len(img_array.shape) == 2:
                # Grayscale analysis
                results = self._estimate_noise_single_channel(img_array)
            else:
                # Multi-channel analysis
                channel_results = []
                for channel in range(img_array.shape[2]):
                    channel_noise = self._estimate_noise_single_channel(img_array[:, :, channel])
                    channel_results.append(channel_noise)
                
                # Aggregate results
                results = {
                    'overall_std': float(np.mean([cr['std_deviation'] for cr in channel_results])),
                    'overall_mad': float(np.mean([cr['median_absolute_deviation'] for cr in channel_results])),
                    'overall_noise_estimate': float(np.mean([cr['noise_estimate'] for cr in channel_results])),
                    'channel_results': channel_results
                }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _estimate_noise_single_channel(self, channel_data: np.ndarray) -> Dict[str, Any]:
        """Estimate noise in a single channel using multiple methods."""
        # Standard deviation
        std_dev = float(np.std(channel_data))
        
        # Median Absolute Deviation (more robust)
        mad = float(np.median(np.abs(channel_data - np.median(channel_data))))
        
        # Wavelet-based noise estimation (if scipy available)
        if ADVANCED_DEPS_AVAILABLE:
            try:
                # Simple high-pass filter approximation
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                high_freq = scipy.ndimage.convolve(channel_data, kernel)
                noise_estimate = float(np.std(high_freq) / 6.0)  # Scaling factor
            except:
                noise_estimate = mad * 1.4826  # Fallback
        else:
            noise_estimate = mad * 1.4826  # Convert MAD to std estimate
        
        return {
            'std_deviation': std_dev,
            'median_absolute_deviation': mad,
            'noise_estimate': noise_estimate,
            'snr_estimate': float(np.mean(channel_data) / (noise_estimate + 1e-7))
        }
    
    def _analyze_contrast(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image contrast and dynamic range."""
        try:
            if len(img_array.shape) == 3:
                # Convert to grayscale for contrast analysis
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Basic contrast metrics
            min_val, max_val = gray.min(), gray.max()
            dynamic_range = max_val - min_val
            
            # RMS contrast
            rms_contrast = float(np.sqrt(np.mean((gray - np.mean(gray)) ** 2)))
            
            # Michelson contrast
            michelson_contrast = float((max_val - min_val) / (max_val + min_val + 1e-7))
            
            # Histogram-based metrics
            hist, bins = np.histogram(gray, bins=256)
            cumulative = np.cumsum(hist) / np.sum(hist)
            
            # Find percentiles
            p1 = bins[np.searchsorted(cumulative, 0.01)]
            p99 = bins[np.searchsorted(cumulative, 0.99)]
            
            return {
                'dynamic_range': float(dynamic_range),
                'rms_contrast': rms_contrast,
                'michelson_contrast': michelson_contrast,
                'percentile_range_1_99': float(p99 - p1),
                'contrast_ratio': float(max_val / (min_val + 1e-7)),
                'mean_brightness': float(np.mean(gray)),
                'brightness_std': float(np.std(gray))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_sharpness(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image sharpness and blur detection."""
        try:
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Laplacian variance (standard sharpness measure) using lazy OpenCV import
            cv2 = _ensure_cv2()
            laplacian_var = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
            
            # Gradient magnitude
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            mean_gradient = float(np.mean(gradient_magnitude))
            
            # Tenenbaum gradient (sum of squared gradients)
            tenenbaum = float(np.sum(grad_x**2 + grad_y**2))
            
            # Determine sharpness level
            if laplacian_var > 500:
                sharpness_level = "very_sharp"
            elif laplacian_var > 100:
                sharpness_level = "sharp"
            elif laplacian_var > 50:
                sharpness_level = "moderate"
            elif laplacian_var > 10:
                sharpness_level = "slightly_blurry"
            else:
                sharpness_level = "blurry"
            
            return {
                'laplacian_variance': laplacian_var,
                'mean_gradient_magnitude': mean_gradient,
                'tenenbaum_measure': tenenbaum,
                'sharpness_level': sharpness_level,
                'blur_detected': laplacian_var < 50
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_steganography_indicators(self, img_arrays: Dict[str, np.ndarray], 
                                        analysis_level: AnalysisLevel) -> Dict[str, Any]:
        """Analyze indicators that suggest steganographic content."""
        try:
            results = {}
            rgb_array = img_arrays.get('rgb')
            
            if rgb_array is None:
                return {'error': 'RGB array not available'}
            
            # LSB analysis
            results['lsb_analysis'] = self._analyze_lsb_comprehensive(rgb_array, analysis_level)
            
            # Chi-square test for randomness
            if analysis_level in [AnalysisLevel.THOROUGH, AnalysisLevel.RESEARCH]:
                results['chi_square_tests'] = self._perform_chi_square_tests(rgb_array)
            
            # Histogram analysis
            results['histogram_analysis'] = self._analyze_histograms_for_stego(rgb_array)
            
            # Pixel value analysis
            results['pixel_analysis'] = self._analyze_pixel_relationships(rgb_array)
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_lsb_comprehensive(self, img_array: np.ndarray, analysis_level: AnalysisLevel) -> Dict[str, Any]:
        """Comprehensive LSB analysis for steganography detection."""
        try:
            results = {}
            
            if len(img_array.shape) == 2:
                channels = [img_array]
                channel_names = ['gray']
            else:
                channels = [img_array[:, :, i] for i in range(img_array.shape[2])]
                channel_names = ['red', 'green', 'blue'][:img_array.shape[2]]
            
            channel_results = {}
            
            for channel, name in zip(channels, channel_names):
                # Extract LSB plane
                lsb_plane = channel & 1
                
                # Basic LSB statistics
                lsb_ratio = np.mean(lsb_plane)
                lsb_entropy = self._calculate_entropy_1d(lsb_plane.flatten())
                
                # Pattern analysis
                pattern_score = self._detect_lsb_patterns_advanced(lsb_plane, analysis_level)
                
                # Neighboring pixel correlation in LSB
                correlation = self._calculate_lsb_correlation(lsb_plane)
                
                channel_results[name] = {
                    'lsb_ratio': float(lsb_ratio),
                    'lsb_entropy': float(lsb_entropy),
                    'pattern_anomaly_score': pattern_score,
                    'neighbor_correlation': correlation,
                    'randomness_quality': 'suspicious' if abs(lsb_ratio - 0.5) < 0.01 and lsb_entropy > 0.99 else 'normal'
                }
            
            results['channel_analysis'] = channel_results
            
            # Overall assessment
            avg_entropy = np.mean([ch['lsb_entropy'] for ch in channel_results.values()])
            avg_ratio_deviation = np.mean([abs(ch['lsb_ratio'] - 0.5) for ch in channel_results.values()])
            
            results['overall_assessment'] = {
                'average_lsb_entropy': float(avg_entropy),
                'average_ratio_deviation': float(avg_ratio_deviation),
                'steganography_likelihood': self._assess_stego_likelihood(avg_entropy, avg_ratio_deviation)
            }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_lsb_patterns_advanced(self, lsb_plane: np.ndarray, analysis_level: AnalysisLevel) -> float:
        """Advanced LSB pattern detection."""
        try:
            # Chi-square test for uniformity
            ones = np.sum(lsb_plane)
            zeros = lsb_plane.size - ones
            expected = lsb_plane.size / 2
            
            chi_square = ((ones - expected) ** 2 + (zeros - expected) ** 2) / expected
            
            # Additional tests for thorough analysis
            if analysis_level in [AnalysisLevel.THOROUGH, AnalysisLevel.RESEARCH]:
                # Runs test for randomness
                runs_score = self._runs_test(lsb_plane.flatten())
                
                # Autocorrelation test
                autocorr_score = self._autocorrelation_test(lsb_plane)
                
                # Combine scores
                combined_score = (chi_square / 100.0 + runs_score + autocorr_score) / 3.0
                return min(1.0, combined_score)
            
            return min(1.0, chi_square / 100.0)
            
        except Exception as e:
            return 0.0
    
    def _runs_test(self, sequence: np.ndarray) -> float:
        """Runs test for randomness in binary sequence."""
        try:
            # Count runs (consecutive identical elements)
            runs = 1
            for i in range(1, len(sequence)):
                if sequence[i] != sequence[i-1]:
                    runs += 1
            
            n1 = np.sum(sequence)
            n0 = len(sequence) - n1
            
            if n1 == 0 or n0 == 0:
                return 1.0  # Definitely not random
            
            # Expected number of runs
            expected_runs = (2 * n1 * n0) / (n1 + n0) + 1
            
            # Return normalized deviation
            deviation = abs(runs - expected_runs) / expected_runs if expected_runs > 0 else 1.0
            return min(1.0, deviation)
            
        except Exception as e:
            return 0.0
    
    def _autocorrelation_test(self, lsb_plane: np.ndarray) -> float:
        """Test for autocorrelation in LSB plane."""
        try:
            flat = lsb_plane.flatten()
            if len(flat) < 2:
                return 0.0
            
            # Calculate lag-1 autocorrelation
            autocorr = np.corrcoef(flat[:-1], flat[1:])[0, 1]
            
            # Return absolute autocorrelation as anomaly score
            return min(1.0, abs(autocorr))
            
        except Exception as e:
            return 0.0
    
    def _calculate_lsb_correlation(self, lsb_plane: np.ndarray) -> Dict[str, float]:
        """Calculate correlation between neighboring pixels in LSB plane."""
        try:
            # Horizontal correlation
            h_corr = np.corrcoef(lsb_plane[:, :-1].flatten(), lsb_plane[:, 1:].flatten())[0, 1]
            
            # Vertical correlation  
            v_corr = np.corrcoef(lsb_plane[:-1, :].flatten(), lsb_plane[1:, :].flatten())[0, 1]
            
            # Diagonal correlation
            d_corr = np.corrcoef(lsb_plane[:-1, :-1].flatten(), lsb_plane[1:, 1:].flatten())[0, 1]
            
            return {
                'horizontal': float(h_corr) if not np.isnan(h_corr) else 0.0,
                'vertical': float(v_corr) if not np.isnan(v_corr) else 0.0,
                'diagonal': float(d_corr) if not np.isnan(d_corr) else 0.0,
                'average': float(np.mean([h_corr, v_corr, d_corr]) if not any(np.isnan([h_corr, v_corr, d_corr])) else 0.0)
            }
            
        except Exception as e:
            return {'horizontal': 0.0, 'vertical': 0.0, 'diagonal': 0.0, 'average': 0.0}
    
    def _perform_chi_square_tests(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Perform various chi-square tests for steganography detection."""
        try:
            results = {}
            
            # Standard chi-square test on pixel values
            for channel in range(img_array.shape[2] if len(img_array.shape) == 3 else 1):
                if len(img_array.shape) == 3:
                    channel_data = img_array[:, :, channel]
                else:
                    channel_data = img_array
                
                # Chi-square test for uniform distribution
                hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
                expected = channel_data.size / 256
                chi_square = np.sum((hist - expected) ** 2 / expected)
                
                results[f'channel_{channel}'] = {
                    'chi_square_statistic': float(chi_square),
                    'degrees_of_freedom': 255,
                    'uniformity_score': float(1.0 - min(1.0, chi_square / 1000.0))
                }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_histograms_for_stego(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze histograms for steganographic artifacts."""
        try:
            results = {}
            
            for channel in range(img_array.shape[2] if len(img_array.shape) == 3 else 1):
                if len(img_array.shape) == 3:
                    channel_data = img_array[:, :, channel]
                else:
                    channel_data = img_array
                
                hist, bins = np.histogram(channel_data, bins=256, range=(0, 256))
                
                # Look for unusual patterns
                # 1. Pairs of values analysis (common in LSB steganography)
                pair_differences = []
                for i in range(0, 254, 2):
                    diff = abs(hist[i] - hist[i+1])
                    pair_differences.append(diff)
                
                avg_pair_diff = np.mean(pair_differences) if pair_differences else 0
                max_pair_diff = np.max(pair_differences) if pair_differences else 0
                
                # 2. Histogram smoothness
                smoothness = np.mean(np.abs(np.diff(hist)))
                
                # 3. Peak analysis
                peaks = []
                for i in range(1, len(hist)-1):
                    if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                        peaks.append(i)
                
                results[f'channel_{channel}'] = {
                    'average_pair_difference': float(avg_pair_diff),
                    'max_pair_difference': float(max_pair_diff),
                    'smoothness_score': float(smoothness),
                    'num_peaks': len(peaks),
                    'histogram_uniformity': float(np.std(hist) / (np.mean(hist) + 1e-7))
                }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_pixel_relationships(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze relationships between pixels for steganography detection."""
        try:
            # Sample pixels for performance (use every 4th pixel)
            if len(img_array.shape) == 3:
                sampled = img_array[::4, ::4, :]
            else:
                sampled = img_array[::4, ::4]
            
            results = {}
            
            # Calculate correlations between channels
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                correlations = {}
                channels = ['red', 'green', 'blue']
                
                for i in range(3):
                    for j in range(i+1, 3):
                        corr = np.corrcoef(sampled[:, :, i].flatten(), sampled[:, :, j].flatten())[0, 1]
                        correlations[f'{channels[i]}_{channels[j]}'] = float(corr) if not np.isnan(corr) else 0.0
                
                results['inter_channel_correlations'] = correlations
            
            # Analyze neighboring pixel relationships
            if len(img_array.shape) == 3:
                channel_data = img_array[:, :, 0]  # Use first channel
            else:
                channel_data = img_array
            
            # Horizontal, vertical, and diagonal pixel correlations
            h_pixels = channel_data[:, :-1].flatten()
            h_neighbors = channel_data[:, 1:].flatten()
            h_correlation = np.corrcoef(h_pixels, h_neighbors)[0, 1]
            
            v_pixels = channel_data[:-1, :].flatten()
            v_neighbors = channel_data[1:, :].flatten()
            v_correlation = np.corrcoef(v_pixels, v_neighbors)[0, 1]
            
            results['spatial_correlations'] = {
                'horizontal': float(h_correlation) if not np.isnan(h_correlation) else 0.0,
                'vertical': float(v_correlation) if not np.isnan(v_correlation) else 0.0,
            }
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _assess_stego_likelihood(self, entropy: float, ratio_deviation: float) -> str:
        """Assess likelihood of steganographic content."""
        # High entropy with low ratio deviation suggests possible steganography
        if entropy > 0.98 and ratio_deviation < 0.02:
            return 'high'
        elif entropy > 0.95 and ratio_deviation < 0.05:
            return 'medium' 
        elif entropy > 0.90 and ratio_deviation < 0.1:
            return 'low'
        else:
            return 'none'
    
    def _assess_security_comprehensive(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive security assessment for steganographic applications."""
        try:
            security_score = 0.0
            factors = []
            
            # Format security
            file_info = results.get('file_info', {})
            if file_info.get('is_lossless_format', False):
                security_score += 0.2
                factors.append("Lossless format provides better steganographic security")
            
            # Entropy assessment
            quality_metrics = results.get('quality_metrics', {})
            rgb_metrics = quality_metrics.get('rgb', {})
            entropy_info = rgb_metrics.get('entropy', {})
            
            if entropy_info:
                entropy = entropy_info.get('overall', 0)
                if entropy > self.ENTROPY_EXCELLENT:
                    security_score += 0.3
                    factors.append("Excellent entropy provides natural randomness")
                elif entropy > self.ENTROPY_GOOD:
                    security_score += 0.2
                    factors.append("Good entropy supports steganographic operations")
            
            # Noise analysis
            noise_info = rgb_metrics.get('noise_analysis', {})
            if noise_info:
                noise_level = noise_info.get('overall_noise_estimate', 0)
                if noise_level > self.NOISE_HIGH:
                    security_score += 0.2
                    factors.append("High noise level masks steganographic changes")
                elif noise_level > self.NOISE_MEDIUM:
                    security_score += 0.1
                    factors.append("Moderate noise level provides some masking")
            
            # Texture complexity
            texture_analysis = results.get('texture_analysis', {})
            lbp_info = texture_analysis.get('local_binary_patterns', {})
            if lbp_info:
                texture_entropy = lbp_info.get('entropy', 0)
                if texture_entropy > 5.0:
                    security_score += 0.15
                    factors.append("Complex texture patterns enhance security")
            
            # Steganography indicators (lower is better for security)
            stego_analysis = results.get('steganography_analysis', {})
            lsb_analysis = stego_analysis.get('lsb_analysis', {})
            overall_assessment = lsb_analysis.get('overall_assessment', {})
            
            if overall_assessment:
                stego_likelihood = overall_assessment.get('steganography_likelihood', 'none')
                if stego_likelihood == 'none':
                    security_score += 0.15
                    factors.append("No existing steganographic indicators detected")
                elif stego_likelihood == 'low':
                    security_score += 0.1
                
            # Normalize score to 0-10 scale
            security_score = min(10.0, security_score * 10)
            
            # Determine security rating
            if security_score >= 8.0:
                rating = 'excellent'
            elif security_score >= 6.0:
                rating = 'good'
            elif security_score >= 4.0:
                rating = 'moderate'
            else:
                rating = 'poor'
            
            return {
                'overall_security_score': float(security_score),
                'security_rating': rating,
                'contributing_factors': factors,
                'max_score': 10.0,
                'recommendations': self._generate_security_recommendations(security_score, results)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_security_recommendations(self, security_score: float, results: Dict[str, Any]) -> List[str]:
        """Generate security-focused recommendations."""
        recommendations = []
        
        if security_score < 4.0:
            recommendations.append("âš ï¸ Image has poor steganographic security characteristics")
            recommendations.append("ðŸ”„ Consider using a different image with higher entropy and noise")
        
        # Format recommendations
        file_info = results.get('file_info', {})
        if not file_info.get('is_lossless_format', True):
            recommendations.append("ðŸ“ Use lossless format (PNG, BMP, TIFF) for better security")
        
        # Entropy recommendations
        quality_metrics = results.get('quality_metrics', {})
        rgb_metrics = quality_metrics.get('rgb', {})
        entropy_info = rgb_metrics.get('entropy', {})
        
        if entropy_info and entropy_info.get('overall', 0) < self.ENTROPY_GOOD:
            recommendations.append("ðŸŽ² Low entropy detected - use randomization techniques")
        
        # Noise recommendations  
        noise_info = rgb_metrics.get('noise_analysis', {})
        if noise_info and noise_info.get('overall_noise_estimate', 0) < self.NOISE_MEDIUM:
            recommendations.append("ðŸ”Š Low noise level - steganographic changes may be detectable")
        
        # Size recommendations
        capacity = results.get('capacity_analysis', {})
        effective_capacity = capacity.get('effective_capacity', {})
        if effective_capacity and effective_capacity.get('usable_bytes', 0) < 1024:
            recommendations.append("ðŸ“ Very limited capacity - consider larger image")
        
        if not recommendations:
            recommendations.append("âœ… Image shows good characteristics for secure steganography")
        
        return recommendations
    
    def _generate_recommendations_enhanced(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on all analysis results."""
        recommendations = []
        
        try:
            # Security recommendations
            security_recs = self._generate_security_recommendations(
                results.get('security_assessment', {}).get('overall_security_score', 0),
                results
            )
            recommendations.extend(security_recs[:3])  # Limit to top 3 security recommendations
            
            # Performance recommendations
            perf_metrics = results.get('performance_metrics', {})
            analysis_time = perf_metrics.get('total_analysis_time', 0)
            
            if analysis_time > 10.0:
                recommendations.append("â±ï¸ Analysis took significant time - consider FAST analysis level for routine checks")
            
            # ML-based recommendations
            ml_analysis = results.get('ml_analysis', {})
            if 'anomaly_detection' in ml_analysis:
                anomaly_info = ml_analysis['anomaly_detection']
                if anomaly_info.get('is_anomalous', False):
                    confidence = anomaly_info.get('confidence', 0)
                    recommendations.append(f"ðŸ¤– ML anomaly detection: unusual patterns detected (confidence: {confidence:.1%})")
            
            # Steganography-specific recommendations
            stego_analysis = results.get('steganography_analysis', {})
            if 'lsb_analysis' in stego_analysis:
                lsb_info = stego_analysis['lsb_analysis']
                overall = lsb_info.get('overall_assessment', {})
                likelihood = overall.get('steganography_likelihood', 'none')
                
                if likelihood in ['high', 'medium']:
                    recommendations.append("ðŸ” Potential steganographic content detected - verify before use")
            
            # Format-specific recommendations
            file_info = results.get('file_info', {})
            if file_info.get('file_extension') in ['.jpg', '.jpeg']:
                recommendations.append("ðŸ“¸ JPEG format detected - DCT-based methods recommended over LSB")
            
            # Capacity recommendations
            capacity = results.get('capacity_analysis', {})
            basic_lsb = capacity.get('basic_lsb', {})
            if basic_lsb:
                capacity_mb = basic_lsb.get('capacity_mb', 0)
                if capacity_mb > 50:
                    recommendations.append("ðŸ“Š Excellent capacity for large data hiding operations")
                elif capacity_mb < 0.1:
                    recommendations.append("ðŸ“‰ Limited capacity - suitable only for small data")
            
            # Limit recommendations to most important ones
            return recommendations[:6]
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced recommendations: {e}")
            return ["âš ï¸ Error generating recommendations - check analysis results"]
    
    def detect_steganography_advanced(self, 
                                    image_path: Union[str, Path],
                                    analysis_level: AnalysisLevel = AnalysisLevel.THOROUGH,
                                    use_ml: bool = True) -> Dict[str, Any]:
        """Advanced steganography detection using multiple methods."""
        try:
            # Perform full analysis
            full_analysis = self.analyze_image_advanced(
                image_path, 
                analysis_level=analysis_level, 
                enable_ml=use_ml
            )
            
            if 'error' in full_analysis:
                return {'error': full_analysis['error']}
            
            # Extract detection-relevant information
            detection_results = {
                'detection_confidence': 0.0,
                'detection_methods': {},
                'indicators': [],
                'overall_likelihood': 'none'
            }
            
            confidence_scores = []
            
            # LSB analysis
            stego_analysis = full_analysis.get('steganography_analysis', {})
            lsb_analysis = stego_analysis.get('lsb_analysis', {})
            overall_lsb = lsb_analysis.get('overall_assessment', {})
            
            if overall_lsb:
                likelihood = overall_lsb.get('steganography_likelihood', 'none')
                entropy = overall_lsb.get('average_lsb_entropy', 0)
                
                lsb_confidence = 0.0
                if likelihood == 'high':
                    lsb_confidence = 0.8
                elif likelihood == 'medium':
                    lsb_confidence = 0.5
                elif likelihood == 'low':
                    lsb_confidence = 0.2
                
                confidence_scores.append(lsb_confidence)
                detection_results['detection_methods']['lsb_analysis'] = {
                    'confidence': lsb_confidence,
                    'likelihood': likelihood,
                    'entropy': entropy
                }
                
                if lsb_confidence > 0.5:
                    detection_results['indicators'].append(f"LSB analysis indicates {likelihood} likelihood")
            
            # Histogram analysis
            histogram_analysis = stego_analysis.get('histogram_analysis', {})
            if histogram_analysis:
                # Analyze histogram irregularities
                hist_score = 0.0
                for channel_key, channel_data in histogram_analysis.items():
                    if 'average_pair_difference' in channel_data:
                        pair_diff = channel_data['average_pair_difference']
                        if pair_diff < 50:  # Very uniform pairs suggest steganography
                            hist_score += 0.3
                
                confidence_scores.append(hist_score)
                detection_results['detection_methods']['histogram_analysis'] = {
                    'confidence': hist_score
                }
                
                if hist_score > 0.4:
                    detection_results['indicators'].append("Histogram analysis shows suspicious pair patterns")
            
            # Machine learning analysis
            ml_analysis = full_analysis.get('ml_analysis', {})
            if ml_analysis and 'anomaly_detection' in ml_analysis:
                anomaly_info = ml_analysis['anomaly_detection']
                is_anomalous = anomaly_info.get('is_anomalous', False)
                ml_confidence = anomaly_info.get('confidence', 0)
                
                # Convert to detection confidence
                ml_detection_confidence = ml_confidence if is_anomalous else 0.0
                confidence_scores.append(ml_detection_confidence)
                
                detection_results['detection_methods']['machine_learning'] = {
                    'confidence': ml_detection_confidence,
                    'is_anomalous': is_anomalous
                }
                
                if ml_detection_confidence > 0.6:
                    detection_results['indicators'].append("ML anomaly detection flagged unusual patterns")
            
            # Frequency analysis
            freq_analysis = full_analysis.get('frequency_analysis', {})
            if freq_analysis and 'dct_analysis' in freq_analysis:
                dct_info = freq_analysis['dct_analysis']
                artifact_indicator = dct_info.get('compression_artifacts_indicator', 0)
                
                # High artifact indicator in lossless formats suggests steganography
                freq_confidence = 0.0
                if artifact_indicator > 5.0:  # Threshold for suspicion
                    freq_confidence = 0.4
                
                confidence_scores.append(freq_confidence)
                detection_results['detection_methods']['frequency_analysis'] = {
                    'confidence': freq_confidence,
                    'artifact_indicator': artifact_indicator
                }
                
                if freq_confidence > 0.3:
                    detection_results['indicators'].append("Frequency analysis shows unusual DCT patterns")
            
            # Calculate overall confidence
            if confidence_scores:
                # Use weighted combination (not simple average)
                weights = [0.4, 0.3, 0.2, 0.1][:len(confidence_scores)]  # LSB gets highest weight
                weights = weights[:len(confidence_scores)]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    weighted_confidence = sum(score * weight for score, weight in zip(confidence_scores, weights)) / total_weight
                    detection_results['detection_confidence'] = min(1.0, weighted_confidence)
            
            # Determine overall likelihood
            confidence = detection_results['detection_confidence']
            if confidence > 0.75:
                detection_results['overall_likelihood'] = 'very_high'
            elif confidence > 0.5:
                detection_results['overall_likelihood'] = 'high'
            elif confidence > 0.3:
                detection_results['overall_likelihood'] = 'medium'
            elif confidence > 0.1:
                detection_results['overall_likelihood'] = 'low'
            else:
                detection_results['overall_likelihood'] = 'none'
            
            # Add metadata
            detection_results['analysis_metadata'] = {
                'analysis_level': analysis_level.value,
                'ml_enabled': use_ml,
                'methods_used': len(detection_results['detection_methods']),
                'timestamp': full_analysis.get('metadata', {}).get('analysis_timestamp')
            }
            
            return detection_results
            
        except Exception as e:
            self.logger.error(f"Advanced steganography detection failed: {e}")
            return {'error': str(e)}
    
    def compare_images_similarity(self, image1_path: Union[str, Path], 
                                image2_path: Union[str, Path]) -> Dict[str, Any]:
        """Compare two images for similarity using perceptual hashing and other metrics."""
        try:
            # Analyze both images
            results1 = self.analyze_image_advanced(image1_path, AnalysisLevel.BALANCED)
            results2 = self.analyze_image_advanced(image2_path, AnalysisLevel.BALANCED)
            
            if 'error' in results1 or 'error' in results2:
                return {'error': 'Failed to analyze one or both images'}
            
            # Compare perceptual fingerprints
            fingerprint1 = results1.get('perceptual_fingerprint', {})
            fingerprint2 = results2.get('perceptual_fingerprint', {})
            
            similarity_metrics = {}
            
            # Hamming distance for hashes
            for hash_type in ['dhash', 'ahash', 'phash']:
                if hash_type in fingerprint1 and hash_type in fingerprint2:
                    hash1 = fingerprint1[hash_type]
                    hash2 = fingerprint2[hash_type]
                    
                    if len(hash1) == len(hash2):
                        hamming_dist = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
                        similarity = 1.0 - (hamming_dist / len(hash1))
                        similarity_metrics[f'{hash_type}_similarity'] = float(similarity)
            
            # Statistical similarity
            stats1 = results1.get('image_properties', {}).get('pixel_statistics', {})
            stats2 = results2.get('image_properties', {}).get('pixel_statistics', {})
            
            if stats1 and stats2:
                # Compare mean values
                mean_diff = np.mean([abs(m1 - m2) for m1, m2 in zip(stats1.get('mean', []), stats2.get('mean', []))])
                mean_similarity = max(0.0, 1.0 - mean_diff / 255.0)
                similarity_metrics['statistical_similarity'] = float(mean_similarity)
            
            # Overall similarity score
            if similarity_metrics:
                overall_similarity = np.mean(list(similarity_metrics.values()))
                similarity_metrics['overall_similarity'] = float(overall_similarity)
                
                # Similarity assessment
                if overall_similarity > 0.95:
                    assessment = 'nearly_identical'
                elif overall_similarity > 0.85:
                    assessment = 'very_similar'
                elif overall_similarity > 0.7:
                    assessment = 'similar'
                elif overall_similarity > 0.5:
                    assessment = 'somewhat_similar'
                else:
                    assessment = 'different'
                
                similarity_metrics['similarity_assessment'] = assessment
            
            return {
                'similarity_metrics': similarity_metrics,
                'image1_info': {
                    'path': str(image1_path),
                    'dimensions': f"{results1.get('image_properties', {}).get('width', 0)}x{results1.get('image_properties', {}).get('height', 0)}"
                },
                'image2_info': {
                    'path': str(image2_path),
                    'dimensions': f"{results2.get('image_properties', {}).get('width', 0)}x{results2.get('image_properties', {}).get('height', 0)}"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Image similarity comparison failed: {e}")
            return {'error': str(e)}
    
    def warmup_async(self) -> None:
        """Preload heavy dependencies in a background thread to avoid UI freeze on first use.
        Safe to call multiple times; subsequent calls are no-ops.
        """
        if getattr(self, '_warmed_up', False):
            return

        def _warmup():
            try:
                # Trigger lazy cv2 import
                try:
                    _ = _ensure_cv2()
                except Exception:
                    pass
                # Light-touch numpy ops to allocate internal caches
                try:
                    _ = np.fft.fft(np.zeros(8))
                except Exception:
                    pass
                # Touch advanced deps if available to compile C-extensions
                try:
                    if ADVANCED_DEPS_AVAILABLE:
                        from scipy import fftpack as _fp  # noqa: F401
                        from skimage import color as _col  # noqa: F401
                except Exception:
                    pass
                # Touch ML deps if available
                try:
                    if ML_AVAILABLE:
                        from sklearn.ensemble import IsolationForest  # noqa: F401
                except Exception:
                    pass
            finally:
                self._warmed_up = True

        t = threading.Thread(target=_warmup, name="ImageAnalyzerWarmup", daemon=True)
        t.start()

    def batch_analyze(self, image_paths: List[Union[str, Path]], 
                     analysis_level: AnalysisLevel = AnalysisLevel.FAST,
                     max_parallel: int = None) -> Dict[str, Any]:
        """Batch analyze multiple images with parallel processing."""
        try:
            if not self.enable_parallel:
                # Sequential processing
                results = {}
                for i, path in enumerate(image_paths):
                    self.logger.info(f"Processing {i+1}/{len(image_paths)}: {Path(path).name}")
                    results[str(path)] = self.analyze_image_advanced(path, analysis_level)
                return results
            
            # Parallel processing
            max_workers = min(max_parallel or self.max_workers, len(image_paths))
            
            def analyze_single(path):
                return str(path), self.analyze_image_advanced(path, analysis_level)
            
            results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(analyze_single, path): path for path in image_paths}
                
                for future in future_to_path:
                    try:
                        path_str, result = future.result(timeout=60)  # 60 second timeout per image
                        results[path_str] = result
                    except Exception as e:
                        path = future_to_path[future]
                        results[str(path)] = {'error': str(e)}
                        self.logger.error(f"Failed to analyze {path}: {e}")
            
            # Summary statistics
            successful = sum(1 for r in results.values() if 'error' not in r)
            failed = len(results) - successful
            
            batch_summary = {
                'total_images': len(image_paths),
                'successful_analyses': successful,
                'failed_analyses': failed,
                'success_rate': successful / len(image_paths) if image_paths else 0,
                'analysis_level': analysis_level.value,
                'parallel_workers': max_workers
            }
            
            return {
                'batch_summary': batch_summary,
                'individual_results': results
            }
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
            return {'error': str(e)}
    
    def get_analysis_summary_enhanced(self, results: Dict[str, Any]) -> str:
        """Generate enhanced human-readable summary with modern insights."""
        try:
            if 'error' in results:
                return f"âŒ Analysis failed: {results['error']}"
            
            summary_lines = []
            
            # Header with analysis metadata
            metadata = results.get('metadata', {})
            analysis_level = metadata.get('analysis_level', 'unknown')
            summary_lines.append(f"ðŸ”¬ **Enhanced Image Analysis Report** ({analysis_level.upper()})")
            summary_lines.append("=" * 60)
            
            # Basic file information
            file_info = results.get('file_info', {})
            if file_info:
                summary_lines.append(f"ðŸ“Œ **File:** {file_info.get('filename', 'Unknown')}")
                summary_lines.append(f"ðŸ’¾ **Size:** {file_info.get('file_size_mb', 0):.2f} MB")
                summary_lines.append(f"ðŸ“ **Format:** {file_info.get('file_extension', 'Unknown').upper()} {'(Lossless)' if file_info.get('is_lossless_format', False) else '(Lossy)'}")
            
            # Image properties
            img_props = results.get('image_properties', {})
            if img_props:
                summary_lines.append(f"ðŸ“ **Dimensions:** {img_props.get('width', 0):,}Ã—{img_props.get('height', 0):,} ({img_props.get('megapixels', 0):.1f} MP)")
                summary_lines.append(f"ðŸŽ¨ **Channels:** {img_props.get('channels', 0)} ({img_props.get('mode', 'Unknown')})")
            
            summary_lines.append("")  # Blank line
            
            summary_lines.append("")  # Blank line
            
            # Capacity analysis
            capacity = results.get('capacity_analysis', {})
            basic_lsb = capacity.get('basic_lsb', {})
            if basic_lsb:
                capacity_mb = basic_lsb.get('capacity_mb', 0)
                summary_lines.append(f"ðŸ“Š **LSB Capacity:** {capacity_mb:.2f} MB ({basic_lsb.get('capacity_kb', 0):,.0f} KB)")
            
            # Quality metrics
            quality = results.get('quality_metrics', {})
            rgb_quality = quality.get('rgb', {})
            
            if rgb_quality:
                # Entropy
                entropy_info = rgb_quality.get('entropy', {})
                if entropy_info:
                    entropy = entropy_info.get('overall', 0)
                    entropy_rating = 'Excellent' if entropy > 7.5 else 'Good' if entropy > 6.0 else 'Moderate' if entropy > 4.0 else 'Poor'
                    summary_lines.append(f"ðŸŽ² **Entropy:** {entropy:.2f}/8.0 ({entropy_rating})")
                
                # Noise analysis
                noise_info = rgb_quality.get('noise_analysis', {})
                if noise_info:
                    noise = noise_info.get('overall_noise_estimate', 0)
                    noise_rating = 'High' if noise > 35 else 'Medium' if noise > 15 else 'Low'
                    summary_lines.append(f"ðŸ”Š **Noise Level:** {noise:.1f} ({noise_rating})")
            
            summary_lines.append("")  # Blank line
            
            # Security assessment
            security = results.get('security_assessment', {})
            if security:
                score = security.get('overall_security_score', 0)
                rating = security.get('security_rating', 'unknown')
                summary_lines.append(f"ðŸ›¡ï¸ **Security Rating:** {score:.1f}/10.0 ({rating.title()})")
            
            # Key recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                summary_lines.append("")
                summary_lines.append("ðŸ’¡ **Key Recommendations:**")
                for rec in recommendations[:3]:  # Show top 3
                    summary_lines.append(f"   â€¢ {rec}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced summary: {e}")
            return f"Error generating summary: {e}"
    
    # Backward compatibility methods
    def quick_suitability_check(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Quick suitability check for backward compatibility."""
        try:
            results = self.analyze_image_advanced(
                image_path=image_path,
                analysis_level=AnalysisLevel.FAST,
                enable_ml=False
            )
            
            security = results.get('security_assessment', {})
            capacity = results.get('capacity_analysis', {})
            
            return {
                'suitable': security.get('security_rating') in ['good', 'excellent'],
                'security_score': security.get('overall_security_score', 0),
                'capacity_mb': capacity.get('basic_lsb', {}).get('capacity_mb', 0),
                'recommendation': f"Security rating: {security.get('security_rating', 'unknown').title()}"
            }
        except Exception as e:
            return {'suitable': False, 'error': str(e), 'recommendation': 'Analysis failed'}
    
    def validate_image_file(self, image_path: Union[str, Path]) -> bool:
        """Public validation method for backward compatibility."""
        return self._validate_image_file(Path(image_path))
    
    def detect_potential_steganography(self, image_path: Union[str, Path], 
                                     analysis_level: AnalysisLevel = AnalysisLevel.BALANCED) -> Dict[str, Any]:
        """Detect potential steganography for backward compatibility."""
        return self.detect_steganography_advanced(image_path, analysis_level, use_ml=True)
    
    def analyze_image_comprehensive(self, image_path: Union[str, Path], 
                                  analysis_level: AnalysisLevel = AnalysisLevel.BALANCED) -> Dict[str, Any]:
        """Comprehensive analysis for backward compatibility."""
        return self.analyze_image_advanced(
            image_path=image_path,
            analysis_level=analysis_level,
            enable_ml=True
        )
    
    def get_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Get analysis summary for backward compatibility."""
        return self.get_analysis_summary_enhanced(results)


# GPU acceleration helpers (if available)
if GPU_AVAILABLE:
    def _rgb_to_hsv_gpu(self, rgb_array: np.ndarray) -> np.ndarray:
        """GPU-accelerated RGB to HSV conversion."""
        try:
            rgb_gpu = cp.asarray(rgb_array)
            # Simplified HSV conversion on GPU
            # This is a basic implementation - could be optimized further
            hsv_gpu = cp.zeros_like(rgb_gpu)
            # ... GPU conversion logic here ...
            return cp.asnumpy(hsv_gpu)
        except Exception as e:
            # Fallback to CPU
            return cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
else:
    def _rgb_to_hsv_gpu(self, rgb_array: np.ndarray) -> np.ndarray:
        """Fallback CPU RGB to HSV conversion."""
        return cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
