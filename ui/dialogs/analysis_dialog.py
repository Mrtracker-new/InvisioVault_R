"""
Analysis Dialog
Image analysis dialog for capacity assessment and steganographic detection.
"""

import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QTextEdit, QProgressBar, QMessageBox,
    QTabWidget, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPixmap, QPalette
from PySide6.QtCharts import QChart, QChartView, QBarSeries, QBarSet, QValueAxis, QBarCategoryAxis

from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler
from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel

# Optional dependencies - graceful fallback if not available
try:
    from PIL import Image
    import numpy as np
    PILImage = Image
except ImportError:
    Image = None
    np = None
    PILImage = None

try:
    import scipy  # type: ignore # Optional dependency for advanced analysis
    import scipy.ndimage  # type: ignore
    import scipy.signal   # type: ignore
except ImportError:
    scipy = None
    # Create dummy modules to prevent attribute errors
    class DummyModule:
        def __getattr__(self, name):
            def dummy_func(*args, **kwargs):
                raise ImportError(f"scipy is not installed. Install with: pip install scipy")
            return dummy_func
    
    if TYPE_CHECKING:
        import scipy
        import scipy.ndimage
        import scipy.signal
    else:
        scipy = DummyModule()  # type: ignore


class AnalysisWorkerThread(QThread):
    """Optimized worker thread for image analysis operations."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    analysis_completed = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self, image_path: str, analysis_quality: str = "balanced"):
        super().__init__()
        self.image_path = image_path
        self.analysis_quality = analysis_quality  # "fast", "balanced", "thorough"
        self.logger = Logger()
        self._cancelled = False
    
    def cancel(self):
        """Cancel the analysis operation."""
        self._cancelled = True
    
    def run(self):
        """Execute the optimized image analysis."""
        try:
            self.status_updated.emit("Initializing analysis...")
            self.progress_updated.emit(5)
            
            if self._cancelled:
                return
            
            analysis_results = self._analyze_image_optimized()
            
            if not self._cancelled:
                self.status_updated.emit("Analysis completed successfully!")
                self.progress_updated.emit(100)
                self.analysis_completed.emit(analysis_results)
            
        except Exception as e:
            if not self._cancelled:
                self.logger.error(f"Image analysis failed: {e}")
                self.error_occurred.emit(str(e))
    
    def _analyze_image_optimized(self) -> Dict:
        """Perform optimized image analysis with configurable quality levels."""
        if not PILImage or np is None:
            raise ImportError("PIL and numpy are required for image analysis")
        
        try:
            self.status_updated.emit("Loading image metadata...")
            self.progress_updated.emit(10)
            
            # Get basic file info first (fast)
            file_path = Path(self.image_path)
            file_size = file_path.stat().st_size
            
            # Quick image format check
            with PILImage.open(self.image_path) as img:
                original_format = img.format
                width, height = img.size
                mode = img.mode
                
                # Convert to RGB for analysis
                if mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                    mode = 'RGB'
                
                channels = 3 if mode == 'RGB' else 4
                
                # Optimized sampling for large images
                if self.analysis_quality == "fast" and (width > 2048 or height > 2048):
                    # Sample image for faster processing
                    sample_size = (min(1024, width), min(1024, height))
                    img_sampled = img.resize(sample_size, PILImage.Resampling.LANCZOS)
                    img_array = np.array(img_sampled)
                    is_sampled = True
                    sample_ratio = (width * height) / (sample_size[0] * sample_size[1])
                else:
                    img_array = np.array(img)
                    is_sampled = False
                    sample_ratio = 1.0
            
            if self._cancelled:
                return {}
            
            self.status_updated.emit("Calculating capacity...")
            self.progress_updated.emit(25)
            
            # Calculate theoretical capacity
            total_pixels = width * height
            lsb_capacity_bits = total_pixels * 3  # RGB channels
            lsb_capacity_bytes = lsb_capacity_bits // 8
            
            if self._cancelled:
                return {}
            
            self.status_updated.emit("Analyzing image properties...")
            self.progress_updated.emit(40)
            
            # Optimized statistical analysis
            stats = self._calculate_statistics_optimized(img_array, is_sampled)
            
            if self._cancelled:
                return {}
            
            self.status_updated.emit("Performing quality assessment...")
            self.progress_updated.emit(60)
            
            # Quality metrics (optimized for different analysis levels)
            quality_metrics = self._assess_quality_optimized(img_array, self.analysis_quality)
            
            if self._cancelled:
                return {}
            
            # LSB analysis (skip for fast mode to save time)
            if self.analysis_quality != "fast":
                self.status_updated.emit("Analyzing LSB patterns...")
                self.progress_updated.emit(80)
                lsb_analysis = self._analyze_lsb_patterns_optimized(img_array)
            else:
                lsb_analysis = self._generate_fast_lsb_analysis(img_array)
            
            if self._cancelled:
                return {}
            
            self.status_updated.emit("Computing suitability score...")
            self.progress_updated.emit(90)
            
            # Suitability assessment
            suitability = self._assess_suitability_optimized(stats, quality_metrics, lsb_analysis)
            
            # Build comprehensive results
            return {
                'basic_info': {
                    'file_path': self.image_path,
                    'file_size': file_size,
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'total_pixels': total_pixels,
                    'bits_per_pixel': channels * 8,
                    'format': original_format,
                    'mode': mode,
                    'is_sampled': is_sampled,
                    'sample_ratio': sample_ratio if is_sampled else 1.0
                },
                'capacity': {
                    'lsb_capacity_bits': lsb_capacity_bits,
                    'lsb_capacity_bytes': lsb_capacity_bytes,
                    'lsb_capacity_kb': lsb_capacity_bytes / 1024,
                    'lsb_capacity_mb': lsb_capacity_bytes / (1024 * 1024),
                    'capacity_ratio': lsb_capacity_bytes / file_size if file_size > 0 else 0,
                    'effective_capacity_bytes': int(lsb_capacity_bytes * 0.85),  # Account for overhead
                },
                'statistics': stats,
                'lsb_analysis': lsb_analysis,
                'quality_metrics': quality_metrics,
                'suitability': suitability,
                'analysis_quality': self.analysis_quality
            }
            
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")
    
    def _calculate_statistics_optimized(self, img_array, is_sampled: bool) -> Dict:
        """Optimized statistical calculation."""
        if np is None:
            raise ImportError("numpy is required for statistical analysis")
        
        # Use efficient numpy operations
        if len(img_array.shape) == 3:
            channels = img_array.shape[2]
        else:
            channels = 1
            img_array = img_array.reshape(img_array.shape[0], img_array.shape[1], 1)
        
        # Calculate statistics efficiently
        channel_stats = []
        for i in range(channels):
            channel = img_array[:, :, i]
            
            # Vectorized calculations
            channel_stats.append({
                'mean': float(np.mean(channel)),
                'std': float(np.std(channel)),
                'min': int(np.min(channel)),
                'max': int(np.max(channel)),
                'median': float(np.median(channel)),
                'variance': float(np.var(channel))
            })
        
        # Overall statistics from flattened array
        flat = img_array.flatten()
        entropy = self._calculate_entropy_optimized(flat)
        
        return {
            'overall': {
                'mean': float(np.mean(flat)),
                'std': float(np.std(flat)),
                'min': int(np.min(flat)),
                'max': int(np.max(flat)),
                'median': float(np.median(flat)),
                'variance': float(np.var(flat))
            },
            'channels': channel_stats,
            'entropy': entropy,
            'is_sampled': is_sampled
        }
    
    def _calculate_entropy_optimized(self, data) -> float:
        """Optimized entropy calculation using numpy operations."""
        if np is None:
            return 0.0
        
        # Use numpy's histogram with optimized bins
        hist, _ = np.histogram(data, bins=256, range=(0, 256), density=True)
        
        # Remove zero probabilities and calculate entropy in one step
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small value to avoid log(0)
        return float(entropy)
    
    def _assess_quality_optimized(self, img_array, quality_level: str) -> Dict:
        """Optimized quality assessment based on analysis level."""
        if np is None:
            return {'noise_level': 0.0, 'sharpness': 0.0, 'texture_complexity': 0.0}
        
        # Convert to grayscale for efficiency
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Basic texture complexity (always calculated)
        texture_complexity = float(np.std(gray))
        
        if quality_level == "fast":
            # Quick approximations for fast analysis
            # Sample-based noise estimation
            h, w = gray.shape
            sample_points = min(10000, h * w // 4)
            sample_indices = np.random.choice(h * w, sample_points, replace=False)
            sample_data = gray.flat[sample_indices]
            noise_level = float(np.std(np.diff(sample_data)))
            sharpness = float(np.var(np.gradient(sample_data)))
        
        else:
            # More thorough analysis for balanced/thorough modes
            # Sobel edge detection for sharpness
            if quality_level == "thorough" and scipy is not None:
                try:
                    sobel_x = scipy.ndimage.sobel(gray, axis=1)
                    sobel_y = scipy.ndimage.sobel(gray, axis=0)
                    sharpness = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))
                except (ImportError, AttributeError):
                    # Fallback to simple gradient
                    grad_x = np.abs(np.diff(gray, axis=1))
                    grad_y = np.abs(np.diff(gray, axis=0))
                    sharpness = float(np.mean(grad_x) + np.mean(grad_y))
            else:
                # Simple gradient for balanced mode or when scipy unavailable
                grad_x = np.abs(np.diff(gray, axis=1))
                grad_y = np.abs(np.diff(gray, axis=0))
                sharpness = float(np.mean(grad_x) + np.mean(grad_y))
            
            # Noise level using Laplacian
            laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            
            # Apply convolution efficiently
            if scipy is not None:
                try:
                    filtered = scipy.signal.convolve2d(gray, laplacian_kernel, mode='valid')
                    noise_level = float(np.std(filtered))
                except (ImportError, AttributeError):
                    # Fallback to manual convolution approximation
                    noise_level = float(np.std(np.abs(np.diff(gray.flatten()))))
            else:
                # Fallback to simple noise estimation
                noise_level = float(np.std(np.abs(np.diff(gray.flatten()))))
        
        return {
            'noise_level': noise_level,
            'sharpness': sharpness,
            'texture_complexity': texture_complexity,
            'quality_level': quality_level
        }
    
    def _analyze_lsb_patterns_optimized(self, img_array) -> Dict:
        """Optimized LSB pattern analysis."""
        if np is None:
            return self._generate_fast_lsb_analysis(img_array)
        
        channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
        lsb_stats = []
        
        for i in range(min(channels, 3)):  # Analyze up to RGB channels
            channel = img_array[:, :, i] if channels > 1 else img_array.squeeze()
            
            # Extract LSBs efficiently
            lsbs = channel & 1
            lsb_flat = lsbs.flatten()
            
            # Vectorized calculations
            ones_ratio = float(np.mean(lsb_flat))
            
            # Chi-square test
            expected = len(lsb_flat) / 2
            observed_ones = np.sum(lsb_flat)
            observed_zeros = len(lsb_flat) - observed_ones
            chi_square = float(((observed_ones - expected) ** 2 + 
                              (observed_zeros - expected) ** 2) / expected)
            
            # Efficient difference calculations
            if channel.shape[1] > 1:
                avg_horizontal_diff = float(np.mean(np.abs(np.diff(channel, axis=1))))
            else:
                avg_horizontal_diff = 0.0
            
            if channel.shape[0] > 1:
                avg_vertical_diff = float(np.mean(np.abs(np.diff(channel, axis=0))))
            else:
                avg_vertical_diff = 0.0
            
            lsb_stats.append({
                'ones_ratio': ones_ratio,
                'chi_square': chi_square,
                'avg_horizontal_diff': avg_horizontal_diff,
                'avg_vertical_diff': avg_vertical_diff
            })
        
        # Overall assessment
        if lsb_stats:
            avg_ones_ratio = np.mean([s['ones_ratio'] for s in lsb_stats])
            avg_chi_square = np.mean([s['chi_square'] for s in lsb_stats])
        else:
            avg_ones_ratio, avg_chi_square = 0.5, 0.0
        
        # Improved suspicion detection
        randomness_suspicion = (
            "High" if abs(avg_ones_ratio - 0.5) > 0.05 
            else "Medium" if abs(avg_ones_ratio - 0.5) > 0.02 
            else "Low"
        )
        
        chi_suspicion = (
            "High" if avg_chi_square > 10.83 
            else "Medium" if avg_chi_square > 3.84 
            else "Low"
        )
        
        return {
            'channels': lsb_stats,
            'average_ones_ratio': float(avg_ones_ratio),
            'average_chi_square': float(avg_chi_square),
            'randomness_suspicion': randomness_suspicion,
            'chi_suspicion': chi_suspicion
        }
    
    def _generate_fast_lsb_analysis(self, img_array) -> Dict:
        """Generate quick LSB analysis for fast mode."""
        # Quick approximation without full analysis
        if np is not None and len(img_array.shape) == 3:
            # Sample-based analysis for speed
            sample_size = min(10000, img_array.size // 10)
            sample_data = np.random.choice(img_array.flatten(), sample_size)
            lsbs = sample_data & 1
            ones_ratio = float(np.mean(lsbs))
            chi_square = 2.0 * abs(ones_ratio - 0.5) * sample_size
        else:
            ones_ratio = 0.5
            chi_square = 0.0
        
        return {
            'channels': [{'ones_ratio': ones_ratio, 'chi_square': chi_square, 
                         'avg_horizontal_diff': 1.0, 'avg_vertical_diff': 1.0}],
            'average_ones_ratio': ones_ratio,
            'average_chi_square': chi_square,
            'randomness_suspicion': "Low",
            'chi_suspicion': "Low",
            'note': "Fast analysis - limited accuracy"
        }
    
    def _assess_suitability_optimized(self, stats: Dict, quality: Dict, lsb: Dict) -> Dict:
        """Optimized suitability assessment with better scoring."""
        score = 0
        reasons = []
        warnings = []
        
        # Enhanced entropy assessment
        entropy = stats.get('entropy', 0)
        if entropy > 7.5:
            score += 35
            reasons.append("Excellent entropy level - ideal for steganography")
        elif entropy > 7.0:
            score += 30
            reasons.append("Very good entropy level")
        elif entropy > 6.0:
            score += 20
            reasons.append("Good entropy level")
        elif entropy > 5.0:
            score += 10
            reasons.append("Moderate entropy level")
        else:
            score += 2
            reasons.append("Low entropy - data may be easily detectable")
            warnings.append("Consider using images with more visual complexity")
        
        # Texture complexity assessment
        texture = quality.get('texture_complexity', 0)
        if texture > 50:
            score += 25
            reasons.append("Excellent texture complexity")
        elif texture > 30:
            score += 20
            reasons.append("Good texture complexity")
        elif texture > 15:
            score += 15
            reasons.append("Moderate texture complexity")
        else:
            score += 5
            reasons.append("Low texture complexity")
            warnings.append("Images with more detail provide better hiding capability")
        
        # Noise level assessment (sweet spot analysis)
        noise = quality.get('noise_level', 0)
        if 15 <= noise <= 40:
            score += 20
            reasons.append("Optimal noise level for steganography")
        elif 5 <= noise < 15 or 40 < noise <= 60:
            score += 15
            reasons.append("Good noise level")
        elif noise > 80:
            score += 8
            reasons.append("High noise level - may affect data integrity")
            warnings.append("Very noisy images may corrupt hidden data")
        else:
            score += 10
            reasons.append("Low noise level - acceptable")
        
        # LSB distribution assessment
        ones_ratio = lsb.get('average_ones_ratio', 0.5)
        if 0.48 <= ones_ratio <= 0.52:
            score += 15
            reasons.append("Perfect LSB distribution")
        elif 0.45 <= ones_ratio <= 0.55:
            score += 12
            reasons.append("Good LSB distribution")
        elif 0.4 <= ones_ratio <= 0.6:
            score += 8
            reasons.append("Acceptable LSB distribution")
        else:
            score += 3
            reasons.append("Poor LSB distribution")
            warnings.append("Unusual bit patterns may raise suspicion")
        
        # File format bonus
        image_path_lower = self.image_path.lower()
        if image_path_lower.endswith('.png'):
            score += 10
            reasons.append("PNG format - lossless compression ideal for steganography")
        elif image_path_lower.endswith(('.bmp', '.tiff', '.tif')):
            score += 8
            reasons.append("Lossless format - good for steganography")
        elif image_path_lower.endswith(('.jpg', '.jpeg')):
            score += 2
            warnings.append("JPEG compression may interfere with hidden data")
        
        # Determine rating with more granular scoring
        if score >= 85:
            rating = "Excellent"
        elif score >= 70:
            rating = "Very Good"
        elif score >= 55:
            rating = "Good"
        elif score >= 40:
            rating = "Fair"
        elif score >= 25:
            rating = "Poor"
        else:
            rating = "Very Poor"
        
        return {
            'score': score,
            'rating': rating,
            'reasons': reasons,
            'warnings': warnings,
            'max_score': 100
        }

    def _analyze_image(self) -> Dict:
        """Perform comprehensive image analysis."""
        if not PILImage or np is None:
            raise ImportError("PIL and numpy are required for image analysis")
        
        # Load image
        try:
            with PILImage.open(self.image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get basic properties
                width, height = img.size
                channels = len(img.getbands())
                
                # Convert to numpy array for analysis
                img_array = np.array(img)
        except Exception as e:
            raise Exception(f"Failed to load image: {e}")
        
        self.status_updated.emit("Analyzing image properties...")
        self.progress_updated.emit(25)
        
        # Basic image properties
        file_size = Path(self.image_path).stat().st_size
        total_pixels = width * height
        bits_per_pixel = 24 if channels == 3 else 32  # Assume RGB or RGBA
        
        # Calculate theoretical capacity (LSB steganography)
        # For RGB, we can use 1 LSB per color channel = 3 bits per pixel
        lsb_capacity_bits = total_pixels * 3  # 3 channels (RGB)
        lsb_capacity_bytes = lsb_capacity_bits // 8
        
        self.status_updated.emit("Performing statistical analysis...")
        self.progress_updated.emit(50)
        
        # Statistical analysis
        stats = self._calculate_statistics(img_array)
        
        self.status_updated.emit("Analyzing LSB patterns...")
        self.progress_updated.emit(70)
        
        # LSB analysis
        lsb_analysis = self._analyze_lsb_patterns(img_array)
        
        self.status_updated.emit("Calculating quality metrics...")
        self.progress_updated.emit(85)
        
        # Quality assessment
        quality_metrics = self._assess_quality(img_array)
        
        # Steganographic suitability
        suitability = self._assess_suitability(stats, quality_metrics, lsb_analysis)
        
        return {
            'basic_info': {
                'file_path': self.image_path,
                'file_size': file_size,
                'width': width,
                'height': height,
                'channels': channels,
                'total_pixels': total_pixels,
                'bits_per_pixel': bits_per_pixel
            },
            'capacity': {
                'lsb_capacity_bits': lsb_capacity_bits,
                'lsb_capacity_bytes': lsb_capacity_bytes,
                'lsb_capacity_kb': lsb_capacity_bytes / 1024,
                'lsb_capacity_mb': lsb_capacity_bytes / (1024 * 1024),
                'capacity_ratio': lsb_capacity_bytes / file_size if file_size > 0 else 0
            },
            'statistics': stats,
            'lsb_analysis': lsb_analysis,
            'quality_metrics': quality_metrics,
            'suitability': suitability
        }
    
    def _calculate_statistics(self, img_array) -> Dict:
        """Calculate statistical properties of the image."""
        if np is None:
            raise ImportError("numpy is required for statistical analysis")
            
        # Flatten array for overall statistics
        flat = img_array.flatten()
        
        # Per-channel statistics
        channel_stats = []
        for i in range(img_array.shape[2]):
            channel = img_array[:, :, i].flatten()
            channel_stats.append({
                'mean': float(np.mean(channel)),
                'std': float(np.std(channel)),
                'min': int(np.min(channel)),
                'max': int(np.max(channel)),
                'median': float(np.median(channel)),
                'variance': float(np.var(channel))
            })
        
        # Overall statistics
        return {
            'overall': {
                'mean': float(np.mean(flat)),
                'std': float(np.std(flat)),
                'min': int(np.min(flat)),
                'max': int(np.max(flat)),
                'median': float(np.median(flat)),
                'variance': float(np.var(flat))
            },
            'channels': channel_stats,
            'entropy': self._calculate_entropy(flat)
        }
    
    def _calculate_entropy(self, data) -> float:
        """Calculate Shannon entropy of the data."""
        if np is None:
            raise ImportError("numpy is required for entropy calculation")
            
        # Calculate histogram
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        
        # Normalize to get probabilities
        hist = hist / len(data)
        
        # Remove zero probabilities
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        return float(entropy)
    
    def _analyze_lsb_patterns(self, img_array) -> Dict:
        """Analyze LSB patterns to detect potential steganographic content."""
        if np is None:
            raise ImportError("numpy is required for LSB pattern analysis")
            
        lsb_stats = []
        
        # Analyze each channel
        for i in range(img_array.shape[2]):
            channel = img_array[:, :, i]
            
            # Extract LSBs
            lsbs = channel & 1
            
            # Calculate LSB statistics
            lsb_flat = lsbs.flatten()
            ones_ratio = np.sum(lsb_flat) / len(lsb_flat)
            
            # Chi-square test for randomness
            expected = len(lsb_flat) / 2
            observed_ones = np.sum(lsb_flat)
            observed_zeros = len(lsb_flat) - observed_ones
            
            chi_square = ((observed_ones - expected) ** 2 + 
                         (observed_zeros - expected) ** 2) / expected
            
            # Analyze adjacent pixel differences
            diff_horizontal = np.abs(np.diff(channel, axis=1))
            diff_vertical = np.abs(np.diff(channel, axis=0))
            
            lsb_stats.append({
                'ones_ratio': float(ones_ratio),
                'chi_square': float(chi_square),
                'avg_horizontal_diff': float(np.mean(diff_horizontal)),
                'avg_vertical_diff': float(np.mean(diff_vertical))
            })
        
        # Overall LSB assessment
        avg_ones_ratio = np.mean([s['ones_ratio'] for s in lsb_stats])
        avg_chi_square = np.mean([s['chi_square'] for s in lsb_stats])
        
        # Suspicion levels (basic heuristics)
        randomness_suspicion = "High" if avg_ones_ratio > 0.55 or avg_ones_ratio < 0.45 else "Low"
        chi_suspicion = "High" if avg_chi_square > 10.83 else "Medium" if avg_chi_square > 3.84 else "Low"
        
        return {
            'channels': lsb_stats,
            'average_ones_ratio': float(avg_ones_ratio),
            'average_chi_square': float(avg_chi_square),
            'randomness_suspicion': randomness_suspicion,
            'chi_suspicion': chi_suspicion
        }
    
    def _assess_quality(self, img_array) -> Dict:
        """Assess image quality metrics."""
        if np is None:
            raise ImportError("numpy is required for quality assessment")
            
        # Calculate noise level (standard deviation of Laplacian)
        gray = np.mean(img_array, axis=2)
        laplacian = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        
        # Apply Laplacian filter
        filtered = np.zeros_like(gray)
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                filtered[i, j] = np.sum(gray[i-1:i+2, j-1:j+2] * laplacian)
        
        noise_level = np.std(filtered)
        
        # Calculate sharpness (variance of Laplacian)
        sharpness = np.var(filtered)
        
        # Assess texture complexity
        texture_complexity = np.std(gray)
        
        return {
            'noise_level': float(noise_level),
            'sharpness': float(sharpness),
            'texture_complexity': float(texture_complexity)
        }
    
    def _assess_suitability(self, stats: Dict, quality: Dict, lsb: Dict) -> Dict:
        """Assess overall suitability for steganography."""
        score = 0
        reasons = []
        
        # Entropy assessment (higher is better for steganography)
        if stats['entropy'] > 7.0:
            score += 30
            reasons.append("Good entropy level (high information content)")
        elif stats['entropy'] > 6.0:
            score += 20
            reasons.append("Moderate entropy level")
        else:
            score += 5
            reasons.append("Low entropy - may not hide data well")
        
        # Texture complexity (higher is better)
        if quality['texture_complexity'] > 30:
            score += 25
            reasons.append("High texture complexity - good for hiding data")
        elif quality['texture_complexity'] > 15:
            score += 15
            reasons.append("Moderate texture complexity")
        else:
            score += 5
            reasons.append("Low texture complexity - data may be detectable")
        
        # Noise level (moderate is best)
        if 10 < quality['noise_level'] < 50:
            score += 20
            reasons.append("Optimal noise level for steganography")
        elif quality['noise_level'] > 80:
            score += 5
            reasons.append("High noise level - may affect data integrity")
        else:
            score += 10
            reasons.append("Low noise level - acceptable but not ideal")
        
        # LSB randomness (close to 0.5 is better)
        ones_ratio = lsb['average_ones_ratio']
        if 0.48 <= ones_ratio <= 0.52:
            score += 15
            reasons.append("Excellent LSB distribution")
        elif 0.45 <= ones_ratio <= 0.55:
            score += 10
            reasons.append("Good LSB distribution")
        else:
            score += 2
            reasons.append("Poor LSB distribution - may raise suspicion")
        
        # File format bonus (PNG is lossless)
        if self.image_path.lower().endswith('.png'):
            score += 10
            reasons.append("PNG format - lossless compression")
        
        # Determine overall rating
        if score >= 80:
            rating = "Excellent"
        elif score >= 60:
            rating = "Good"
        elif score >= 40:
            rating = "Fair"
        else:
            rating = "Poor"
        
        return {
            'score': score,
            'rating': rating,
            'reasons': reasons
        }


class AnalysisDialog(QDialog):
    """Dialog for comprehensive image analysis and capacity assessment."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Analysis & Capacity Assessment")
        self.setModal(True)
        self.resize(1100, 800)
        
        # Initialize components
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        
        # State variables
        self.worker_thread = None
        self.analysis_results = None
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🔍 Image Analysis & Capacity Assessment")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Analyze images for steganographic capacity, detect existing hidden data, "
            "and assess suitability for steganographic operations."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Image selection
        selection_group = QGroupBox("📁 Image Selection & Analysis Settings")
        selection_layout = QVBoxLayout(selection_group)
        
        self.image_label = QLabel("No image selected")
        self.image_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.image_label.setMinimumHeight(60)
        
        # Analysis quality selection
        quality_layout = QHBoxLayout()
        quality_label = QLabel("Analysis Quality:")
        quality_layout.addWidget(quality_label)
        
        from PySide6.QtWidgets import QComboBox
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Fast", "Balanced", "Thorough"])
        self.quality_combo.setCurrentText("Balanced")
        self.quality_combo.setToolTip(
            "Fast: Quick analysis with sampling for large images\n"
            "Balanced: Good balance between speed and accuracy\n"
            "Thorough: Complete analysis with advanced algorithms"
        )
        quality_layout.addWidget(self.quality_combo)
        quality_layout.addStretch()
        
        selection_button_layout = QHBoxLayout()
        self.select_image_button = QPushButton("📁 Select Image")
        self.select_image_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        
        self.analyze_button = QPushButton("🔍 Analyze Image")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        
        self.cancel_button = QPushButton("⏹ Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #d32f2f; }
        """)
        
        selection_button_layout.addWidget(self.select_image_button)
        selection_button_layout.addStretch()
        selection_button_layout.addWidget(self.cancel_button)
        selection_button_layout.addWidget(self.analyze_button)
        
        selection_layout.addWidget(self.image_label)
        selection_layout.addLayout(quality_layout)
        selection_layout.addLayout(selection_button_layout)
        layout.addWidget(selection_group)
        
        # Progress section
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)
        
        # Results area
        self.results_tabs = QTabWidget()
        self.results_tabs.setVisible(False)
        
        # Create result tabs
        self.overview_tab = self.create_overview_tab()
        self.capacity_tab = self.create_capacity_tab()
        self.statistics_tab = self.create_statistics_tab()
        self.detection_tab = self.create_detection_tab()
        
        self.results_tabs.addTab(self.overview_tab, "📊 Overview")
        self.results_tabs.addTab(self.capacity_tab, "💾 Capacity")
        self.results_tabs.addTab(self.statistics_tab, "📈 Statistics")
        self.results_tabs.addTab(self.detection_tab, "🔍 Detection")
        
        layout.addWidget(self.results_tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.export_button = QPushButton("📄 Export Report")
        self.export_button.setEnabled(False)
        self.close_button = QPushButton("Close")
        
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
    
    def create_overview_tab(self) -> QWidget:
        """Create the overview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Basic info table
        self.basic_info_table = QTableWidget()
        self.basic_info_table.setColumnCount(2)
        self.basic_info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.basic_info_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(QLabel("Basic Information:"))
        layout.addWidget(self.basic_info_table)
        
        # Suitability assessment
        suitability_group = QGroupBox("Steganographic Suitability")
        suitability_layout = QVBoxLayout(suitability_group)
        
        self.suitability_rating = QLabel("Not analyzed")
        self.suitability_rating.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        
        self.suitability_reasons = QTextEdit()
        self.suitability_reasons.setReadOnly(True)
        self.suitability_reasons.setMaximumHeight(120)
        
        suitability_layout.addWidget(QLabel("Overall Rating:"))
        suitability_layout.addWidget(self.suitability_rating)
        suitability_layout.addWidget(QLabel("Assessment Details:"))
        suitability_layout.addWidget(self.suitability_reasons)
        
        layout.addWidget(suitability_group)
        
        return widget
    
    def create_capacity_tab(self) -> QWidget:
        """Create the capacity analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Capacity table
        self.capacity_table = QTableWidget()
        self.capacity_table.setColumnCount(2)
        self.capacity_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.capacity_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(QLabel("Storage Capacity (LSB Steganography):"))
        layout.addWidget(self.capacity_table)
        
        # Visual representation would go here (chart)
        info_label = QLabel(
            "💡 Tip: LSB steganography uses the least significant bit of each color channel. "
            "The actual usable capacity may be lower due to encryption overhead and error correction."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            "color: #666; font-size: 11px; padding: 10px; "
            "background-color: #f0f8ff; border-radius: 5px; margin-top: 10px;"
        )
        layout.addWidget(info_label)
        
        return widget
    
    def create_statistics_tab(self) -> QWidget:
        """Create the statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistics table
        self.statistics_table = QTableWidget()
        self.statistics_table.setColumnCount(4)
        self.statistics_table.setHorizontalHeaderLabels(["Channel", "Mean", "Std Dev", "Entropy"])
        self.statistics_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(QLabel("Statistical Analysis:"))
        layout.addWidget(self.statistics_table)
        
        # Quality metrics
        quality_group = QGroupBox("Quality Metrics")
        quality_layout = QVBoxLayout(quality_group)
        
        self.quality_table = QTableWidget()
        self.quality_table.setColumnCount(2)
        self.quality_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.quality_table.horizontalHeader().setStretchLastSection(True)
        
        quality_layout.addWidget(self.quality_table)
        layout.addWidget(quality_group)
        
        return widget
    
    def create_detection_tab(self) -> QWidget:
        """Create the steganographic detection tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Detection results
        detection_group = QGroupBox("LSB Pattern Analysis")
        detection_layout = QVBoxLayout(detection_group)
        
        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(3)
        self.detection_table.setHorizontalHeaderLabels(["Test", "Result", "Suspicion Level"])
        self.detection_table.horizontalHeader().setStretchLastSection(True)
        
        detection_layout.addWidget(self.detection_table)
        layout.addWidget(detection_group)
        
        # Analysis explanation
        explanation = QLabel(
            "🔍 Detection Analysis:\n"
            "• Chi-Square Test: Measures randomness of LSB distribution\n"
            "• Ones Ratio: Proportion of 1s in LSB planes (should be ~0.5 for natural images)\n"
            "• High suspicion levels may indicate presence of hidden data"
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet(
            "color: #444; font-size: 11px; padding: 15px; "
            "background-color: #f9f9f9; border-radius: 5px; margin-top: 10px;"
        )
        layout.addWidget(explanation)
        
        return widget
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        self.select_image_button.clicked.connect(self.select_image)
        self.analyze_button.clicked.connect(self.analyze_image)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.export_button.clicked.connect(self.export_report)
        self.close_button.clicked.connect(self.accept)
    
    def select_image(self):
        """Select an image for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image for Analysis", "",
            "Image Files (*.png *.bmp *.tiff *.tif *.jpg *.jpeg);;All Files (*)"
        )
        
        if file_path:
            self.image_path = file_path
            self.image_label.setText(f"Selected: {Path(file_path).name}")
            self.analyze_button.setEnabled(True)
            
            # Clear previous results
            self.results_tabs.setVisible(False)
            self.export_button.setEnabled(False)
    
    def analyze_image(self):
        """Start image analysis."""
        if not hasattr(self, 'image_path'):
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return
        
        # Get analysis quality setting
        quality_map = {"Fast": "fast", "Balanced": "balanced", "Thorough": "thorough"}
        selected_quality = quality_map.get(self.quality_combo.currentText(), "balanced")
        
        # Start worker thread with quality setting
        self.worker_thread = AnalysisWorkerThread(self.image_path, selected_quality)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.status_label.setText)
        self.worker_thread.analysis_completed.connect(self.on_analysis_completed)
        self.worker_thread.error_occurred.connect(self.on_analysis_error)
        
        # Update UI state for analysis
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.analyze_button.setVisible(False)
        self.cancel_button.setVisible(True)
        self.quality_combo.setEnabled(False)
        
        # Start analysis
        self.worker_thread.start()
        
        # Update status
        estimated_time = {
            "fast": "10-30 seconds",
            "balanced": "30-60 seconds", 
            "thorough": "1-3 minutes"
        }.get(selected_quality, "unknown")
        self.logger.info(f"Starting {selected_quality} analysis, estimated time: {estimated_time}")
    
    def on_analysis_completed(self, results: Dict):
        """Handle analysis completion."""
        self.analysis_results = results
        
        # Update UI
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(True)
        self.status_label.setText("Analysis completed successfully!")
        
        # Populate results
        self.populate_results(results)
        
        # Show results tabs
        self.results_tabs.setVisible(True)
        self.export_button.setEnabled(True)
    
    def populate_results(self, results: Dict):
        """Populate the results tabs with analysis data."""
        # Overview tab
        self.populate_basic_info(results['basic_info'])
        self.populate_suitability(results['suitability'])
        
        # Capacity tab
        self.populate_capacity(results['capacity'])
        
        # Statistics tab
        self.populate_statistics(results['statistics'], results['quality_metrics'])
        
        # Detection tab
        self.populate_detection(results['lsb_analysis'])
    
    def populate_basic_info(self, basic_info: Dict):
        """Populate basic information table."""
        info_items = [
            ("File Path", Path(basic_info['file_path']).name),
            ("File Size", f"{basic_info['file_size']:,} bytes"),
            ("Dimensions", f"{basic_info['width']} × {basic_info['height']}"),
            ("Channels", str(basic_info['channels'])),
            ("Total Pixels", f"{basic_info['total_pixels']:,}"),
            ("Bits per Pixel", str(basic_info['bits_per_pixel']))
        ]
        
        self.basic_info_table.setRowCount(len(info_items))
        for i, (prop, value) in enumerate(info_items):
            self.basic_info_table.setItem(i, 0, QTableWidgetItem(prop))
            self.basic_info_table.setItem(i, 1, QTableWidgetItem(value))
        
        self.basic_info_table.resizeColumnsToContents()
    
    def populate_suitability(self, suitability: Dict):
        """Populate suitability assessment."""
        rating = suitability['rating']
        score = suitability['score']
        
        # Set rating with color coding
        self.suitability_rating.setText(f"{rating} ({score}/100)")
        
        if rating == "Excellent":
            color = "#4CAF50"
        elif rating == "Good":
            color = "#FF9800"
        elif rating == "Fair":
            color = "#FFC107"
        else:
            color = "#F44336"
        
        self.suitability_rating.setStyleSheet(
            f"font-size: 16px; font-weight: bold; padding: 10px; color: {color};"
        )
        
        # Set reasons
        reasons_text = "\n".join([f"• {reason}" for reason in suitability['reasons']])
        self.suitability_reasons.setText(reasons_text)
    
    def populate_capacity(self, capacity: Dict):
        """Populate capacity information."""
        capacity_items = [
            ("LSB Capacity (bits)", f"{capacity['lsb_capacity_bits']:,}"),
            ("LSB Capacity (bytes)", f"{capacity['lsb_capacity_bytes']:,}"),
            ("LSB Capacity (KB)", f"{capacity['lsb_capacity_kb']:.1f}"),
            ("LSB Capacity (MB)", f"{capacity['lsb_capacity_mb']:.2f}"),
            ("Capacity/File Size Ratio", f"{capacity['capacity_ratio']:.2f}")
        ]
        
        self.capacity_table.setRowCount(len(capacity_items))
        for i, (metric, value) in enumerate(capacity_items):
            self.capacity_table.setItem(i, 0, QTableWidgetItem(metric))
            self.capacity_table.setItem(i, 1, QTableWidgetItem(value))
        
        self.capacity_table.resizeColumnsToContents()
    
    def populate_statistics(self, statistics: Dict, quality: Dict):
        """Populate statistics and quality metrics."""
        # Statistics table
        channels = ["Overall"] + [f"Channel {i+1}" for i in range(len(statistics['channels']))]
        stats_data = [statistics['overall']] + statistics['channels']
        
        self.statistics_table.setRowCount(len(channels))
        for i, (channel, stats) in enumerate(zip(channels, stats_data)):
            self.statistics_table.setItem(i, 0, QTableWidgetItem(channel))
            self.statistics_table.setItem(i, 1, QTableWidgetItem(f"{stats['mean']:.2f}"))
            self.statistics_table.setItem(i, 2, QTableWidgetItem(f"{stats['std']:.2f}"))
            if channel == "Overall":
                self.statistics_table.setItem(i, 3, QTableWidgetItem(f"{statistics['entropy']:.3f}"))
            else:
                self.statistics_table.setItem(i, 3, QTableWidgetItem("N/A"))
        
        self.statistics_table.resizeColumnsToContents()
        
        # Quality metrics
        quality_items = [
            ("Noise Level", f"{quality['noise_level']:.2f}"),
            ("Sharpness", f"{quality['sharpness']:.2f}"),
            ("Texture Complexity", f"{quality['texture_complexity']:.2f}")
        ]
        
        self.quality_table.setRowCount(len(quality_items))
        for i, (metric, value) in enumerate(quality_items):
            self.quality_table.setItem(i, 0, QTableWidgetItem(metric))
            self.quality_table.setItem(i, 1, QTableWidgetItem(value))
        
        self.quality_table.resizeColumnsToContents()
    
    def populate_detection(self, lsb_analysis: Dict):
        """Populate detection analysis."""
        detection_items = [
            ("Ones Ratio Test", f"{lsb_analysis['average_ones_ratio']:.3f}", lsb_analysis['randomness_suspicion']),
            ("Chi-Square Test", f"{lsb_analysis['average_chi_square']:.2f}", lsb_analysis['chi_suspicion'])
        ]
        
        self.detection_table.setRowCount(len(detection_items))
        for i, (test, result, suspicion) in enumerate(detection_items):
            self.detection_table.setItem(i, 0, QTableWidgetItem(test))
            self.detection_table.setItem(i, 1, QTableWidgetItem(result))
            
            suspicion_item = QTableWidgetItem(suspicion)
            if suspicion == "High":
                suspicion_item.setBackground(QPalette().color(QPalette.ColorRole.BrightText))
            elif suspicion == "Medium":
                suspicion_item.setBackground(QPalette().color(QPalette.ColorRole.Light))
            
            self.detection_table.setItem(i, 2, suspicion_item)
        
        self.detection_table.resizeColumnsToContents()
    
    def export_report(self):
        """Export analysis report to file."""
        if not self.analysis_results:
            QMessageBox.warning(self, "Warning", "No analysis results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Report", "analysis_report.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                self._write_report(file_path)
                QMessageBox.information(self, "Success", "Report exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export report:\n{str(e)}")
    
    def _write_report(self, file_path: str):
        """Write analysis report to file."""
        results = self.analysis_results
        
        if not results:
            return
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("InvisioVault - Image Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            
            # Basic info
            f.write("BASIC INFORMATION\n")
            f.write("-" * 20 + "\n")
            if 'basic_info' in results:
                basic = results['basic_info']
                f.write(f"File: {Path(basic['file_path']).name}\n")
                f.write(f"Size: {basic['file_size']:,} bytes\n")
                f.write(f"Dimensions: {basic['width']} × {basic['height']}\n")
                f.write(f"Channels: {basic['channels']}\n")
                f.write(f"Total Pixels: {basic['total_pixels']:,}\n\n")
            
            # Capacity
            f.write("CAPACITY ANALYSIS\n")
            f.write("-" * 20 + "\n")
            if 'capacity' in results:
                capacity = results['capacity']
                f.write(f"LSB Capacity: {capacity['lsb_capacity_bytes']:,} bytes ({capacity['lsb_capacity_mb']:.2f} MB)\n")
                f.write(f"Capacity Ratio: {capacity['capacity_ratio']:.2f}\n\n")
            
            # Suitability
            f.write("SUITABILITY ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            if 'suitability' in results:
                suitability = results['suitability']
                f.write(f"Rating: {suitability['rating']} ({suitability['score']}/100)\n")
                f.write("Reasons:\n")
                for reason in suitability['reasons']:
                    f.write(f"  • {reason}\n")
                f.write("\n")
            
            # Statistics
            f.write("STATISTICAL ANALYSIS\n")
            f.write("-" * 20 + "\n")
            if 'statistics' in results:
                stats = results['statistics']
                f.write(f"Entropy: {stats['entropy']:.3f}\n")
                f.write(f"Overall Mean: {stats['overall']['mean']:.2f}\n")
                f.write(f"Overall Std Dev: {stats['overall']['std']:.2f}\n\n")
            
            # Detection
            f.write("DETECTION ANALYSIS\n")
            f.write("-" * 20 + "\n")
            if 'lsb_analysis' in results:
                lsb = results['lsb_analysis']
                f.write(f"Average Ones Ratio: {lsb['average_ones_ratio']:.3f}\n")
                f.write(f"Chi-Square Value: {lsb['average_chi_square']:.2f}\n")
                f.write(f"Randomness Suspicion: {lsb['randomness_suspicion']}\n")
                f.write(f"Chi-Square Suspicion: {lsb['chi_suspicion']}\n")
    
    def cancel_analysis(self):
        """Cancel the ongoing analysis."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.worker_thread.wait(3000)  # Wait up to 3 seconds for thread to finish
            
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()  # Force terminate if still running
                self.worker_thread.wait(1000)
        
        # Reset UI state
        self.progress_bar.setVisible(False)
        self.analyze_button.setVisible(True)
        self.analyze_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.quality_combo.setEnabled(True)
        self.status_label.setText("Analysis cancelled")
        
        self.logger.info("Analysis cancelled by user")
    
    
    def on_analysis_error(self, error_message: str):
        """Handle analysis error."""
        # Reset UI state
        self.progress_bar.setVisible(False)
        self.analyze_button.setVisible(True)
        self.analyze_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.quality_combo.setEnabled(True)
        self.status_label.setText(f"Error: {error_message}")
        
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n{error_message}")
