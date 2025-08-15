"""
Image Analyzer
Comprehensive image analysis for steganographic capacity, quality assessment, and anomaly detection.
"""

import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
from enum import Enum

# Optional dependencies - graceful fallback if not available
try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None

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

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from core.crypto_utils import CryptoUtils


class AnalysisLevel(Enum):
    """Analysis complexity levels."""
    FAST = "fast"           # Basic analysis for quick assessment
    BALANCED = "balanced"   # Comprehensive analysis with good performance
    THOROUGH = "thorough"   # Deep analysis with all available metrics


class QualityMetric(Enum):
    """Image quality assessment metrics."""
    ENTROPY = "entropy"
    NOISE_LEVEL = "noise_level"
    TEXTURE_COMPLEXITY = "texture_complexity"
    COLOR_DISTRIBUTION = "color_distribution"
    EDGE_DENSITY = "edge_density"
    COMPRESSION_ARTIFACTS = "compression_artifacts"


class ImageAnalyzer:
    """Comprehensive image analysis for steganographic applications."""
    
    # Supported image formats for analysis
    SUPPORTED_FORMATS = {'.png', '.bmp', '.tiff', '.tif'}
    
    # Analysis thresholds and constants
    ENTROPY_THRESHOLD_HIGH = 6.5  # High entropy indicates good randomness
    ENTROPY_THRESHOLD_LOW = 4.0   # Low entropy indicates poor randomness
    NOISE_THRESHOLD_HIGH = 30.0   # High noise level
    NOISE_THRESHOLD_LOW = 10.0    # Low noise level
    SUITABILITY_EXCELLENT = 8.5   # Excellent suitability score
    SUITABILITY_GOOD = 6.5        # Good suitability score
    SUITABILITY_POOR = 4.0        # Poor suitability score
    
    def __init__(self):
        """Initialize the image analyzer."""
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.crypto_utils = CryptoUtils()
        
        # Verify required dependencies
        if not Image or not np:
            self.logger.warning("PIL and numpy not available - limited functionality")
        
        self.logger.info("ImageAnalyzer initialized")
    
    def validate_image_file(self, image_path: Union[str, Path]) -> bool:
        """Validate image file for analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image is valid for analysis
        """
        try:
            path = Path(image_path)
            
            if not path.exists():
                self.logger.error(f"Image file not found: {path}")
                return False
            
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                self.logger.warning(f"Unsupported format: {path.suffix}")
                return False
            
            # Try to open the image
            if Image:
                with Image.open(path) as img:
                    # Basic validation
                    if img.size[0] < 10 or img.size[1] < 10:
                        self.logger.error("Image too small for analysis")
                        return False
                    
                    if img.mode not in ['RGB', 'RGBA', 'L']:
                        self.logger.warning(f"Unusual color mode: {img.mode}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating image: {e}")
            return False
    
    def analyze_image_comprehensive(self, image_path: Union[str, Path], 
                                  analysis_level: AnalysisLevel = AnalysisLevel.BALANCED) -> Dict[str, Any]:
        """Perform comprehensive image analysis.
        
        Args:
            image_path: Path to the image file
            analysis_level: Level of analysis complexity
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            path = Path(image_path)
            
            if not self.validate_image_file(path):
                raise ValueError(f"Invalid image file: {path}")
            
            self.logger.info(f"Starting {analysis_level.value} analysis of {path.name}")
            
            # Initialize results structure
            results = {
                'file_info': self._analyze_file_info(path),
                'image_properties': {},
                'capacity_analysis': {},
                'quality_metrics': {},
                'suitability_assessment': {},
                'lsb_analysis': {},
                'security_assessment': {},
                'recommendations': []
            }
            
            if not Image or not np:
                results['error'] = "PIL and numpy required for full analysis"
                return results
            
            # Load and prepare image
            with Image.open(path) as img:
                # Basic image properties
                results['image_properties'] = self._analyze_image_properties(img)
                
                # Convert to numpy array for advanced analysis
                img_array = self._prepare_image_array(img, analysis_level)
                
                # Capacity analysis
                results['capacity_analysis'] = self._analyze_capacity(img, img_array)
                
                # Quality metrics based on analysis level
                if analysis_level in [AnalysisLevel.BALANCED, AnalysisLevel.THOROUGH]:
                    results['quality_metrics'] = self._analyze_quality_metrics(img_array, analysis_level)
                    results['lsb_analysis'] = self._analyze_lsb_patterns(img_array, analysis_level)
                    
                if analysis_level == AnalysisLevel.THOROUGH:
                    results['security_assessment'] = self._analyze_security_aspects(img_array)
                
                # Suitability assessment
                results['suitability_assessment'] = self._assess_overall_suitability(
                    results['image_properties'],
                    results['capacity_analysis'],
                    results['quality_metrics'],
                    results['lsb_analysis']
                )
                
                # Generate recommendations
                results['recommendations'] = self._generate_recommendations(results)
            
            self.logger.info(f"Analysis completed for {path.name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during image analysis: {e}")
            self.error_handler.handle_exception(e)
            return {'error': str(e)}
    
    def _analyze_file_info(self, path: Path) -> Dict[str, Any]:
        """Analyze basic file information."""
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
                'format_supported': path.suffix.lower() in self.SUPPORTED_FORMATS
            }
        except Exception as e:
            self.logger.error(f"Error analyzing file info: {e}")
            return {'error': str(e)}
    
    def _analyze_image_properties(self, img: Image.Image) -> Dict[str, Any]:
        """Analyze basic image properties."""
        try:
            width, height = img.size
            return {
                'width': width,
                'height': height,
                'total_pixels': width * height,
                'aspect_ratio': width / height if height > 0 else 0,
                'mode': img.mode,
                'channels': len(img.getbands()),
                'bits_per_pixel': len(img.getbands()) * 8,
                'format': img.format,
                'has_transparency': img.mode in ['RGBA', 'LA'] or 'transparency' in img.info
            }
        except Exception as e:
            self.logger.error(f"Error analyzing image properties: {e}")
            return {'error': str(e)}
    
    def _prepare_image_array(self, img: Image.Image, analysis_level: AnalysisLevel):
        """Prepare image array for analysis with optional optimization."""
        # Convert to RGB if necessary
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        
        width, height = img.size
        
        # Optimize for large images based on analysis level
        if analysis_level == AnalysisLevel.FAST:
            if width > 2048 or height > 2048:
                # Sample down for faster processing
                max_size = 1024
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
        elif analysis_level == AnalysisLevel.THOROUGH:
            # Even for thorough analysis, optimize very large images to prevent UI freezing
            if width > 4096 or height > 4096:
                self.logger.info(f"Large image detected ({width}x{height}), optimizing for thorough analysis")
                max_size = 3072  # Larger than balanced but still manageable
                ratio = min(max_size / width, max_size / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                self.logger.info(f"Resized to {new_size[0]}x{new_size[1]} for better performance")
        
        return np.array(img)
    
    def _analyze_capacity(self, img: Image.Image, img_array) -> Dict[str, Any]:
        """Analyze steganographic capacity."""
        try:
            width, height = img.size
            channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
            
            # LSB capacity calculations
            total_pixels = width * height
            lsb_bits_per_pixel = channels  # 1 bit per channel
            total_lsb_bits = total_pixels * lsb_bits_per_pixel
            
            # Reserve space for headers and metadata
            header_overhead_bits = 256  # Estimated header size
            usable_lsb_bits = max(0, total_lsb_bits - header_overhead_bits)
            
            # Convert to bytes
            lsb_capacity_bytes = usable_lsb_bits // 8
            
            return {
                'lsb_total_bits': total_lsb_bits,
                'lsb_usable_bits': usable_lsb_bits,
                'lsb_capacity_bytes': lsb_capacity_bytes,
                'lsb_capacity_kb': lsb_capacity_bytes / 1024,
                'lsb_capacity_mb': lsb_capacity_bytes / (1024 * 1024),
                'capacity_ratio': lsb_capacity_bytes / (width * height * channels) if total_pixels > 0 else 0,
                'estimated_overhead_bits': header_overhead_bits,
                'theoretical_max_bytes': total_lsb_bits // 8
            }
        except Exception as e:
            self.logger.error(f"Error analyzing capacity: {e}")
            return {'error': str(e)}
    
    def _analyze_quality_metrics(self, img_array, analysis_level: AnalysisLevel) -> Dict[str, Any]:
        """Analyze image quality metrics."""
        try:
            metrics = {}
            
            # Entropy analysis
            metrics['entropy'] = self._calculate_entropy(img_array)
            
            # Noise level analysis
            metrics['noise_level'] = self._calculate_noise_level(img_array)
            
            # Texture complexity
            if analysis_level in [AnalysisLevel.BALANCED, AnalysisLevel.THOROUGH]:
                metrics['texture_complexity'] = self._calculate_texture_complexity(img_array)
            
            # Color distribution
            metrics['color_distribution'] = self._analyze_color_distribution(img_array)
            
            # Edge density (for thorough analysis)
            if analysis_level == AnalysisLevel.THOROUGH:
                metrics['edge_density'] = self._calculate_edge_density(img_array)
                metrics['compression_artifacts'] = self._detect_compression_artifacts(img_array)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing quality metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_entropy(self, img_array) -> Dict[str, Any]:
        """Calculate Shannon entropy for image channels."""
        try:
            if len(img_array.shape) == 2:
                # Grayscale
                hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
                hist = hist[hist > 0]
                prob = hist / hist.sum()
                entropy = -np.sum(prob * np.log2(prob))
                return {'overall': float(entropy), 'channels': [float(entropy)]}
            else:
                # Color image
                entropies = []
                for channel in range(img_array.shape[2]):
                    hist, _ = np.histogram(img_array[:, :, channel], bins=256, range=(0, 256))
                    hist = hist[hist > 0]
                    if len(hist) > 0:
                        prob = hist / hist.sum()
                        entropy = -np.sum(prob * np.log2(prob))
                        entropies.append(float(entropy))
                    else:
                        entropies.append(0.0)
                
                overall_entropy = np.mean(entropies)
                return {
                    'overall': float(overall_entropy),
                    'channels': entropies,
                    'max_entropy': float(max(entropies)) if entropies else 0.0,
                    'min_entropy': float(min(entropies)) if entropies else 0.0
                }
        except Exception as e:
            self.logger.error(f"Error calculating entropy: {e}")
            return {'error': str(e)}
    
    def _calculate_noise_level(self, img_array) -> Dict[str, Any]:
        """Calculate noise level using standard deviation."""
        try:
            if len(img_array.shape) == 2:
                # Grayscale
                noise = float(np.std(img_array))
                return {'overall': noise, 'channels': [noise]}
            else:
                # Color image
                channel_noise = []
                for channel in range(img_array.shape[2]):
                    noise = float(np.std(img_array[:, :, channel]))
                    channel_noise.append(noise)
                
                overall_noise = float(np.mean(channel_noise))
                return {
                    'overall': overall_noise,
                    'channels': channel_noise,
                    'max_noise': float(max(channel_noise)),
                    'min_noise': float(min(channel_noise))
                }
        except Exception as e:
            self.logger.error(f"Error calculating noise level: {e}")
            return {'error': str(e)}
    
    def _calculate_texture_complexity(self, img_array) -> Dict[str, Any]:
        """Calculate texture complexity using local variance."""
        try:
            if len(img_array.shape) == 3:
                # Convert to grayscale for texture analysis
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Calculate local variance using a sliding window
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            
            # Mean filter and variance calculation
            try:
                if hasattr(scipy, 'ndimage') and scipy.ndimage:
                    mean_filtered = scipy.ndimage.convolve(gray, kernel)
                    variance = scipy.ndimage.convolve((gray - mean_filtered) ** 2, kernel)
                    complexity = float(np.mean(variance))
                else:
                    # Fallback without scipy
                    complexity = float(np.var(gray))
            except (ImportError, AttributeError):
                # Fallback without scipy
                complexity = float(np.var(gray))
            
            return {
                'complexity_score': complexity,
                'texture_quality': 'high' if complexity > 100 else 'medium' if complexity > 50 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating texture complexity: {e}")
            return {'error': str(e)}
    
    def _analyze_color_distribution(self, img_array) -> Dict[str, Any]:
        """Analyze color distribution and uniformity."""
        try:
            if len(img_array.shape) == 2:
                # Grayscale
                unique_values = len(np.unique(img_array))
                return {
                    'unique_colors': unique_values,
                    'color_uniformity': unique_values / 256.0,
                    'distribution_type': 'grayscale'
                }
            else:
                # Color image
                channel_stats = []
                total_unique = 0
                
                for channel in range(img_array.shape[2]):
                    unique_vals = len(np.unique(img_array[:, :, channel]))
                    channel_stats.append({
                        'unique_values': unique_vals,
                        'mean': float(np.mean(img_array[:, :, channel])),
                        'std': float(np.std(img_array[:, :, channel]))
                    })
                    total_unique += unique_vals
                
                return {
                    'channel_statistics': channel_stats,
                    'total_unique_values': total_unique,
                    'average_unique_per_channel': total_unique / img_array.shape[2],
                    'color_diversity_score': min(10.0, total_unique / (256 * img_array.shape[2]) * 10)
                }
        except Exception as e:
            self.logger.error(f"Error analyzing color distribution: {e}")
            return {'error': str(e)}
    
    def _calculate_edge_density(self, img_array) -> Dict[str, Any]:
        """Calculate edge density using gradient magnitude."""
        try:
            if len(img_array.shape) == 3:
                # Convert to grayscale
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Calculate gradients
            try:
                if hasattr(scipy, 'ndimage') and scipy.ndimage:
                    grad_x = scipy.ndimage.sobel(gray, axis=1)
                    grad_y = scipy.ndimage.sobel(gray, axis=0)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                else:
                    # Simple gradient calculation without scipy
                    grad_x = np.gradient(gray, axis=1)
                    grad_y = np.gradient(gray, axis=0)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            except (ImportError, AttributeError):
                # Simple gradient calculation without scipy
                grad_x = np.gradient(gray, axis=1)
                grad_y = np.gradient(gray, axis=0)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            edge_density = float(np.mean(gradient_magnitude))
            edge_threshold = np.percentile(gradient_magnitude, 95)
            strong_edges = float(np.sum(gradient_magnitude > edge_threshold) / gradient_magnitude.size)
            
            return {
                'edge_density': edge_density,
                'strong_edge_ratio': strong_edges,
                'edge_quality': 'high' if edge_density > 20 else 'medium' if edge_density > 10 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating edge density: {e}")
            return {'error': str(e)}
    
    def _detect_compression_artifacts(self, img_array) -> Dict[str, Any]:
        """Detect potential compression artifacts."""
        try:
            # Simple artifact detection based on block patterns
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Check for 8x8 block patterns (JPEG artifacts)
            block_variance = []
            height, width = gray.shape
            
            for i in range(0, height-8, 8):
                for j in range(0, width-8, 8):
                    block = gray[i:i+8, j:j+8]
                    block_variance.append(np.var(block))
            
            if block_variance:
                avg_block_variance = np.mean(block_variance)
                variance_std = np.std(block_variance)
                
                # Artifacts likely if block variances are very uniform
                artifact_likelihood = max(0.0, 1.0 - (variance_std / avg_block_variance)) if avg_block_variance > 0 else 0.0
                block_variance_uniformity = float(variance_std / avg_block_variance) if avg_block_variance > 0 else 0
            else:
                artifact_likelihood = 0.0
                avg_block_variance = 0
                variance_std = 0
                block_variance_uniformity = 0
            
            return {
                'artifact_likelihood': float(artifact_likelihood),
                'block_variance_uniformity': block_variance_uniformity,
                'artifact_assessment': 'likely' if artifact_likelihood > 0.7 else 'possible' if artifact_likelihood > 0.4 else 'unlikely'
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting compression artifacts: {e}")
            return {'error': str(e)}
    
    def _analyze_lsb_patterns(self, img_array, analysis_level: AnalysisLevel) -> Dict[str, Any]:
        """Analyze LSB (Least Significant Bit) patterns."""
        try:
            results = {}
            
            if len(img_array.shape) == 2:
                # Grayscale
                channels = [img_array]
                channel_names = ['gray']
            else:
                # Color
                channels = [img_array[:, :, i] for i in range(img_array.shape[2])]
                channel_names = ['red', 'green', 'blue'] if img_array.shape[2] == 3 else [f'channel_{i}' for i in range(img_array.shape[2])]
            
            channel_results = {}
            for channel, name in zip(channels, channel_names):
                # Extract LSB plane
                lsb_plane = channel & 1
                
                # LSB statistics
                lsb_entropy = self._calculate_entropy(lsb_plane.astype(np.uint8) * 255)['overall']
                lsb_uniformity = np.abs(np.mean(lsb_plane) - 0.5)  # Should be ~0.5 for random data
                
                # Pattern detection (for balanced and thorough analysis)
                if analysis_level in [AnalysisLevel.BALANCED, AnalysisLevel.THOROUGH]:
                    pattern_score = self._detect_lsb_patterns(lsb_plane)
                else:
                    pattern_score = 0.0
                
                channel_results[name] = {
                    'lsb_entropy': float(lsb_entropy),
                    'lsb_uniformity': float(lsb_uniformity),
                    'pattern_anomaly_score': float(pattern_score),
                    'randomness_quality': 'good' if lsb_entropy > 0.9 and lsb_uniformity < 0.1 else 'suspicious'
                }
            
            results['channel_analysis'] = channel_results
            
            # Overall LSB assessment
            avg_entropy = np.mean([ch['lsb_entropy'] for ch in channel_results.values()])
            avg_uniformity = np.mean([ch['lsb_uniformity'] for ch in channel_results.values()])
            avg_pattern_score = np.mean([ch['pattern_anomaly_score'] for ch in channel_results.values()])
            
            results['overall_lsb_assessment'] = {
                'average_lsb_entropy': float(avg_entropy),
                'average_uniformity_deviation': float(avg_uniformity),
                'average_pattern_score': float(avg_pattern_score),
                'steganography_likelihood': 'high' if avg_pattern_score > 0.7 else 'medium' if avg_pattern_score > 0.3 else 'low'
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing LSB patterns: {e}")
            return {'error': str(e)}
    
    def _detect_lsb_patterns(self, lsb_plane) -> float:
        """Detect anomalous patterns in LSB plane."""
        try:
            # Chi-square test for randomness
            ones = np.sum(lsb_plane)
            zeros = lsb_plane.size - ones
            expected = lsb_plane.size / 2
            
            chi_square = ((ones - expected) ** 2 + (zeros - expected) ** 2) / expected
            
            # Convert to 0-1 score (higher = more suspicious)
            pattern_score = min(1.0, chi_square / 100.0)
            
            return pattern_score
            
        except Exception as e:
            self.logger.error(f"Error detecting LSB patterns: {e}")
            return 0.0
    
    def _analyze_security_aspects(self, img_array) -> Dict[str, Any]:
        """Analyze security-related aspects for steganography."""
        try:
            security_metrics = {}
            
            # Predictability analysis
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Calculate autocorrelation to measure predictability
            try:
                if hasattr(scipy, 'signal') and scipy.signal:
                    # Simple autocorrelation approximation
                    autocorr = scipy.signal.correlate2d(gray, gray, mode='same')
                    predictability = float(np.max(autocorr) / np.mean(autocorr))
                else:
                    # Fallback: use variance as predictability measure
                    predictability = float(np.var(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0.0
            except (ImportError, AttributeError):
                # Fallback: use variance as predictability measure
                predictability = float(np.var(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0.0
            
            security_metrics['predictability_score'] = predictability
            security_metrics['unpredictability_rating'] = 'high' if predictability < 2 else 'medium' if predictability < 5 else 'low'
            
            # Steganalysis resistance estimation
            entropy_metrics = self._calculate_entropy(img_array)
            noise_metrics = self._calculate_noise_level(img_array)
            
            resistance_score = (
                (entropy_metrics['overall'] / 8.0) * 0.4 +  # Entropy contributes 40%
                min(1.0, (noise_metrics['overall'] / 50.0)) * 0.3 +  # Noise contributes 30%
                min(1.0, 1.0 / predictability) * 0.3  # Unpredictability contributes 30%
            )
            
            security_metrics['steganalysis_resistance'] = float(resistance_score)
            security_metrics['resistance_rating'] = 'excellent' if resistance_score > 0.8 else 'good' if resistance_score > 0.6 else 'moderate' if resistance_score > 0.4 else 'poor'
            
            return security_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing security aspects: {e}")
            return {'error': str(e)}
    
    def _assess_overall_suitability(self, image_props: Dict, capacity: Dict, 
                                  quality: Dict, lsb_analysis: Dict) -> Dict[str, Any]:
        """Assess overall suitability for steganography."""
        try:
            # Initialize suitability factors
            factors = {
                'capacity_score': 0.0,
                'quality_score': 0.0,
                'security_score': 0.0,
                'format_score': 0.0
            }
            
            # Capacity scoring (0-10)
            if 'lsb_capacity_kb' in capacity:
                capacity_kb = capacity['lsb_capacity_kb']
                if capacity_kb > 1000:  # > 1MB
                    factors['capacity_score'] = 10.0
                elif capacity_kb > 100:  # > 100KB
                    factors['capacity_score'] = 8.0
                elif capacity_kb > 10:   # > 10KB
                    factors['capacity_score'] = 6.0
                elif capacity_kb > 1:    # > 1KB
                    factors['capacity_score'] = 4.0
                else:
                    factors['capacity_score'] = 2.0
            
            # Quality scoring (0-10)
            if 'entropy' in quality and 'noise_level' in quality:
                entropy_score = min(10.0, (quality['entropy']['overall'] / 8.0) * 10)
                noise_score = min(10.0, (quality['noise_level']['overall'] / 50.0) * 10)
                factors['quality_score'] = (entropy_score + noise_score) / 2
            
            # Security scoring (0-10) based on LSB analysis
            if 'overall_lsb_assessment' in lsb_analysis:
                lsb_entropy = lsb_analysis['overall_lsb_assessment'].get('average_lsb_entropy', 0)
                lsb_uniformity = lsb_analysis['overall_lsb_assessment'].get('average_uniformity_deviation', 1)
                security_score = (lsb_entropy * 5) + ((1 - lsb_uniformity) * 5)
                factors['security_score'] = min(10.0, security_score)
            
            # Format scoring
            format_name = image_props.get('format', '').upper()
            if format_name in ['PNG', 'BMP', 'TIFF']:
                factors['format_score'] = 10.0
            elif format_name in ['JPEG', 'JPG']:
                factors['format_score'] = 3.0  # Lossy compression problematic
            else:
                factors['format_score'] = 5.0  # Unknown format
            
            # Calculate overall suitability (weighted average)
            weights = {
                'capacity_score': 0.25,  # 25%
                'quality_score': 0.35,   # 35%
                'security_score': 0.30,  # 30%
                'format_score': 0.10     # 10%
            }
            
            overall_score = sum(factors[key] * weights[key] for key in factors)
            
            # Determine rating
            if overall_score >= self.SUITABILITY_EXCELLENT:
                rating = 'excellent'
                recommendation = 'Ideal for steganographic operations'
            elif overall_score >= self.SUITABILITY_GOOD:
                rating = 'good'
                recommendation = 'Well-suited for steganography'
            elif overall_score >= self.SUITABILITY_POOR:
                rating = 'moderate'
                recommendation = 'Acceptable for steganography with precautions'
            else:
                rating = 'poor'
                recommendation = 'Not recommended for steganographic use'
            
            return {
                'overall_score': float(overall_score),
                'rating': rating,
                'recommendation': recommendation,
                'factor_scores': factors,
                'factor_weights': weights,
                'max_score': 10.0
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing suitability: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        try:
            # Format recommendations
            format_name = results['image_properties'].get('format', '').upper()
            if format_name in ['JPEG', 'JPG']:
                recommendations.append("⚠️ JPEG format detected - use PNG or BMP for better steganographic security")
            elif format_name not in ['PNG', 'BMP', 'TIFF']:
                recommendations.append("❓ Unusual format detected - verify steganographic compatibility")
            
            # Capacity recommendations
            capacity = results.get('capacity_analysis', {})
            if 'lsb_capacity_kb' in capacity:
                capacity_kb = capacity['lsb_capacity_kb']
                if capacity_kb < 1:
                    recommendations.append("📏 Very low capacity - consider using a larger image")
                elif capacity_kb > 10000:  # > 10MB
                    recommendations.append("📊 Very high capacity - excellent for large data hiding")
            
            # Quality recommendations
            quality = results.get('quality_metrics', {})
            if 'entropy' in quality:
                entropy = quality['entropy']['overall']
                if entropy < self.ENTROPY_THRESHOLD_LOW:
                    recommendations.append("🎲 Low entropy detected - consider using randomization techniques")
                elif entropy > self.ENTROPY_THRESHOLD_HIGH:
                    recommendations.append("✨ High entropy - excellent natural randomness")
            
            if 'noise_level' in quality:
                noise = quality['noise_level']['overall']
                if noise < self.NOISE_THRESHOLD_LOW:
                    recommendations.append("🔇 Low noise level - steganographic changes may be more detectable")
                elif noise > self.NOISE_THRESHOLD_HIGH:
                    recommendations.append("🔊 High noise level - good for masking steganographic modifications")
            
            # LSB analysis recommendations
            lsb = results.get('lsb_analysis', {})
            if 'overall_lsb_assessment' in lsb:
                lsb_assessment = lsb['overall_lsb_assessment']
                if lsb_assessment.get('steganography_likelihood') == 'high':
                    recommendations.append("🚨 Possible existing steganographic content detected")
                elif lsb_assessment.get('average_lsb_entropy', 0) < 0.8:
                    recommendations.append("🎯 LSB planes show patterns - good baseline for steganography")
            
            # Security recommendations
            security = results.get('security_assessment', {})
            if 'steganalysis_resistance' in security:
                resistance = security['steganalysis_resistance']
                if resistance < 0.4:
                    recommendations.append("🛡️ Low steganalysis resistance - use advanced hiding techniques")
                elif resistance > 0.8:
                    recommendations.append("🔒 Excellent steganalysis resistance - highly secure for steganography")
            
            # Suitability-based recommendations
            suitability = results.get('suitability_assessment', {})
            if 'rating' in suitability:
                rating = suitability['rating']
                if rating == 'poor':
                    recommendations.append("❌ Image not recommended for steganographic use")
                elif rating == 'excellent':
                    recommendations.append("🏆 Ideal image for advanced steganographic operations")
            
            # General recommendations
            if not recommendations:
                recommendations.append("✅ Image analysis complete - see detailed metrics above")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["⚠️ Error generating recommendations - check analysis results"]
    
    def quick_suitability_check(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Perform quick suitability assessment.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Quick assessment results
        """
        try:
            if not self.validate_image_file(image_path):
                return {'suitable': False, 'reason': 'Invalid image file'}
            
            path = Path(image_path)
            
            # Quick format check
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                return {
                    'suitable': False, 
                    'reason': f'Unsupported format: {path.suffix}',
                    'recommendation': 'Use PNG, BMP, or TIFF format'
                }
            
            if Image:
                with Image.open(path) as img:
                    width, height = img.size
                    
                    # Quick capacity check
                    estimated_capacity = (width * height * len(img.getbands())) // 8
                    
                    if estimated_capacity < 1024:  # Less than 1KB
                        return {
                            'suitable': False,
                            'reason': 'Insufficient capacity',
                            'estimated_capacity_bytes': estimated_capacity,
                            'recommendation': 'Use a larger image'
                        }
                    
                    return {
                        'suitable': True,
                        'estimated_capacity_bytes': estimated_capacity,
                        'estimated_capacity_kb': estimated_capacity / 1024,
                        'dimensions': f'{width}x{height}',
                        'format': img.format,
                        'recommendation': 'Image appears suitable for steganography'
                    }
            
            return {'suitable': True, 'reason': 'Basic validation passed'}
            
        except Exception as e:
            self.logger.error(f"Error in quick suitability check: {e}")
            return {'suitable': False, 'reason': f'Analysis error: {e}'}
    
    def detect_potential_steganography(self, image_path: Union[str, Path], 
                                     analysis_level: AnalysisLevel = AnalysisLevel.BALANCED) -> Dict[str, Any]:
        """Detect potential steganographic content in image.
        
        Args:
            image_path: Path to image file
            analysis_level: Level of analysis to perform
            
        Returns:
            Detection results
        """
        try:
            # Perform comprehensive analysis
            results = self.analyze_image_comprehensive(image_path, analysis_level)
            
            if 'error' in results:
                return {'detection_confidence': 0.0, 'error': results['error']}
            
            # Extract relevant metrics for steganography detection
            detection_indicators = []
            confidence_factors = []
            
            # LSB analysis indicators
            lsb_analysis = results.get('lsb_analysis', {})
            if 'overall_lsb_assessment' in lsb_analysis:
                lsb_data = lsb_analysis['overall_lsb_assessment']
                
                # Check LSB entropy (should be ~1.0 for random steganographic data)
                lsb_entropy = lsb_data.get('average_lsb_entropy', 0)
                if lsb_entropy > 0.95:
                    detection_indicators.append("High LSB entropy suggests possible steganographic content")
                    confidence_factors.append(0.3)
                
                # Check uniformity deviation
                uniformity_dev = lsb_data.get('average_uniformity_deviation', 0)
                if uniformity_dev < 0.05:  # Very uniform
                    detection_indicators.append("LSB planes show unusual uniformity")
                    confidence_factors.append(0.2)
                
                # Pattern anomaly score
                pattern_score = lsb_data.get('average_pattern_score', 0)
                if pattern_score > 0.5:
                    detection_indicators.append("Anomalous patterns detected in LSB planes")
                    confidence_factors.append(0.4)
            
            # Calculate overall detection confidence
            if confidence_factors:
                detection_confidence = min(1.0, sum(confidence_factors))
            else:
                detection_confidence = 0.0
            
            # Determine likelihood rating
            if detection_confidence > 0.7:
                likelihood = 'high'
            elif detection_confidence > 0.4:
                likelihood = 'medium'
            elif detection_confidence > 0.1:
                likelihood = 'low'
            else:
                likelihood = 'none'
            
            return {
                'detection_confidence': float(detection_confidence),
                'steganography_likelihood': likelihood,
                'detection_indicators': detection_indicators,
                'analysis_timestamp': results.get('analysis_timestamp'),
                'recommendation': self._get_detection_recommendation(likelihood, detection_confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting steganography: {e}")
            return {
                'detection_confidence': 0.0,
                'error': str(e),
                'steganography_likelihood': 'unknown'
            }
    
    def _get_detection_recommendation(self, likelihood: str, confidence: float) -> str:
        """Get recommendation based on detection results."""
        if likelihood == 'high':
            return f"Strong indicators of steganographic content (confidence: {confidence:.1%})"
        elif likelihood == 'medium':
            return f"Possible steganographic content detected (confidence: {confidence:.1%})"
        elif likelihood == 'low':
            return f"Weak indicators of steganographic content (confidence: {confidence:.1%})"
        else:
            return "No significant indicators of steganographic content found"
    
    def get_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable analysis summary.
        
        Args:
            results: Results from analyze_image_comprehensive
            
        Returns:
            Formatted summary string
        """
        try:
            if 'error' in results:
                return f"Analysis failed: {results['error']}"
            
            summary_lines = []
            
            # Basic info
            if 'file_info' in results:
                file_info = results['file_info']
                summary_lines.append(f"📄 File: {file_info.get('filename', 'Unknown')}")
                summary_lines.append(f"💾 Size: {file_info.get('file_size_kb', 0):.1f} KB")
            
            # Image properties
            if 'image_properties' in results:
                props = results['image_properties']
                summary_lines.append(f"📐 Dimensions: {props.get('width', 0)}×{props.get('height', 0)} ({props.get('total_pixels', 0):,} pixels)")
                summary_lines.append(f"🎨 Format: {props.get('format', 'Unknown')} ({props.get('channels', 0)} channels)")
            
            # Capacity
            if 'capacity_analysis' in results:
                capacity = results['capacity_analysis']
                capacity_kb = capacity.get('lsb_capacity_kb', 0)
                summary_lines.append(f"📊 LSB Capacity: {capacity_kb:.1f} KB ({capacity.get('lsb_capacity_mb', 0):.2f} MB)")
            
            # Quality metrics
            if 'quality_metrics' in results:
                quality = results['quality_metrics']
                if 'entropy' in quality:
                    entropy = quality['entropy']['overall']
                    summary_lines.append(f"🎲 Entropy: {entropy:.2f}/8.0 ({'High' if entropy > 6.5 else 'Medium' if entropy > 4.0 else 'Low'})")
                
                if 'noise_level' in quality:
                    noise = quality['noise_level']['overall']
                    summary_lines.append(f"🔊 Noise Level: {noise:.1f} ({'High' if noise > 30 else 'Medium' if noise > 10 else 'Low'})")
            
            # Suitability
            if 'suitability_assessment' in results:
                suitability = results['suitability_assessment']
                score = suitability.get('overall_score', 0)
                rating = suitability.get('rating', 'unknown')
                summary_lines.append(f"⭐ Suitability: {score:.1f}/10.0 ({rating.title()})")
                summary_lines.append(f"💡 {suitability.get('recommendation', '')}")
            
            # Recommendations
            if 'recommendations' in results:
                recommendations = results['recommendations']
                if recommendations:
                    summary_lines.append("\n📋 Recommendations:")
                    for rec in recommendations[:3]:  # Show first 3 recommendations
                        summary_lines.append(f"  • {rec}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"
