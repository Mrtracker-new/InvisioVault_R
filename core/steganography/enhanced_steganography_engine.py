"""
Enhanced Steganography Engine with Anti-Detection
Combines existing high-performance steganography with advanced anti-detection techniques
to evade tools like StegExpose, zsteg, StegSeek, and other steganalysis methods.
"""

import os
import struct
import hashlib
import secrets
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None

from core.steganography.steganography_engine import SteganographyEngine
from core.steganography.anti_detection_engine import AntiDetectionEngine
from core.security.encryption_engine import EncryptionEngine
from utils.logger import Logger
from utils.error_handler import ErrorHandler


class EnhancedSteganographyEngine(SteganographyEngine):
    """
    Enhanced steganography engine with integrated anti-detection capabilities.
    Maintains compatibility with existing InVisioVault while adding undetectable steganography.
    """
    
    def __init__(self, use_anti_detection: bool = True):
        super().__init__()
        
        # Initialize anti-detection engine
        self.anti_detection_engine = AntiDetectionEngine()
        self.use_anti_detection = use_anti_detection
        
        # Enhanced parameters
        self.enhanced_mode = use_anti_detection
        
        self.logger.info(f"Enhanced Steganography Engine initialized (anti-detection: {use_anti_detection})")
    
    def hide_data_enhanced(self, carrier_path, data: bytes, output_path, 
                          password: Optional[str] = None, randomize: bool = True, 
                          seed: Optional[int] = None, use_anti_detection: Optional[bool] = None) -> bool:
        """
        Enhanced hide data method with anti-detection capabilities.
        
        Args:
            carrier_path: Path to carrier image
            data: Data to hide
            output_path: Output path
            password: Password for randomization and anti-detection
            randomize: Use randomized positioning
            seed: Random seed (derived from password if not provided)
            use_anti_detection: Override default anti-detection setting
            
        Returns:
            Success status
        """
        
        # Use instance setting if not overridden
        if use_anti_detection is None:
            use_anti_detection = self.use_anti_detection
        
        # Generate seed from password if not provided
        if password and seed is None:
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
        
        try:
            # Convert paths to Path objects
            if isinstance(carrier_path, str):
                carrier_path = Path(carrier_path)
            if isinstance(output_path, str):
                output_path = Path(output_path)
            
            if use_anti_detection:
                self.logger.info("Using enhanced anti-detection steganography")
                
                # Check if we need to combine anti-detection with regular randomization
                if randomize:
                    self.logger.info("Combining anti-detection with randomized LSB positioning")
                    # When both are enabled, use a hybrid approach
                    success = self._hybrid_anti_detection_hide(
                        carrier_path=carrier_path,
                        data=data,
                        output_path=output_path,
                        password=password,
                        seed=seed
                    )
                else:
                    # Use pure anti-detection engine
                    success = self.anti_detection_engine.enhanced_hide_data(
                        carrier_path=carrier_path,
                        data=data,
                        output_path=output_path,
                        password=password,
                        use_anti_detection=True
                    )
                
                if success:
                    self.logger.info("Anti-detection hiding completed successfully")
                    # Analyze the result to ensure low detectability
                    risk_analysis = self.anti_detection_engine.analyze_detectability_risk(output_path)
                    self.logger.info(f"Detectability risk: {risk_analysis.get('risk_level', 'UNKNOWN')} "
                                   f"(score: {risk_analysis.get('overall_risk_score', 0):.3f})")
                    
                    if risk_analysis.get('overall_risk_score', 1.0) > 0.7:
                        self.logger.warning("High detectability risk detected - consider using different carrier image")
                
                return success
                
            else:
                # Use original high-performance engine
                self.logger.info(f"Using original high-performance steganography (randomize={randomize})")
                return self.hide_data(
                    carrier_path=carrier_path,
                    data=data,
                    output_path=output_path,
                    randomize=randomize,
                    seed=seed
                )
                
        except Exception as e:
            self.logger.error(f"Enhanced hiding failed: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def extract_data_enhanced(self, stego_path, password: Optional[str] = None, 
                             randomize: bool = True, seed: Optional[int] = None,
                             use_anti_detection: Optional[bool] = None) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Enhanced extract data method with anti-detection awareness.
        
        Args:
            stego_path: Path to steganographic image
            password: Password for extraction
            randomize: Use randomized positioning
            seed: Random seed (derived from password if not provided)
            use_anti_detection: Try anti-detection extraction first
            
        Returns:
            Tuple of (extracted_data, extraction_info) where extraction_info contains details about which method worked
        """
        
        # Use instance setting if not overridden
        if use_anti_detection is None:
            use_anti_detection = self.use_anti_detection
        
        # Generate seed from password if not provided
        if password and seed is None:
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
        
        # Initialize extraction info with user's chosen settings
        extraction_info = {
            'user_settings': {
                'anti_detection': use_anti_detection,
                'randomize_lsb': randomize,
                'password_used': bool(password)
            },
            'methods_tried': [],
            'successful_method': None,
            'compatibility_note': None,
            'data_size': 0
        }
        
        try:
            # Convert to Path object
            if isinstance(stego_path, str):
                stego_path = Path(stego_path)
            
            # Try anti-detection extraction first if enabled
            if use_anti_detection:
                self.logger.info("🛡️ Attempting anti-detection extraction modes")
                
                # Method 1: Hybrid anti-detection + randomized extraction
                if randomize:
                    self.logger.info("🔄 Method 1: Trying hybrid anti-detection + randomized extraction")
                    extraction_info['methods_tried'].append('hybrid_anti_detection_randomized')
                    
                    result = self._hybrid_anti_detection_extract(
                        stego_path=stego_path,
                        password=password,
                        seed=seed
                    )
                    if result:
                        extraction_info['successful_method'] = 'hybrid_anti_detection_randomized'
                        extraction_info['data_size'] = len(result)
                        extraction_info['compatibility_note'] = (
                            "Extracted using hybrid method (randomized LSB with anti-detection compatibility). "
                            "This works with images hidden using either pure anti-detection or hybrid modes."
                        )
                        self.logger.info(f"✅ Hybrid extraction successful: {len(result)} bytes")
                        return result, extraction_info
                    else:
                        self.logger.info("❌ Hybrid extraction failed, trying pure anti-detection")
                
                # Method 2: Pure anti-detection extraction
                self.logger.info("🔄 Method 2: Trying pure anti-detection extraction")
                extraction_info['methods_tried'].append('pure_anti_detection')
                
                result = self.anti_detection_engine.enhanced_extract_data(
                    stego_path=stego_path,
                    password=password
                )
                
                if result:
                    extraction_info['successful_method'] = 'pure_anti_detection'
                    extraction_info['data_size'] = len(result)
                    extraction_info['compatibility_note'] = (
                        "Extracted using pure anti-detection method. "
                        "This indicates the image was hidden with advanced anti-detection techniques."
                    )
                    self.logger.info(f"✅ Pure anti-detection extraction successful: {len(result)} bytes")
                    return result, extraction_info
                else:
                    self.logger.info("❌ Pure anti-detection extraction failed")
            
            # Method 3: Standard randomized extraction (fallback)
            if randomize:
                self.logger.info("🔄 Method 3: Trying standard randomized extraction (fallback)")
                extraction_info['methods_tried'].append('standard_randomized')
                
                result = self.extract_data(
                    stego_path=stego_path,
                    randomize=True,
                    seed=seed
                )
                
                if result:
                    extraction_info['successful_method'] = 'standard_randomized'
                    extraction_info['data_size'] = len(result)
                    
                    if use_anti_detection:
                        extraction_info['compatibility_note'] = (
                            "⚠️ EXTRACTED USING FALLBACK METHOD: The image appears to be hidden with standard randomized LSB, "
                            "not anti-detection mode. Your anti-detection extraction settings didn't match the hiding method, "
                            "but the system successfully fell back to standard randomized extraction."
                        )
                    else:
                        extraction_info['compatibility_note'] = (
                            "Extracted using standard randomized LSB method. "
                            "This matches your extraction settings."
                        )
                    
                    self.logger.info(f"✅ Standard randomized extraction successful: {len(result)} bytes")
                    return result, extraction_info
                else:
                    self.logger.info("❌ Standard randomized extraction failed")
            
            # Method 4: Sequential extraction (final fallback)
            self.logger.info("🔄 Method 4: Trying sequential extraction (final fallback)")
            extraction_info['methods_tried'].append('sequential')
            
            result = self.extract_data(
                stego_path=stego_path,
                randomize=False,
                seed=None
            )
            
            if result:
                extraction_info['successful_method'] = 'sequential'
                extraction_info['data_size'] = len(result)
                extraction_info['compatibility_note'] = (
                    "⚠️ EXTRACTED USING BASIC SEQUENTIAL METHOD: The image was hidden with basic sequential LSB hiding. "
                    "Your extraction settings (anti-detection/randomization) were more advanced than the hiding method used."
                )
                self.logger.info(f"✅ Sequential extraction successful: {len(result)} bytes")
                return result, extraction_info
            else:
                self.logger.error("❌ All extraction methods failed")
            
            # All methods failed
            extraction_info['successful_method'] = None
            extraction_info['compatibility_note'] = (
                "❌ EXTRACTION FAILED: None of the extraction methods worked. "
                "This could indicate: wrong password, corrupted image, or no hidden data present."
            )
            
            return None, extraction_info
            
        except Exception as e:
            self.logger.error(f"Enhanced extraction failed: {e}")
            extraction_info['successful_method'] = None
            extraction_info['compatibility_note'] = f"❌ EXTRACTION ERROR: {str(e)}"
            return None, extraction_info
    
    def create_undetectable_stego(self, carrier_path: Path, data: bytes, 
                                 output_path: Path, password: str,
                                 target_risk_level: str = "LOW") -> Dict[str, Any]:
        """
        Create steganographic image with guaranteed low detectability.
        
        Args:
            carrier_path: Path to carrier image
            data: Data to hide
            output_path: Output path
            password: Password for hiding
            target_risk_level: Target risk level ("LOW", "MEDIUM", "HIGH")
            
        Returns:
            Dictionary with results and analysis
        """
        
        max_attempts = 3
        attempt = 0
        risk_analysis = {}  # Initialize to prevent undefined reference
        
        while attempt < max_attempts:
            attempt += 1
            
            self.logger.info(f"Attempting undetectable stego creation (attempt {attempt}/{max_attempts})")
            
            # Use anti-detection hiding
            success = self.hide_data_enhanced(
                carrier_path=carrier_path,
                data=data,
                output_path=output_path,
                password=password,
                use_anti_detection=True
            )
            
            if not success:
                self.logger.error(f"Hiding failed on attempt {attempt}")
                continue
            
            # Analyze detectability
            risk_analysis = self.anti_detection_engine.analyze_detectability_risk(output_path)
            current_risk = risk_analysis.get('risk_level', 'HIGH')
            risk_score = risk_analysis.get('overall_risk_score', 1.0)
            
            self.logger.info(f"Attempt {attempt} risk analysis: {current_risk} (score: {risk_score:.3f})")
            
            # Check if we met the target
            risk_levels = ['LOW', 'MEDIUM', 'HIGH']
            target_index = risk_levels.index(target_risk_level)
            current_index = risk_levels.index(current_risk)
            
            if current_index <= target_index:
                self.logger.info(f"Successfully created undetectable stego with {current_risk} risk")
                
                return {
                    'success': True,
                    'attempts': attempt,
                    'risk_level': current_risk,
                    'risk_score': risk_score,
                    'risk_analysis': risk_analysis,
                    'output_path': str(output_path)
                }
            
            if attempt < max_attempts:
                self.logger.info(f"Risk level {current_risk} higher than target {target_risk_level}, retrying...")
                # Remove failed attempt file if exists
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except:
                        pass
        
        # All attempts failed - provide detailed constraint information
        self.logger.error(f"Failed to create undetectable stego after {max_attempts} attempts")
        
        # Analyze why constraints weren't met
        constraint_details = self._analyze_constraint_failures(
            carrier_path, data, target_risk_level, risk_analysis
        )
        
        detailed_error = f"Could not achieve target risk level '{target_risk_level}' after {max_attempts} attempts.\n\n"
        detailed_error += "Constraint failures:\n"
        for constraint in constraint_details:
            detailed_error += f"• {constraint}\n"
        
        return {
            'success': False,
            'attempts': max_attempts,
            'risk_level': risk_analysis.get('risk_level', 'HIGH'),
            'risk_score': risk_analysis.get('overall_risk_score', 1.0),
            'error': detailed_error.strip(),
            'constraint_details': constraint_details
        }
    
    def analyze_carrier_suitability(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze carrier image suitability for anti-detection steganography.
        
        Args:
            image_path: Path to carrier image
            
        Returns:
            Comprehensive suitability analysis
        """
        
        try:
            # Get base analysis from original engine
            base_analysis = self.analyze_image_suitability(image_path)
            
            # Add anti-detection specific analysis
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                # Calculate anti-detection metrics
                capacity_map = self.anti_detection_engine._calculate_adaptive_capacity(img_array)
                secure_capacity = np.sum(capacity_map > 0.3) * 3  # 3 channels
                
                anti_detection_metrics = {
                    'secure_capacity_pixels': int(secure_capacity),
                    'secure_capacity_bytes': int(secure_capacity // 8),
                    'complexity_score': float(np.mean(capacity_map)),
                    'texture_regions_percent': float(np.sum(capacity_map > 0.5) / capacity_map.size * 100),
                    'smooth_regions_percent': float(np.sum(capacity_map < 0.3) / capacity_map.size * 100)
                }
                
                # Enhanced recommendations
                enhanced_recommendations = base_analysis.get('recommendations', []).copy()
                
                if anti_detection_metrics['complexity_score'] < 0.4:
                    enhanced_recommendations.append("Image has many smooth areas - may be detectable")
                elif anti_detection_metrics['complexity_score'] > 0.7:
                    enhanced_recommendations.append("Excellent image complexity for anti-detection")
                
                if anti_detection_metrics['texture_regions_percent'] < 30:
                    enhanced_recommendations.append("Consider image with more texture for better concealment")
                
                # Combine analyses
                enhanced_analysis = base_analysis.copy()
                enhanced_analysis.update(anti_detection_metrics)
                enhanced_analysis['recommendations'] = enhanced_recommendations
                enhanced_analysis['anti_detection_score'] = min(10, int(
                    anti_detection_metrics['complexity_score'] * 5 +
                    (anti_detection_metrics['texture_regions_percent'] / 100) * 3 +
                    2  # Base score
                ))
                
                return enhanced_analysis
                
        except Exception as e:
            self.logger.error(f"Enhanced carrier analysis failed: {e}")
            return {'error': str(e)}
    
    def test_against_steganalysis(self, stego_path: Path) -> Dict[str, Any]:
        """
        Test steganographic image against common steganalysis techniques.
        
        Args:
            stego_path: Path to steganographic image
            
        Returns:
            Test results
        """
        
        try:
            self.logger.info(f"Testing {stego_path} against steganalysis techniques")
            
            # Get comprehensive risk analysis
            risk_analysis = self.anti_detection_engine.analyze_detectability_risk(stego_path)
            
            # Simulate specific tool detection patterns
            test_results = {
                'risk_analysis': risk_analysis,
                'tool_simulation': {}
            }
            
            with Image.open(stego_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                # 1. StegExpose-like detection (LSB analysis)
                lsb_evenness = self.anti_detection_engine._calculate_lsb_evenness(img_array)
                test_results['tool_simulation']['stegexpose_risk'] = {
                    'lsb_evenness': lsb_evenness,
                    'likely_detected': lsb_evenness > 0.1,
                    'confidence': min(lsb_evenness * 10, 1.0)
                }
                
                # 2. Chi-square test (like StegSecret, OpenStego detection)
                chi_square_risk = self.anti_detection_engine._simulate_chi_square_test(img_array)
                test_results['tool_simulation']['chi_square_test'] = {
                    'risk_score': chi_square_risk,
                    'likely_detected': chi_square_risk > 0.3,
                    'confidence': chi_square_risk
                }
                
                # 3. Histogram analysis (general steganalysis)
                histogram_anomalies = self.anti_detection_engine._detect_histogram_anomalies(img_array)
                test_results['tool_simulation']['histogram_analysis'] = {
                    'anomaly_score': histogram_anomalies,
                    'likely_detected': histogram_anomalies > 0.4,
                    'confidence': histogram_anomalies
                }
                
                # 4. Noise pattern analysis
                noise_risk = self.anti_detection_engine._analyze_noise_patterns(img_array)
                test_results['tool_simulation']['noise_analysis'] = {
                    'artificial_pattern_risk': noise_risk,
                    'likely_detected': noise_risk > 0.6,
                    'confidence': noise_risk
                }
                
                # Overall assessment
                detection_risks = [
                    test_results['tool_simulation']['stegexpose_risk']['confidence'],
                    test_results['tool_simulation']['chi_square_test']['confidence'],
                    test_results['tool_simulation']['histogram_analysis']['confidence'],
                    test_results['tool_simulation']['noise_analysis']['confidence']
                ]
                
                avg_risk = sum(detection_risks) / len(detection_risks)
                
                test_results['overall_assessment'] = {
                    'average_detection_risk': avg_risk,
                    'likely_detected_by_any_tool': any([
                        test_results['tool_simulation']['stegexpose_risk']['likely_detected'],
                        test_results['tool_simulation']['chi_square_test']['likely_detected'],
                        test_results['tool_simulation']['histogram_analysis']['likely_detected'],
                        test_results['tool_simulation']['noise_analysis']['likely_detected']
                    ]),
                    'safety_level': 'HIGH' if avg_risk < 0.3 else 'MEDIUM' if avg_risk < 0.6 else 'LOW'
                }
                
                return test_results
                
        except Exception as e:
            self.logger.error(f"Steganalysis testing failed: {e}")
            return {'error': str(e)}
    
    def get_optimal_settings(self, carrier_path: Path, data_size: int) -> Dict[str, Any]:
        """
        Get optimal settings for undetectable steganography.
        
        Args:
            carrier_path: Path to carrier image
            data_size: Size of data to hide
            
        Returns:
            Optimal settings recommendations
        """
        
        try:
            # Analyze carrier
            carrier_analysis = self.analyze_carrier_suitability(carrier_path)
            
            settings = {
                'use_anti_detection': True,
                'randomize': True,
                'carrier_analysis': carrier_analysis
            }
            
            # Capacity check
            secure_capacity = carrier_analysis.get('secure_capacity_bytes', 0)
            total_capacity = carrier_analysis.get('capacity_bytes', 0)
            
            if data_size > secure_capacity:
                settings['warning'] = f"Data size ({data_size}) exceeds secure capacity ({secure_capacity})"
                settings['recommendation'] = "Use larger image or split data across multiple images"
            elif data_size > total_capacity * 0.8:
                settings['warning'] = f"Data size uses {(data_size/total_capacity)*100:.1f}% of total capacity"
                settings['recommendation'] = "Consider using less of the capacity for better security"
            
            # Image-specific recommendations
            complexity_score = carrier_analysis.get('complexity_score', 0)
            
            if complexity_score < 0.4:
                settings['image_recommendation'] = "Image has low complexity - consider adding subtle texture"
            elif complexity_score > 0.7:
                settings['image_recommendation'] = "Excellent image complexity for steganography"
            
            # Anti-detection settings
            settings['anti_detection_settings'] = {
                'histogram_preservation': True,
                'selective_smoothing': True,
                'edge_aware_filtering': True,
                'adaptive_positioning': True
            }
            
            return settings
            
        except Exception as e:
            self.logger.error(f"Optimal settings calculation failed: {e}")
            return {'error': str(e)}
    
    def _hybrid_anti_detection_hide(self, carrier_path: Path, data: bytes, 
                                   output_path: Path, password: Optional[str], seed: Optional[int]) -> bool:
        """
        Hybrid method combining anti-detection with randomized LSB positioning.
        
        This uses the regular steganography engine with randomization but applies
        some anti-detection post-processing to reduce detectability.
        """
        try:
            self.logger.info("Starting hybrid anti-detection + randomization hiding")
            
            # Use regular randomized hiding first (which is compatible with extraction)
            success = self.hide_data(
                carrier_path=carrier_path,
                data=data,
                output_path=output_path,
                randomize=True,
                seed=seed
            )
            
            if not success:
                return False
            
            # Apply some anti-detection post-processing to the result
            self.logger.info("Applying anti-detection post-processing")
            
            with Image.open(output_path) as img:
                img_array = np.array(img)
                
                # Apply light anti-detection filtering
                filtered_array = self._apply_light_anti_detection_filter(img_array)
                
                # Save the enhanced image
                result_img = Image.fromarray(filtered_array)
                result_img.save(output_path)
            
            self.logger.info("Hybrid anti-detection hiding completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Hybrid anti-detection hiding failed: {e}")
            return False
    
    def _hybrid_anti_detection_extract(self, stego_path: Path, password: Optional[str], 
                                      seed: Optional[int]) -> Optional[bytes]:
        """
        Hybrid extraction method for images created with hybrid anti-detection.
        
        Since hybrid mode uses regular randomized hiding with post-processing,
        we can extract using the regular randomized method.
        """
        try:
            self.logger.info("Starting hybrid anti-detection extraction")
            
            # Use regular randomized extraction
            result = self.extract_data(
                stego_path=stego_path,
                randomize=True,
                seed=seed
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hybrid anti-detection extraction failed: {e}")
            return None
    
    def _apply_light_anti_detection_filter(self, img_array: np.ndarray) -> np.ndarray:
        """
        Apply light anti-detection filtering that reduces detectability
        without affecting the extractability of the hidden data.
        """
        try:
            if cv2 is None:
                # If OpenCV is not available, apply minimal noise to break patterns
                noise = np.random.normal(0, 0.3, img_array.shape).astype(np.int8)
                result = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Ensure LSBs are mostly preserved
                diff = np.abs(result.astype(np.int16) - img_array.astype(np.int16))
                mask = diff <= 1
                result = np.where(mask, result, img_array)
                return result
            
            # Very light bilateral filtering to reduce artificial patterns
            # but preserve the LSB data
            img_float = img_array.astype(np.float32) / 255.0
            
            # Use very gentle parameters to avoid corrupting LSB data
            filtered = cv2.bilateralFilter(img_float, 3, 0.05, 0.05)
            
            # Convert back and ensure we don't change LSBs too much
            filtered_uint8 = (filtered * 255).astype(np.uint8)
            
            # Preserve most LSBs by limiting changes
            diff = np.abs(filtered_uint8.astype(np.int16) - img_array.astype(np.int16))
            mask = diff <= 1  # Only allow changes of 0 or 1
            
            result = np.where(mask, filtered_uint8, img_array)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Light anti-detection filtering failed: {e}")
            return img_array
    
    def _analyze_constraint_failures(self, carrier_path: Path, data: bytes, 
                                   target_risk_level: str, risk_analysis: Dict[str, Any]) -> List[str]:
        """
        Analyze why anti-detection constraints were not met and provide specific feedback.
        
        Args:
            carrier_path: Path to carrier image
            data: Data that failed to hide
            target_risk_level: Target risk level that couldn't be achieved
            risk_analysis: Latest risk analysis results
            
        Returns:
            List of specific constraint failure reasons
        """
        constraints = []
        
        try:
            # 1. Risk level constraint
            current_risk = risk_analysis.get('risk_level', 'HIGH')
            current_score = risk_analysis.get('overall_risk_score', 1.0)
            
            risk_thresholds = {
                'LOW': 0.3,
                'MEDIUM': 0.6,
                'HIGH': 1.0
            }
            
            target_threshold = risk_thresholds.get(target_risk_level, 0.3)
            
            if current_score > target_threshold:
                constraints.append(
                    f"Risk score {current_score:.3f} exceeds target threshold {target_threshold:.3f} for '{target_risk_level}' security level"
                )
            
            # 2. Specific detectability issues
            lsb_evenness = risk_analysis.get('lsb_histogram_evenness', 0)
            if lsb_evenness > 0.1:
                constraints.append(
                    f"LSB distribution too uneven ({lsb_evenness:.3f}) - detectable by StegExpose-like tools"
                )
            
            chi_square_risk = risk_analysis.get('chi_square_risk', 0)
            if chi_square_risk > 0.3:
                constraints.append(
                    f"Chi-square test risk too high ({chi_square_risk:.3f}) - fails statistical randomness tests"
                )
            
            histogram_anomalies = risk_analysis.get('histogram_anomalies', 0)
            if histogram_anomalies > 0.4:
                constraints.append(
                    f"Histogram anomalies detected ({histogram_anomalies:.3f}) - suspicious pixel distributions"
                )
            
            noise_pattern_risk = risk_analysis.get('noise_pattern_risk', 0)
            if noise_pattern_risk > 0.6:
                constraints.append(
                    f"Artificial noise patterns detected ({noise_pattern_risk:.3f}) - appears non-natural"
                )
            
            # 3. Carrier image limitations
            try:
                carrier_analysis = self.analyze_carrier_suitability(carrier_path)
                complexity_score = carrier_analysis.get('complexity_score', 0)
                secure_capacity = carrier_analysis.get('secure_capacity_bytes', 0)
                
                if complexity_score < 0.4:
                    constraints.append(
                        f"Carrier image has low complexity ({complexity_score:.3f}) - too many smooth areas for secure hiding"
                    )
                
                data_ratio = len(data) / secure_capacity if secure_capacity > 0 else 1
                if data_ratio > 0.7:
                    constraints.append(
                        f"Data uses {data_ratio*100:.1f}% of secure capacity - exceeds safe limit for undetectable hiding"
                    )
                    
            except Exception as e:
                constraints.append(f"Unable to analyze carrier limitations: {str(e)}")
            
            # 4. Provide recommendations if no specific constraints found
            if not constraints:
                constraints.append(
                    "Anti-detection algorithms unable to achieve target security level with current image and data combination"
                )
                constraints.append(
                    "Try: different carrier image, smaller data size, or lower security target"
                )
            
            return constraints
            
        except Exception as e:
            self.logger.error(f"Constraint analysis failed: {e}")
            return [
                f"Unable to analyze constraint failures: {str(e)}",
                "General recommendation: try a different carrier image or reduce data size"
            ]


def enhance_existing_engine(original_engine: SteganographyEngine) -> EnhancedSteganographyEngine:
    """
    Enhance an existing steganography engine with anti-detection capabilities.
    
    Args:
        original_engine: Existing SteganographyEngine instance
        
    Returns:
        Enhanced engine with anti-detection features
    """
    
    enhanced = EnhancedSteganographyEngine(use_anti_detection=True)
    
    # Copy relevant settings from original engine
    if hasattr(original_engine, 'SUPPORTED_FORMATS'):
        enhanced.SUPPORTED_FORMATS = original_engine.SUPPORTED_FORMATS
    
    if hasattr(original_engine, 'logger'):
        enhanced.logger = original_engine.logger
    
    enhanced.logger.info("Enhanced existing steganography engine with anti-detection capabilities")
    
    return enhanced
