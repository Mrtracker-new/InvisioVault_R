"""
Image Analysis Operation
Performs analysis on images to assess steganographic capacity and detect anomalies.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime

from operations.base_operation import BaseOperation, OperationType
from core.steganography.steganography_engine import SteganographyEngine
from core.analyzers.image_analyzer import ImageAnalyzer, AnalysisLevel
from core.utils.file_manager import FileManager
from utils.logger import Logger
from utils.error_handler import ErrorHandler, UserInputError, InvisioVaultError


class AnalysisOperation(BaseOperation):
    """Image analysis operation for steganography."""
    
    def __init__(self, operation_id: Optional[str] = None):
        super().__init__(operation_id)
        self.operation_type = OperationType.ANALYZE
        self.steg_engine = SteganographyEngine()
        self.image_analyzer = ImageAnalyzer()
        # Note: file_manager, logger, error_handler are inherited from BaseOperation
        
        # Operation parameters
        self.image_path: Optional[Path] = None
        self.analysis_level: str = 'basic'  # 'basic', 'full', or 'comprehensive'
        
        # Results
        self.analysis_results: Dict[str, Any] = {}
    
    def configure(self, image_path: str, analysis_level: str = 'basic'):
        """Configure the analysis operation parameters.
        
        Args:
            image_path: Path to the image to analyze
            analysis_level: Level of analysis:
                - 'basic': Quick suitability check and basic file info
                - 'full': Comprehensive analysis with quality metrics and LSB analysis
                - 'comprehensive': Thorough analysis including security assessment
        """
        try:
            self.image_path = Path(image_path)
            self.analysis_level = analysis_level
            
            # Validate analysis level
            valid_levels = ['basic', 'full', 'comprehensive']
            if analysis_level not in valid_levels:
                self.logger.warning(f"Invalid analysis level '{analysis_level}', using 'basic'")
                self.analysis_level = 'basic'
            
            self.logger.info(f"Analysis operation configured for image: {self.image_path} (level: {self.analysis_level})")
            
        except Exception as e:
            self.logger.error(f"Error configuring analysis operation: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def validate_inputs(self) -> bool:
        """Validate operation inputs.
        
        Returns:
            True if inputs are valid
        
        Raises:
            ValidationError: If validation fails
        """
        try:
            if not self.image_path or not self.image_path.exists():
                raise UserInputError("Image not found", field="image_path")
            
            # Use enhanced ImageAnalyzer for comprehensive validation
            if not self.image_analyzer._validate_image_file(self.image_path):
                raise UserInputError("Invalid image format for analysis", field="image_path")
            
            self.logger.info("Analysis operation inputs validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise
    
    def execute(self) -> bool:
        """Execute the analysis operation (required by BaseOperation).
        
        Returns:
            True if operation succeeded, False otherwise
        """
        try:
            # Step 1: Basic analysis (50% progress)
            self.analysis_results = self._perform_basic_analysis()
            self.update_progress(50)
            
            # Step 2: Full analysis if requested (100% progress)
            if self.analysis_level in ['full', 'comprehensive']:
                self.update_status("Performing full analysis...")
                full_results = self._perform_full_analysis()
                self.analysis_results.update(full_results)
            
            self.update_progress(100)
            self.analysis_results['completed_at'] = datetime.now().isoformat()
            
            self.logger.info("Analysis operation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis operation failed: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def run_analysis(self, progress_callback: Optional[Callable[[float], None]] = None,
                    status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Run the analysis operation with callbacks.
        
        Args:
            progress_callback: Callback for progress updates
            status_callback: Callback for status updates
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Set up callbacks
            if progress_callback:
                self.set_progress_callback(lambda p: progress_callback(p / 100.0))
            if status_callback:
                self.set_status_callback(status_callback)
            
            # Start the operation using the base class method
            success = self.start()
            
            if success:
                return self.analysis_results
            else:
                raise InvisioVaultError(f"Analysis operation failed: {self.error_message}")
            
        except Exception as e:
            self.logger.error(f"Analysis operation failed: {e}")
            self.error_handler.handle_exception(e)
            raise InvisioVaultError(f"Analysis operation failed: {e}")
    
    def _perform_basic_analysis(self) -> Dict[str, Any]:
        """Perform basic analysis using enhanced ImageAnalyzer.
        
        Returns:
            Dictionary with basic analysis results
        """
        try:
            if self.image_path is None:
                raise UserInputError("Image path not configured")
            
            self.update_status("Performing quick analysis...")
            
            # Use enhanced ImageAnalyzer for fast analysis
            analysis_results = self.image_analyzer.analyze_image_advanced(
                image_path=self.image_path,
                analysis_level=AnalysisLevel.FAST,
                enable_ml=False  # Disable ML for basic analysis for speed
            )
            
            # Get basic file metadata
            metadata = self.file_manager.get_file_metadata(self.image_path)
            
            # Extract key information from enhanced analysis
            file_info = analysis_results.get('file_info', {})
            capacity = analysis_results.get('capacity_analysis', {})
            security = analysis_results.get('security_assessment', {})
            
            # Create compatibility layer for existing code
            basic_recommendation = "Suitable for steganography"
            if security.get('security_rating') == 'poor':
                basic_recommendation = "Poor security characteristics - not recommended"
            elif security.get('security_rating') == 'moderate':
                basic_recommendation = "Moderate suitability for steganography"
            elif security.get('security_rating') in ['good', 'excellent']:
                basic_recommendation = "Good candidate for steganography"
                
            # Combine results in expected format
            results = {
                'analysis_type': 'basic',
                'analysis_timestamp': datetime.now().isoformat(),
                'file_info': {
                    'file_name': self.image_path.name,
                    'file_path': str(self.image_path),
                    'file_size_bytes': file_info.get('file_size_bytes', metadata['size']),
                    'file_size_kb': file_info.get('file_size_kb', metadata['size'] / 1024),
                    'mime_type': metadata['mime_type'],
                    'last_modified': metadata.get('modified', 'Unknown'),
                    'format': file_info.get('file_extension', ''),
                    'is_lossless': file_info.get('is_lossless_format', False)
                },
                'capacity_analysis': capacity,
                'security_assessment': security,
                'basic_recommendation': basic_recommendation,
                'enhanced_analysis_available': True
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during basic analysis: {e}")
            raise
    
    def _perform_full_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive analysis using enhanced ImageAnalyzer.
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            if self.image_path is None:
                raise UserInputError("Image path not configured")
            
            # Determine analysis level based on configuration
            if self.analysis_level == 'comprehensive':
                analysis_level = AnalysisLevel.THOROUGH
                enable_ml = True  # Enable ML for comprehensive analysis
            else:
                analysis_level = AnalysisLevel.BALANCED
                enable_ml = False  # Keep ML off for balanced performance
            
            # Progressive updates with progress callback
            def progress_callback(progress: float):
                # Convert 0.0-1.0 to our progress range
                current_progress = 55 + (progress * 35)  # 55% to 90%
                self.update_progress(int(current_progress))
            
            # Status updates based on analysis level
            if self.analysis_level == 'comprehensive':
                self.update_status("Performing thorough analysis with ML detection...")
            else:
                self.update_status(f"Performing {analysis_level.value} image analysis...")
            
            # Perform comprehensive analysis using enhanced ImageAnalyzer
            comprehensive_results = self.image_analyzer.analyze_image_advanced(
                image_path=self.image_path,
                analysis_level=analysis_level,
                enable_ml=enable_ml,
                progress_callback=progress_callback
            )
            
            self.update_progress(90)
            self.update_status("Generating analysis summary...")
            
            # Generate human-readable summary
            analysis_summary = self.image_analyzer.get_analysis_summary_enhanced(comprehensive_results)
            
            self.update_progress(92)
            
            # Perform dedicated steganography detection if comprehensive
            detection_results = None
            if self.analysis_level == 'comprehensive':
                self.update_status("Running advanced steganography detection...")
                detection_results = self.image_analyzer.detect_steganography_advanced(
                    image_path=self.image_path,
                    analysis_level=analysis_level,
                    use_ml=enable_ml
                )
                self.update_progress(95)
            
            self.update_status("Finalizing analysis report...")
            
            # Combine all results
            full_results = {
                'analysis_type': 'comprehensive' if self.analysis_level == 'comprehensive' else 'full',
                'analysis_level': analysis_level.value,
                'comprehensive_analysis': comprehensive_results,
                'human_readable_summary': analysis_summary,
                'detailed_recommendations': comprehensive_results.get('recommendations', []),
                'performance_metrics': comprehensive_results.get('performance_metrics', {}),
                'enhanced_features_used': {
                    'machine_learning': enable_ml,
                    'advanced_texture_analysis': analysis_level in [AnalysisLevel.THOROUGH, AnalysisLevel.RESEARCH],
                    'frequency_domain_analysis': analysis_level in [AnalysisLevel.THOROUGH, AnalysisLevel.RESEARCH],
                    'perceptual_fingerprinting': analysis_level != AnalysisLevel.LIGHTNING
                }
            }
            
            # Add steganography detection results if available
            if detection_results:
                full_results['steganography_detection'] = detection_results
            
            return full_results
            
        except Exception as e:
            self.logger.error(f"Error during full analysis: {e}")
            raise
    
    def _get_image_dimensions(self) -> Optional[Dict[str, int]]:
        """Get image dimensions (width, height).
        
        Returns:
            Dictionary with width and height, or None
        """
        try:
            from PIL import Image
            # Check if image_path is None before using it
            if self.image_path is None:
                return None
            with Image.open(self.image_path) as img:
                return {'width': img.width, 'height': img.height}
        except Exception as e:
            self.logger.warning(f"Could not get image dimensions: {e}")
            return None
    
    def _assess_steganography_likelihood(self, lsb_analysis: Dict, entropy: float) -> str:
        """Assess likelihood of steganography based on analysis.
        
        Args:
            lsb_analysis: LSB analysis results
            entropy: Entropy score
        
        Returns:
            Likelihood assessment string ('Low', 'Medium', 'High')
        """
        score = 0
        
        # High entropy can indicate encrypted data
        if entropy > 7.5:
            score += 1
        
        # Check if LSB planes look random
        # (A simple check for non-zero variance)
        if lsb_analysis['red_lsb_variance'] > 1 or \
           lsb_analysis['green_lsb_variance'] > 1 or \
           lsb_analysis['blue_lsb_variance'] > 1:
            score += 1
        
        if score >= 2:
            return 'High'
        elif score == 1:
            return 'Medium'
        else:
            return 'Low'
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get operation summary.
        
        Returns:
            Dictionary with operation summary
        """
        return {
            'operation_type': 'analysis',
            'operation_id': self.operation_id,
            'status': self.status.value,
            'image_path': str(self.image_path) if self.image_path else None,
            'analysis_level': self.analysis_level,
            'results': self.analysis_results,
            'progress': self.progress,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
    
    # Convenience methods
    def get_human_readable_summary(self) -> str:
        """Get human-readable analysis summary.
        
        Returns:
            Formatted summary string
        """
        if 'human_readable_summary' in self.analysis_results:
            return self.analysis_results['human_readable_summary']
        elif 'basic_recommendation' in self.analysis_results:
            return f"Basic Analysis: {self.analysis_results['basic_recommendation']}"
        else:
            return "Analysis not yet completed"
    
    def get_suitability_rating(self) -> Optional[str]:
        """Get overall suitability rating for steganography.
        
        Returns:
            Suitability rating string or None if not available
        """
        try:
            # Check comprehensive analysis first
            if 'comprehensive_analysis' in self.analysis_results:
                suitability = self.analysis_results['comprehensive_analysis'].get('suitability_assessment', {})
                return suitability.get('rating')
            # Fall back to basic analysis
            elif 'quick_suitability_check' in self.analysis_results:
                quick_check = self.analysis_results['quick_suitability_check']
                return 'suitable' if quick_check.get('suitable', False) else 'unsuitable'
            return None
        except Exception:
            return None
    
    def get_capacity_estimate(self) -> Optional[Dict[str, Any]]:
        """Get steganographic capacity estimate.
        
        Returns:
            Capacity information or None if not available
        """
        try:
            # Check comprehensive analysis first
            if 'comprehensive_analysis' in self.analysis_results:
                return self.analysis_results['comprehensive_analysis'].get('capacity_analysis')
            # Fall back to basic analysis
            elif 'quick_suitability_check' in self.analysis_results:
                quick_check = self.analysis_results['quick_suitability_check']
                if 'estimated_capacity_bytes' in quick_check:
                    return {
                        'estimated_capacity_bytes': quick_check['estimated_capacity_bytes'],
                        'estimated_capacity_kb': quick_check.get('estimated_capacity_kb', 0)
                    }
            return None
        except Exception:
            return None
    
    def get_recommendations(self) -> List[str]:
        """Get analysis recommendations.
        
        Returns:
            List of recommendation strings
        """
        try:
            recommendations = []
            
            # Get comprehensive recommendations
            if 'detailed_recommendations' in self.analysis_results:
                recommendations.extend(self.analysis_results['detailed_recommendations'])
            
            # Get basic recommendation
            if 'basic_recommendation' in self.analysis_results:
                recommendations.append(self.analysis_results['basic_recommendation'])
            
            return recommendations if recommendations else ["No specific recommendations available"]
        except Exception:
            return ["Error retrieving recommendations"]
    
    def has_potential_steganography(self) -> bool:
        """Check if potential steganographic content was detected.
        
        Returns:
            True if potential steganography detected
        """
        try:
            if 'steganography_detection' in self.analysis_results:
                detection = self.analysis_results['steganography_detection']
                likelihood = detection.get('steganography_likelihood', 'none')
                return likelihood in ['medium', 'high']
            return False
        except Exception:
            return False
