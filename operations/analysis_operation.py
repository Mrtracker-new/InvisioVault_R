"""
Image Analysis Operation
Performs analysis on images to assess steganographic capacity and detect anomalies.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

from operations.base_operation import BaseOperation, OperationType
from core.steganography_engine import SteganographyEngine
from core.file_manager import FileManager
from utils.logger import Logger
from utils.error_handler import ErrorHandler, UserInputError, InvisioVaultError


class AnalysisOperation(BaseOperation):
    """Image analysis operation for steganography."""
    
    def __init__(self, operation_id: Optional[str] = None):
        super().__init__(operation_id)
        self.operation_type = OperationType.ANALYZE
        self.steg_engine = SteganographyEngine()
        # Note: file_manager, logger, error_handler are inherited from BaseOperation
        
        # Operation parameters
        self.image_path: Optional[Path] = None
        self.analysis_level: str = 'basic'  # 'basic' or 'full'
        
        # Results
        self.analysis_results: Dict[str, Any] = {}
    
    def configure(self, image_path: str, analysis_level: str = 'basic'):
        """Configure the analysis operation parameters.
        
        Args:
            image_path: Path to the image to analyze
            analysis_level: Level of analysis ('basic' or 'full')
        """
        try:
            self.image_path = Path(image_path)
            self.analysis_level = analysis_level
            self.logger.info(f"Analysis operation configured for image: {self.image_path}")
            
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
            
            if not self.file_manager.validate_image_file(self.image_path):
                raise UserInputError("Invalid image format", field="image_path")
            
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
            if self.analysis_level == 'full':
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
        """Perform basic analysis.
        
        Returns:
            Dictionary with basic analysis results
        """
        try:
            # Check if image_path is None before using it
            if self.image_path is None:
                raise UserInputError("Image path not configured")
                
            metadata = self.file_manager.get_file_metadata(self.image_path)
            capacity = self.steg_engine.calculate_capacity(self.image_path)
            
            results = {
                'file_name': self.image_path.name if self.image_path else 'Unknown',
                'file_path': str(self.image_path),
                'file_size': metadata['size'],
                'mime_type': metadata['mime_type'],
                'last_modified': metadata.get('modified', 'Unknown'),
                'steganographic_capacity_bytes': capacity,
                'image_dimensions': self._get_image_dimensions()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during basic analysis: {e}")
            raise
    
    def _perform_full_analysis(self) -> Dict[str, Any]:
        """Perform full analysis.
        
        Returns:
            Dictionary with full analysis results
        """
        try:
            # Example of more advanced analysis (can be expanded)
            # Use available method for analysis
            if self.image_path is None:
                raise UserInputError("Image path not configured")
                
            suitability_analysis = self.steg_engine.analyze_image_suitability(self.image_path)
            
            # Extract entropy from suitability analysis
            entropy = suitability_analysis.get('entropy', 0.0)
            
            # Mock LSB analysis data based on available information
            lsb_analysis = {
                'red_lsb_variance': suitability_analysis.get('noise_level', 0) * 0.1,
                'green_lsb_variance': suitability_analysis.get('noise_level', 0) * 0.12,
                'blue_lsb_variance': suitability_analysis.get('noise_level', 0) * 0.08,
                'analysis_method': 'derived_from_suitability'
            }
            
            results = {
                'lsb_plane_analysis': lsb_analysis,
                'entropy_analysis': entropy,
                'potential_steganography': self._assess_steganography_likelihood(lsb_analysis, entropy)
            }
            
            return results
            
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
