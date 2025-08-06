"""
Hide Operation
Implements steganographic hiding of files within images.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from operations.base_operation import BaseOperation, OperationStatus
from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine
from core.file_manager import FileManager
from utils.file_utils import FileUtils, CompressionType
from utils.config_manager import ConfigManager
from utils.logger import Logger
from utils.error_handler import ErrorHandler, ValidationError, OperationError


class HideOperation(BaseOperation):
    """Steganographic file hiding operation."""
    
    def __init__(self, operation_id: Optional[str] = None):
        super().__init__(operation_id)
        self.steg_engine = SteganographyEngine()
        self.encryption_engine = EncryptionEngine()
        self.file_manager = FileManager()
        self.file_utils = FileUtils()
        self.config = ConfigManager()
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        
        # Operation parameters
        self.cover_image_path: Optional[Path] = None
        self.files_to_hide: List[Path] = []
        self.output_image_path: Optional[Path] = None
        self.password: Optional[str] = None
        self.encryption_key: Optional[bytes] = None
        self.compression_type: str = CompressionType.GZIP
        self.compression_level: int = 6
        self.use_encryption: bool = True
        self.two_factor_enabled: bool = False
        self.keyfile_path: Optional[Path] = None
        self.use_decoy: bool = False
        self.decoy_data: Optional[bytes] = None
        
        # Results
        self.hidden_files_info: List[Dict[str, Any]] = []
        self.capacity_used: float = 0.0
        self.total_hidden_bytes: int = 0
    
    def configure(self, cover_image_path: str, files_to_hide: List[str], 
                 output_image_path: str, password: Optional[str] = None,
                 use_encryption: bool = True, compression_type: str = CompressionType.GZIP,
                 compression_level: int = 6, two_factor_enabled: bool = False,
                 keyfile_path: Optional[str] = None, use_decoy: bool = False,
                 decoy_data: Optional[bytes] = None):
        """Configure the hide operation parameters.
        
        Args:
            cover_image_path: Path to cover image
            files_to_hide: List of file paths to hide
            output_image_path: Path for output steganographic image
            password: Password for encryption
            use_encryption: Whether to encrypt hidden data
            compression_type: Type of compression to use
            compression_level: Compression level (1-9)
            two_factor_enabled: Whether to use two-factor authentication
            keyfile_path: Path to keyfile for two-factor auth
            use_decoy: Whether to use decoy data
            decoy_data: Decoy data bytes
        """
        try:
            self.cover_image_path = Path(cover_image_path)
            self.files_to_hide = [Path(f) for f in files_to_hide]
            self.output_image_path = Path(output_image_path)
            self.password = password
            self.use_encryption = use_encryption
            self.compression_type = compression_type
            self.compression_level = compression_level
            self.two_factor_enabled = two_factor_enabled
            self.keyfile_path = Path(keyfile_path) if keyfile_path else None
            self.use_decoy = use_decoy
            self.decoy_data = decoy_data
            
            # Generate encryption key if needed
            if self.use_encryption and self.password:
                salt = self.encryption_engine.generate_salt()
                self.encryption_key = self.encryption_engine.derive_key_from_password(
                    self.password, salt, keyfile_path=str(self.keyfile_path) if self.keyfile_path else None
                )
            
            self.logger.info(f"Hide operation configured: {len(self.files_to_hide)} files to hide")
            
        except Exception as e:
            self.logger.error(f"Error configuring hide operation: {e}")
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
            # Validate cover image
            if not self.cover_image_path or not self.cover_image_path.exists():
                raise ValidationError("Cover image not found", file_path=str(self.cover_image_path))
            
            if not self.file_manager.validate_image_file(self.cover_image_path):
                raise ValidationError("Invalid cover image format", file_path=str(self.cover_image_path))
            
            # Validate files to hide
            if not self.files_to_hide:
                raise ValidationError("No files specified to hide")
            
            total_size = 0
            for file_path in self.files_to_hide:
                if not file_path.exists():
                    raise ValidationError(f"File to hide not found: {file_path}", file_path=str(file_path))
                
                if not self.file_manager.validate_data_file(file_path):
                    raise ValidationError(f"Invalid file to hide: {file_path}", file_path=str(file_path))
                
                total_size += file_path.stat().st_size
            
            # Check capacity
            image_capacity = self.steg_engine.calculate_capacity(self.cover_image_path)
            estimated_size = self._estimate_payload_size(total_size)
            
            if estimated_size > image_capacity:
                raise ValidationError(
                    f"Files too large for cover image. Need {estimated_size} bytes, "
                    f"but image capacity is {image_capacity} bytes"
                )
            
            # Validate output path
            if self.output_image_path:
                self.output_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Validate encryption settings
            if self.use_encryption and not self.password:
                raise ValidationError("Password required when encryption is enabled")
            
            # Validate keyfile if two-factor is enabled
            if self.two_factor_enabled:
                if not self.keyfile_path or not self.keyfile_path.exists():
                    raise ValidationError("Keyfile required when two-factor authentication is enabled")
            
            self.logger.info("Hide operation inputs validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise
    
    def execute(self, progress_callback: Optional[Callable[[float], None]] = None,
               status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Execute the hide operation.
        
        Args:
            progress_callback: Callback for progress updates (0.0 to 1.0)
            status_callback: Callback for status updates
            
        Returns:
            Dictionary containing operation results
        """
        try:
            self.start()
            
            if progress_callback:
                progress_callback(0.0)
            if status_callback:
                status_callback("Preparing files for hiding...")
            
            # Step 1: Prepare payload data (20% progress)
            payload_data = self._prepare_payload_data()
            if progress_callback:
                progress_callback(0.2)
            
            # Step 2: Compress data if requested (40% progress)
            if status_callback:
                status_callback("Compressing data...")
            
            if self.compression_type != CompressionType.NONE:
                payload_data = self.file_utils.compress_data(
                    payload_data, self.compression_type, self.compression_level
                )
            
            if progress_callback:
                progress_callback(0.4)
            
            # Step 3: Encrypt data if requested (60% progress)
            if self.use_encryption and self.encryption_key:
                if status_callback:
                    status_callback("Encrypting data...")
                payload_data = self.encryption_engine.encrypt_data(payload_data, self.encryption_key)
            
            if progress_callback:
                progress_callback(0.6)
            
            # Step 4: Add decoy data if requested (70% progress)
            if self.use_decoy and self.decoy_data:
                if status_callback:
                    status_callback("Adding decoy data...")
                payload_data = self._add_decoy_data(payload_data)
            
            if progress_callback:
                progress_callback(0.7)
            
            # Step 5: Hide data in image (90% progress)
            if status_callback:
                status_callback("Hiding data in image...")
            
            result_image = self.steg_engine.hide_data(
                self.cover_image_path, payload_data, self.output_image_path
            )
            
            if progress_callback:
                progress_callback(0.9)
            
            # Step 6: Finalize and save results (100% progress)
            if status_callback:
                status_callback("Finalizing...")
            
            # Calculate statistics
            self.total_hidden_bytes = len(payload_data)
            image_capacity = self.steg_engine.calculate_capacity(self.cover_image_path)
            self.capacity_used = (self.total_hidden_bytes / image_capacity) * 100 if image_capacity > 0 else 0
            
            # Prepare file info
            for file_path in self.files_to_hide:
                file_info = {
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'type': self.file_manager.get_file_metadata(file_path)['mime_type'],
                    'hidden_at': datetime.now().isoformat()
                }
                self.hidden_files_info.append(file_info)
            
            if progress_callback:
                progress_callback(1.0)
            
            result = {
                'success': True,
                'output_image': str(result_image),
                'hidden_files': self.hidden_files_info,
                'total_hidden_bytes': self.total_hidden_bytes,
                'capacity_used_percent': round(self.capacity_used, 2),
                'compression_type': self.compression_type,
                'encryption_used': self.use_encryption,
                'two_factor_used': self.two_factor_enabled,
                'decoy_used': self.use_decoy,
                'operation_id': self.operation_id,
                'completed_at': datetime.now().isoformat()
            }
            
            self.complete()
            self.logger.info(f"Hide operation completed successfully: {result['total_hidden_bytes']} bytes hidden")
            
            return result
            
        except Exception as e:
            self.fail(str(e))
            self.logger.error(f"Hide operation failed: {e}")
            self.error_handler.handle_exception(e)
            raise OperationError(f"Hide operation failed: {e}")
    
    def _prepare_payload_data(self) -> bytes:
        """Prepare payload data from files to hide.
        
        Returns:
            Combined payload data
        """
        try:
            payload_parts = []
            
            # Add file count header
            file_count = len(self.files_to_hide)
            payload_parts.append(file_count.to_bytes(4, 'big'))
            
            # Add each file
            for file_path in self.files_to_hide:
                # File name length and name
                file_name = file_path.name.encode('utf-8')
                payload_parts.append(len(file_name).to_bytes(4, 'big'))
                payload_parts.append(file_name)
                
                # File data length and data
                file_data = self.file_manager.read_file_bytes(file_path)
                payload_parts.append(len(file_data).to_bytes(8, 'big'))
                payload_parts.append(file_data)
            
            return b''.join(payload_parts)
            
        except Exception as e:
            self.logger.error(f"Error preparing payload data: {e}")
            raise
    
    def _add_decoy_data(self, payload_data: bytes) -> bytes:
        """Add decoy data to payload.
        
        Args:
            payload_data: Original payload data
            
        Returns:
            Payload with decoy data
        """
        try:
            # Add decoy marker and data
            decoy_marker = b'DECOY'
            decoy_length = len(self.decoy_data).to_bytes(4, 'big')
            
            # Combine: marker + decoy_length + decoy_data + payload
            return decoy_marker + decoy_length + self.decoy_data + payload_data
            
        except Exception as e:
            self.logger.error(f"Error adding decoy data: {e}")
            raise
    
    def _estimate_payload_size(self, raw_size: int) -> int:
        """Estimate final payload size after processing.
        
        Args:
            raw_size: Raw file size
            
        Returns:
            Estimated final size
        """
        # Account for file headers (name lengths, data lengths, etc.)
        overhead_per_file = 16  # 4 bytes for name length + 8 bytes for data length + extra
        header_overhead = overhead_per_file * len(self.files_to_hide) + 4  # 4 bytes for file count
        
        estimated_size = raw_size + header_overhead
        
        # Account for compression (estimate 70% of original size)
        if self.compression_type != CompressionType.NONE:
            estimated_size = int(estimated_size * 0.7)
        
        # Account for encryption padding (AES block size is 16 bytes)
        if self.use_encryption:
            block_size = 16
            estimated_size = ((estimated_size // block_size) + 1) * block_size
        
        # Account for decoy data
        if self.use_decoy and self.decoy_data:
            estimated_size += len(self.decoy_data) + 9  # 5 bytes marker + 4 bytes length
        
        return estimated_size
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get operation summary.
        
        Returns:
            Dictionary with operation summary
        """
        return {
            'operation_type': 'hide',
            'operation_id': self.operation_id,
            'status': self.status.value,
            'cover_image': str(self.cover_image_path) if self.cover_image_path else None,
            'files_to_hide_count': len(self.files_to_hide),
            'output_image': str(self.output_image_path) if self.output_image_path else None,
            'use_encryption': self.use_encryption,
            'compression_type': self.compression_type,
            'two_factor_enabled': self.two_factor_enabled,
            'use_decoy': self.use_decoy,
            'capacity_used_percent': round(self.capacity_used, 2) if hasattr(self, 'capacity_used') else 0,
            'total_hidden_bytes': self.total_hidden_bytes if hasattr(self, 'total_hidden_bytes') else 0,
            'progress': self.progress,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
