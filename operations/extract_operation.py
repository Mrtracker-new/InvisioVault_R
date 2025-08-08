"""
Extract Operation
Implements extraction of files from steganographic images.
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
from utils.error_handler import ErrorHandler, UserInputError, InvisioVaultError


class ExtractOperation(BaseOperation):
    """Steganographic file extraction operation."""
    
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
        self.steganographic_image_path: Optional[Path] = None
        self.output_directory: Optional[Path] = None
        self.password: Optional[str] = None
        self.encryption_key: Optional[bytes] = None
        self.two_factor_enabled: bool = False
        self.keyfile_path: Optional[Path] = None
        self.expected_compression: Optional[str] = None
        self.detect_decoy: bool = False
        
        # Results
        self.extracted_files: List[Dict[str, Any]] = []
        self.total_extracted_bytes: int = 0
        self.decoy_data: Optional[bytes] = None
        self.decoy_detected: bool = False
        self.extraction_results: Dict[str, Any] = {}
    
    def configure(self, steganographic_image_path: str, output_directory: str,
                 password: Optional[str] = None, two_factor_enabled: bool = False,
                 keyfile_path: Optional[str] = None, expected_compression: Optional[str] = None,
                 detect_decoy: bool = False):
        """Configure the extract operation parameters.
        
        Args:
            steganographic_image_path: Path to steganographic image
            output_directory: Directory to extract files to
            password: Password for decryption
            two_factor_enabled: Whether to use two-factor authentication
            keyfile_path: Path to keyfile for two-factor auth
            expected_compression: Expected compression type
            detect_decoy: Whether to detect and handle decoy data
        """
        try:
            self.steganographic_image_path = Path(steganographic_image_path)
            self.output_directory = Path(output_directory)
            self.password = password
            self.two_factor_enabled = two_factor_enabled
            self.keyfile_path = Path(keyfile_path) if keyfile_path else None
            self.expected_compression = expected_compression
            self.detect_decoy = detect_decoy
            
            # Generate encryption key if needed
            if self.password:
                salt = self.encryption_engine.generate_salt()
                self.encryption_key = self.encryption_engine.derive_key(
                    self.password, salt
                )
            
            self.logger.info(f"Extract operation configured for image: {self.steganographic_image_path}")
            
        except Exception as e:
            self.logger.error(f"Error configuring extract operation: {e}")
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
            # Validate steganographic image
            if not self.steganographic_image_path or not self.steganographic_image_path.exists():
                raise UserInputError(
                    "Steganographic image not found"
                )
            
            if not self.file_manager.validate_image_file(self.steganographic_image_path):
                raise UserInputError(
                    "Invalid steganographic image format"
                )
            
            # Validate output directory
            if not self.output_directory:
                raise UserInputError("Output directory not specified")
            
            # Create output directory if it doesn't exist
            self.output_directory.mkdir(parents=True, exist_ok=True)
            
            # Validate keyfile if two-factor is enabled
            if self.two_factor_enabled:
                if not self.keyfile_path or not self.keyfile_path.exists():
                    raise UserInputError("Keyfile required when two-factor authentication is enabled")
            
            self.logger.info("Extract operation inputs validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise
    
    def execute(self) -> bool:
        """Execute the extract operation (required by BaseOperation).
        
        Returns:
            True if operation succeeded, False otherwise
        """
        try:
            # Step 1: Extract raw data from image (30% progress)
            raw_payload = self.steg_engine.extract_data(self.steganographic_image_path)
            
            if not raw_payload:
                self.logger.error("No hidden data found in image")
                return False
            
            self.update_progress(30)
            
            # Step 2: Handle decoy data if enabled (40% progress)
            payload_data = raw_payload
            if self.detect_decoy:
                self.update_status("Checking for decoy data...")
                payload_data = self._handle_decoy_data(payload_data)
            
            self.update_progress(40)
            
            # Step 3: Decrypt data if password provided (60% progress)
            if self.password and self.encryption_key:
                self.update_status("Decrypting data...")
                
                try:
                    payload_data = self.encryption_engine.decrypt(payload_data, self.password)
                except Exception as e:
                    self.logger.error(f"Decryption failed: {e}")
                    return False
            
            self.update_progress(60)
            
            # Step 4: Decompress data (70% progress)
            self.update_status("Decompressing data...")
            payload_data = self._decompress_payload(payload_data)
            self.update_progress(70)
            
            # Step 5: Parse and extract files (90% progress)
            self.update_status("Parsing and extracting files...")
            self._extract_files_from_payload(payload_data)
            self.update_progress(90)
            
            # Step 6: Finalize and save results (100% progress)
            self.update_status("Finalizing extraction...")
            
            # Calculate total extracted bytes
            self.total_extracted_bytes = sum(file_info['size'] for file_info in self.extracted_files)
            
            # Store results
            self.extraction_results = {
                'success': True,
                'extracted_files': self.extracted_files,
                'output_directory': str(self.output_directory),
                'total_extracted_bytes': self.total_extracted_bytes,
                'files_count': len(self.extracted_files),
                'decoy_detected': self.decoy_detected,
                'decoy_data_size': len(self.decoy_data) if self.decoy_data else 0,
                'operation_id': self.operation_id,
                'completed_at': datetime.now().isoformat()
            }
            
            self.update_progress(100)
            self.logger.info(f"Extract operation completed successfully: {len(self.extracted_files)} files extracted")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Extract operation failed: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def run_extraction(self, progress_callback: Optional[Callable[[float], None]] = None,
                      status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Run the extraction operation with callbacks.
        
        Args:
            progress_callback: Callback for progress updates (0.0 to 1.0)
            status_callback: Callback for status updates
            
        Returns:
            Dictionary containing operation results
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
                return self.extraction_results
            else:
                raise InvisioVaultError(f"Extract operation failed: {self.error_message}")
            
        except Exception as e:
            self.logger.error(f"Extract operation failed: {e}")
            self.error_handler.handle_exception(e)
            raise InvisioVaultError(f"Extract operation failed: {e}")
    
    def _handle_decoy_data(self, payload_data: bytes) -> bytes:
        """Handle decoy data detection and removal.
        
        Args:
            payload_data: Raw payload data
            
        Returns:
            Payload data with decoy removed
        """
        try:
            decoy_marker = b'DECOY'
            
            if payload_data.startswith(decoy_marker):
                self.decoy_detected = True
                
                # Extract decoy length (4 bytes after marker)
                if len(payload_data) < 9:  # 5 bytes marker + 4 bytes length
                    raise InvisioVaultError("Invalid decoy data format")
                
                decoy_length = int.from_bytes(payload_data[5:9], 'big')
                
                if len(payload_data) < 9 + decoy_length:
                    raise InvisioVaultError("Incomplete decoy data")
                
                # Extract decoy data
                self.decoy_data = payload_data[9:9+decoy_length]
                
                # Return payload after decoy data
                return payload_data[9+decoy_length:]
            
            return payload_data
            
        except Exception as e:
            self.logger.warning(f"Error handling decoy data: {e}")
            return payload_data
    
    def _decompress_payload(self, payload_data: bytes) -> bytes:
        """Decompress payload data.
        
        Args:
            payload_data: Compressed payload data
            
        Returns:
            Decompressed payload data
        """
        try:
            # If expected compression is specified, use it
            if self.expected_compression and self.expected_compression != CompressionType.NONE:
                return self.file_utils.decompress_data(payload_data, self.expected_compression)
            
            # Try to auto-detect compression
            compression_types = [CompressionType.GZIP]
            
            # Add other compression types if available
            try:
                import bz2
                compression_types.append(CompressionType.BZ2)
            except ImportError:
                pass
            
            try:
                import lzma
                compression_types.append(CompressionType.XZ)
            except ImportError:
                pass
            
            # Try each compression type
            for comp_type in compression_types:
                try:
                    decompressed = self.file_utils.decompress_data(payload_data, comp_type)
                    self.logger.info(f"Successfully decompressed using {comp_type}")
                    return decompressed
                except Exception:
                    continue
            
            # If no decompression worked, assume data is not compressed
            self.logger.info("No compression detected, using raw data")
            return payload_data
            
        except Exception as e:
            self.logger.warning(f"Error during decompression: {e}")
            return payload_data
    
    def _extract_files_from_payload(self, payload_data: bytes):
        """Extract individual files from payload data.
        
        Args:
            payload_data: Decompressed payload data
        """
        try:
            offset = 0
            
            # Read file count (4 bytes)
            if len(payload_data) < 4:
                raise InvisioVaultError("Invalid payload data: too short")
            
            file_count = int.from_bytes(payload_data[offset:offset+4], 'big')
            offset += 4
            
            if file_count <= 0 or file_count > 1000:  # Sanity check
                raise InvisioVaultError(f"Invalid file count: {file_count}")
            
            self.logger.info(f"Extracting {file_count} files from payload")
            
            # Extract each file
            for i in range(file_count):
                if offset >= len(payload_data):
                    raise InvisioVaultError(f"Unexpected end of payload at file {i}")
                
                # Read file name length (4 bytes)
                if offset + 4 > len(payload_data):
                    raise InvisioVaultError(f"Cannot read filename length for file {i}")
                
                name_length = int.from_bytes(payload_data[offset:offset+4], 'big')
                offset += 4
                
                # Read file name
                if offset + name_length > len(payload_data):
                    raise InvisioVaultError(f"Cannot read filename for file {i}")
                
                file_name = payload_data[offset:offset+name_length].decode('utf-8')
                offset += name_length
                
                # Read file data length (8 bytes)
                if offset + 8 > len(payload_data):
                    raise InvisioVaultError(f"Cannot read data length for file {i}: {file_name}")
                
                data_length = int.from_bytes(payload_data[offset:offset+8], 'big')
                offset += 8
                
                # Read file data
                if offset + data_length > len(payload_data):
                    raise InvisioVaultError(f"Cannot read data for file {i}: {file_name}")
                
                file_data = payload_data[offset:offset+data_length]
                offset += data_length
                
                # Save file
                output_path = self._save_extracted_file(file_name, file_data)
                
                # Add to extracted files list
                file_info = {
                    'name': file_name,
                    'path': str(output_path),
                    'size': len(file_data),
                    'type': self.file_manager.get_file_metadata(output_path)['mime_type'],
                    'extracted_at': datetime.now().isoformat()
                }
                self.extracted_files.append(file_info)
                
                self.logger.info(f"Extracted file {i+1}/{file_count}: {file_name} ({len(file_data)} bytes)")
            
        except Exception as e:
            self.logger.error(f"Error extracting files from payload: {e}")
            raise
    
    def _save_extracted_file(self, file_name: str, file_data: bytes) -> Path:
        """Save extracted file data to disk.
        
        Args:
            file_name: Name of the file
            file_data: File data bytes
            
        Returns:
            Path to saved file
        """
        try:
            # Sanitize filename
            safe_filename = self._sanitize_filename(file_name)
            
            # Create unique filename if file already exists
            if self.output_directory is None:
                raise InvisioVaultError("Output directory not configured")
                
            output_path = self.output_directory / safe_filename
            counter = 1
            while output_path.exists():
                name_parts = safe_filename.rsplit('.', 1)
                if len(name_parts) == 2:
                    new_name = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                else:
                    new_name = f"{safe_filename}_{counter}"
                if self.output_directory is None:
                    raise InvisioVaultError("Output directory not configured")
                output_path = self.output_directory / new_name
                counter += 1
            
            # Save file
            self.file_manager.write_file_bytes(output_path, file_data)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving extracted file {file_name}: {e}")
            raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe saving.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace dangerous characters
        import re
        
        # Replace dangerous characters with underscores
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        safe_name = safe_name.strip('. ')
        
        # Ensure filename is not empty
        if not safe_name:
            safe_name = 'extracted_file'
        
        # Limit length
        if len(safe_name) > 200:
            name_parts = safe_name.rsplit('.', 1)
            if len(name_parts) == 2:
                safe_name = name_parts[0][:190] + '.' + name_parts[1]
            else:
                safe_name = safe_name[:200]
        
        return safe_name
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get operation summary.
        
        Returns:
            Dictionary with operation summary
        """
        return {
            'operation_type': 'extract',
            'operation_id': self.operation_id,
            'status': self.status.value,
            'steganographic_image': str(self.steganographic_image_path) if self.steganographic_image_path else None,
            'output_directory': str(self.output_directory) if self.output_directory else None,
            'extracted_files_count': len(self.extracted_files),
            'total_extracted_bytes': self.total_extracted_bytes if hasattr(self, 'total_extracted_bytes') else 0,
            'two_factor_enabled': self.two_factor_enabled,
            'decoy_detected': self.decoy_detected if hasattr(self, 'decoy_detected') else False,
            'progress': self.progress,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
    
    def save_decoy_data(self, output_path: Path) -> bool:
        """Save extracted decoy data to file.
        
        Args:
            output_path: Path to save decoy data
            
        Returns:
            True if saved successfully
        """
        try:
            if not self.decoy_data:
                return False
            
            self.file_manager.write_file_bytes(output_path, self.decoy_data)
            self.logger.info(f"Decoy data saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving decoy data: {e}")
            return False
