"""
File Operations & Validation Manager
Provides secure file operations, validation, and metadata management.
"""

import os
import shutil
import mimetypes
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
import stat

from utils.logger import Logger
from utils.error_handler import ErrorHandler, FileAccessError
from core.crypto_utils import CryptoUtils


class FileManager:
    """Secure file operations and validation manager."""
    
    # Supported image formats for steganography
    SUPPORTED_IMAGE_FORMATS = {
        '.png': 'image/png',
        '.bmp': 'image/bmp', 
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff'
    }
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        'image': 500 * 1024 * 1024,  # 500MB
        'data': 100 * 1024 * 1024,   # 100MB
        'keyfile': 10 * 1024 * 1024  # 10MB
    }
    
    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.crypto_utils = CryptoUtils()
    
    def validate_file_path(self, file_path: Union[str, Path], must_exist: bool = True) -> Path:
        """Validate file path and return Path object.
        
        Args:
            file_path: Path to validate
            must_exist: Whether file must exist
            
        Returns:
            Validated Path object
            
        Raises:
            FileAccessError: If path is invalid or file doesn't exist when required
        """
        try:
            path = Path(file_path).resolve()
            
            if must_exist and not path.exists():
                raise FileAccessError(f"File does not exist: {path}", file_path=str(path))
            
            if must_exist and not path.is_file():
                raise FileAccessError(f"Path is not a file: {path}", file_path=str(path))
            
            return path
            
        except Exception as e:
            self.logger.error(f"File path validation failed: {e}")
            raise FileAccessError(f"Invalid file path: {file_path}", file_path=str(file_path))
    
    def validate_directory_path(self, dir_path: Union[str, Path], must_exist: bool = True, create: bool = False) -> Path:
        """Validate directory path.
        
        Args:
            dir_path: Directory path to validate
            must_exist: Whether directory must exist
            create: Whether to create directory if it doesn't exist
            
        Returns:
            Validated Path object
            
        Raises:
            FileAccessError: If path is invalid
        """
        try:
            path = Path(dir_path).resolve()
            
            if not path.exists():
                if create:
                    path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created directory: {path}")
                elif must_exist:
                    raise FileAccessError(f"Directory does not exist: {path}", file_path=str(path))
            
            if path.exists() and not path.is_dir():
                raise FileAccessError(f"Path is not a directory: {path}", file_path=str(path))
            
            return path
            
        except Exception as e:
            self.logger.error(f"Directory path validation failed: {e}")
            raise FileAccessError(f"Invalid directory path: {dir_path}", file_path=str(dir_path))
    
    def validate_image_file(self, image_path: Union[str, Path]) -> Tuple[Path, Dict[str, Any]]:
        """Validate image file for steganography.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (validated_path, metadata)
            
        Raises:
            FileAccessError: If image is invalid
        """
        try:
            path = self.validate_file_path(image_path)
            
            # Check file extension
            if path.suffix.lower() not in self.SUPPORTED_IMAGE_FORMATS:
                raise FileAccessError(
                    f"Unsupported image format: {path.suffix}. "
                    f"Supported formats: {', '.join(self.SUPPORTED_IMAGE_FORMATS.keys())}",
                    file_path=str(path)
                )
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.MAX_FILE_SIZES['image']:
                raise FileAccessError(
                    f"Image file too large: {file_size / 1024 / 1024:.1f}MB. "
                    f"Maximum allowed: {self.MAX_FILE_SIZES['image'] / 1024 / 1024:.1f}MB",
                    file_path=str(path)
                )
            
            # Get file metadata
            metadata = self.get_file_metadata(path)
            
            # Verify MIME type
            expected_mime = self.SUPPORTED_IMAGE_FORMATS[path.suffix.lower()]
            if metadata.get('mime_type') and not metadata['mime_type'].startswith('image/'):
                self.logger.warning(f"MIME type mismatch: expected image, got {metadata['mime_type']}")
            
            self.logger.info(f"Validated image file: {path} ({file_size} bytes)")
            return path, metadata
            
        except FileAccessError:
            raise
        except Exception as e:
            self.logger.error(f"Image validation failed: {e}")
            raise FileAccessError(f"Image validation failed: {e}", file_path=str(image_path))
    
    def validate_data_file(self, data_path: Union[str, Path]) -> Tuple[Path, Dict[str, Any]]:
        """Validate data file for hiding.
        
        Args:
            data_path: Path to data file
            
        Returns:
            Tuple of (validated_path, metadata)
            
        Raises:
            FileAccessError: If data file is invalid
        """
        try:
            path = self.validate_file_path(data_path)
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.MAX_FILE_SIZES['data']:
                raise FileAccessError(
                    f"Data file too large: {file_size / 1024 / 1024:.1f}MB. "
                    f"Maximum allowed: {self.MAX_FILE_SIZES['data'] / 1024 / 1024:.1f}MB",
                    file_path=str(path)
                )
            
            if file_size == 0:
                raise FileAccessError("Data file is empty", file_path=str(path))
            
            # Get file metadata
            metadata = self.get_file_metadata(path)
            
            self.logger.info(f"Validated data file: {path} ({file_size} bytes)")
            return path, metadata
            
        except FileAccessError:
            raise
        except Exception as e:
            self.logger.error(f"Data file validation failed: {e}")
            raise FileAccessError(f"Data file validation failed: {e}", file_path=str(data_path))
    
    def get_file_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive file metadata.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary containing file metadata
        """
        try:
            path = Path(file_path)
            stat_info = path.stat()
            
            # Get MIME type
            mime_type, encoding = mimetypes.guess_type(str(path))
            
            metadata = {
                'name': path.name,
                'stem': path.stem,
                'suffix': path.suffix,
                'size': stat_info.st_size,
                'size_mb': stat_info.st_size / 1024 / 1024,
                'created': datetime.fromtimestamp(stat_info.st_ctime),
                'modified': datetime.fromtimestamp(stat_info.st_mtime),
                'accessed': datetime.fromtimestamp(stat_info.st_atime),
                'mime_type': mime_type,
                'encoding': encoding,
                'permissions': stat.filemode(stat_info.st_mode),
                'is_readable': os.access(path, os.R_OK),
                'is_writable': os.access(path, os.W_OK),
                'hash_sha256': None  # Will be calculated if requested
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error getting file metadata: {e}")
            return {'error': str(e)}
    
    def calculate_file_hash(self, file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
        """Calculate file hash.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm to use
            
        Returns:
            Hex-encoded hash digest
        """
        try:
            path = self.validate_file_path(file_path)
            return self.crypto_utils.hash_file(path, algorithm)
            
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def create_backup(self, file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
        """Create backup of file.
        
        Args:
            file_path: Path to file to backup
            backup_dir: Directory to store backup (default: same directory)
            
        Returns:
            Path to backup file
        """
        try:
            source_path = self.validate_file_path(file_path)
            
            if backup_dir:
                backup_dir_path = self.validate_directory_path(backup_dir, create=True)
            else:
                backup_dir_path = source_path.parent
            
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{source_path.stem}_backup_{timestamp}{source_path.suffix}"
            backup_path = backup_dir_path / backup_name
            
            # Copy file
            shutil.copy2(source_path, backup_path)
            
            self.logger.info(f"Created backup: {source_path} -> {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def secure_delete(self, file_path: Union[str, Path], passes: int = 3) -> bool:
        """Securely delete file by overwriting with random data.
        
        Args:
            file_path: Path to file to delete
            passes: Number of overwrite passes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = self.validate_file_path(file_path)
            file_size = path.stat().st_size
            
            self.logger.info(f"Securely deleting file: {path} ({passes} passes)")
            
            # Overwrite file with random data
            with open(path, 'r+b') as f:
                for i in range(passes):
                    f.seek(0)
                    random_data = self.crypto_utils.generate_secure_random_bytes(file_size)
                    f.write(random_data)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
            
            # Delete the file
            path.unlink()
            
            self.logger.info(f"Securely deleted file: {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error securely deleting file: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def create_temporary_file(self, suffix: str = '', prefix: str = 'invault_', directory: Optional[str] = None) -> Path:
        """Create temporary file.
        
        Args:
            suffix: File suffix/extension
            prefix: File prefix
            directory: Directory to create temp file in
            
        Returns:
            Path to temporary file
        """
        try:
            # Create temporary file
            fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=directory)
            os.close(fd)  # Close file descriptor
            
            temp_file = Path(temp_path)
            self.logger.debug(f"Created temporary file: {temp_file}")
            return temp_file
            
        except Exception as e:
            self.logger.error(f"Error creating temporary file: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path], preserve_metadata: bool = True) -> Path:
        """Copy file to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
            preserve_metadata: Whether to preserve metadata
            
        Returns:
            Path to copied file
        """
        try:
            source_path = self.validate_file_path(source)
            dest_path = Path(destination)
            
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            if preserve_metadata:
                shutil.copy2(source_path, dest_path)
            else:
                shutil.copy(source_path, dest_path)
            
            self.logger.info(f"Copied file: {source_path} -> {dest_path}")
            return dest_path
            
        except Exception as e:
            self.logger.error(f"Error copying file: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> Path:
        """Move file to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            Path to moved file
        """
        try:
            source_path = self.validate_file_path(source)
            dest_path = Path(destination)
            
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            moved_path = shutil.move(str(source_path), str(dest_path))
            
            self.logger.info(f"Moved file: {source_path} -> {moved_path}")
            return Path(moved_path)
            
        except Exception as e:
            self.logger.error(f"Error moving file: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def read_file_bytes(self, file_path: Union[str, Path], max_size: Optional[int] = None) -> bytes:
        """Read file contents as bytes.
        
        Args:
            file_path: Path to file
            max_size: Maximum size to read (None for no limit)
            
        Returns:
            File contents as bytes
        """
        try:
            path = self.validate_file_path(file_path)
            
            file_size = path.stat().st_size
            if max_size and file_size > max_size:
                raise FileAccessError(
                    f"File too large: {file_size} bytes (max: {max_size})",
                    file_path=str(path)
                )
            
            with open(path, 'rb') as f:
                data = f.read()
            
            self.logger.debug(f"Read file: {path} ({len(data)} bytes)")
            return data
            
        except Exception as e:
            self.logger.error(f"Error reading file: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def write_file_bytes(self, file_path: Union[str, Path], data: bytes, create_backup: bool = False) -> Path:
        """Write bytes to file.
        
        Args:
            file_path: Path to file
            data: Data to write
            create_backup: Whether to create backup if file exists
            
        Returns:
            Path to written file
        """
        try:
            path = Path(file_path)
            
            # Create backup if requested and file exists
            if create_backup and path.exists():
                self.create_backup(path)
            
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            self.logger.info(f"Wrote file: {path} ({len(data)} bytes)")
            return path
            
        except Exception as e:
            self.logger.error(f"Error writing file: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def get_available_space(self, path: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
        """Get available disk space.
        
        Args:
            path: Path to check (file or directory)
            
        Returns:
            Dictionary with space information
        """
        try:
            path = Path(path)
            if path.is_file():
                path = path.parent
            
            # Get disk usage statistics
            usage = shutil.disk_usage(path)
            
            return {
                'total': usage.total,
                'used': usage.used,
                'free': usage.free,
                'total_mb': round(usage.total / 1024 / 1024, 2),
                'used_mb': round(usage.used / 1024 / 1024, 2),
                'free_mb': round(usage.free / 1024 / 1024, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting disk space: {e}")
            return {'error': str(e)}
    
    def list_directory(self, dir_path: Union[str, Path], pattern: str = '*', recursive: bool = False) -> List[Path]:
        """List files in directory.
        
        Args:
            dir_path: Directory path
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        try:
            path = self.validate_directory_path(dir_path)
            
            if recursive:
                files = list(path.rglob(pattern))
            else:
                files = list(path.glob(pattern))
            
            # Filter only files (not directories)
            files = [f for f in files if f.is_file()]
            
            self.logger.debug(f"Listed {len(files)} files in {path}")
            return files
            
        except Exception as e:
            self.logger.error(f"Error listing directory: {e}")
            self.error_handler.handle_exception(e)
            return []
    
    def cleanup_temporary_files(self, temp_dir: Optional[Union[str, Path]] = None, max_age_hours: int = 24):
        """Clean up old temporary files.
        
        Args:
            temp_dir: Directory to clean (default: system temp)
            max_age_hours: Maximum age of files to keep
        """
        try:
            if temp_dir:
                temp_path = Path(temp_dir)
            else:
                temp_path = Path(tempfile.gettempdir())
            
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_hours * 3600
            
            # Find old InvisioVault temporary files
            for temp_file in temp_path.glob('invault_*'):
                try:
                    if temp_file.is_file():
                        file_age = current_time - temp_file.stat().st_mtime
                        if file_age > max_age_seconds:
                            temp_file.unlink()
                            self.logger.debug(f"Cleaned up old temp file: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean temp file {temp_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning temporary files: {e}")
