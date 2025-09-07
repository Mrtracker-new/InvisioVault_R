"""
File Utilities
Provides file compression, archive handling, and advanced file operations.
"""

import os
import gzip
import zipfile
import tarfile
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime
import mimetypes

from utils.logger import Logger
from utils.error_handler import ErrorHandler, FileAccessError
from core.security.crypto_utils import CryptoUtils

try:
    import lzma
    HAS_LZMA = True
except ImportError:
    HAS_LZMA = False

try:
    import bz2
    HAS_BZ2 = True
except ImportError:
    HAS_BZ2 = False


class CompressionType:
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"
    TAR_XZ = "tar.xz"
    BZ2 = "bz2"
    XZ = "xz"


class FileUtils:
    """Advanced file utilities with compression support."""
    
    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.crypto_utils = CryptoUtils()
    
    def compress_data(self, data: bytes, compression_type: str = CompressionType.GZIP, 
                     compression_level: int = 6) -> bytes:
        """Compress data using specified compression type.
        
        Args:
            data: Data to compress
            compression_type: Type of compression
            compression_level: Compression level (1-9)
            
        Returns:
            Compressed data
        """
        try:
            if compression_type == CompressionType.NONE:
                return data
            
            elif compression_type == CompressionType.GZIP:
                return gzip.compress(data, compresslevel=compression_level)
            
            elif compression_type == CompressionType.BZ2 and HAS_BZ2:
                return bz2.compress(data, compresslevel=compression_level)
            
            elif compression_type == CompressionType.XZ and HAS_LZMA:
                return lzma.compress(data, preset=compression_level)
            
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")
                
        except Exception as e:
            self.logger.error(f"Error compressing data: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def decompress_data(self, compressed_data: bytes, compression_type: str) -> bytes:
        """Decompress data using specified compression type.
        
        Args:
            compressed_data: Compressed data
            compression_type: Type of compression used
            
        Returns:
            Decompressed data
        """
        try:
            if compression_type == CompressionType.NONE:
                return compressed_data
            
            elif compression_type == CompressionType.GZIP:
                return gzip.decompress(compressed_data)
            
            elif compression_type == CompressionType.BZ2 and HAS_BZ2:
                return bz2.decompress(compressed_data)
            
            elif compression_type == CompressionType.XZ and HAS_LZMA:
                return lzma.decompress(compressed_data)
            
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")
                
        except Exception as e:
            self.logger.error(f"Error decompressing data: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def compress_file(self, file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None,
                     compression_type: str = CompressionType.GZIP, compression_level: int = 6) -> Path:
        """Compress a file.
        
        Args:
            file_path: Path to file to compress
            output_path: Output path (auto-generated if None)
            compression_type: Type of compression
            compression_level: Compression level
            
        Returns:
            Path to compressed file
        """
        try:
            input_path = Path(file_path)
            
            if not input_path.exists():
                raise FileAccessError(f"Input file not found: {input_path}", file_path=str(input_path))
            
            # Generate output path if not provided
            if output_path is None:
                if compression_type == CompressionType.GZIP:
                    output_path = input_path.with_suffix(input_path.suffix + '.gz')
                elif compression_type == CompressionType.BZ2:
                    output_path = input_path.with_suffix(input_path.suffix + '.bz2')
                elif compression_type == CompressionType.XZ:
                    output_path = input_path.with_suffix(input_path.suffix + '.xz')
                else:
                    output_path = input_path.with_suffix(input_path.suffix + '.compressed')
            else:
                output_path = Path(output_path)
            
            # Read and compress data
            with open(input_path, 'rb') as f:
                data = f.read()
            
            compressed_data = self.compress_data(data, compression_type, compression_level)
            
            # Write compressed data
            with open(output_path, 'wb') as f:
                f.write(compressed_data)
            
            original_size = len(data)
            compressed_size = len(compressed_data)
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            self.logger.info(f"Compressed {input_path} -> {output_path} "
                           f"({original_size} -> {compressed_size} bytes, {ratio:.1f}% reduction)")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error compressing file: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def decompress_file(self, compressed_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None,
                       compression_type: Optional[str] = None) -> Path:
        """Decompress a file.
        
        Args:
            compressed_path: Path to compressed file
            output_path: Output path (auto-generated if None)
            compression_type: Type of compression (auto-detected if None)
            
        Returns:
            Path to decompressed file
        """
        try:
            input_path = Path(compressed_path)
            
            if not input_path.exists():
                raise FileAccessError(f"Compressed file not found: {input_path}", file_path=str(input_path))
            
            # Auto-detect compression type if not provided
            if compression_type is None:
                compression_type = self._detect_compression_type(input_path)
            
            # Generate output path if not provided
            if output_path is None:
                output_path = self._generate_decompressed_filename(input_path, compression_type)
            else:
                output_path = Path(output_path)
            
            # Read and decompress data
            with open(input_path, 'rb') as f:
                compressed_data = f.read()
            
            data = self.decompress_data(compressed_data, compression_type)
            
            # Write decompressed data
            with open(output_path, 'wb') as f:
                f.write(data)
            
            self.logger.info(f"Decompressed {input_path} -> {output_path} "
                           f"({len(compressed_data)} -> {len(data)} bytes)")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error decompressing file: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def create_archive(self, files: List[Union[str, Path]], archive_path: Union[str, Path],
                      compression_type: str = CompressionType.ZIP, compression_level: int = 6) -> Path:
        """Create an archive from multiple files.
        
        Args:
            files: List of file paths to include
            archive_path: Path for the output archive
            compression_type: Type of archive/compression
            compression_level: Compression level
            
        Returns:
            Path to created archive
        """
        try:
            archive_path = Path(archive_path)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            if compression_type == CompressionType.ZIP:
                return self._create_zip_archive(files, archive_path, compression_level)
            
            elif compression_type in [CompressionType.TAR, CompressionType.TAR_GZ, 
                                     CompressionType.TAR_BZ2, CompressionType.TAR_XZ]:
                return self._create_tar_archive(files, archive_path, compression_type)
            
            else:
                raise ValueError(f"Unsupported archive type: {compression_type}")
                
        except Exception as e:
            self.logger.error(f"Error creating archive: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def extract_archive(self, archive_path: Union[str, Path], extract_to: Union[str, Path],
                       password: Optional[str] = None) -> List[Path]:
        """Extract files from an archive.
        
        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to
            password: Password for encrypted archives
            
        Returns:
            List of extracted file paths
        """
        try:
            archive_path = Path(archive_path)
            extract_dir = Path(extract_to)
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            if not archive_path.exists():
                raise FileAccessError(f"Archive not found: {archive_path}", file_path=str(archive_path))
            
            # Detect archive type
            archive_type = self._detect_archive_type(archive_path)
            
            if archive_type == "zip":
                return self._extract_zip_archive(archive_path, extract_dir, password)
            
            elif archive_type.startswith("tar"):
                return self._extract_tar_archive(archive_path, extract_dir)
            
            else:
                raise ValueError(f"Unsupported archive type: {archive_type}")
                
        except Exception as e:
            self.logger.error(f"Error extracting archive: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def get_compression_ratio(self, original_data: bytes, compressed_data: bytes) -> float:
        """Calculate compression ratio.
        
        Args:
            original_data: Original data
            compressed_data: Compressed data
            
        Returns:
            Compression ratio as percentage
        """
        if len(original_data) == 0:
            return 0.0
        
        return (1 - len(compressed_data) / len(original_data)) * 100
    
    def get_optimal_compression(self, data: bytes) -> Tuple[str, int]:
        """Find optimal compression type and level for data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Tuple of (compression_type, compression_level)
        """
        try:
            results = []
            
            # Test different compression types
            test_types = [CompressionType.GZIP]
            if HAS_BZ2:
                test_types.append(CompressionType.BZ2)
            if HAS_LZMA:
                test_types.append(CompressionType.XZ)
            
            for comp_type in test_types:
                for level in [1, 6, 9]:  # Test low, medium, high compression
                    try:
                        compressed = self.compress_data(data, comp_type, level)
                        ratio = self.get_compression_ratio(data, compressed)
                        results.append((comp_type, level, ratio, len(compressed)))
                    except Exception as e:
                        self.logger.debug(f"Compression test failed for {comp_type} level {level}: {e}")
            
            if not results:
                return CompressionType.GZIP, 6
            
            # Find best compression (highest ratio, breaking ties with smaller size)
            best = max(results, key=lambda x: (x[2], -x[3]))
            return best[0], best[1]
            
        except Exception as e:
            self.logger.warning(f"Error finding optimal compression: {e}")
            return CompressionType.GZIP, 6
    
    def split_file(self, file_path: Union[str, Path], chunk_size: int, 
                  output_dir: Optional[Union[str, Path]] = None) -> List[Path]:
        """Split a large file into smaller chunks.
        
        Args:
            file_path: Path to file to split
            chunk_size: Size of each chunk in bytes
            output_dir: Directory for chunks (same as input if None)
            
        Returns:
            List of chunk file paths
        """
        try:
            input_path = Path(file_path)
            
            if not input_path.exists():
                raise FileAccessError(f"File not found: {input_path}", file_path=str(input_path))
            
            if output_dir is None:
                output_dir = input_path.parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            chunk_paths = []
            chunk_index = 0
            
            with open(input_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    chunk_filename = f"{input_path.stem}.part{chunk_index:03d}"
                    chunk_path = output_dir / chunk_filename
                    
                    with open(chunk_path, 'wb') as chunk_file:
                        chunk_file.write(chunk)
                    
                    chunk_paths.append(chunk_path)
                    chunk_index += 1
            
            self.logger.info(f"Split {input_path} into {len(chunk_paths)} chunks")
            return chunk_paths
            
        except Exception as e:
            self.logger.error(f"Error splitting file: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def join_files(self, chunk_paths: List[Union[str, Path]], output_path: Union[str, Path]) -> Path:
        """Join file chunks back into a single file.
        
        Args:
            chunk_paths: List of chunk file paths
            output_path: Path for joined file
            
        Returns:
            Path to joined file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as output_file:
                for chunk_path in chunk_paths:
                    chunk_path = Path(chunk_path)
                    if not chunk_path.exists():
                        raise FileAccessError(f"Chunk not found: {chunk_path}", file_path=str(chunk_path))
                    
                    with open(chunk_path, 'rb') as chunk_file:
                        output_file.write(chunk_file.read())
            
            self.logger.info(f"Joined {len(chunk_paths)} chunks into {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error joining files: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def _detect_compression_type(self, file_path: Path) -> str:
        """Detect compression type from file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.gz':
            return CompressionType.GZIP
        elif suffix == '.bz2':
            return CompressionType.BZ2
        elif suffix == '.xz':
            return CompressionType.XZ
        else:
            return CompressionType.GZIP  # Default
    
    def _detect_archive_type(self, file_path: Path) -> str:
        """Detect archive type from file extension."""
        suffix = file_path.suffix.lower()
        suffixes = ''.join(file_path.suffixes).lower()
        
        if suffix == '.zip':
            return "zip"
        elif suffixes.endswith('.tar.gz') or suffixes.endswith('.tgz'):
            return "tar.gz"
        elif suffixes.endswith('.tar.bz2') or suffixes.endswith('.tbz'):
            return "tar.bz2"
        elif suffixes.endswith('.tar.xz'):
            return "tar.xz"
        elif suffix == '.tar':
            return "tar"
        else:
            return "zip"  # Default
    
    def _generate_decompressed_filename(self, compressed_path: Path, compression_type: str) -> Path:
        """Generate filename for decompressed file."""
        if compression_type == CompressionType.GZIP and compressed_path.suffix == '.gz':
            return compressed_path.with_suffix('')
        elif compression_type == CompressionType.BZ2 and compressed_path.suffix == '.bz2':
            return compressed_path.with_suffix('')
        elif compression_type == CompressionType.XZ and compressed_path.suffix == '.xz':
            return compressed_path.with_suffix('')
        else:
            return compressed_path.with_suffix('.decompressed')
    
    def _create_zip_archive(self, files: List[Union[str, Path]], archive_path: Path, 
                           compression_level: int) -> Path:
        """Create ZIP archive."""
        compression = zipfile.ZIP_DEFLATED if compression_level > 0 else zipfile.ZIP_STORED
        
        with zipfile.ZipFile(archive_path, 'w', compression=compression, compresslevel=compression_level) as zf:
            for file_path in files:
                file_path = Path(file_path)
                if file_path.exists():
                    zf.write(file_path, file_path.name)
                else:
                    self.logger.warning(f"File not found, skipping: {file_path}")
        
        self.logger.info(f"Created ZIP archive: {archive_path}")
        return archive_path
    
    def _create_tar_archive(self, files: List[Union[str, Path]], archive_path: Path, 
                           compression_type: str) -> Path:
        """Create TAR archive."""
        from typing import Literal
        
        # Map compression types to proper tarfile modes
        mode: Literal['w', 'w:gz', 'w:bz2', 'w:xz']
        if compression_type == CompressionType.TAR:
            mode = 'w'
        elif compression_type == CompressionType.TAR_GZ:
            mode = 'w:gz'
        elif compression_type == CompressionType.TAR_BZ2:
            mode = 'w:bz2'
        elif compression_type == CompressionType.TAR_XZ:
            mode = 'w:xz'
        else:
            mode = 'w'  # Default to uncompressed
        
        with tarfile.open(archive_path, mode) as tf:
            for file_path in files:
                file_path = Path(file_path)
                if file_path.exists():
                    tf.add(file_path, arcname=file_path.name)
                else:
                    self.logger.warning(f"File not found, skipping: {file_path}")
        
        self.logger.info(f"Created TAR archive: {archive_path}")
        return archive_path
    
    def _extract_zip_archive(self, archive_path: Path, extract_dir: Path, 
                            password: Optional[str] = None) -> List[Path]:
        """Extract ZIP archive."""
        extracted_files = []
        
        with zipfile.ZipFile(archive_path, 'r') as zf:
            if password:
                zf.setpassword(password.encode('utf-8'))
            
            for member in zf.namelist():
                # Security check: prevent path traversal
                if os.path.isabs(member) or '..' in member:
                    self.logger.warning(f"Skipping potentially unsafe path: {member}")
                    continue
                
                extract_path = extract_dir / member
                zf.extract(member, extract_dir)
                extracted_files.append(extract_path)
        
        self.logger.info(f"Extracted {len(extracted_files)} files from ZIP archive")
        return extracted_files
    
    def _extract_tar_archive(self, archive_path: Path, extract_dir: Path) -> List[Path]:
        """Extract TAR archive."""
        extracted_files = []
        
        with tarfile.open(archive_path, 'r:*') as tf:
            for member in tf.getmembers():
                # Security check: prevent path traversal
                if os.path.isabs(member.name) or '..' in member.name:
                    self.logger.warning(f"Skipping potentially unsafe path: {member.name}")
                    continue
                
                tf.extract(member, extract_dir)
                extracted_files.append(extract_dir / member.name)
        
        self.logger.info(f"Extracted {len(extracted_files)} files from TAR archive")
        return extracted_files
