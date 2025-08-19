"""
Video Steganography Engine
Implements advanced video steganography techniques for hiding data in video files.
"""

import os
import sys
import tempfile
import shutil
import struct
import hashlib
import secrets
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import warnings

import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import cv2
    import ffmpeg
except ImportError as e:
    print(f"Warning: Video dependencies not fully installed: {e}")
    print("Please install: pip install opencv-python ffmpeg-python")

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from core.encryption_engine import EncryptionEngine, SecurityLevel
from core.multimedia_analyzer import MultimediaAnalyzer


class VideoSteganographyEngine:
    """Advanced video steganography implementation using LSB in selected frames."""
    
    MAGIC_HEADER = b'INVV_VID'  # InvisioVault Video magic bytes
    VERSION = b'\x01\x00'  # Version 1.0
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MAXIMUM):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.security_level = security_level
        self.encryption_engine = EncryptionEngine(security_level)
        self.analyzer = MultimediaAnalyzer()
        
        # Video processing parameters
        self.frame_skip = 10  # Use every 10th frame to avoid detection
        self.max_temp_frames = 1000  # Limit frames in memory
        
        self.logger.info(f"Video steganography engine initialized with {security_level.value} security")
    
    def hide_data_in_video(self, video_path: Path, data: bytes, output_path: Path, 
                          password: str, compression_quality: int = 95) -> bool:
        """
        Hide data in video file using LSB steganography in selected frames.
        
        Args:
            video_path: Path to carrier video file
            data: Data to hide (will be encrypted)
            output_path: Output video path
            password: Password for encryption and randomization
            compression_quality: Video compression quality (0-100)
            
        Returns:
            Success status
        """
        temp_dir = None
        try:
            self.logger.info(f"Starting video steganography: {video_path.name}")
            
            # Validate input
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if not self.analyzer.is_video_file(video_path):
                raise ValueError(f"Unsupported video format: {video_path.suffix}")
            
            # Analyze video capacity
            analysis = self.analyzer.analyze_video_file(video_path)
            if 'error' in analysis:
                raise Exception(f"Video analysis failed: {analysis['error']}")
            
            capacity = analysis['capacity_bytes']
            if len(data) > capacity:
                raise ValueError(f"Data too large: {len(data)} bytes exceeds capacity {capacity} bytes")
            
            # Encrypt and prepare data
            encrypted_data = self._prepare_data_for_hiding(data, password)
            
            # Create temporary directory for frame processing
            temp_dir = Path(tempfile.mkdtemp(prefix="invv_video_"))
            
            # Extract frames for processing
            frame_files = self._extract_frames(video_path, temp_dir)
            if not frame_files:
                raise Exception("Failed to extract video frames")
            
            # Select frames for steganography (every nth frame)
            selected_frames = frame_files[::self.frame_skip]
            
            # Verify we have enough frames
            total_capacity_needed = len(encrypted_data)
            estimated_capacity = self._estimate_frame_capacity(selected_frames[:5])  # Check first 5 frames
            total_estimated = estimated_capacity * len(selected_frames)
            
            if total_capacity_needed > total_estimated:
                raise ValueError(f"Insufficient capacity in selected frames: {total_estimated} < {total_capacity_needed}")
            
            # Hide data in selected frames
            self._hide_data_in_frames(selected_frames, encrypted_data, password)
            
            # Reassemble video with modified frames
            success = self._reassemble_video(
                video_path, temp_dir, output_path, compression_quality
            )
            
            if success:
                self.logger.info(f"Video steganography completed successfully: {output_path.name}")
                return True
            else:
                raise Exception("Failed to reassemble video")
            
        except Exception as e:
            self.logger.error(f"Video steganography failed: {e}")
            return False
        
        finally:
            # Clean up temporary files
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp directory: {e}")
    
    def extract_data_from_video(self, video_path: Path, password: str) -> Optional[bytes]:
        """
        Extract hidden data from video file.
        
        Args:
            video_path: Path to video file containing hidden data
            password: Password for decryption
            
        Returns:
            Extracted data or None if failed
        """
        temp_dir = None
        try:
            self.logger.info(f"Starting video data extraction: {video_path.name}")
            
            # Validate input
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if not self.analyzer.is_video_file(video_path):
                raise ValueError(f"Unsupported video format: {video_path.suffix}")
            
            # Create temporary directory
            temp_dir = Path(tempfile.mkdtemp(prefix="invv_extract_"))
            
            # Extract frames
            frame_files = self._extract_frames(video_path, temp_dir)
            if not frame_files:
                raise Exception("Failed to extract video frames")
            
            # Select frames that were used for steganography
            selected_frames = frame_files[::self.frame_skip]
            
            # Extract data from frames
            encrypted_data = self._extract_data_from_frames(selected_frames, password)
            
            if not encrypted_data:
                raise Exception("No hidden data found in video")
            
            # Decrypt and verify data
            original_data = self._extract_and_decrypt_data(encrypted_data, password)
            
            self.logger.info(f"Video data extraction completed: {len(original_data)} bytes")
            return original_data
            
        except Exception as e:
            self.logger.error(f"Video data extraction failed: {e}")
            return None
        
        finally:
            # Clean up temporary files
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp directory: {e}")
    
    def _prepare_data_for_hiding(self, data: bytes, password: str) -> bytes:
        """Encrypt and prepare data with header for hiding."""
        try:
            # Encrypt the data
            encrypted_data = self.encryption_engine.encrypt_with_metadata(data, password)
            
            # Create header: magic + version + size + checksum
            data_size = len(encrypted_data)
            checksum = hashlib.md5(encrypted_data).digest()
            
            header = (
                self.MAGIC_HEADER +
                self.VERSION +
                struct.pack('<Q', data_size) +  # 8 bytes for size
                checksum  # 16 bytes for MD5
            )
            
            # Combine header + encrypted data
            prepared_data = header + encrypted_data
            
            self.logger.debug(f"Data prepared for hiding: {len(prepared_data)} bytes total")
            return prepared_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            raise
    
    def _extract_frames(self, video_path: Path, output_dir: Path) -> List[Path]:
        """Extract video frames to temporary directory."""
        try:
            # Create frames output directory
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            # Use OpenCV to extract frames
            cap = cv2.VideoCapture(str(video_path))
            frame_files = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame as PNG (lossless)
                frame_path = frames_dir / f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                frame_files.append(frame_path)
                frame_count += 1
                
                # Limit frames to prevent memory issues
                if frame_count >= self.max_temp_frames:
                    self.logger.warning(f"Limiting extraction to {self.max_temp_frames} frames")
                    break
            
            cap.release()
            
            self.logger.info(f"Extracted {len(frame_files)} frames from video")
            return frame_files
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            return []
    
    def _estimate_frame_capacity(self, frame_files: List[Path]) -> int:
        """Estimate average capacity per frame."""
        if not frame_files:
            return 0
        
        total_capacity = 0
        valid_frames = 0
        
        for frame_file in frame_files[:5]:  # Check first 5 frames
            try:
                img = cv2.imread(str(frame_file))
                if img is not None:
                    height, width, channels = img.shape
                    # 1 LSB per channel per pixel, minus header space
                    capacity = (height * width * channels) // 8 - 64  # 64 bytes header buffer
                    total_capacity += capacity
                    valid_frames += 1
            except Exception:
                continue
        
        return total_capacity // valid_frames if valid_frames > 0 else 0
    
    def _hide_data_in_frames(self, frame_files: List[Path], data: bytes, password: str):
        """Hide data across selected frames using LSB steganography."""
        try:
            # Generate deterministic random sequence from password
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            
            # Convert data to bit array
            data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            total_bits = len(data_bits)
            
            bits_hidden = 0
            
            for frame_idx, frame_file in enumerate(frame_files):
                if bits_hidden >= total_bits:
                    break
                
                # Load frame
                img = cv2.imread(str(frame_file))
                if img is None:
                    continue
                
                height, width, channels = img.shape
                
                # Calculate how many bits to hide in this frame
                remaining_bits = total_bits - bits_hidden
                max_bits_per_frame = (height * width * channels) // 4  # Use 25% of capacity
                bits_to_hide = min(remaining_bits, max_bits_per_frame)
                
                if bits_to_hide <= 0:
                    continue
                
                # Generate random positions for this frame
                total_positions = height * width * channels
                positions = rng.choice(total_positions, size=bits_to_hide, replace=False)
                
                # Hide bits in LSBs
                flat_img = img.flatten()
                for i, pos in enumerate(positions):
                    bit_to_hide = data_bits[bits_hidden + i]
                    flat_img[pos] = (flat_img[pos] & 0xFE) | bit_to_hide
                
                # Reshape and save modified frame
                modified_img = flat_img.reshape(height, width, channels)
                cv2.imwrite(str(frame_file), modified_img)
                
                bits_hidden += bits_to_hide
                
                if frame_idx % 50 == 0:  # Progress logging
                    progress = (bits_hidden / total_bits) * 100
                    self.logger.debug(f"Hiding progress: {progress:.1f}%")
            
            if bits_hidden < total_bits:
                raise Exception(f"Could not hide all data: {bits_hidden}/{total_bits} bits")
            
            self.logger.info("Data successfully hidden in video frames")
            
        except Exception as e:
            self.logger.error(f"Failed to hide data in frames: {e}")
            raise
    
    def _extract_data_from_frames(self, frame_files: List[Path], password: str) -> Optional[bytes]:
        """Extract hidden data from frames."""
        try:
            # Generate same random sequence used for hiding
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            
            # Try to read header from first frame to determine data size
            if not frame_files:
                return None
            
            # Read header from first frame
            header_data = self._extract_header_from_frame(frame_files[0], rng)
            if not header_data:
                return None
            
            # Parse header
            magic = header_data[:8]
            if magic != self.MAGIC_HEADER:
                return None
            
            version = header_data[8:10]
            data_size = struct.unpack('<Q', header_data[10:18])[0]
            checksum = header_data[18:34]
            
            # Calculate total bits needed
            header_bits = len(header_data) * 8
            data_bits_needed = data_size * 8
            total_bits_needed = header_bits + data_bits_needed
            
            # Extract all bits
            extracted_bits = []
            bits_extracted = 0
            
            for frame_idx, frame_file in enumerate(frame_files):
                if bits_extracted >= total_bits_needed:
                    break
                
                img = cv2.imread(str(frame_file))
                if img is None:
                    continue
                
                height, width, channels = img.shape
                
                # Calculate bits to extract from this frame
                remaining_bits = total_bits_needed - bits_extracted
                max_bits_per_frame = (height * width * channels) // 4
                bits_to_extract = min(remaining_bits, max_bits_per_frame)
                
                if bits_to_extract <= 0:
                    continue
                
                # Generate same random positions
                total_positions = height * width * channels
                positions = rng.choice(total_positions, size=bits_to_extract, replace=False)
                
                # Extract LSBs
                flat_img = img.flatten()
                frame_bits = []
                for pos in positions:
                    frame_bits.append(flat_img[pos] & 1)
                
                extracted_bits.extend(frame_bits)
                bits_extracted += len(frame_bits)
            
            if bits_extracted < total_bits_needed:
                raise Exception(f"Could not extract enough bits: {bits_extracted}/{total_bits_needed}")
            
            # Convert bits back to bytes
            bit_array = np.array(extracted_bits[:total_bits_needed], dtype=np.uint8)
            byte_array = np.packbits(bit_array)
            
            # Skip header and get encrypted data
            header_bytes = len(header_data)
            encrypted_data = bytes(byte_array[header_bytes:header_bytes + data_size])
            
            # Verify checksum
            actual_checksum = hashlib.md5(encrypted_data).digest()
            if actual_checksum != checksum:
                raise Exception("Data corruption detected - checksum mismatch")
            
            return encrypted_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract data from frames: {e}")
            return None
    
    def _extract_header_from_frame(self, frame_file: Path, rng: np.random.RandomState) -> Optional[bytes]:
        """Extract header data from first frame."""
        try:
            img = cv2.imread(str(frame_file))
            if img is None:
                return None
            
            height, width, channels = img.shape
            
            # Header size in bits (magic + version + size + checksum)
            header_size_bits = (8 + 2 + 8 + 16) * 8  # 34 bytes * 8 bits
            
            # Generate positions for header
            total_positions = height * width * channels
            positions = rng.choice(total_positions, size=header_size_bits, replace=False)
            
            # Extract header bits
            flat_img = img.flatten()
            header_bits = []
            for pos in positions:
                header_bits.append(flat_img[pos] & 1)
            
            # Convert to bytes
            bit_array = np.array(header_bits, dtype=np.uint8)
            header_bytes = np.packbits(bit_array)
            
            return bytes(header_bytes)
            
        except Exception:
            return None
    
    def _reassemble_video(self, original_path: Path, frames_dir: Path, 
                         output_path: Path, quality: int) -> bool:
        """Reassemble video from modified frames."""
        try:
            frames_path = frames_dir / "frames"
            
            # Get original video properties
            probe = ffmpeg.probe(str(original_path))
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            fps = eval(video_stream['r_frame_rate'])  # Convert fraction to float
            
            # Use ffmpeg to create video from frames
            input_pattern = str(frames_path / "frame_%06d.png")
            
            # Build ffmpeg command
            stream = ffmpeg.input(input_pattern, framerate=fps)
            
            # Copy audio from original video if present
            audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
            if audio_streams:
                audio_input = ffmpeg.input(str(original_path))
                stream = ffmpeg.output(
                    stream, audio_input['a'],
                    str(output_path),
                    vcodec='libx264',
                    crf=quality,
                    acodec='copy'
                )
            else:
                stream = ffmpeg.output(
                    stream,
                    str(output_path),
                    vcodec='libx264',
                    crf=quality
                )
            
            # Run ffmpeg
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            return output_path.exists()
            
        except Exception as e:
            self.logger.error(f"Video reassembly failed: {e}")
            return False
    
    def _extract_and_decrypt_data(self, encrypted_data: bytes, password: str) -> bytes:
        """Decrypt extracted data."""
        try:
            # Decrypt the data
            original_data = self.encryption_engine.decrypt_with_metadata(encrypted_data, password)
            
            self.logger.debug(f"Data decrypted successfully: {len(original_data)} bytes")
            return original_data
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def calculate_capacity(self, video_path: Path) -> int:
        """Calculate video steganography capacity in bytes."""
        try:
            analysis = self.analyzer.analyze_video_file(video_path)
            return analysis.get('capacity_bytes', 0)
        except Exception:
            return 0
    
    def validate_video_format(self, video_path: Path) -> bool:
        """Validate if video format is supported."""
        return self.analyzer.is_video_file(video_path)
