"""
Video Steganography Engine
Implements advanced video steganography techniques for hiding data in video files.
"""

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

# Type imports for optional dependencies
try:
    import cv2
    import ffmpeg  # type: ignore # ffmpeg-python package
except ImportError as e:
    print(f"Warning: Video dependencies not fully installed: {e}")
    print("Please install: pip install opencv-python ffmpeg-python")
    cv2 = None  # type: ignore
    ffmpeg = None  # type: ignore

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from core.security.encryption_engine import EncryptionEngine, SecurityLevel
from core.analyzers.multimedia_analyzer import MultimediaAnalyzer


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
        
        # Check if video dependencies are available
        self.dependencies_available = cv2 is not None and ffmpeg is not None
        if not self.dependencies_available:
            self.logger.warning("Video dependencies not available. Install with: pip install opencv-python ffmpeg-python")
        
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
        if not self.dependencies_available:
            self.logger.error("Video dependencies not available. Install with: pip install opencv-python ffmpeg-python")
            return False
            
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
        if not self.dependencies_available:
            self.logger.error("Video dependencies not available. Install with: pip install opencv-python ffmpeg-python")
            return None
            
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
            # Ensure cv2 is available (should be checked by dependencies_available)
            assert cv2 is not None, "OpenCV not available"
            
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
                assert cv2 is not None, "OpenCV not available"
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
            # Generate deterministic random sequence from password - CRITICAL: use consistent seed
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            self.logger.info(f"Video hiding using seed: {seed}")
            
            # Convert data to bit array
            data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            total_bits = len(data_bits)
            self.logger.info(f"Hiding {len(data)} bytes ({total_bits} bits) across {len(frame_files)} frames")
            
            # Log first few bytes for debugging
            first_bytes = data[:20] if len(data) >= 20 else data
            self.logger.debug(f"First 20 bytes of data to hide: {first_bytes.hex()}")
            
            bits_hidden = 0
            modified_frames = []
            
            # Process each frame with consistent random state
            for frame_idx, frame_file in enumerate(frame_files):
                if bits_hidden >= total_bits:
                    break
                
                # Load frame
                assert cv2 is not None, "OpenCV not available"
                img = cv2.imread(str(frame_file))
                if img is None:
                    self.logger.warning(f"Could not load frame: {frame_file}")
                    continue
                
                height, width, channels = img.shape
                self.logger.debug(f"Frame {frame_idx}: {height}x{width}x{channels}")
                
                # Calculate how many bits to hide in this frame
                remaining_bits = total_bits - bits_hidden
                max_bits_per_frame = (height * width * channels) // 4  # Use 25% of capacity
                bits_to_hide = min(remaining_bits, max_bits_per_frame)
                
                if bits_to_hide <= 0:
                    continue
                
                # CRITICAL: Create fresh RNG for each frame to ensure consistency
                frame_seed = seed + frame_idx  # Deterministic but unique per frame
                frame_rng = np.random.RandomState(frame_seed)
                
                # Generate random positions for this frame
                total_positions = height * width * channels
                positions = frame_rng.choice(total_positions, size=bits_to_hide, replace=False)
                
                # Log first few positions for debugging
                self.logger.debug(f"Frame {frame_idx}: hiding {bits_to_hide} bits at positions {positions[:10]}...")
                
                # Hide bits in LSBs
                flat_img = img.flatten().copy()  # Ensure we work with a copy
                for i, pos in enumerate(positions):
                    bit_to_hide = data_bits[bits_hidden + i]
                    original_value = flat_img[pos]
                    flat_img[pos] = (int(flat_img[pos]) & 0xFE) | int(bit_to_hide)
                    
                    # Debug first few modifications
                    if i < 5:
                        self.logger.debug(f"Position {pos}: {original_value} -> {flat_img[pos]} (bit: {bit_to_hide})")
                
                # Reshape and save modified frame  
                modified_img = flat_img.reshape(height, width, channels)
                
                # Write frame with PNG compression settings to ensure lossless storage
                write_success = cv2.imwrite(str(frame_file), modified_img, 
                                          [cv2.IMWRITE_PNG_COMPRESSION, 0])  # No compression
                if not write_success:
                    self.logger.error(f"Failed to write modified frame: {frame_file}")
                    raise Exception(f"Could not write frame {frame_file}")
                
                # Verify frame was written correctly by reading it back
                verification_img = cv2.imread(str(frame_file))
                if verification_img is None:
                    raise Exception(f"Cannot read back written frame: {frame_file}")
                
                # Quick verification - check that data changed
                if np.array_equal(img, verification_img):
                    self.logger.warning(f"Frame {frame_idx} appears unchanged after modification")
                
                modified_frames.append(frame_file)
                bits_hidden += bits_to_hide
                
                if frame_idx % 10 == 0:  # More frequent progress logging
                    progress = (bits_hidden / total_bits) * 100
                    self.logger.info(f"Hiding progress: {progress:.1f}% ({len(modified_frames)} frames modified)")
            
            if bits_hidden < total_bits:
                raise Exception(f"Could not hide all data: {bits_hidden}/{total_bits} bits")
            
            self.logger.info(f"Data successfully hidden in {len(modified_frames)} video frames")
            
        except Exception as e:
            self.logger.error(f"Failed to hide data in frames: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _extract_data_from_frames(self, frame_files: List[Path], password: str) -> Optional[bytes]:
        """Extract hidden data from frames - MUST exactly match hiding logic."""
        try:
            # Generate SAME random sequence used for hiding
            seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
            self.logger.info(f"Video extraction using seed: {seed}")
            
            if not frame_files:
                self.logger.error("No frame files provided for extraction")
                return None
            
            # We need to simulate the hiding process exactly to know where bits were placed
            # This is a 2-pass approach: first find the data size, then extract exactly that amount
            
            # PASS 1: Extract header to determine data size
            # Extract enough bits to get the header from sequential frames
            header_bits_needed = 34 * 8  # 272 bits for header
            header_bits = []
            
            for frame_idx, frame_file in enumerate(frame_files):
                if len(header_bits) >= header_bits_needed:
                    break
                    
                assert cv2 is not None, "OpenCV not available"
                img = cv2.imread(str(frame_file))
                if img is None:
                    continue
                
                height, width, channels = img.shape
                
                # Calculate how many bits we need from this frame for header
                remaining_header_bits = header_bits_needed - len(header_bits)
                max_bits_per_frame = (height * width * channels) // 4  # Same 25% capacity as hiding
                bits_to_extract = min(remaining_header_bits, max_bits_per_frame)
                
                # Use SAME frame-specific seed as hiding
                frame_seed = seed + frame_idx
                frame_rng = np.random.RandomState(frame_seed)
                
                # Generate SAME positions as hiding would
                total_positions = height * width * channels
                positions = frame_rng.choice(total_positions, size=bits_to_extract, replace=False)
                
                # Extract LSBs from these positions
                flat_img = img.flatten()
                for pos in positions:
                    if pos < len(flat_img):
                        header_bits.append(int(flat_img[pos]) & 1)
            
            if len(header_bits) < header_bits_needed:
                self.logger.error(f"Not enough header bits extracted: {len(header_bits)} < {header_bits_needed}")
                return None
            
            # Parse header
            header_bit_array = np.array(header_bits[:header_bits_needed], dtype=np.uint8)
            header_bytes = np.packbits(header_bit_array)
            
            if len(header_bytes) < 34:
                self.logger.error(f"Header too short after packing: {len(header_bytes)} < 34 bytes")
                return None
            
            magic = bytes(header_bytes[:8])
            version = bytes(header_bytes[8:10])
            data_size = struct.unpack('<Q', bytes(header_bytes[10:18]))[0]
            checksum = bytes(header_bytes[18:34])
            
            self.logger.info(f"Header parsed - magic: {magic}, data size: {data_size} bytes")
            
            if magic != self.MAGIC_HEADER:
                self.logger.error(f"Invalid magic header: {magic} != {self.MAGIC_HEADER}")
                return None
            
            if data_size <= 0 or data_size > 100 * 1024 * 1024:  # Max 100MB
                self.logger.error(f"Invalid data size: {data_size}")
                return None
            
            # PASS 2: Now extract the exact total bits needed (header + data)
            total_bits_needed = (34 + data_size) * 8
            self.logger.info(f"Extracting {total_bits_needed} total bits using exact hiding logic")
            
            all_bits = []
            bits_extracted = 0
            
            # Extract bits using EXACT same frame-by-frame logic as hiding
            for frame_idx, frame_file in enumerate(frame_files):
                if bits_extracted >= total_bits_needed:
                    break
                
                assert cv2 is not None, "OpenCV not available"
                img = cv2.imread(str(frame_file))
                if img is None:
                    continue
                
                height, width, channels = img.shape
                
                # Calculate how many bits to extract (SAME logic as hiding)
                remaining_bits = total_bits_needed - bits_extracted
                max_bits_per_frame = (height * width * channels) // 4  # Same 25% capacity
                bits_to_extract = min(remaining_bits, max_bits_per_frame)
                
                if bits_to_extract <= 0:
                    continue
                
                # Use SAME frame-specific seed as hiding
                frame_seed = seed + frame_idx
                frame_rng = np.random.RandomState(frame_seed)
                
                # Generate SAME positions as hiding would
                total_positions = height * width * channels
                positions = frame_rng.choice(total_positions, size=bits_to_extract, replace=False)
                
                if frame_idx == 0:  # Debug first frame
                    self.logger.debug(f"Frame {frame_idx}: extracting {bits_to_extract} bits at positions {positions[:10]}...")
                
                # Extract LSBs from these positions
                flat_img = img.flatten()
                frame_bits = []
                for pos in positions:
                    if pos < len(flat_img):
                        frame_bits.append(int(flat_img[pos]) & 1)
                
                all_bits.extend(frame_bits)
                bits_extracted += len(frame_bits)
                
                if frame_idx % 10 == 0:
                    progress = (bits_extracted / total_bits_needed) * 100
                    self.logger.info(f"Extraction progress: {progress:.1f}%")
            
            if bits_extracted < total_bits_needed:
                self.logger.error(f"Not enough bits extracted: {bits_extracted} < {total_bits_needed}")
                return None
            
            # Convert to bytes
            bit_array = np.array(all_bits[:total_bits_needed], dtype=np.uint8)
            byte_array = np.packbits(bit_array)
            
            self.logger.info(f"Converted {len(all_bits)} bits to {len(byte_array)} bytes")
            
            # Skip header (34 bytes) and extract encrypted data
            if len(byte_array) < 34 + data_size:
                self.logger.error(f"Not enough bytes: {len(byte_array)} < {34 + data_size}")
                return None
            
            encrypted_data = bytes(byte_array[34:34 + data_size])
            
            # Verify checksum
            actual_checksum = hashlib.md5(encrypted_data).digest()
            if actual_checksum != checksum:
                self.logger.error(f"Checksum mismatch: expected {checksum.hex()}, got {actual_checksum.hex()}")
                return None
            
            self.logger.info(f"Successfully extracted and verified {len(encrypted_data)} bytes of encrypted data")
            return encrypted_data
            
        except Exception as e:
            self.logger.error(f"Video extraction failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _extract_header_from_frame(self, frame_file: Path, rng: np.random.RandomState) -> Optional[bytes]:
        """Extract header data from first frame."""
        try:
            assert cv2 is not None, "OpenCV not available"
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
                header_bits.append(int(flat_img[pos]) & 1)
            
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
            # Ensure ffmpeg is available (should be checked by dependencies_available)
            assert ffmpeg is not None, "ffmpeg not available"
            
            frames_path = frames_dir / "frames"
            
            # Get original video properties
            assert ffmpeg is not None, "FFmpeg module is required for video operations"
            probe = ffmpeg.probe(str(original_path))  # type: ignore
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            fps = eval(video_stream['r_frame_rate'])  # Convert fraction to float
            
            self.logger.debug(f"Video properties: FPS={fps}, Total frames={video_stream.get('nb_frames', 'unknown')}")
            
            # Verify extracted frames exist
            frame_files = list(frames_path.glob("frame_*.png"))
            if not frame_files:
                raise Exception("No frames found for reassembly")
            
            self.logger.info(f"Reassembling video from {len(frame_files)} frames")
            
            # Use ffmpeg to create video from frames - with proper frame handling
            input_pattern = str(frames_path / "frame_%06d.png")
            
            # Build ffmpeg command with additional options for stability
            assert ffmpeg is not None, "FFmpeg module is required for input stream"
            stream = ffmpeg.input(  # type: ignore
                input_pattern, 
                framerate=fps,
                start_number=0  # Start from frame 0
            )
            
            # Copy audio from original video if present
            audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
            
            # Use FFV1 codec to truly preserve LSB data  
            # FFV1 with bgr0 preserves exact RGB pixel values (100% LSB preservation)
            encoding_params = {
                'vcodec': 'ffv1',         # FFV1 lossless codec
                'level': 3,               # FFV1 version 3 (best compression)
                'pix_fmt': 'bgr0',        # BGRA format (4 channels)
                'slices': 24,             # Enable parallel processing
                'slicecrc': 1,            # Enable CRC error detection
            }
            
            if audio_streams:
                self.logger.debug("Including audio stream from original video")
                assert ffmpeg is not None, "FFmpeg module is required for audio input"
                audio_input = ffmpeg.input(str(original_path))  # type: ignore
                stream = ffmpeg.output(  # type: ignore
                    stream, audio_input['a'],
                    str(output_path),
                    acodec='copy',
                    **encoding_params
                )
            else:
                self.logger.debug("Video has no audio stream")
                assert ffmpeg is not None, "FFmpeg module is required for video output"
                stream = ffmpeg.output(  # type: ignore
                    stream,
                    str(output_path),
                    **encoding_params
                )
            
            # Run ffmpeg with better error handling
            assert ffmpeg is not None, "FFmpeg module is required for compilation and execution"
            self.logger.debug(f"Running ffmpeg command: {' '.join(ffmpeg.compile(stream))}")  # type: ignore
            
            try:
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)  # type: ignore
            except ffmpeg.Error as e:  # type: ignore
                error_output = e.stderr.decode() if e.stderr else "Unknown ffmpeg error"
                self.logger.error(f"FFmpeg error: {error_output}")
                raise Exception(f"Video encoding failed: {error_output}")
            
            # Verify output file was created and has content
            if not output_path.exists():
                raise Exception("Output video file was not created")
            
            output_size = output_path.stat().st_size
            if output_size == 0:
                raise Exception("Output video file is empty")
            
            self.logger.info(f"Video reassembly completed: {output_path.name} ({output_size:,} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"Video reassembly failed: {e}")
            # Clean up failed output file
            if output_path.exists():
                try:
                    output_path.unlink()
                except:
                    pass
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
    
    def test_steganography_roundtrip(self, video_path: Path, test_data: bytes, 
                                    password: str) -> bool:
        """Test if steganography works by doing a complete hide/extract cycle."""
        try:
            self.logger.info(f"Testing steganography round-trip with {len(test_data)} bytes")
            
            # Create temporary output file
            temp_dir = Path(tempfile.mkdtemp(prefix="invv_test_"))
            test_output = temp_dir / "test_output.mp4"
            
            try:
                # Hide test data
                hide_success = self.hide_data_in_video(
                    video_path, test_data, test_output, password
                )
                
                if not hide_success:
                    self.logger.error("Test hiding failed")
                    return False
                
                # Extract test data
                extracted_data = self.extract_data_from_video(test_output, password)
                
                if extracted_data is None:
                    self.logger.error("Test extraction failed")
                    return False
                
                # Compare data
                if extracted_data == test_data:
                    self.logger.info("Steganography round-trip test PASSED!")
                    return True
                else:
                    self.logger.error(
                        f"Test data mismatch: original {len(test_data)} bytes, "
                        f"extracted {len(extracted_data)} bytes"
                    )
                    return False
                    
            finally:
                # Clean up test files
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            self.logger.error(f"Round-trip test failed: {e}")
            return False
    
    def _verify_frame_write(self, frame_path: Path, original_img: np.ndarray, 
                           expected_img: np.ndarray) -> bool:
        """Verify that a frame was written correctly to disk."""
        try:
            # Read the frame back from disk
            assert cv2 is not None, "OpenCV not available"
            written_img = cv2.imread(str(frame_path))
            if written_img is None:
                self.logger.error(f"Could not read back written frame: {frame_path}")
                return False
            
            # Check basic properties match
            if written_img.shape != expected_img.shape:
                self.logger.error(f"Frame shape mismatch: expected {expected_img.shape}, got {written_img.shape}")
                return False
            
            # Check that the data is actually different from original (modifications were applied)
            diff_count = np.sum(written_img != original_img)
            if diff_count == 0:
                self.logger.warning(f"Frame appears unchanged from original (this may be normal for small data)")
            
            # Check that written frame matches what we expected to write
            exact_match = np.array_equal(written_img, expected_img)
            if not exact_match:
                # Allow for small differences due to compression/encoding
                max_diff = np.max(np.abs(written_img.astype(int) - expected_img.astype(int)))
                if max_diff > 2:  # Allow up to 2 pixel value difference
                    self.logger.error(f"Frame content mismatch: max difference {max_diff}")
                    return False
                else:
                    self.logger.debug(f"Frame verification passed with minor differences (max diff: {max_diff})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Frame verification failed with exception: {e}")
            return False
