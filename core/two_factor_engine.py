"""
Two-Factor Steganography Engine
Distributes data across multiple images for enhanced security and redundancy.
"""

import os
import json
import struct
import hashlib
import secrets
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image

from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel
from utils.logger import Logger
from utils.error_handler import ErrorHandler, EncryptionError


class TwoFactorEngine:
    """Multi-image data distribution with redundancy and error correction."""
    
    MAGIC_HEADER = b'INV2'  # InvisioVault 2-Factor
    VERSION = b'\x01\x00'
    MIN_IMAGES = 2
    MAX_IMAGES = 8
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.stego_engine = SteganographyEngine()
        self.encryption_engine = EncryptionEngine(security_level)
        
        self.logger.info("Two-Factor Engine initialized")
    
    def calculate_distribution_capacity(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Calculate total capacity for multi-image distribution.
        
        Args:
            image_paths: List of carrier image paths
            
        Returns:
            Dictionary with capacity information
        """
        try:
            capacities = []
            total_capacity = 0
            
            for image_path in image_paths:
                if self.stego_engine.validate_image_format(image_path):
                    capacity = self.stego_engine.calculate_capacity(image_path)
                    capacities.append(capacity)
                    total_capacity += capacity
                else:
                    capacities.append(0)
            
            # Reserve space for headers and redundancy
            header_overhead = len(image_paths) * 100  # Approximate per-image overhead
            redundancy_overhead = int(total_capacity * 0.1)  # 10% for redundancy
            
            usable_capacity = max(0, total_capacity - header_overhead - redundancy_overhead)
            
            return {
                'image_count': len(image_paths),
                'individual_capacities': capacities,
                'total_capacity': total_capacity,
                'usable_capacity': usable_capacity,
                'header_overhead': header_overhead,
                'redundancy_overhead': redundancy_overhead
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating distribution capacity: {e}")
            return {'usable_capacity': 0, 'error': str(e)}
    
    def distribute_data(self, data: bytes, password: str, carrier_paths: List[Path], 
                       output_paths: List[Path], redundancy_level: int = 1) -> bool:
        """Distribute data across multiple images with redundancy.
        
        Args:
            data: Data to distribute
            password: Password for encryption
            carrier_paths: List of carrier image paths
            output_paths: List of output steganographic image paths
            redundancy_level: Level of redundancy (1-3)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not (self.MIN_IMAGES <= len(carrier_paths) <= self.MAX_IMAGES):
                self.logger.error(f"Image count must be between {self.MIN_IMAGES} and {self.MAX_IMAGES}")
                return False
            
            if len(carrier_paths) != len(output_paths):
                self.logger.error("Number of carrier and output paths must match")
                return False
            
            # Check capacity
            capacity_info = self.calculate_distribution_capacity(carrier_paths)
            if len(data) > capacity_info['usable_capacity']:
                self.logger.error(f"Data too large: {len(data)} > {capacity_info['usable_capacity']}")
                return False
            
            # Encrypt the data
            encrypted_data = self.encryption_engine.encrypt_with_metadata(data, password)
            
            # Create manifest
            manifest = self._create_manifest(len(carrier_paths), len(encrypted_data), redundancy_level)
            
            # Split data into chunks with redundancy
            chunks = self._split_data_with_redundancy(encrypted_data, len(carrier_paths), redundancy_level)
            
            # Hide data in each image
            success_count = 0
            for i, (carrier_path, output_path, chunk) in enumerate(zip(carrier_paths, output_paths, chunks)):
                # Create payload with manifest and chunk
                payload = self._create_chunk_payload(manifest, i, chunk)
                
                # Hide in image
                if self.stego_engine.hide_data(
                    carrier_path=carrier_path,
                    data=payload,
                    output_path=output_path,
                    randomize=True,
                    seed=self.stego_engine.generate_random_seed()
                ):
                    success_count += 1
                    self.logger.debug(f"Successfully hid chunk {i+1} in {output_path}")
                else:
                    self.logger.error(f"Failed to hide chunk {i+1} in {output_path}")
            
            if success_count == len(carrier_paths):
                self.logger.info(f"Successfully distributed data across {len(carrier_paths)} images")
                return True
            else:
                self.logger.error(f"Only {success_count}/{len(carrier_paths)} images processed successfully")
                return False
            
        except Exception as e:
            self.logger.error(f"Error distributing data: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def reconstruct_data(self, stego_paths: List[Path], password: str) -> Optional[bytes]:
        """Reconstruct data from multiple steganographic images.
        
        Args:
            stego_paths: List of steganographic image paths
            password: Password for decryption
            
        Returns:
            Reconstructed data if successful, None otherwise
        """
        try:
            # Extract chunks from all images
            chunks = {}
            manifest = None
            
            for i, stego_path in enumerate(stego_paths):
                try:
                    # Extract payload
                    payload = self.stego_engine.extract_data(stego_path, randomize=True)
                    if not payload:
                        self.logger.warning(f"No data found in {stego_path}")
                        continue
                    
                    # Parse payload
                    parsed_manifest, chunk_index, chunk_data = self._parse_chunk_payload(payload)
                    if parsed_manifest and chunk_data is not None:
                        if manifest is None:
                            manifest = parsed_manifest
                        
                        chunks[chunk_index] = chunk_data
                        self.logger.debug(f"Extracted chunk {chunk_index} from {stego_path}")
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting from {stego_path}: {e}")
                    continue
            
            if not manifest or not chunks:
                self.logger.error("No valid chunks found")
                return None
            
            # Reconstruct data from chunks
            encrypted_data = self._reconstruct_from_chunks(chunks, manifest)
            if not encrypted_data:
                self.logger.error("Failed to reconstruct data from chunks")
                return None
            
            # Decrypt the reconstructed data
            try:
                decrypted_data = self.encryption_engine.decrypt_with_metadata(encrypted_data, password)
                self.logger.info(f"Successfully reconstructed {len(decrypted_data)} bytes")
                return decrypted_data
            except Exception as e:
                self.logger.error(f"Decryption failed: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error reconstructing data: {e}")
            return None
    
    def _create_manifest(self, image_count: int, data_size: int, redundancy_level: int) -> Dict[str, Any]:
        """Create manifest with distribution information.
        
        Args:
            image_count: Number of images
            data_size: Size of encrypted data
            redundancy_level: Level of redundancy
            
        Returns:
            Manifest dictionary
        """
        return {
            'version': 1,
            'image_count': image_count,
            'data_size': data_size,
            'redundancy_level': redundancy_level,
            'chunk_size': data_size // image_count,
            'checksum': hashlib.sha256(str(image_count + data_size).encode()).hexdigest()[:16]
        }
    
    def _split_data_with_redundancy(self, data: bytes, num_chunks: int, redundancy_level: int) -> List[bytes]:
        """Split data into chunks with redundancy.
        
        Args:
            data: Data to split
            num_chunks: Number of chunks
            redundancy_level: Level of redundancy
            
        Returns:
            List of data chunks
        """
        chunk_size = len(data) // num_chunks
        chunks = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(data)
            chunk = data[start:end]
            
            # Add redundancy information
            if redundancy_level > 1:
                # Simple XOR redundancy for now
                redundancy_data = self._generate_redundancy(chunk, redundancy_level)
                chunk = chunk + redundancy_data
            
            chunks.append(chunk)
        
        return chunks
    
    def _generate_redundancy(self, data: bytes, level: int) -> bytes:
        """Generate redundancy data for error correction.
        
        Args:
            data: Original data
            level: Redundancy level
            
        Returns:
            Redundancy data
        """
        # Simple implementation - XOR with shifted version
        redundancy = bytearray()
        for _ in range(level - 1):
            xor_data = bytes(b ^ ((b << 1) & 0xFF) for b in data)
            redundancy.extend(xor_data[:len(data)//4])  # Use quarter size for redundancy
        
        return bytes(redundancy)
    
    def _create_chunk_payload(self, manifest: Dict[str, Any], chunk_index: int, chunk_data: bytes) -> bytes:
        """Create payload for a single chunk.
        
        Args:
            manifest: Distribution manifest
            chunk_index: Index of this chunk
            chunk_data: Chunk data
            
        Returns:
            Complete payload
        """
        # Serialize manifest
        manifest_json = json.dumps(manifest, separators=(',', ':')).encode('utf-8')
        manifest_size = struct.pack('<H', len(manifest_json))  # 2 bytes
        chunk_index_bytes = struct.pack('<B', chunk_index)      # 1 byte
        chunk_size = struct.pack('<I', len(chunk_data))        # 4 bytes
        
        # Combine payload
        payload = (
            self.MAGIC_HEADER +      # 4 bytes
            self.VERSION +           # 2 bytes
            manifest_size +          # 2 bytes
            chunk_index_bytes +      # 1 byte
            chunk_size +             # 4 bytes
            manifest_json +          # Variable
            chunk_data               # Variable
        )
        
        return payload
    
    def _parse_chunk_payload(self, payload: bytes) -> Tuple[Optional[Dict], Optional[int], Optional[bytes]]:
        """Parse chunk payload to extract components.
        
        Args:
            payload: Chunk payload
            
        Returns:
            Tuple of (manifest, chunk_index, chunk_data) or None values on error
        """
        try:
            if len(payload) < 13:  # Minimum header size
                return None, None, None
            
            offset = 0
            
            # Check magic header
            magic = payload[offset:offset + len(self.MAGIC_HEADER)]
            if magic != self.MAGIC_HEADER:
                self.logger.error("Invalid two-factor magic header")
                return None, None, None
            offset += len(self.MAGIC_HEADER)
            
            # Check version
            version = payload[offset:offset + len(self.VERSION)]
            offset += len(self.VERSION)
            
            # Extract sizes and index
            manifest_size = struct.unpack('<H', payload[offset:offset + 2])[0]
            offset += 2
            chunk_index = struct.unpack('<B', payload[offset:offset + 1])[0]
            offset += 1
            chunk_size = struct.unpack('<I', payload[offset:offset + 4])[0]
            offset += 4
            
            # Extract manifest
            manifest_json = payload[offset:offset + manifest_size]
            offset += manifest_size
            
            # Extract chunk data
            chunk_data = payload[offset:offset + chunk_size]
            
            # Parse manifest JSON
            manifest = json.loads(manifest_json.decode('utf-8'))
            
            return manifest, chunk_index, chunk_data
            
        except Exception as e:
            self.logger.error(f"Error parsing chunk payload: {e}")
            return None, None, None
    
    def _reconstruct_from_chunks(self, chunks: Dict[int, bytes], manifest: Dict[str, Any]) -> Optional[bytes]:
        """Reconstruct original data from chunks.
        
        Args:
            chunks: Dictionary mapping chunk index to chunk data
            manifest: Distribution manifest
            
        Returns:
            Reconstructed data or None on error
        """
        try:
            # Check if we have enough chunks
            required_chunks = manifest['image_count']
            if len(chunks) < required_chunks // 2:  # Need at least half the chunks
                self.logger.error(f"Insufficient chunks: {len(chunks)} < {required_chunks // 2}")
                return None
            
            # Reconstruct data in order
            reconstructed = bytearray()
            
            for i in range(required_chunks):
                if i in chunks:
                    chunk_data = chunks[i]
                    
                    # Remove redundancy data if present
                    if manifest['redundancy_level'] > 1:
                        # Remove redundancy suffix
                        original_size = manifest['chunk_size']
                        if i == required_chunks - 1:  # Last chunk might be smaller
                            remaining = manifest['data_size'] - (i * original_size)
                            chunk_data = chunk_data[:remaining]
                        else:
                            chunk_data = chunk_data[:original_size]
                    
                    reconstructed.extend(chunk_data)
                else:
                    # Try to recover missing chunk using redundancy
                    self.logger.warning(f"Missing chunk {i}, attempting recovery")
                    # For now, we'll fail if any chunk is missing
                    # In a full implementation, we'd use the redundancy data
                    return None
            
            # Verify size
            if len(reconstructed) != manifest['data_size']:
                self.logger.error(f"Size mismatch: {len(reconstructed)} != {manifest['data_size']}")
                return None
            
            return bytes(reconstructed)
            
        except Exception as e:
            self.logger.error(f"Error reconstructing from chunks: {e}")
            return None
    
    def analyze_distribution(self, stego_paths: List[Path]) -> Dict[str, Any]:
        """Analyze multi-image distribution.
        
        Args:
            stego_paths: List of steganographic image paths
            
        Returns:
            Analysis results
        """
        analysis = {
            'total_images': len(stego_paths),
            'valid_chunks': 0,
            'chunks_found': [],
            'manifest': None,
            'reconstructable': False
        }
        
        try:
            for i, stego_path in enumerate(stego_paths):
                try:
                    payload = self.stego_engine.extract_data(stego_path, randomize=True)
                    if payload:
                        manifest, chunk_index, chunk_data = self._parse_chunk_payload(payload)
                        if manifest and chunk_data is not None:
                            analysis['valid_chunks'] += 1
                            analysis['chunks_found'].append(chunk_index)
                            if analysis['manifest'] is None:
                                analysis['manifest'] = manifest
                except Exception:
                    continue
            
            # Check if reconstructable
            if analysis['manifest']:
                required_chunks = analysis['manifest']['image_count']
                analysis['reconstructable'] = analysis['valid_chunks'] >= (required_chunks // 2)
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
