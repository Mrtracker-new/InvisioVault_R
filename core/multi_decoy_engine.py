"""
Multi-Decoy Engine - Multiple Dataset Steganography with Plausible Deniability
Hides multiple separate datasets in a single image with different passwords and priority levels.
"""

import os
import struct
import hashlib
import secrets
import json
import tempfile
import zipfile
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

import numpy as np
from PIL import Image

from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel
from utils.logger import Logger
from utils.error_handler import ErrorHandler, EncryptionError


class MultiDecoyEngine:
    """Multi-dataset steganography implementation with layered plausible deniability."""
    
    MAGIC_HEADER = b'INVMD'  # InvisioVault Multi-Decoy
    VERSION = b'\x02\x00'
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MAXIMUM):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.stego_engine = SteganographyEngine()
        self.encryption_engine = EncryptionEngine(security_level)
        
        self.logger.info("Multi-Decoy Engine initialized")
    
    def calculate_multi_capacity(self, image_path: Path, num_datasets: int = 5) -> Dict[str, int]:
        """Calculate capacity for multiple datasets in an image.
        
        Args:
            image_path: Path to the carrier image
            num_datasets: Expected number of datasets
            
        Returns:
            Dictionary with capacity information
        """
        try:
            total_capacity = self.stego_engine.calculate_capacity(image_path)
            
            # Reserve space for headers and metadata per dataset
            header_overhead_per_dataset = (
                len(self.MAGIC_HEADER) + len(self.VERSION) + 
                4 + 4 + 4 + 32 + 256  # dataset_id_len + data_size + priority + checksum + metadata
            )
            total_header_overhead = header_overhead_per_dataset * num_datasets
            
            available_bytes = total_capacity - total_header_overhead
            
            # Calculate per-dataset capacity (equal distribution)
            per_dataset_capacity = available_bytes // num_datasets if num_datasets > 0 else available_bytes
            
            return {
                'total_capacity': total_capacity,
                'available_capacity': available_bytes,
                'per_dataset_capacity': per_dataset_capacity,
                'max_datasets': available_bytes // (header_overhead_per_dataset + 1000),  # Min 1KB per dataset
                'header_overhead': total_header_overhead
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating multi-decoy capacity: {e}")
            return {'total_capacity': 0, 'available_capacity': 0, 'per_dataset_capacity': 0, 'max_datasets': 0}
    
    def hide_multiple_datasets(self, carrier_path: Path, datasets: List[Dict], output_path: Path) -> bool:
        """Hide multiple datasets with layered steganography.
        
        Args:
            carrier_path: Path to carrier image
            datasets: List of dataset configurations with 'name', 'password', 'priority', 'files'
            output_path: Path to save the steganographic image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not datasets:
                self.logger.error("No datasets provided")
                return False
            
            # Sort datasets by priority (1 = outermost/least secure, 5 = innermost/most secure)
            sorted_datasets = sorted(datasets, key=lambda x: x['priority'])
            
            # Check capacity
            capacity = self.calculate_multi_capacity(carrier_path, len(datasets))
            
            # Process each dataset and prepare encrypted data
            processed_datasets = []
            for i, dataset in enumerate(sorted_datasets):
                self.logger.info(f"Processing dataset {i+1}/{len(datasets)}: {dataset['name']}")
                
                # Create archive from files
                archive_data = self._create_dataset_archive(dataset['files'])
                if not archive_data:
                    self.logger.error(f"Failed to create archive for dataset {dataset['name']}")
                    return False
                
                # Check individual dataset size
                if len(archive_data) > capacity['per_dataset_capacity']:
                    self.logger.error(f"Dataset {dataset['name']} too large: {len(archive_data)} > {capacity['per_dataset_capacity']}")
                    return False
                
                # Encrypt dataset
                encrypted_data = self.encryption_engine.encrypt_with_metadata(
                    archive_data, dataset['password']
                )
                
                # Create dataset metadata
                dataset_metadata = {
                    'dataset_id': dataset['name'],
                    'priority': dataset['priority'],
                    'decoy_type': dataset.get('decoy_type', 'standard'),
                    'file_count': len(dataset['files']),
                    'original_size': len(archive_data),
                    'encrypted_size': len(encrypted_data)
                }
                
                processed_datasets.append({
                    'metadata': dataset_metadata,
                    'encrypted_data': encrypted_data,
                    'password': dataset['password']
                })
            
            # Create layered payload
            layered_payload = self._create_layered_payload(processed_datasets)
            
            # Hide the layered payload using non-randomized approach for reliable extraction
            # Sequential hiding ensures extraction works consistently
            success = self.stego_engine.hide_data(
                carrier_path=carrier_path,
                data=layered_payload,
                output_path=output_path,
                randomize=False,
                seed=None
            )
            
            if success:
                self.logger.info(f"Successfully created multi-decoy image with {len(datasets)} datasets")
                return True
            else:
                self.logger.error("Failed to hide layered payload")
                return False
                
        except Exception as e:
            self.logger.error(f"Error hiding multiple datasets: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def extract_dataset(self, stego_path: Path, password: str, output_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract a specific dataset using its password.
        
        Args:
            stego_path: Path to steganographic image
            password: Password for the dataset to extract
            output_dir: Directory to extract files to
            
        Returns:
            Dataset metadata if successful, None otherwise
        """
        try:
            # Extract the layered payload using non-randomized approach matching hide method
            payload = self.stego_engine.extract_data(stego_path, randomize=False)
            if not payload:
                self.logger.error("No steganographic data found")
                return None
            
            # Parse layered payload
            datasets = self._parse_layered_payload(payload)
            if not datasets:
                self.logger.error("Failed to parse layered payload")
                return None
            
            # Try to decrypt each dataset with the provided password
            for dataset_info in datasets:
                try:
                    decrypted_data = self.encryption_engine.decrypt_with_metadata(
                        dataset_info['encrypted_data'], password
                    )
                    
                    # Successfully decrypted - extract files
                    dataset_output_dir = output_dir / dataset_info['metadata']['dataset_id']
                    dataset_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    extracted_files = self._extract_dataset_files(decrypted_data, dataset_output_dir)
                    if extracted_files:
                        self.logger.info(f"Successfully extracted dataset: {dataset_info['metadata']['dataset_id']}")
                        
                        # Add extracted file information to metadata
                        result_metadata = dataset_info['metadata'].copy()
                        result_metadata['extracted_files'] = [
                            {'path': str(file_path), 'name': file_path.name} for file_path in extracted_files
                        ]
                        result_metadata['extraction_path'] = str(dataset_output_dir)
                        
                        return result_metadata
                        
                except Exception as decrypt_error:
                    # Failed to decrypt with this password - try next dataset
                    continue
            
            self.logger.error("No dataset could be decrypted with the provided password")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting dataset: {e}")
            return None
    
    def list_datasets(self, stego_path: Path) -> List[Dict[str, Any]]:
        """List metadata of all datasets in the image (without decrypting).
        
        Args:
            stego_path: Path to steganographic image
            
        Returns:
            List of dataset metadata
        """
        try:
            # Use non-randomized extraction matching hide method
            payload = self.stego_engine.extract_data(stego_path, randomize=False)
            if not payload:
                return []
            
            datasets = self._parse_layered_payload(payload)
            return [d['metadata'] for d in datasets] if datasets else []
            
        except Exception as e:
            self.logger.error(f"Error listing datasets: {e}")
            return []
    
    def _create_dataset_archive(self, file_paths: List[str]) -> Optional[bytes]:
        """Create a ZIP archive from the dataset files.
        
        Args:
            file_paths: List of file paths to include
            
        Returns:
            Archive data as bytes, or None on error
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                temp_zip_path = Path(temp_file.name)
            
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for file_path in file_paths:
                    if Path(file_path).exists():
                        archive.write(file_path, Path(file_path).name)
                    else:
                        self.logger.warning(f"File not found: {file_path}")
            
            with open(temp_zip_path, 'rb') as f:
                archive_data = f.read()
            
            temp_zip_path.unlink()  # Clean up
            return archive_data
            
        except Exception as e:
            self.logger.error(f"Error creating dataset archive: {e}")
            return None
    
    def _create_layered_payload(self, processed_datasets: List[Dict]) -> bytes:
        """Create a layered payload with all encrypted datasets.
        
        Args:
            processed_datasets: List of processed dataset dictionaries
            
        Returns:
            Layered payload bytes
        """
        # Create header
        payload = self.MAGIC_HEADER + self.VERSION
        
        # Add dataset count
        payload += struct.pack('<I', len(processed_datasets))
        
        # Add each dataset
        for dataset in processed_datasets:
            metadata_json = json.dumps(dataset['metadata']).encode('utf-8')
            encrypted_data = dataset['encrypted_data']
            
            # Create dataset entry
            dataset_entry = (
                struct.pack('<I', len(metadata_json)) +  # Metadata length
                metadata_json +                          # Metadata
                struct.pack('<I', len(encrypted_data)) + # Data length
                hashlib.sha256(encrypted_data).digest()[:8] +  # Checksum (8 bytes)
                encrypted_data                           # Encrypted data
            )
            
            payload += dataset_entry
        
        return payload
    
    def _parse_layered_payload(self, payload: bytes) -> Optional[List[Dict]]:
        """Parse layered payload to extract dataset information.
        
        Args:
            payload: Layered payload bytes
            
        Returns:
            List of dataset dictionaries or None on error
        """
        try:
            if len(payload) < 10:  # Minimum header size
                return None
            
            offset = 0
            
            # Check magic header
            magic = payload[offset:offset + len(self.MAGIC_HEADER)]
            if magic != self.MAGIC_HEADER:
                self.logger.error("Invalid multi-decoy magic header")
                return None
            offset += len(self.MAGIC_HEADER)
            
            # Check version
            version = payload[offset:offset + len(self.VERSION)]
            offset += len(self.VERSION)
            
            # Get dataset count
            dataset_count = struct.unpack('<I', payload[offset:offset + 4])[0]
            offset += 4
            
            datasets = []
            
            # Parse each dataset
            for i in range(dataset_count):
                if offset >= len(payload):
                    break
                
                # Get metadata length and data
                metadata_len = struct.unpack('<I', payload[offset:offset + 4])[0]
                offset += 4
                
                metadata_json = payload[offset:offset + metadata_len].decode('utf-8')
                offset += metadata_len
                metadata = json.loads(metadata_json)
                
                # Get encrypted data length
                data_len = struct.unpack('<I', payload[offset:offset + 4])[0]
                offset += 4
                
                # Get checksum
                checksum = payload[offset:offset + 8]
                offset += 8
                
                # Get encrypted data
                encrypted_data = payload[offset:offset + data_len]
                offset += data_len
                
                # Verify checksum
                if hashlib.sha256(encrypted_data).digest()[:8] != checksum:
                    self.logger.error(f"Checksum mismatch for dataset {i}")
                    continue
                
                datasets.append({
                    'metadata': metadata,
                    'encrypted_data': encrypted_data
                })
            
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error parsing layered payload: {e}")
            return None
    
    def _extract_dataset_files(self, archive_data: bytes, output_dir: Path) -> List[Path]:
        """Extract files from dataset archive.
        
        Args:
            archive_data: ZIP archive data
            output_dir: Directory to extract to
            
        Returns:
            List of extracted file paths, empty list on failure
        """
        extracted_files = []
        try:
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                temp_zip_path = Path(temp_file.name)
            
            with open(temp_zip_path, 'wb') as f:
                f.write(archive_data)
            
            with zipfile.ZipFile(temp_zip_path, 'r') as archive:
                for member in archive.namelist():
                    # Extract each file and track its path
                    archive.extract(member, output_dir)
                    extracted_file_path = output_dir / member
                    if extracted_file_path.exists():
                        extracted_files.append(extracted_file_path)
            
            temp_zip_path.unlink()  # Clean up
            return extracted_files
            
        except Exception as e:
            self.logger.error(f"Error extracting dataset files: {e}")
            return []
    
    def _generate_deterministic_seed(self, image_path: Path, payload_data: Optional[bytes]) -> int:
        """Generate a deterministic seed based on image properties.
        
        Args:
            image_path: Path to the image file
            payload_data: Optional payload data (for hiding, None for extraction)
            
        Returns:
            Deterministic seed for consistent hide/extract operations
        """
        try:
            # Get image file stats for deterministic properties
            file_size = image_path.stat().st_size
            
            # Load image to get dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
            
            # Create deterministic seed components
            seed_components = [
                str(file_size).encode('utf-8'),
                str(width).encode('utf-8'),
                str(height).encode('utf-8'),
                mode.encode('utf-8'),
                str(image_path.name).encode('utf-8'),
                b'MULTI_DECOY_SEED_V2'  # Version identifier
            ]
            
            # Combine all components
            combined_data = b''.join(seed_components)
            
            # Generate deterministic hash
            seed_hash = hashlib.sha256(combined_data).digest()
            
            # Convert to integer seed (32-bit for numpy compatibility)
            seed = int.from_bytes(seed_hash[:4], byteorder='big') % (2**31 - 1)
            
            self.logger.debug(f"Generated deterministic seed: {seed} for image {image_path.name}")
            return seed
            
        except Exception as e:
            self.logger.error(f"Error generating deterministic seed: {e}")
            # Fallback seed based on image path hash
            fallback_hash = hashlib.md5(str(image_path).encode('utf-8')).digest()
            return int.from_bytes(fallback_hash[:4], byteorder='big') % (2**31 - 1)
