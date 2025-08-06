"""
Decoy Engine - Dual-Data Steganography with Plausible Deniability
Hides two separate datasets in a single image with different passwords.
"""

import os
import struct
import hashlib
import secrets
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

import numpy as np
from PIL import Image

from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel
from utils.logger import Logger
from utils.error_handler import ErrorHandler, EncryptionError


class DecoyEngine:
    """Dual-data steganography implementation with plausible deniability."""
    
    MAGIC_HEADER = b'INVD'  # InvisioVault Decoy
    VERSION = b'\x01\x00'
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.stego_engine = SteganographyEngine()
        self.encryption_engine = EncryptionEngine(security_level)
        
        self.logger.info("Decoy Engine initialized")
    
    def calculate_decoy_capacity(self, image_path: Path) -> Dict[str, int]:
        """Calculate capacity for decoy and real data in an image.
        
        Args:
            image_path: Path to the carrier image
            
        Returns:
            Dictionary with capacity information
        """
        try:
            total_capacity = self.stego_engine.calculate_capacity(image_path)
            
            # Reserve space for dual headers and metadata
            header_overhead = (len(self.MAGIC_HEADER) + len(self.VERSION) + 
                              4 + 4 + 32) * 8  # flags + sizes + checksums
            
            available_bits = (total_capacity * 8) - header_overhead
            available_bytes = available_bits // 8
            
            # Split capacity - 30% for decoy, 70% for real data
            decoy_capacity = int(available_bytes * 0.3)
            real_capacity = available_bytes - decoy_capacity
            
            return {
                'total_capacity': total_capacity,
                'decoy_capacity': decoy_capacity,
                'real_capacity': real_capacity,
                'overhead_bytes': header_overhead // 8
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating decoy capacity: {e}")
            return {'total_capacity': 0, 'decoy_capacity': 0, 'real_capacity': 0}
    
    def hide_dual_data(self, carrier_path: Path, decoy_data: bytes, decoy_password: str,
                       real_data: bytes, real_password: str, output_path: Path) -> bool:
        """Hide both decoy and real data in a single image.
        
        Args:
            carrier_path: Path to carrier image
            decoy_data: Data for decoy (accessible with decoy password)
            decoy_password: Password for decoy data
            real_data: Real hidden data
            real_password: Password for real data
            output_path: Path to save the steganographic image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check capacity
            capacity = self.calculate_decoy_capacity(carrier_path)
            if len(decoy_data) > capacity['decoy_capacity']:
                self.logger.error(f"Decoy data too large: {len(decoy_data)} > {capacity['decoy_capacity']}")
                return False
            
            if len(real_data) > capacity['real_capacity']:
                self.logger.error(f"Real data too large: {len(real_data)} > {capacity['real_capacity']}")
                return False
            
            # Encrypt both datasets
            encrypted_decoy = self.encryption_engine.encrypt_with_metadata(decoy_data, decoy_password)
            encrypted_real = self.encryption_engine.encrypt_with_metadata(real_data, real_password)
            
            # Create combined payload with headers
            payload = self._create_dual_payload(encrypted_decoy, encrypted_real)
            
            # Hide the combined payload using LSB steganography
            success = self.stego_engine.hide_data(
                carrier_path=carrier_path,
                data=payload,
                output_path=output_path,
                randomize=True,
                seed=self.stego_engine.generate_random_seed()
            )
            
            if success:
                self.logger.info(f"Successfully created decoy steganographic image at {output_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error hiding dual data: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def extract_decoy_data(self, stego_path: Path, password: str) -> Optional[bytes]:
        """Extract decoy data from steganographic image.
        
        Args:
            stego_path: Path to steganographic image
            password: Password for decoy data
            
        Returns:
            Decoy data if successful, None otherwise
        """
        try:
            # Extract the combined payload
            payload = self.stego_engine.extract_data(stego_path, randomize=True)
            if not payload:
                return None
            
            # Parse the dual payload
            decoy_data, real_data = self._parse_dual_payload(payload)
            if not decoy_data:
                return None
            
            # Try to decrypt decoy data
            try:
                decrypted_data = self.encryption_engine.decrypt_with_metadata(decoy_data, password)
                self.logger.info("Successfully extracted decoy data")
                return decrypted_data
            except Exception:
                self.logger.error("Failed to decrypt with provided password")
                return None
            
        except Exception as e:
            self.logger.error(f"Error extracting decoy data: {e}")
            return None
    
    def extract_real_data(self, stego_path: Path, password: str) -> Optional[bytes]:
        """Extract real hidden data from steganographic image.
        
        Args:
            stego_path: Path to steganographic image
            password: Password for real data
            
        Returns:
            Real data if successful, None otherwise
        """
        try:
            # Extract the combined payload
            payload = self.stego_engine.extract_data(stego_path, randomize=True)
            if not payload:
                return None
            
            # Parse the dual payload
            decoy_data, real_data = self._parse_dual_payload(payload)
            if not real_data:
                return None
            
            # Try to decrypt real data
            try:
                decrypted_data = self.encryption_engine.decrypt_with_metadata(real_data, password)
                self.logger.info("Successfully extracted real hidden data")
                return decrypted_data
            except Exception:
                self.logger.error("Failed to decrypt with provided password")
                return None
            
        except Exception as e:
            self.logger.error(f"Error extracting real data: {e}")
            return None
    
    def _create_dual_payload(self, encrypted_decoy: bytes, encrypted_real: bytes) -> bytes:
        """Create combined payload with both encrypted datasets.
        
        Args:
            encrypted_decoy: Encrypted decoy data
            encrypted_real: Encrypted real data
            
        Returns:
            Combined payload
        """
        # Create header
        decoy_size = struct.pack('<I', len(encrypted_decoy))  # 4 bytes
        real_size = struct.pack('<I', len(encrypted_real))    # 4 bytes
        decoy_checksum = hashlib.sha256(encrypted_decoy).digest()[:4]  # 4 bytes
        real_checksum = hashlib.sha256(encrypted_real).digest()[:4]    # 4 bytes
        
        # Combine all components
        payload = (
            self.MAGIC_HEADER +     # 4 bytes
            self.VERSION +          # 2 bytes
            decoy_size +            # 4 bytes
            real_size +             # 4 bytes
            decoy_checksum +        # 4 bytes
            real_checksum +         # 4 bytes
            encrypted_decoy +       # Variable
            encrypted_real          # Variable
        )
        
        return payload
    
    def _parse_dual_payload(self, payload: bytes) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Parse combined payload to extract both datasets.
        
        Args:
            payload: Combined payload
            
        Returns:
            Tuple of (decoy_data, real_data) or (None, None) on error
        """
        try:
            # Validate header
            if len(payload) < 22:  # Minimum header size
                return None, None
            
            offset = 0
            
            # Check magic header
            magic = payload[offset:offset + len(self.MAGIC_HEADER)]
            if magic != self.MAGIC_HEADER:
                self.logger.error("Invalid decoy magic header")
                return None, None
            offset += len(self.MAGIC_HEADER)
            
            # Check version
            version = payload[offset:offset + len(self.VERSION)]
            offset += len(self.VERSION)
            
            # Extract sizes
            decoy_size = struct.unpack('<I', payload[offset:offset + 4])[0]
            offset += 4
            real_size = struct.unpack('<I', payload[offset:offset + 4])[0]
            offset += 4
            
            # Extract checksums
            decoy_checksum = payload[offset:offset + 4]
            offset += 4
            real_checksum = payload[offset:offset + 4]
            offset += 4
            
            # Extract data
            encrypted_decoy = payload[offset:offset + decoy_size]
            offset += decoy_size
            encrypted_real = payload[offset:offset + real_size]
            
            # Verify checksums
            if hashlib.sha256(encrypted_decoy).digest()[:4] != decoy_checksum:
                self.logger.error("Decoy data checksum mismatch")
                return None, None
            
            if hashlib.sha256(encrypted_real).digest()[:4] != real_checksum:
                self.logger.error("Real data checksum mismatch")
                return None, None
            
            return encrypted_decoy, encrypted_real
            
        except Exception as e:
            self.logger.error(f"Error parsing dual payload: {e}")
            return None, None
    
    def analyze_decoy_image(self, image_path: Path) -> Dict[str, Any]:
        """Analyze whether an image contains decoy data.
        
        Args:
            image_path: Path to image to analyze
            
        Returns:
            Analysis results
        """
        try:
            # Try to extract payload
            payload = self.stego_engine.extract_data(image_path, randomize=True)
            if not payload:
                return {
                    'has_decoy': False,
                    'error': 'No steganographic data found'
                }
            
            # Check if it's a decoy image
            if len(payload) < len(self.MAGIC_HEADER):
                return {
                    'has_decoy': False,
                    'has_regular_stego': True
                }
            
            magic = payload[:len(self.MAGIC_HEADER)]
            has_decoy = (magic == self.MAGIC_HEADER)
            
            if has_decoy:
                # Parse to get more details
                decoy_data, real_data = self._parse_dual_payload(payload)
                return {
                    'has_decoy': True,
                    'decoy_size': len(decoy_data) if decoy_data else 0,
                    'real_size': len(real_data) if real_data else 0,
                    'total_payload_size': len(payload)
                }
            else:
                return {
                    'has_decoy': False,
                    'has_regular_stego': True,
                    'payload_size': len(payload)
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing decoy image: {e}")
            return {
                'has_decoy': False,
                'error': str(e)
            }
