"""
AES-256 Encryption Engine with PBKDF2 Key Derivation
Provides secure encryption/decryption with multiple security levels.
"""

import os
import secrets
import hashlib
from typing import Tuple, Optional, Union
from enum import Enum

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.backends import default_backend

from utils.logger import Logger
from utils.error_handler import ErrorHandler


class SecurityLevel(Enum):
    """Security levels for encryption operations."""
    STANDARD = "standard"  # 100k iterations
    HIGH = "high"         # 500k iterations
    MAXIMUM = "maximum"    # 1M iterations + additional entropy


class EncryptionEngine:
    """AES-256 encryption with PBKDF2 key derivation."""
    
    # Algorithm specifications
    KEY_SIZE = 32  # 256 bits
    IV_SIZE = 16   # 128 bits for AES block size
    SALT_SIZE = 16 # 128 bits
    
    # Security level configurations
    SECURITY_CONFIGS = {
        SecurityLevel.STANDARD: {
            'iterations': 100000,
            'additional_entropy': False
        },
        SecurityLevel.HIGH: {
            'iterations': 500000,
            'additional_entropy': False
        },
        SecurityLevel.MAXIMUM: {
            'iterations': 1000000,
            'additional_entropy': True
        }
    }
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.config = self.SECURITY_CONFIGS[security_level]
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.backend = default_backend()
    
    def derive_key(self, password: Union[str, bytes], salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        try:
            if isinstance(password, str):
                password = password.encode('utf-8')
            
            # Create PBKDF2 key derivation function
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.KEY_SIZE,
                salt=salt,
                iterations=self.config['iterations'],
                backend=self.backend
            )
            
            key = kdf.derive(password)
            
            # Add additional entropy for maximum security
            if self.config['additional_entropy']:
                # Mix with SHA-512 for additional entropy
                extended_key = hashlib.sha512(key + salt + password).digest()[:self.KEY_SIZE]
                key = bytes(a ^ b for a, b in zip(key, extended_key))
            
            self.logger.debug(f"Key derived with {self.config['iterations']} iterations")
            return key
            
        except Exception as e:
            self.logger.error(f"Error deriving key: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def generate_salt(self) -> bytes:
        """Generate cryptographically secure random salt."""
        return secrets.token_bytes(self.SALT_SIZE)
    
    def generate_iv(self) -> bytes:
        """Generate cryptographically secure random IV."""
        return secrets.token_bytes(self.IV_SIZE)
    
    def encrypt(self, data: bytes, password: Union[str, bytes], 
                salt: Optional[bytes] = None) -> Tuple[bytes, bytes, bytes]:
        """Encrypt data using AES-256-CBC.
        
        Args:
            data: Data to encrypt
            password: Password for key derivation
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (encrypted_data, salt, iv)
        """
        try:
            # Generate salt and IV if not provided
            if salt is None:
                salt = self.generate_salt()
            iv = self.generate_iv()
            
            # Derive encryption key
            key = self.derive_key(password, salt)
            
            # Apply PKCS7 padding
            padder = padding.PKCS7(128).padder()  # 128-bit block size for AES
            padded_data = padder.update(data) + padder.finalize()
            
            # Create cipher and encrypt
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Clear sensitive data from memory
            key = b'\x00' * len(key)
            
            self.logger.debug(f"Encrypted {len(data)} bytes")
            return encrypted_data, salt, iv
            
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def decrypt(self, encrypted_data: bytes, password: Union[str, bytes], 
                salt: bytes, iv: bytes) -> bytes:
        """Decrypt data using AES-256-CBC.
        
        Args:
            encrypted_data: Data to decrypt
            password: Password for key derivation
            salt: Salt used during encryption
            iv: Initialization vector used during encryption
            
        Returns:
            Decrypted data
        """
        try:
            # Derive decryption key
            key = self.derive_key(password, salt)
            
            # Create cipher and decrypt
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Remove PKCS7 padding
            unpadder = padding.PKCS7(128).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
            
            # Clear sensitive data from memory
            key = b'\x00' * len(key)
            
            self.logger.debug(f"Decrypted {len(data)} bytes")
            return data
            
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def encrypt_with_metadata(self, data: bytes, password: Union[str, bytes]) -> bytes:
        """Encrypt data and prepend metadata (salt + iv + encrypted_data).
        
        Args:
            data: Data to encrypt
            password: Password for encryption
            
        Returns:
            Complete encrypted package with metadata
        """
        try:
            encrypted_data, salt, iv = self.encrypt(data, password)
            
            # Package format: [SALT][IV][ENCRYPTED_DATA]
            package = salt + iv + encrypted_data
            
            self.logger.debug(f"Created encrypted package of {len(package)} bytes")
            return package
            
        except Exception as e:
            self.logger.error(f"Error creating encrypted package: {e}")
            raise
    
    def decrypt_with_metadata(self, package: bytes, password: Union[str, bytes]) -> bytes:
        """Decrypt data from package with embedded metadata.
        
        Args:
            package: Complete encrypted package
            password: Password for decryption
            
        Returns:
            Decrypted data
        """
        try:
            # Validate package size
            min_size = self.SALT_SIZE + self.IV_SIZE
            if len(package) < min_size:
                raise ValueError(f"Package too small: {len(package)} < {min_size}")
            
            # Extract components
            salt = package[:self.SALT_SIZE]
            iv = package[self.SALT_SIZE:self.SALT_SIZE + self.IV_SIZE]
            encrypted_data = package[self.SALT_SIZE + self.IV_SIZE:]
            
            # Decrypt
            data = self.decrypt(encrypted_data, password, salt, iv)
            
            self.logger.debug(f"Extracted data from {len(package)}-byte package")
            return data
            
        except Exception as e:
            self.logger.error(f"Error extracting from package: {e}")
            raise
    
    def verify_password(self, password: Union[str, bytes], package: bytes) -> bool:
        """Verify if password can decrypt the package.
        
        Args:
            password: Password to test
            package: Encrypted package to test against
            
        Returns:
            True if password is correct, False otherwise
        """
        try:
            # Try to decrypt a small portion to verify password
            self.decrypt_with_metadata(package, password)
            return True
            
        except Exception:
            return False
    
    def get_security_info(self) -> dict:
        """Get information about current security configuration."""
        return {
            'security_level': self.security_level.value,
            'iterations': self.config['iterations'],
            'key_size_bits': self.KEY_SIZE * 8,
            'algorithm': 'AES-256-CBC',
            'kdf': 'PBKDF2-HMAC-SHA256',
            'additional_entropy': self.config['additional_entropy']
        }
    
    def estimate_derivation_time(self, iterations: Optional[int] = None) -> float:
        """Estimate key derivation time in seconds.
        
        Args:
            iterations: Number of iterations (uses current config if None)
            
        Returns:
            Estimated time in seconds
        """
        if iterations is None:
            iterations = self.config['iterations']
        
        # Ensure we have a valid number for calculation
        if iterations is None or iterations <= 0:
            iterations = 100000  # Safe default
        
        # Very rough estimate: ~1000 iterations per millisecond on modern hardware
        # This varies greatly depending on the system
        base_time_ms = iterations / 1000
        return base_time_ms / 1000  # Convert to seconds
    
    def clear_memory(self):
        """Clear sensitive data from memory (placeholder for secure cleanup)."""
        # In a production environment, this would implement secure memory clearing
        # Python's garbage collector makes this challenging, but we can at least
        # ensure variables are overwritten
        pass
