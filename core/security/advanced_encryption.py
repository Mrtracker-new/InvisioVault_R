"""
Advanced Encryption Engine with Keyfile Authentication
Extends the standard encryption engine with two-factor (password + keyfile) authentication.
"""

import os
import secrets
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Union

from core.security.encryption_engine import EncryptionEngine, SecurityLevel
from utils.logger import Logger
from utils.error_handler import ErrorHandler, EncryptionError


class AdvancedEncryptionEngine(EncryptionEngine):
    """Keyfile-based two-factor authentication for encryption.
    
    Inherits from EncryptionEngine and adds keyfile functionality.
    """
    
    KEYFILE_HEADER = b'INVK'  # InvisioVault Keyfile
    VERSION = b'\x01\x00'
    MIN_KEYFILE_SIZE = 256 * 1024  # 256KB
    MAX_KEYFILE_SIZE = 1 * 1024 * 1024 # 1MB
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        super().__init__(security_level)
        self.logger.info("Advanced Encryption Engine initialized")
    
    def generate_keyfile(self, file_path: Path, size_kb: int = 256) -> bool:
        """Generate a cryptographically secure keyfile.
        
        Args:
            file_path: Path to save the keyfile
            size_kb: Size of the keyfile in kilobytes (256-1024)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            size_bytes = size_kb * 1024
            if not (self.MIN_KEYFILE_SIZE <= size_bytes <= self.MAX_KEYFILE_SIZE):
                raise ValueError(f"Keyfile size must be between {self.MIN_KEYFILE_SIZE//1024}KB and {self.MAX_KEYFILE_SIZE//1024}KB")
            
            # Generate random data
            random_data = secrets.token_bytes(size_bytes - len(self.KEYFILE_HEADER) - len(self.VERSION))
            
            # Create keyfile with header
            with open(file_path, 'wb') as f:
                f.write(self.KEYFILE_HEADER)
                f.write(self.VERSION)
                f.write(random_data)
            
            self.logger.info(f"Successfully generated keyfile at {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating keyfile: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def validate_keyfile(self, file_path: Path) -> bool:
        """Validate the integrity and format of a keyfile.
        
        Args:
            file_path: Path to the keyfile
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not file_path.exists():
                self.logger.error(f"Keyfile not found: {file_path}")
                return False
            
            file_size = file_path.stat().st_size
            if not (self.MIN_KEYFILE_SIZE <= file_size <= self.MAX_KEYFILE_SIZE):
                self.logger.error(f"Invalid keyfile size: {file_size} bytes")
                return False
            
            with open(file_path, 'rb') as f:
                header = f.read(len(self.KEYFILE_HEADER))
                version = f.read(len(self.VERSION))
            
            if header != self.KEYFILE_HEADER:
                self.logger.error("Invalid keyfile header")
                return False
            
            if version != self.VERSION:
                self.logger.warning(f"Keyfile version mismatch: expected {self.VERSION}, got {version}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating keyfile: {e}")
            return False
    
    def get_keyfile_hash(self, file_path: Path) -> Optional[bytes]:
        """Get SHA-256 hash of keyfile content for verification.
        
        Args:
            file_path: Path to the keyfile
        
        Returns:
            SHA-256 hash of the keyfile content, or None on error
        """
        try:
            if not self.validate_keyfile(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                keyfile_content = f.read()
            
            return hashlib.sha256(keyfile_content).digest()
            
        except Exception as e:
            self.logger.error(f"Error hashing keyfile: {e}")
            return None
    
    def derive_key_with_keyfile(self, password: Union[str, bytes], keyfile_path: Path, salt: bytes) -> bytes:
        """Derive encryption key using both password and keyfile.
        
        Args:
            password: User password
            keyfile_path: Path to the keyfile
            salt: Salt for key derivation
            
        Returns:
            Derived encryption key
        """
        try:
            if not self.validate_keyfile(keyfile_path):
                raise EncryptionError("Invalid or corrupted keyfile provided")
            
            # Derive initial key from password
            password_key = self.derive_key(password, salt)
            
            # Get keyfile hash
            keyfile_hash = self.get_keyfile_hash(keyfile_path)
            if not keyfile_hash:
                raise EncryptionError("Failed to process keyfile")
            
            # Combine password key and keyfile hash for final key
            # Using HMAC for a secure combination
            combined_key = hashlib.pbkdf2_hmac(
                'sha256',
                password=password_key,
                salt=keyfile_hash,
                iterations=1,  # Only one iteration needed for mixing
                dklen=self.KEY_SIZE
            )
            
            self.logger.debug("Key derived with password and keyfile")
            return combined_key
            
        except Exception as e:
            self.logger.error(f"Error deriving key with keyfile: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def encrypt_with_keyfile(self, data: bytes, password: Union[str, bytes], keyfile_path: Path) -> bytes:
        """Encrypt data using two-factor (password + keyfile) authentication.
        
        Args:
            data: Data to encrypt
            password: User password
            keyfile_path: Path to the keyfile
            
        Returns:
            Complete encrypted package with metadata
        """
        try:
            salt = self.generate_salt()
            iv = self.generate_iv()
            
            # Derive key with keyfile
            key = self.derive_key_with_keyfile(password, keyfile_path, salt)
            
            # Encrypt data (re-implementing to use the new key)
            from cryptography.hazmat.primitives import padding
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()
            
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Clear sensitive data
            key = b'\x00' * len(key)
            
            # Package format: [SALT][IV][ENCRYPTED_DATA]
            package = salt + iv + encrypted_data
            
            self.logger.debug(f"Created 2FA encrypted package of {len(package)} bytes")
            return package
            
        except Exception as e:
            self.logger.error(f"Encryption with keyfile failed: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def decrypt_with_keyfile(self, package: bytes, password: Union[str, bytes], keyfile_path: Path) -> bytes:
        """Decrypt data using two-factor (password + keyfile) authentication.
        
        Args:
            package: Encrypted package
            password: User password
            keyfile_path: Path to the keyfile
            
        Returns:
            Decrypted data
        """
        try:
            # Extract components from package
            min_size = self.SALT_SIZE + self.IV_SIZE
            if len(package) < min_size:
                raise ValueError(f"Package too small: {len(package)} < {min_size}")
            
            salt = package[:self.SALT_SIZE]
            iv = package[self.SALT_SIZE:self.SALT_SIZE + self.IV_SIZE]
            encrypted_data = package[self.SALT_SIZE + self.IV_SIZE:]
            
            # Derive key with keyfile
            key = self.derive_key_with_keyfile(password, keyfile_path, salt)
            
            # Decrypt data (re-implementing to use the new key)
            from cryptography.hazmat.primitives import padding
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            unpadder = padding.PKCS7(128).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
            
            # Clear sensitive data
            key = b'\x00' * len(key)
            
            self.logger.debug(f"Decrypted 2FA package successfully")
            return data
            
        except Exception as e:
            self.logger.error(f"Decryption with keyfile failed: {e}")
            self.error_handler.handle_exception(e)
            raise
