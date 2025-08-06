"""
Cryptographic Utilities
Provides cryptographic helper functions and secure random generation.
"""

import os
import secrets
import hashlib
import hmac
from typing import Tuple, Optional, Union, List
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

from utils.logger import Logger
from utils.error_handler import ErrorHandler


class CryptoUtils:
    """Cryptographic utility functions."""
    
    def __init__(self):
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.backend = default_backend()
    
    @staticmethod
    def generate_secure_random_bytes(length: int) -> bytes:
        """Generate cryptographically secure random bytes.
        
        Args:
            length: Number of bytes to generate
            
        Returns:
            Secure random bytes
        """
        return secrets.token_bytes(length)
    
    @staticmethod
    def generate_secure_random_int(bits: int = 32) -> int:
        """Generate cryptographically secure random integer.
        
        Args:
            bits: Number of bits for the integer
            
        Returns:
            Secure random integer
        """
        return secrets.randbits(bits)
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure token string.
        
        Args:
            length: Length of the token in bytes
            
        Returns:
            Secure hex token string
        """
        return secrets.token_hex(length)
    
    @staticmethod
    def hash_data(data: bytes, algorithm: str = 'sha256') -> bytes:
        """Hash data using specified algorithm.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm ('sha256', 'sha512', 'sha3_256', 'sha3_512')
            
        Returns:
            Hash digest
        """
        algorithms = {
            'sha256': hashlib.sha256,
            'sha512': hashlib.sha512,
            'sha3_256': hashlib.sha3_256,
            'sha3_512': hashlib.sha3_512,
            'blake2b': hashlib.blake2b,
            'blake2s': hashlib.blake2s
        }
        
        if algorithm not in algorithms:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        return algorithms[algorithm](data).digest()
    
    @staticmethod
    def hash_file(file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate hash of a file.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use
            
        Returns:
            Hex-encoded hash digest
        """
        algorithms = {
            'sha256': hashlib.sha256,
            'sha512': hashlib.sha512,
            'sha3_256': hashlib.sha3_256,
            'sha3_512': hashlib.sha3_512
        }
        
        if algorithm not in algorithms:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hasher = algorithms[algorithm]()
        
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    @staticmethod
    def hmac_digest(key: bytes, message: bytes, algorithm: str = 'sha256') -> bytes:
        """Generate HMAC digest.
        
        Args:
            key: Secret key
            message: Message to authenticate
            algorithm: Hash algorithm to use
            
        Returns:
            HMAC digest
        """
        algorithms = {
            'sha256': hashlib.sha256,
            'sha512': hashlib.sha512,
            'sha3_256': hashlib.sha3_256,
            'sha3_512': hashlib.sha3_512
        }
        
        if algorithm not in algorithms:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        return hmac.new(key, message, algorithms[algorithm]).digest()
    
    @staticmethod
    def verify_hmac(key: bytes, message: bytes, expected_digest: bytes, algorithm: str = 'sha256') -> bool:
        """Verify HMAC digest.
        
        Args:
            key: Secret key
            message: Original message
            expected_digest: Expected HMAC digest
            algorithm: Hash algorithm used
            
        Returns:
            True if HMAC is valid, False otherwise
        """
        try:
            calculated_digest = CryptoUtils.hmac_digest(key, message, algorithm)
            return hmac.compare_digest(calculated_digest, expected_digest)
        except Exception:
            return False
    
    def derive_key_pbkdf2(self, password: Union[str, bytes], salt: bytes, 
                         iterations: int = 100000, key_length: int = 32) -> bytes:
        """Derive key from password using PBKDF2.
        
        Args:
            password: Password to derive from
            salt: Salt value
            iterations: Number of iterations
            key_length: Desired key length in bytes
            
        Returns:
            Derived key
        """
        try:
            if isinstance(password, str):
                password = password.encode('utf-8')
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=iterations,
                backend=self.backend
            )
            
            return kdf.derive(password)
            
        except Exception as e:
            self.logger.error(f"Error deriving key: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate RSA key pair.
        
        Args:
            key_size: RSA key size in bits
            
        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=self.backend
            )
            
            public_key = private_key.public_key()
            
            # Serialize private key
            private_pem = private_key.private_key(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            # Serialize public key
            public_pem = public_key.public_key(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return private_pem, public_pem
            
        except Exception as e:
            self.logger.error(f"Error generating RSA keypair: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    @staticmethod
    def secure_compare(a: Union[str, bytes], b: Union[str, bytes]) -> bool:
        """Constant-time comparison to prevent timing attacks.
        
        Args:
            a: First value to compare
            b: Second value to compare
            
        Returns:
            True if values are equal, False otherwise
        """
        if isinstance(a, str):
            a = a.encode('utf-8')
        if isinstance(b, str):
            b = b.encode('utf-8')
        
        return hmac.compare_digest(a, b)
    
    @staticmethod
    def xor_bytes(data1: bytes, data2: bytes) -> bytes:
        """XOR two byte sequences.
        
        Args:
            data1: First byte sequence
            data2: Second byte sequence
            
        Returns:
            XORed result
        
        Raises:
            ValueError: If sequences have different lengths
        """
        if len(data1) != len(data2):
            raise ValueError("Byte sequences must have the same length")
        
        return bytes(a ^ b for a, b in zip(data1, data2))
    
    @staticmethod
    def pad_data(data: bytes, block_size: int, padding_char: bytes = b'\x00') -> bytes:
        """Pad data to block size.
        
        Args:
            data: Data to pad
            block_size: Target block size
            padding_char: Character to use for padding
            
        Returns:
            Padded data
        """
        padding_length = block_size - (len(data) % block_size)
        if padding_length == block_size:
            padding_length = 0
        
        return data + (padding_char * padding_length)
    
    @staticmethod
    def unpad_data(data: bytes, padding_char: bytes = b'\x00') -> bytes:
        """Remove padding from data.
        
        Args:
            data: Padded data
            padding_char: Padding character used
            
        Returns:
            Unpadded data
        """
        return data.rstrip(padding_char)
    
    @staticmethod
    def entropy_analysis(data: bytes) -> float:
        """Calculate Shannon entropy of data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Entropy value (0.0 to 8.0 for bytes)
        """
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_length = len(data)
        
        for count in byte_counts:
            if count > 0:
                probability = count / data_length
                entropy -= probability * (probability.bit_length() - 1)  # log2(probability)
        
        return entropy
    
    def generate_keyfile_content(self, size: int = 1024, high_entropy: bool = True) -> bytes:
        """Generate keyfile content with specified characteristics.
        
        Args:
            size: Size of keyfile in bytes
            high_entropy: Whether to generate high-entropy content
            
        Returns:
            Keyfile content
        """
        try:
            if high_entropy:
                # Generate truly random content
                content = self.generate_secure_random_bytes(size)
            else:
                # Generate pseudo-random content with patterns
                base_content = self.generate_secure_random_bytes(size // 4)
                content = (base_content * (size // len(base_content) + 1))[:size]
            
            self.logger.info(f"Generated keyfile content: {size} bytes, entropy: {self.entropy_analysis(content):.2f}")
            return content
            
        except Exception as e:
            self.logger.error(f"Error generating keyfile content: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    @staticmethod
    def split_secret(secret: bytes, threshold: int, shares: int) -> List[Tuple[int, bytes]]:
        """Split secret using Shamir's Secret Sharing (simplified version).
        
        Args:
            secret: Secret to split
            threshold: Minimum shares needed to reconstruct
            shares: Total number of shares to create
            
        Returns:
            List of (share_id, share_data) tuples
        
        Note:
            This is a simplified implementation for demonstration.
            Production use should employ a proper SSS library.
        """
        if threshold > shares:
            raise ValueError("Threshold cannot be greater than total shares")
        
        # Simple XOR-based secret sharing (not cryptographically secure)
        # This is just a placeholder - use proper SSS in production
        share_list = []
        
        # Generate random shares
        random_shares = []
        for i in range(shares - 1):
            random_shares.append(CryptoUtils.generate_secure_random_bytes(len(secret)))
        
        # Calculate the final share to make XOR work
        final_share = secret
        for share in random_shares:
            final_share = CryptoUtils.xor_bytes(final_share, share)
        
        # Create share list
        for i, share in enumerate(random_shares):
            share_list.append((i + 1, share))
        share_list.append((shares, final_share))
        
        return share_list
    
    @staticmethod
    def reconstruct_secret(shares: List[Tuple[int, bytes]]) -> bytes:
        """Reconstruct secret from shares (simplified version).
        
        Args:
            shares: List of (share_id, share_data) tuples
            
        Returns:
            Reconstructed secret
        
        Note:
            This is a simplified implementation matching split_secret.
        """
        if len(shares) < 2:
            raise ValueError("At least 2 shares required for reconstruction")
        
        # Simple XOR reconstruction
        secret = shares[0][1]
        for _, share_data in shares[1:]:
            secret = CryptoUtils.xor_bytes(secret, share_data)
        
        return secret
