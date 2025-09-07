"""
Security Service Layer
Simplified security service for offline steganography application.

Author: Rolan (RNR)
Purpose: Minimal security service for local steganography operations
"""

import threading
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from utils.config_manager import ConfigManager
from utils.logger import Logger


class SecurityService:
    """Simplified security service for offline steganography application."""
    
    _instance: Optional['SecurityService'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'SecurityService':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize security service (called only once due to singleton)."""
        if hasattr(self, '_initialized'):
            return
        
        self.logger = Logger()
        self.config = ConfigManager()
        self._initialized = True
        
        self.logger.info("Security Service initialized (offline mode)")
    
    # === SIMPLIFIED AUTHENTICATION FOR OFFLINE USE ===
    
    def authenticate_user(self, password: str, user_id: str = "default") -> tuple[bool, str]:
        """Simplified authentication for offline use - always succeeds."""
        # For offline application, always allow access
        return True, "Authentication successful (offline mode)"
    
    def authenticate_user_with_keyfile(self, password: str, keyfile_path: Path, 
                                     user_id: str = "default") -> tuple[bool, str]:
        """Simplified keyfile authentication for offline use - always succeeds."""
        # For offline application, always allow access
        return True, "Keyfile authentication successful (offline mode)"
    
    def logout_user(self) -> bool:
        """Logout user - simplified for offline use."""
        return True
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated - always True for offline use."""
        return True
    
    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """Get current session information - simplified for offline use."""
        return {
            "session_id": "local_session",
            "user_id": "local",
            "security_level": "standard",
            "authentication_method": "local",
            "privileges": ["all"],
            "created_at": None,
            "last_activity": None
        }
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information - simplified for offline application.
        
        Returns:
            Dictionary with simplified session info (always authenticated for offline use)
        """
        # For offline application, always return authenticated state
        # No complex user management needed for local steganography tool
        return {
            "is_authenticated": True,
            "username": "LocalUser",
            "user_id": "local",
            "session_type": "local",
            "session_id": "local_session",
            "security_level": "standard",
            "authentication_method": "local",
            "privileges": ["all"],
            "created_at": None,
            "last_activity": None
        }
    
    # === SIMPLIFIED AUTHORIZATION FOR OFFLINE USE ===
    
    def can_perform_operation(self, operation: str) -> tuple[bool, str]:
        """Check if current user can perform an operation - always True for offline."""
        return True, "All operations allowed in offline mode"
    
    def require_authentication(self, operation: str = "access") -> tuple[bool, str]:
        """Check authentication requirement - always True for offline."""
        return True, "Access granted (offline mode)"
    
    # === SIMPLIFIED SECURITY STATUS ===
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get simplified security status for offline application."""
        return {
            "authenticated": True,
            "mode": "offline",
            "current_session": self.get_current_session_info(),
            "available_operations": ["hide", "extract", "analyze", "settings"]
        }
    
    def get_security_recommendations(self) -> list[str]:
        """Get basic security recommendations for offline use."""
        return [
            "Use strong passwords for file encryption",
            "Keep keyfiles secure and backed up",
            "Use different passwords for different files",
            "Verify file integrity after steganography operations"
        ]
    
    def generate_security_report(self) -> str:
        """Generate simplified security report for offline mode."""
        return "Security Status: Offline Mode - All operations available locally."
    
    # === SIMPLIFIED UTILITY METHODS ===
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Basic password strength validation for offline use."""
        length = len(password)
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        score = 0
        if length >= 8: score += 1
        if length >= 12: score += 1
        if has_upper and has_lower: score += 1
        if has_digit: score += 1
        if has_special: score += 1
        
        strength_levels = ["Very Weak", "Weak", "Fair", "Good", "Strong"]
        strength = strength_levels[min(score, 4)]
        
        return {
            "valid": length >= 6,
            "strength": strength,
            "score": score,
            "length": length,
            "suggestions": [] if score >= 4 else ["Use at least 8 characters", "Include uppercase and lowercase letters", "Add numbers and special characters"]
        }


# Convenience functions for global access
_security_service: Optional[SecurityService] = None


def get_security_service() -> SecurityService:
    """Get the global security service instance."""
    global _security_service
    if _security_service is None:
        _security_service = SecurityService()
    return _security_service


def requires_auth(operation: str = "access"):
    """Decorator that requires authentication for a function.
    
    Args:
        operation: Operation being performed
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            security = get_security_service()
            authorized, message = security.require_authentication(operation)
            if not authorized:
                raise PermissionError(message)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def is_authenticated() -> bool:
    """Quick check if user is authenticated."""
    return get_security_service().is_authenticated()


def get_current_user() -> Optional[str]:
    """Get current authenticated user ID."""
    session_info = get_security_service().get_current_session_info()
    return session_info["user_id"] if session_info else None
