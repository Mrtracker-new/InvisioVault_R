"""
Centralized Error Management System
Provides comprehensive error handling with user-friendly messages.
"""

import sys
import traceback
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from enum import Enum


class ErrorCategory(Enum):
    """Error categories for classification."""
    FILE_ACCESS = "file_access"
    ENCRYPTION_DECRYPTION = "encryption_decryption"
    IMAGE_PROCESSING = "image_processing"
    NETWORK_IO = "network_io"
    USER_INPUT = "user_input"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InvisioVaultError(Exception):
    """Base exception class for InvisioVault."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 technical_details: Optional[str] = None,
                 recovery_suggestion: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.technical_details = technical_details
        self.recovery_suggestion = recovery_suggestion
        self.timestamp = datetime.now()


class FileAccessError(InvisioVaultError):
    """File access related errors."""
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.FILE_ACCESS, **kwargs)
        self.file_path = file_path


class EncryptionError(InvisioVaultError):
    """Encryption/decryption related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.ENCRYPTION_DECRYPTION, **kwargs)


class ImageProcessingError(InvisioVaultError):
    """Image processing related errors."""
    def __init__(self, message: str, image_path: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.IMAGE_PROCESSING, **kwargs)
        self.image_path = image_path


class UserInputError(InvisioVaultError):
    """User input validation errors."""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.USER_INPUT, severity=ErrorSeverity.LOW, **kwargs)
        self.field = field


class ErrorHandler:
    """Centralized error management system."""
    
    def __init__(self):
        self.error_callbacks: Dict[ErrorCategory, list] = {}
        self.error_count: Dict[ErrorCategory, int] = {}
        self.recent_errors = []
        self.max_recent_errors = 100
        
        # Initialize error counts
        for category in ErrorCategory:
            self.error_count[category] = 0
    
    def register_callback(self, category: ErrorCategory, callback: Callable):
        """Register callback for specific error category."""
        if category not in self.error_callbacks:
            self.error_callbacks[category] = []
        self.error_callbacks[category].append(callback)
    
    def handle_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> InvisioVaultError:
        """Handle any exception and convert to InvisioVaultError."""
        
        # If it's already an InvisioVaultError, just process it
        if isinstance(exception, InvisioVaultError):
            error = exception
        else:
            # Convert standard exceptions to InvisioVaultError
            error = self._convert_exception(exception, context)
        
        # Record the error
        self._record_error(error)
        
        # Execute callbacks
        self._execute_callbacks(error)
        
        return error
    
    def _convert_exception(self, exception: Exception, context: Optional[Dict[str, Any]]) -> InvisioVaultError:
        """Convert standard exceptions to InvisioVaultError."""
        
        exc_type = type(exception)
        message = str(exception)
        technical_details = traceback.format_exc()
        
        # Map common exceptions to categories
        if exc_type in (FileNotFoundError, PermissionError, OSError, IOError):
            return FileAccessError(
                message=self._get_user_friendly_message(exception),
                technical_details=technical_details,
                recovery_suggestion=self._get_recovery_suggestion(exception)
            )
        
        elif exc_type in (ValueError, TypeError):
            if 'password' in message.lower() or 'key' in message.lower():
                return EncryptionError(
                    message="Encryption/decryption failed",
                    technical_details=technical_details,
                    recovery_suggestion="Please check your password and try again"
                )
            else:
                return UserInputError(
                    message=self._get_user_friendly_message(exception),
                    technical_details=technical_details,
                    recovery_suggestion="Please check your input and try again"
                )
        
        elif 'PIL' in str(exc_type) or 'image' in message.lower():
            return ImageProcessingError(
                message="Image processing failed",
                technical_details=technical_details,
                recovery_suggestion="Please try with a different image file"
            )
        
        elif exc_type in (ConnectionError, TimeoutError):
            return InvisioVaultError(
                message="Network operation failed",
                category=ErrorCategory.NETWORK_IO,
                technical_details=technical_details,
                recovery_suggestion="Please check your internet connection and try again"
            )
        
        else:
            # Generic system error
            return InvisioVaultError(
                message="An unexpected error occurred",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.HIGH,
                technical_details=technical_details,
                recovery_suggestion="Please try again or contact support if the problem persists"
            )
    
    def _get_user_friendly_message(self, exception: Exception) -> str:
        """Convert technical exception messages to user-friendly ones."""
        
        message = str(exception).lower()
        
        if 'no such file or directory' in message or 'file not found' in message:
            return "The specified file could not be found"
        
        elif 'permission denied' in message:
            return "Permission denied - please check file permissions"
        
        elif 'disk space' in message or 'no space left' in message:
            return "Insufficient disk space available"
        
        elif 'invalid' in message and 'password' in message:
            return "Invalid password provided"
        
        elif 'corrupt' in message or 'damaged' in message:
            return "The file appears to be corrupted or damaged"
        
        elif 'unsupported' in message:
            return "This file format is not supported"
        
        elif 'timeout' in message:
            return "The operation timed out"
        
        else:
            return str(exception)
    
    def _get_recovery_suggestion(self, exception: Exception) -> str:
        """Provide recovery suggestions based on exception type."""
        
        exc_type = type(exception)
        message = str(exception).lower()
        
        if exc_type == FileNotFoundError:
            return "Please check the file path and ensure the file exists"
        
        elif exc_type == PermissionError:
            return "Please check file permissions or run as administrator"
        
        elif 'disk space' in message:
            return "Please free up disk space and try again"
        
        elif 'password' in message:
            return "Please verify your password and try again"
        
        elif 'image' in message:
            return "Please try with a different image file in PNG, BMP, or TIFF format"
        
        else:
            return "Please try the operation again or contact support"
    
    def _record_error(self, error: InvisioVaultError):
        """Record error for statistics and recent error tracking."""
        
        # Update count
        self.error_count[error.category] += 1
        
        # Add to recent errors
        self.recent_errors.append({
            'timestamp': error.timestamp,
            'category': error.category,
            'severity': error.severity,
            'message': error.message,
            'technical_details': error.technical_details,
            'recovery_suggestion': error.recovery_suggestion
        })
        
        # Keep only recent errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
    
    def _execute_callbacks(self, error: InvisioVaultError):
        """Execute registered callbacks for error category."""
        
        callbacks = self.error_callbacks.get(error.category, [])
        for callback in callbacks:
            try:
                callback(error)
            except Exception as e:
                # Don't let callback errors crash the system
                print(f"Error in error callback: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'total_errors': sum(self.error_count.values()),
            'by_category': {cat.value: count for cat, count in self.error_count.items()},
            'recent_errors_count': len(self.recent_errors),
            'most_common_category': max(self.error_count.items(), key=lambda x: x[1], default=(ErrorCategory.SYSTEM, 0))[0].value if self.error_count else None
        }
    
    def get_recent_errors(self, count: int = 10) -> list:
        """Get recent errors."""
        return self.recent_errors[-count:] if count < len(self.recent_errors) else self.recent_errors
    
    def clear_error_history(self):
        """Clear error history and statistics."""
        self.recent_errors.clear()
        for category in ErrorCategory:
            self.error_count[category] = 0
    
    def create_error_report(self) -> str:
        """Create a comprehensive error report."""
        report_lines = [
            "=== InvisioVault Error Report ===",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "=== Error Statistics ==="
        ]
        
        stats = self.get_error_statistics()
        report_lines.append(f"Total Errors: {stats['total_errors']}")
        report_lines.append("")
        
        report_lines.append("Errors by Category:")
        for category, count in stats['by_category'].items():
            report_lines.append(f"  {category}: {count}")
        
        report_lines.append("")
        report_lines.append("=== Recent Errors ===")
        
        for error in self.recent_errors[-10:]:  # Last 10 errors
            report_lines.extend([
                f"Time: {error['timestamp'].isoformat()}",
                f"Category: {error['category'].value}",
                f"Severity: {error['severity'].value}",
                f"Message: {error['message']}",
                f"Recovery: {error['recovery_suggestion']}",
                "---"
            ])
        
        return "\n".join(report_lines)
    
    @staticmethod
    def setup_global_exception_handler():
        """Set up global exception handler for unhandled exceptions."""
        def handle_global_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Allow Ctrl+C to work normally
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            # Handle other exceptions
            handler = ErrorHandler()
            error = handler.handle_exception(exc_value)
            print(f"Unhandled error: {error.message}")
            if error.recovery_suggestion:
                print(f"Suggestion: {error.recovery_suggestion}")
        
        sys.excepthook = handle_global_exception
