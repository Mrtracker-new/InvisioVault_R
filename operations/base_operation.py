"""
Base Operation Class
Provides abstract base class for all steganography operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum

from utils.logger import Logger
from utils.error_handler import ErrorHandler
from core.file_manager import FileManager
from core.crypto_utils import CryptoUtils


class OperationStatus(Enum):
    """Status of operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OperationType(Enum):
    """Types of operations."""
    HIDE = "hide"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    BATCH = "batch"


class BaseOperation(ABC):
    """Abstract base class for all steganography operations."""
    
    def __init__(self, operation_type: OperationType, operation_id: Optional[str] = None):
        """Initialize base operation.
        
        Args:
            operation_type: Type of operation
            operation_id: Unique operation identifier
        """
        self.operation_type = operation_type
        self.operation_id = operation_id or self._generate_operation_id()
        self.status = OperationStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.progress = 0
        self.error_message: Optional[str] = None
        self.result_data: Optional[Dict[str, Any]] = None
        
        # Initialize utilities
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.file_manager = FileManager()
        self.crypto_utils = CryptoUtils()
        
        # Callbacks
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.status_callback: Optional[Callable[[str], None]] = None
        self.completion_callback: Optional[Callable[[bool, Optional[str]], None]] = None
    
    def _generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        return f"{self.operation_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.crypto_utils.generate_secure_random_int(16):04x}"
    
    def set_progress_callback(self, callback: Callable[[int], None]):
        """Set progress callback function."""
        self.progress_callback = callback
    
    def set_status_callback(self, callback: Callable[[str], None]):
        """Set status callback function."""
        self.status_callback = callback
    
    def set_completion_callback(self, callback: Callable[[bool, Optional[str]], None]):
        """Set completion callback function."""
        self.completion_callback = callback
    
    def update_progress(self, progress: int):
        """Update operation progress.
        
        Args:
            progress: Progress percentage (0-100)
        """
        self.progress = max(0, min(100, progress))
        if self.progress_callback:
            try:
                self.progress_callback(self.progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def update_status(self, status: str):
        """Update operation status message.
        
        Args:
            status: Status message
        """
        self.logger.info(f"[{self.operation_id}] {status}")
        if self.status_callback:
            try:
                self.status_callback(status)
            except Exception as e:
                self.logger.warning(f"Status callback failed: {e}")
    
    def start(self) -> bool:
        """Start the operation.
        
        Returns:
            True if operation started successfully, False otherwise
        """
        try:
            if self.status != OperationStatus.PENDING:
                self.logger.warning(f"Operation {self.operation_id} is not in pending state")
                return False
            
            self.status = OperationStatus.IN_PROGRESS
            self.started_at = datetime.now()
            self.progress = 0
            
            self.update_status(f"Starting {self.operation_type.value} operation")
            self.logger.info(f"Started operation: {self.operation_id}")
            
            # Validate inputs before execution
            validation_result = self.validate_inputs()
            if not validation_result:
                self.fail("Input validation failed")
                return False
            
            # Execute the operation
            success = self.execute()
            
            if success:
                self.complete()
            else:
                self.fail("Operation execution failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error starting operation: {e}")
            self.error_handler.handle_exception(e)
            self.fail(str(e))
            return False
    
    def complete(self, result_data: Optional[Dict[str, Any]] = None):
        """Mark operation as completed.
        
        Args:
            result_data: Optional result data
        """
        self.status = OperationStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 100
        
        if result_data:
            self.result_data = result_data
        
        duration = (self.completed_at - self.started_at).total_seconds() if self.started_at else 0
        self.update_status(f"Operation completed successfully in {duration:.2f}s")
        self.logger.info(f"Completed operation: {self.operation_id}")
        
        if self.completion_callback:
            try:
                self.completion_callback(True, None)
            except Exception as e:
                self.logger.warning(f"Completion callback failed: {e}")
    
    def fail(self, error_message: str):
        """Mark operation as failed.
        
        Args:
            error_message: Error description
        """
        self.status = OperationStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
        
        duration = (self.completed_at - self.started_at).total_seconds() if self.started_at else 0
        self.update_status(f"Operation failed after {duration:.2f}s: {error_message}")
        self.logger.error(f"Failed operation: {self.operation_id} - {error_message}")
        
        if self.completion_callback:
            try:
                self.completion_callback(False, error_message)
            except Exception as e:
                self.logger.warning(f"Completion callback failed: {e}")
    
    def cancel(self):
        """Cancel the operation."""
        if self.status == OperationStatus.IN_PROGRESS:
            self.status = OperationStatus.CANCELLED
            self.completed_at = datetime.now()
            
            self.update_status("Operation cancelled")
            self.logger.info(f"Cancelled operation: {self.operation_id}")
            
            if self.completion_callback:
                try:
                    self.completion_callback(False, "Operation cancelled")
                except Exception as e:
                    self.logger.warning(f"Completion callback failed: {e}")
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get comprehensive status information.
        
        Returns:
            Dictionary containing status information
        """
        duration = None
        if self.started_at:
            end_time = self.completed_at or datetime.now()
            duration = (end_time - self.started_at).total_seconds()
        
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type.value,
            'status': self.status.value,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': duration,
            'error_message': self.error_message,
            'has_result': self.result_data is not None
        }
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get operation result data.
        
        Returns:
            Result data if operation completed successfully, None otherwise
        """
        if self.status == OperationStatus.COMPLETED:
            return self.result_data
        return None
    
    def is_completed(self) -> bool:
        """Check if operation is completed (successfully or with error)."""
        return self.status in (OperationStatus.COMPLETED, OperationStatus.FAILED, OperationStatus.CANCELLED)
    
    def is_successful(self) -> bool:
        """Check if operation completed successfully."""
        return self.status == OperationStatus.COMPLETED
    
    def is_running(self) -> bool:
        """Check if operation is currently running."""
        return self.status == OperationStatus.IN_PROGRESS
    
    @abstractmethod
    def validate_inputs(self) -> bool:
        """Validate operation inputs.
        
        Returns:
            True if inputs are valid, False otherwise
        
        This method must be implemented by subclasses to validate
        their specific input requirements.
        """
        pass
    
    @abstractmethod
    def execute(self) -> bool:
        """Execute the main operation logic.
        
        Returns:
            True if operation succeeded, False otherwise
        
        This method must be implemented by subclasses to perform
        their specific operations.
        """
        pass
    
    def cleanup(self):
        """Cleanup operation resources.
        
        This method can be overridden by subclasses to perform
        specific cleanup tasks like removing temporary files.
        """
        try:
            # Default cleanup - remove temporary files
            self.file_manager.cleanup_temporary_files()
            self.logger.debug(f"Cleaned up resources for operation: {self.operation_id}")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        
        # If an exception occurred and operation is still running, mark as failed
        if exc_type is not None and self.status == OperationStatus.IN_PROGRESS:
            self.fail(f"Exception occurred: {exc_val}")
    
    def __str__(self) -> str:
        """String representation of operation."""
        return f"{self.__class__.__name__}(id={self.operation_id}, status={self.status.value}, progress={self.progress}%)"
    
    def __repr__(self) -> str:
        """Detailed representation of operation."""
        return (
            f"{self.__class__.__name__}("
            f"id='{self.operation_id}', "
            f"type={self.operation_type.value}, "
            f"status={self.status.value}, "
            f"progress={self.progress}%, "
            f"created={self.created_at.isoformat()})"
        )
