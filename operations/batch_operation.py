"""
Batch Operation
Manages batch processing of multiple operations.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import uuid

from operations.base_operation import BaseOperation, OperationStatus
from operations.hide_operation import HideOperation
from operations.extract_operation import ExtractOperation
from operations.analysis_operation import AnalysisOperation
from utils.logger import Logger
from utils.error_handler import ErrorHandler, UserInputError, InvisioVaultError


class BatchOperationType:
    """Batch operation types."""
    HIDE = 'hide'
    EXTRACT = 'extract'
    ANALYSIS = 'analysis'


class BatchOperation(BaseOperation):
    """Batch operation for processing multiple steganographic operations."""
    
    def __init__(self, operation_id: Optional[str] = None):
        super().__init__(operation_id)
        
        # Remove duplicate initialization (inherited from BaseOperation)
        # Note: logger, error_handler are inherited from BaseOperation
        
        # Batch configuration
        self.operation_type: str = BatchOperationType.HIDE
        self.batch_items: List[Dict[str, Any]] = []
        self.continue_on_error: bool = True
        self.max_concurrent: int = 1  # For future parallel processing
        
        # Results
        self.completed_operations: List[Dict[str, Any]] = []
        self.failed_operations: List[Dict[str, Any]] = []
        self.total_success: int = 0
        self.total_failed: int = 0
        self.batch_results: Dict[str, Any] = {}
    
    def configure(self, operation_type: str, batch_items: List[Dict[str, Any]], 
                 continue_on_error: bool = True, max_concurrent: int = 1):
        """Configure the batch operation.
        
        Args:
            operation_type: Type of operation ('hide', 'extract', 'analysis')
            batch_items: List of operation configurations
            continue_on_error: Whether to continue processing after errors
            max_concurrent: Maximum number of concurrent operations
        """
        try:
            self.operation_type = operation_type
            self.batch_items = batch_items
            self.continue_on_error = continue_on_error
            self.max_concurrent = max_concurrent
            
            self.logger.info(f"Batch operation configured: {len(batch_items)} items, type: {operation_type}")
            
        except Exception as e:
            self.logger.error(f"Error configuring batch operation: {e}")
            self.error_handler.handle_exception(e)
            raise
    
    def validate_inputs(self) -> bool:
        """Validate batch operation inputs.
        
        Returns:
            True if inputs are valid
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if not self.batch_items:
                raise UserInputError("No batch items specified", field="batch_items")
            
            if self.operation_type not in [BatchOperationType.HIDE, 
                                         BatchOperationType.EXTRACT, 
                                         BatchOperationType.ANALYSIS]:
                raise UserInputError(f"Invalid operation type: {self.operation_type}", field="operation_type")
            
            # Validate each batch item based on operation type
            for i, item in enumerate(self.batch_items):
                self._validate_batch_item(item, i)
            
            self.logger.info("Batch operation inputs validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise
    
    def execute(self) -> bool:
        """Execute the batch operation (required by BaseOperation).
        
        Returns:
            True if operation succeeded, False otherwise
        """
        try:
            total_items = len(self.batch_items)
            
            # Process each batch item
            for i, item in enumerate(self.batch_items):
                try:
                    self.update_status(f"Processing item {i+1}/{total_items}...")
                    
                    # Create and execute operation
                    operation = self._create_operation(item)
                    result = operation.execute()
                    
                    # Store successful result
                    self.completed_operations.append({
                        'item_index': i,
                        'item_config': item,
                        'operation_id': operation.operation_id,
                        'result': result,
                        'status': 'success'
                    })
                    self.total_success += 1
                    
                except Exception as e:
                    error_info = {
                        'item_index': i,
                        'item_config': item,
                        'error': str(e),
                        'status': 'failed'
                    }
                    self.failed_operations.append(error_info)
                    self.total_failed += 1
                    
                    self.logger.error(f"Batch item {i+1} failed: {e}")
                    
                    if not self.continue_on_error:
                        return False
                
                # Update progress
                self.update_progress(int((i + 1) / total_items * 100))
            
            # Store final results
            self.batch_results = {
                'success': self.total_failed == 0,
                'operation_type': self.operation_type,
                'total_items': total_items,
                'total_success': self.total_success,
                'total_failed': self.total_failed,
                'success_rate': (self.total_success / total_items) * 100 if total_items > 0 else 0,
                'completed_operations': self.completed_operations,
                'failed_operations': self.failed_operations,
                'operation_id': self.operation_id,
                'completed_at': datetime.now().isoformat()
            }
            
            self.logger.info(
                f"Batch operation completed: {self.total_success}/{total_items} succeeded, "
                f"{self.total_failed} failed"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Batch operation failed: {e}")
            self.error_handler.handle_exception(e)
            return False
    
    def run_batch(self, progress_callback: Optional[Callable[[float], None]] = None,
                 status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Run the batch operation with callbacks.
        
        Args:
            progress_callback: Callback for progress updates
            status_callback: Callback for status updates
            
        Returns:
            Dictionary with batch results
        """
        try:
            # Set up callbacks
            if progress_callback:
                self.set_progress_callback(lambda p: progress_callback(p / 100.0))
            if status_callback:
                self.set_status_callback(status_callback)
            
            # Start the operation using the base class method
            success = self.start()
            
            if success:
                return self.batch_results
            else:
                raise InvisioVaultError(f"Batch operation failed: {self.error_message}")
            
        except Exception as e:
            self.logger.error(f"Batch operation failed: {e}")
            self.error_handler.handle_exception(e)
            raise InvisioVaultError(f"Batch operation failed: {e}")
    
    def _validate_batch_item(self, item: Dict[str, Any], index: int):
        """Validate a single batch item.
        
        Args:
            item: Batch item configuration
            index: Item index for error reporting
        """
        if self.operation_type == BatchOperationType.HIDE:
            required_fields = ['cover_image_path', 'files_to_hide', 'output_image_path']
        elif self.operation_type == BatchOperationType.EXTRACT:
            required_fields = ['steganographic_image_path', 'output_directory']
        elif self.operation_type == BatchOperationType.ANALYSIS:
            required_fields = ['image_path']
        else:
            raise UserInputError(f"Unknown operation type: {self.operation_type}", field="operation_type")
        
        for field in required_fields:
            if field not in item:
                raise UserInputError(f"Missing required field '{field}' in batch item {index}", field=field)
        
        # Validate paths exist
        if self.operation_type == BatchOperationType.HIDE:
            if not Path(item['cover_image_path']).exists():
                raise UserInputError(f"Cover image not found in batch item {index}: {item['cover_image_path']}", field="cover_image_path")
            
            for file_path in item['files_to_hide']:
                if not Path(file_path).exists():
                    raise UserInputError(f"File to hide not found in batch item {index}: {file_path}", field="files_to_hide")
        
        elif self.operation_type == BatchOperationType.EXTRACT:
            if not Path(item['steganographic_image_path']).exists():
                raise UserInputError(
                    f"Steganographic image not found in batch item {index}: {item['steganographic_image_path']}",
                    field="steganographic_image_path"
                )
        
        elif self.operation_type == BatchOperationType.ANALYSIS:
            if not Path(item['image_path']).exists():
                raise UserInputError(f"Image not found in batch item {index}: {item['image_path']}", field="image_path")
    
    def _create_operation(self, item: Dict[str, Any]) -> BaseOperation:
        """Create an operation instance based on batch item.
        
        Args:
            item: Batch item configuration
            
        Returns:
            Configured operation instance
        """
        operation_id = f"{self.operation_id}_item_{uuid.uuid4().hex[:8]}"
        
        if self.operation_type == BatchOperationType.HIDE:
            operation = HideOperation(operation_id)
            operation.configure(
                cover_image_path=item['cover_image_path'],
                files_to_hide=item['files_to_hide'],
                output_image_path=item['output_image_path'],
                password=item.get('password'),
                use_encryption=item.get('use_encryption', True),
                compression_type=item.get('compression_type', 'gzip'),
                compression_level=item.get('compression_level', 6),
                two_factor_enabled=item.get('two_factor_enabled', False),
                keyfile_path=item.get('keyfile_path'),
                use_decoy=item.get('use_decoy', False),
                decoy_data=item.get('decoy_data')
            )
        
        elif self.operation_type == BatchOperationType.EXTRACT:
            operation = ExtractOperation(operation_id)
            operation.configure(
                steganographic_image_path=item['steganographic_image_path'],
                output_directory=item['output_directory'],
                password=item.get('password'),
                two_factor_enabled=item.get('two_factor_enabled', False),
                keyfile_path=item.get('keyfile_path'),
                expected_compression=item.get('expected_compression'),
                detect_decoy=item.get('detect_decoy', False)
            )
        
        elif self.operation_type == BatchOperationType.ANALYSIS:
            operation = AnalysisOperation(operation_id)
            operation.configure(
                image_path=item['image_path'],
                analysis_level=item.get('analysis_level', 'basic')
            )
        
        else:
            raise InvisioVaultError(f"Unknown operation type: {self.operation_type}")
        
        # Validate the operation
        operation.validate_inputs()
        
        return operation
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get batch operation summary.
        
        Returns:
            Dictionary with batch operation summary
        """
        return {
            'operation_type': 'batch',
            'batch_operation_type': self.operation_type,
            'operation_id': self.operation_id,
            'status': self.status.value,
            'total_items': len(self.batch_items),
            'total_success': self.total_success,
            'total_failed': self.total_failed,
            'success_rate': (self.total_success / len(self.batch_items)) * 100 if self.batch_items else 0,
            'continue_on_error': self.continue_on_error,
            'progress': self.progress,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """Get detailed batch results including all operations.
        
        Returns:
            Dictionary with detailed results
        """
        return {
            'summary': self.get_operation_summary(),
            'completed_operations': self.completed_operations,
            'failed_operations': self.failed_operations
        }
    
    def retry_failed_operations(self, progress_callback: Optional[Callable[[float], None]] = None,
                               status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Retry failed operations.
        
        Args:
            progress_callback: Callback for progress updates
            status_callback: Callback for status updates
            
        Returns:
            Dictionary with retry results
        """
        if not self.failed_operations:
            return {'message': 'No failed operations to retry'}
        
        retry_items = [op['item_config'] for op in self.failed_operations]
        retry_batch = BatchOperation()
        retry_batch.configure(self.operation_type, retry_items, self.continue_on_error)
        retry_batch.validate_inputs()
        
        return retry_batch.run_batch(progress_callback, status_callback)
