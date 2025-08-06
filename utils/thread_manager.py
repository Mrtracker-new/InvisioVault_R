"""
Thread Management System
Handles background operations with progress reporting and cancellation support.
"""

import threading
import time
from typing import Callable, Any, Optional, Dict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication

from utils.logger import Logger
from utils.error_handler import ErrorHandler


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class BackgroundTask(QObject):
    """Background task with progress reporting."""
    
    # Signals for UI updates
    progress_updated = Signal(int)  # Progress percentage (0-100)
    status_updated = Signal(str)    # Status message
    completed = Signal(object)      # Result object
    failed = Signal(str)           # Error message
    cancelled = Signal()
    
    def __init__(self, task_id: str, function: Callable, *args, **kwargs):
        super().__init__()
        self.task_id = task_id
        self.function = function
        self.args = args
        self.kwargs = kwargs
        
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        
        self._cancelled = threading.Event()
        self.logger = Logger()
    
    def run(self):
        """Execute the background task."""
        try:
            self.status = TaskStatus.RUNNING
            self.start_time = time.time()
            self.status_updated.emit("Starting task...")
            
            # Add progress callback to kwargs if function supports it
            if 'progress_callback' not in self.kwargs:
                self.kwargs['progress_callback'] = self.update_progress
            
            # Add cancellation check to kwargs if function supports it
            if 'is_cancelled' not in self.kwargs:
                self.kwargs['is_cancelled'] = self.is_cancelled
            
            # Execute the function
            self.result = self.function(*self.args, **self.kwargs)
            
            if self.is_cancelled():
                self.status = TaskStatus.CANCELLED
                self.cancelled.emit()
                self.logger.debug(f"Task {self.task_id} was cancelled")
            else:
                self.status = TaskStatus.COMPLETED
                self.progress = 100
                self.progress_updated.emit(100)
                self.completed.emit(self.result)
                self.logger.debug(f"Task {self.task_id} completed successfully")
                
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error = str(e)
            self.failed.emit(self.error)
            self.logger.error(f"Task {self.task_id} failed: {e}")
            
        finally:
            self.end_time = time.time()
    
    def update_progress(self, progress: int, message: str = ""):
        """Update task progress.
        
        Args:
            progress: Progress percentage (0-100)
            message: Optional status message
        """
        self.progress = max(0, min(100, progress))
        self.progress_updated.emit(self.progress)
        
        if message:
            self.status_updated.emit(message)
    
    def cancel(self):
        """Cancel the task."""
        self._cancelled.set()
        self.logger.debug(f"Cancellation requested for task {self.task_id}")
    
    def is_cancelled(self) -> bool:
        """Check if task has been cancelled."""
        return self._cancelled.is_set()
    
    def get_duration(self) -> Optional[float]:
        """Get task execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class ThreadManager(QObject):
    """Background operation management with progress tracking."""
    
    # Signals for global task management
    task_started = Signal(str)     # Task ID
    task_completed = Signal(str)   # Task ID
    task_failed = Signal(str, str) # Task ID, error message
    task_cancelled = Signal(str)   # Task ID
    
    def __init__(self, max_workers: int = 4):
        super().__init__()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, BackgroundTask] = {}
        self.futures: Dict[str, Future] = {}
        
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        
        # Cleanup timer
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self.cleanup_completed_tasks)
        self.cleanup_timer.start(30000)  # Cleanup every 30 seconds
        
        self.logger.info(f"Thread manager initialized with {max_workers} workers")
    
    def submit_task(self, task_id: str, function: Callable, *args, **kwargs) -> BackgroundTask:
        """Submit a task for background execution.
        
        Args:
            task_id: Unique identifier for the task
            function: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            BackgroundTask object for monitoring
        """
        # Cancel existing task with same ID
        if task_id in self.tasks:
            self.cancel_task(task_id)
        
        # Create background task
        task = BackgroundTask(task_id, function, *args, **kwargs)
        
        # Connect signals
        task.completed.connect(lambda result: self._on_task_completed(task_id, result))
        task.failed.connect(lambda error: self._on_task_failed(task_id, error))
        task.cancelled.connect(lambda: self._on_task_cancelled(task_id))
        
        # Submit to executor
        future = self.executor.submit(task.run)
        
        # Store references
        self.tasks[task_id] = task
        self.futures[task_id] = future
        
        self.task_started.emit(task_id)
        self.logger.debug(f"Submitted task {task_id} for execution")
        
        return task
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled, False if not found
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.cancel()
            
            if task_id in self.futures:
                future = self.futures[task_id]
                future.cancel()
            
            self.logger.debug(f"Cancelled task {task_id}")
            return True
        
        return False
    
    def cancel_all_tasks(self):
        """Cancel all running tasks."""
        for task_id in list(self.tasks.keys()):
            self.cancel_task(task_id)
        
        self.logger.debug("Cancelled all tasks")
    
    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            BackgroundTask object or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_running_tasks(self) -> Dict[str, BackgroundTask]:
        """Get all currently running tasks.
        
        Returns:
            Dictionary of running tasks
        """
        return {tid: task for tid, task in self.tasks.items() 
                if task.status == TaskStatus.RUNNING}
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status.
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskStatus or None if not found
        """
        task = self.tasks.get(task_id)
        return task.status if task else None
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> bool:
        """Wait for a task to complete.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if task completed, False if timed out
        """
        if task_id in self.futures:
            try:
                self.futures[task_id].result(timeout=timeout)
                return True
            except Exception:
                return False
        
        return False
    
    def cleanup_completed_tasks(self):
        """Clean up completed, cancelled, or failed tasks."""
        to_remove = []
        
        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED]:
                # Keep task for a while after completion for result retrieval
                if task.end_time and (time.time() - task.end_time) > 300:  # 5 minutes
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            self._remove_task(task_id)
            self.logger.debug(f"Cleaned up completed task {task_id}")
    
    def _on_task_completed(self, task_id: str, result: Any):
        """Handle task completion."""
        self.task_completed.emit(task_id)
        self.logger.debug(f"Task {task_id} completed")
    
    def _on_task_failed(self, task_id: str, error: str):
        """Handle task failure."""
        self.task_failed.emit(task_id, error)
        self.logger.error(f"Task {task_id} failed: {error}")
    
    def _on_task_cancelled(self, task_id: str):
        """Handle task cancellation."""
        self.task_cancelled.emit(task_id)
        self.logger.debug(f"Task {task_id} cancelled")
    
    def _remove_task(self, task_id: str):
        """Remove task and its future."""
        if task_id in self.tasks:
            del self.tasks[task_id]
        if task_id in self.futures:
            del self.futures[task_id]
    
    def shutdown(self, wait: bool = True):
        """Shutdown the thread manager.
        
        Args:
            wait: Whether to wait for running tasks to complete
        """
        self.cleanup_timer.stop()
        
        if not wait:
            self.cancel_all_tasks()
        
        self.executor.shutdown(wait=wait)
        self.logger.info("Thread manager shutdown")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get thread manager statistics.
        
        Returns:
            Statistics dictionary
        """
        status_counts = {}
        for task in self.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'max_workers': self.max_workers,
            'total_tasks': len(self.tasks),
            'running_tasks': len(self.get_running_tasks()),
            'status_counts': status_counts
        }


class ProgressTracker:
    """Helper class for tracking multi-step operation progress."""
    
    def __init__(self, total_steps: int, progress_callback: Optional[Callable] = None):
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_callback = progress_callback
    
    def next_step(self, message: str = ""):
        """Move to the next step.
        
        Args:
            message: Optional progress message
        """
        self.current_step += 1
        progress = int((self.current_step / self.total_steps) * 100)
        
        if self.progress_callback:
            self.progress_callback(progress, message)
    
    def set_step(self, step: int, message: str = ""):
        """Set specific step.
        
        Args:
            step: Step number (0-based)
            message: Optional progress message
        """
        self.current_step = max(0, min(step, self.total_steps))
        progress = int((self.current_step / self.total_steps) * 100)
        
        if self.progress_callback:
            self.progress_callback(progress, message)
    
    def get_progress(self) -> int:
        """Get current progress percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        return int((self.current_step / self.total_steps) * 100)
