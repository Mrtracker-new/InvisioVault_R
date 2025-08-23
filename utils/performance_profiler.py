"""
Performance Profiler
Utility for measuring and analyzing performance bottlenecks in InVisioVault.
"""

import time
import functools
from typing import Dict, List, Callable, Any, Optional
from contextlib import contextmanager
from pathlib import Path

from utils.logger import Logger


class PerformanceProfiler:
    """Singleton performance profiler for timing operations."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(PerformanceProfiler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize profiler (only once)."""
        if not self._initialized:
            self.logger = Logger()
            self.measurements = {}  # operation_name -> list of times
            self.active_timers = {}  # timer_name -> start_time
            PerformanceProfiler._initialized = True
    
    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to ms
            self.record_measurement(operation_name, duration)
    
    def record_measurement(self, operation_name: str, duration_ms: float):
        """Record a performance measurement."""
        if operation_name not in self.measurements:
            self.measurements[operation_name] = []
        
        self.measurements[operation_name].append(duration_ms)
        
        # Log slow operations (>100ms)
        if duration_ms > 100:
            self.logger.warning(f"Slow operation detected: {operation_name} took {duration_ms:.2f}ms")
    
    def start_timer(self, timer_name: str):
        """Start a named timer."""
        self.active_timers[timer_name] = time.perf_counter()
    
    def end_timer(self, timer_name: str) -> float:
        """End a named timer and return duration in ms."""
        if timer_name not in self.active_timers:
            self.logger.warning(f"Timer '{timer_name}' not found")
            return 0.0
        
        start_time = self.active_timers.pop(timer_name)
        duration = (time.perf_counter() - start_time) * 1000
        self.record_measurement(timer_name, duration)
        return duration
    
    def get_statistics(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""
        if operation_name not in self.measurements:
            return {}
        
        times = self.measurements[operation_name]
        return {
            'count': len(times),
            'total_ms': sum(times),
            'average_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'median_ms': sorted(times)[len(times) // 2]
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_statistics(op) for op in self.measurements.keys()}
    
    def print_report(self):
        """Print a performance report."""
        print("\n" + "="*60)
        print("PERFORMANCE PROFILER REPORT")
        print("="*60)
        
        if not self.measurements:
            print("No measurements recorded.")
            return
        
        # Sort operations by average time (slowest first)
        stats = self.get_all_statistics()
        sorted_ops = sorted(stats.items(), key=lambda x: x[1].get('average_ms', 0), reverse=True)
        
        for operation, data in sorted_ops:
            print(f"\n{operation}:")
            print(f"  Count: {data['count']}")
            print(f"  Average: {data['average_ms']:.2f}ms")
            print(f"  Total: {data['total_ms']:.2f}ms")
            print(f"  Range: {data['min_ms']:.2f}ms - {data['max_ms']:.2f}ms")
            print(f"  Median: {data['median_ms']:.2f}ms")
    
    def save_report(self, file_path: Path):
        """Save performance report to file."""
        stats = self.get_all_statistics()
        
        with open(file_path, 'w') as f:
            f.write("Performance Profiler Report\n")
            f.write("=" * 30 + "\n\n")
            
            for operation, data in sorted(stats.items(), key=lambda x: x[1].get('average_ms', 0), reverse=True):
                f.write(f"{operation}:\n")
                f.write(f"  Count: {data['count']}\n")
                f.write(f"  Average: {data['average_ms']:.2f}ms\n")
                f.write(f"  Total: {data['total_ms']:.2f}ms\n")
                f.write(f"  Range: {data['min_ms']:.2f}ms - {data['max_ms']:.2f}ms\n")
                f.write(f"  Median: {data['median_ms']:.2f}ms\n\n")
        
        self.logger.info(f"Performance report saved to: {file_path}")
    
    def clear_measurements(self):
        """Clear all recorded measurements."""
        self.measurements.clear()
        self.active_timers.clear()


def profile_function(operation_name: Optional[str] = None):
    """Decorator for profiling function execution time."""
    def decorator(func: Callable) -> Callable:
        # Resolve the operation name once
        resolved_name = operation_name if operation_name is not None else f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler = PerformanceProfiler()
            with profiler.timer(resolved_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def profile_multimedia_dialog_init():
    """Profile multimedia dialog initialization."""
    profiler = PerformanceProfiler()
    
    # This would be called from the dialog's __init__ method
    with profiler.timer("multimedia_dialog_total_init"):
        
        with profiler.timer("multimedia_dialog_ui_init"):
            # UI initialization would happen here
            pass
        
        with profiler.timer("multimedia_analyzer_init"):
            # Analyzer initialization would happen here  
            pass
        
        with profiler.timer("multimedia_dialog_connections"):
            # Signal connections would happen here
            pass


# Global profiler instance
profiler = PerformanceProfiler()
