"""
Secure Logging System with PII/Sensitive Data Filtering
Provides comprehensive logging with automatic data redaction.
"""

import os
import re
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Logger:
    """Secure logging with sensitive data filtering."""
    
    # Patterns to redact sensitive information
    SENSITIVE_PATTERNS = [
        # Passwords and keys
        (r'password["\']?\s*[:=]\s*["\']?([^"\s,}]+)', 'password=***'),
        (r'pass["\']?\s*[:=]\s*["\']?([^"\s,}]+)', 'pass=***'),
        (r'key["\']?\s*[:=]\s*["\']?([^"\s,}]+)', 'key=***'),
        (r'token["\']?\s*[:=]\s*["\']?([^"\s,}]+)', 'token=***'),
        
        # File paths (partial redaction)
        (r'([A-Za-z]:\\\\[^\\]+\\\\[^\\]+\\\\)([^\\\s]+)', r'\1***'),
        (r'(/[^/]+/[^/]+/)([^/\s]+)', r'\1***'),
        
        # Email addresses
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***@***.***'),
        
        # Credit card numbers (basic pattern)
        (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '****-****-****-****'),
        
        # Social Security Numbers
        (r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****'),
        
        # Phone numbers
        (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '***-***-****'),
    ]
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern for logger."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, log_level: LogLevel = LogLevel.INFO, 
                 log_file: Optional[str] = None, 
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """Initialize logger (only once due to singleton pattern)."""
        if self._initialized:
            return
            
        self.log_level = log_level
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Set up log directory
        if log_file is None:
            log_dir = Path.home() / '.invisiovault' / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir / 'invisiovault.log')
        
        self.log_file = log_file
        
        # Configure logging
        self._setup_logging()
        
        self._initialized = True
        self.info("Logger initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger('InvisioVault')
        self.logger.setLevel(self.log_level.value)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level.value)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings and above to console
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _redact_sensitive_data(self, message: str) -> str:
        """Remove sensitive information from log messages."""
        redacted_message = message
        
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            redacted_message = re.sub(pattern, replacement, redacted_message, flags=re.IGNORECASE)
        
        return redacted_message
    
    def _log(self, level: LogLevel, message: str, *args, **kwargs):
        """Internal logging method with redaction."""
        # Format message if args provided
        if args:
            try:
                message = message % args
            except (TypeError, ValueError):
                # If formatting fails, log as-is
                pass
        
        # Redact sensitive information
        redacted_message = self._redact_sensitive_data(str(message))
        
        # Log the message
        self.logger.log(level.value, redacted_message)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        self._log(LogLevel.ERROR, message, *args, **kwargs)
        self.logger.exception('')  # This adds the traceback
    
    def set_level(self, level: LogLevel):
        """Change logging level."""
        self.log_level = level
        self.logger.setLevel(level.value)
        for handler in self.logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setLevel(level.value)
    
    def get_log_file_path(self) -> str:
        """Get current log file path."""
        return self.log_file
    
    def export_logs(self, output_file: str, start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> bool:
        """Export logs to a file with optional date filtering."""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as source:
                with open(output_file, 'w', encoding='utf-8') as dest:
                    for line in source:
                        # Basic date filtering if requested
                        if start_date or end_date:
                            # Extract timestamp from log line
                            # This is a simple implementation
                            try:
                                timestamp_str = line.split(' - ')[0]
                                log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                                
                                if start_date and log_time < start_date:
                                    continue
                                if end_date and log_time > end_date:
                                    continue
                            except (ValueError, IndexError):
                                # If we can't parse the timestamp, include the line
                                pass
                        
                        dest.write(line)
            
            self.info(f"Logs exported to {output_file}")
            return True
            
        except Exception as e:
            self.error(f"Failed to export logs: {e}")
            return False
    
    def clear_old_logs(self, days: int = 30):
        """Clear log files older than specified days."""
        try:
            log_dir = Path(self.log_file).parent
            cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            for log_file in log_dir.glob('*.log*'):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    self.info(f"Removed old log file: {log_file}")
            
        except Exception as e:
            self.error(f"Error clearing old logs: {e}")
    
    def get_recent_errors(self, count: int = 10) -> list:
        """Get recent error messages from log file."""
        errors = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Search backwards through the file
            for line in reversed(lines):
                if ' - ERROR - ' in line or ' - CRITICAL - ' in line:
                    errors.append(line.strip())
                    if len(errors) >= count:
                        break
            
            return errors[::-1]  # Return in chronological order
            
        except Exception as e:
            self.error(f"Error reading recent errors: {e}")
            return []
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about log file."""
        try:
            log_path = Path(self.log_file)
            if not log_path.exists():
                return {}
            
            stats = {
                'file_size_mb': log_path.stat().st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(log_path.stat().st_ctime),
                'modified': datetime.fromtimestamp(log_path.stat().st_mtime),
                'total_lines': 0,
                'error_count': 0,
                'warning_count': 0
            }
            
            # Count lines and messages
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    stats['total_lines'] += 1
                    if ' - ERROR - ' in line or ' - CRITICAL - ' in line:
                        stats['error_count'] += 1
                    elif ' - WARNING - ' in line:
                        stats['warning_count'] += 1
            
            return stats
            
        except Exception as e:
            self.error(f"Error getting log stats: {e}")
            return {}
