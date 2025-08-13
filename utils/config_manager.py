"""
Configuration Management System
Handles persistent configuration and user settings.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from enum import Enum

from utils.logger import Logger


class ConfigSection(Enum):
    """Configuration sections."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTERFACE = "interface"
    LOGGING = "logging"
    BACKUP = "backup"
    ADVANCED = "advanced"


class ConfigManager:
    """Persistent configuration management."""
    
    DEFAULT_CONFIG = {
        ConfigSection.SECURITY.value: {
            "password_min_length": 8,
            "password_require_uppercase": True,
            "password_require_lowercase": True,
            "password_require_numbers": True,
            "password_require_symbols": True,
            "default_security_level": "standard",
            "enable_keyfile_auth": False,
            "secure_delete_passes": 3,
            "session_timeout_minutes": 30
        },
        ConfigSection.PERFORMANCE.value: {
            "max_threads": 4,
            "memory_limit_mb": 500,
            "compression_level": 6,
            "analysis_quality": "balanced",
            "enable_gpu_acceleration": False,
            "chunk_size_mb": 10
        },
        ConfigSection.INTERFACE.value: {
            "theme": "dark",  # Locked to dark mode
            "language": "en",
            "window_width": 1200,
            "window_height": 800,
            "remember_window_state": True,
            "show_progress_details": True,
            "enable_animations": True,
            "font_size": 12,
            "enable_notifications": True
        },
        ConfigSection.LOGGING.value: {
            "log_level": "INFO",
            "max_log_size_mb": 10,
            "log_retention_days": 30,
            "enable_debug_logging": False,
            "log_performance_metrics": False
        },
        ConfigSection.BACKUP.value: {
            "auto_backup_enabled": True,
            "backup_before_operations": True,
            "backup_retention_count": 5,
            "backup_location": "",  # Empty means default location
            "compress_backups": True
        },
        ConfigSection.ADVANCED.value: {
            "developer_mode": False,
            "experimental_features": False,
            "custom_algorithms": {},
            "plugin_directories": [],
            "enable_telemetry": False,
            "check_for_updates": True
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Path to config file (uses default if None)
        """
        self.logger = Logger()
        
        if config_file is None:
            config_dir = Path.home() / '.invisiovault'
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = str(config_dir / 'config.json')
        
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Validate and migrate config if needed
        self._validate_and_migrate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                merged_config = self._merge_configs(self.DEFAULT_CONFIG, config)
                self.logger.info(f"Configuration loaded from {self.config_file}")
                return merged_config
            else:
                self.logger.info("No config file found, using defaults")
                return self.DEFAULT_CONFIG.copy()
                
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error loading config: {e}")
            self.logger.info("Using default configuration")
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults, preserving structure."""
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged:
                if isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_configs(merged[key], value)
                else:
                    merged[key] = value
            else:
                merged[key] = value
        
        return merged
    
    def _validate_and_migrate_config(self):
        """Validate configuration and migrate if necessary."""
        # Check for missing sections
        for section in ConfigSection:
            if section.value not in self.config:
                self.config[section.value] = self.DEFAULT_CONFIG[section.value].copy()
                self.logger.info(f"Added missing config section: {section.value}")
        
        # Validate security settings
        security = self.config[ConfigSection.SECURITY.value]
        if security['password_min_length'] < 4:
            security['password_min_length'] = 8
            self.logger.warning("Reset minimum password length to 8")
        
        # Validate performance settings
        performance = self.config[ConfigSection.PERFORMANCE.value]
        if performance['max_threads'] < 1:
            performance['max_threads'] = 1
        elif performance['max_threads'] > 16:
            performance['max_threads'] = 16
        
        if performance['memory_limit_mb'] < 100:
            performance['memory_limit_mb'] = 100
        
        # Validate interface settings
        interface = self.config[ConfigSection.INTERFACE.value]
        if interface['window_width'] < 800:
            interface['window_width'] = 800
        if interface['window_height'] < 600:
            interface['window_height'] = 600
        # Ensure theme is always locked to dark mode
        if interface.get('theme') != 'dark':
            interface['theme'] = 'dark'
            self.logger.info("Theme locked to dark mode during validation")
        
        # Save if any changes were made
        self.save_config()
    
    def get(self, section: Union[ConfigSection, str], key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if isinstance(section, ConfigSection):
            section = section.value
        
        try:
            return self.config.get(section, {}).get(key, default)
        except Exception as e:
            self.logger.error(f"Error getting config value {section}.{key}: {e}")
            return default
    
    def set(self, section: Union[ConfigSection, str], key: str, value: Any) -> bool:
        """Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        if isinstance(section, ConfigSection):
            section = section.value
        
        try:
            # Prevent changing theme - always lock to dark mode
            if section == ConfigSection.INTERFACE.value and key == "theme":
                value = "dark"
                self.logger.debug(f"Theme locked to dark mode")
            
            if section not in self.config:
                self.config[section] = {}
            
            self.config[section][key] = value
            self.logger.debug(f"Set config value {section}.{key} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting config value {section}.{key}: {e}")
            return False
    
    def get_section(self, section: Union[ConfigSection, str]) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Configuration section
            
        Returns:
            Section configuration dictionary
        """
        if isinstance(section, ConfigSection):
            section = section.value
        
        return self.config.get(section, {}).copy()
    
    def set_section(self, section: Union[ConfigSection, str], values: Dict[str, Any]) -> bool:
        """Set entire configuration section.
        
        Args:
            section: Configuration section
            values: Dictionary of values to set
            
        Returns:
            True if successful, False otherwise
        """
        if isinstance(section, ConfigSection):
            section = section.value
        
        try:
            # Ensure theme is locked to dark mode for interface section
            if section == ConfigSection.INTERFACE.value and "theme" in values:
                values["theme"] = "dark"
                self.logger.debug(f"Theme locked to dark mode in section update")
            
            self.config[section] = values.copy()
            self.logger.info(f"Updated config section: {section}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting config section {section}: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup of existing config
            if self.config_file.exists():
                backup_file = self.config_file.with_suffix('.json.bak')
                self.config_file.replace(backup_file)
            
            # Write new config
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            return False
    
    def reset_to_defaults(self, section: Optional[Union[ConfigSection, str]] = None) -> bool:
        """Reset configuration to defaults.
        
        Args:
            section: Specific section to reset (resets all if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if section is None:
                # Reset entire configuration
                self.config = self.DEFAULT_CONFIG.copy()
                self.logger.info("Configuration reset to defaults")
            else:
                # Reset specific section
                if isinstance(section, ConfigSection):
                    section = section.value
                
                if section in self.DEFAULT_CONFIG:
                    self.config[section] = self.DEFAULT_CONFIG[section].copy()
                    self.logger.info(f"Config section {section} reset to defaults")
                else:
                    self.logger.error(f"Unknown config section: {section}")
                    return False
            
            return self.save_config()
            
        except Exception as e:
            self.logger.error(f"Error resetting config: {e}")
            return False
    
    def export_config(self, export_file: str) -> bool:
        """Export configuration to file.
        
        Args:
            export_file: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration exported to {export_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting config: {e}")
            return False
    
    def import_config(self, import_file: str) -> bool:
        """Import configuration from file.
        
        Args:
            import_file: Path to import file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(import_file, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # Validate imported config
            self.config = self._merge_configs(self.DEFAULT_CONFIG, imported_config)
            self._validate_and_migrate_config()
            
            self.logger.info(f"Configuration imported from {import_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing config: {e}")
            return False
    
    def get_config_file_path(self) -> str:
        """Get path to configuration file.
        
        Returns:
            Path to configuration file
        """
        return str(self.config_file)
    
    def validate_config(self) -> Dict[str, list]:
        """Validate current configuration.
        
        Returns:
            Dictionary of validation errors by section
        """
        errors = {}
        
        # Validate security section
        security = self.config.get(ConfigSection.SECURITY.value, {})
        security_errors = []
        
        if security.get('password_min_length', 0) < 4:
            security_errors.append("Password minimum length must be at least 4")
        
        if security.get('session_timeout_minutes', 0) < 1:
            security_errors.append("Session timeout must be at least 1 minute")
        
        if security_errors:
            errors[ConfigSection.SECURITY.value] = security_errors
        
        # Validate performance section
        performance = self.config.get(ConfigSection.PERFORMANCE.value, {})
        performance_errors = []
        
        max_threads = performance.get('max_threads', 1)
        if max_threads < 1 or max_threads > 16:
            performance_errors.append("Max threads must be between 1 and 16")
        
        memory_limit = performance.get('memory_limit_mb', 100)
        if memory_limit < 100:
            performance_errors.append("Memory limit must be at least 100MB")
        
        if performance_errors:
            errors[ConfigSection.PERFORMANCE.value] = performance_errors
        
        # Validate interface section
        interface = self.config.get(ConfigSection.INTERFACE.value, {})
        interface_errors = []
        
        if interface.get('window_width', 800) < 800:
            interface_errors.append("Window width must be at least 800px")
        
        if interface.get('window_height', 600) < 600:
            interface_errors.append("Window height must be at least 600px")
        
        if interface_errors:
            errors[ConfigSection.INTERFACE.value] = interface_errors
        
        return errors
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
