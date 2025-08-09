"""
Settings Panel
Comprehensive settings interface for InvisioVault configuration.
"""

import os
from pathlib import Path
from typing import Dict, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTabWidget, QWidget, QSpinBox, QCheckBox,
    QComboBox, QLineEdit, QFileDialog, QMessageBox, QSlider,
    QFormLayout, QButtonGroup, QRadioButton, QTextEdit,
    QDialogButtonBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIcon

from utils.logger import Logger
from utils.config_manager import ConfigManager, ConfigSection
from utils.error_handler import ErrorHandler


class SettingsPanel(QDialog):
    """Settings panel dialog for application configuration."""
    
    settings_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        
        # Track original values for reset functionality
        self.original_config = {}
        self._backup_current_config()
        
        self.init_ui()
        self.load_current_settings()
        
    def init_ui(self):
        """Initialize the settings panel UI."""
        self.setWindowTitle("InvisioVault Settings")
        self.setWindowIcon(QIcon("assets/icons/settings.png"))
        self.setModal(True)
        self.resize(700, 600)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create tabs
        self._create_security_tab()
        self._create_interface_tab()
        self._create_performance_tab()
        self._create_logging_tab()
        self._create_backup_tab()
        self._create_advanced_tab()
        
        layout.addWidget(self.tab_widget)
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        
        # Connect buttons
        button_box.accepted.connect(self.accept_changes)
        button_box.rejected.connect(self.reject_changes)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_changes)
        button_box.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.restore_defaults)
        
        layout.addWidget(button_box)
        
        # Style the dialog
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
            }
            QTabWidget::tab-bar {
                left: 5px;
            }
            QTabBar::tab {
                background-color: #404040;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2b2b2b;
                border-bottom: 2px solid #0078d4;
            }
            QTabBar::tab:hover:!selected {
                background-color: #4a4a4a;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
    
    def _create_security_tab(self):
        """Create the security settings tab."""
        tab = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Password Policy Group
        password_group = QGroupBox("Password Policy")
        password_layout = QFormLayout(password_group)
        
        self.password_min_length = QSpinBox()
        self.password_min_length.setRange(4, 128)
        self.password_min_length.setSuffix(" characters")
        password_layout.addRow("Minimum Length:", self.password_min_length)
        
        self.require_uppercase = QCheckBox("Require uppercase letters")
        password_layout.addRow(self.require_uppercase)
        
        self.require_lowercase = QCheckBox("Require lowercase letters")
        password_layout.addRow(self.require_lowercase)
        
        self.require_numbers = QCheckBox("Require numbers")
        password_layout.addRow(self.require_numbers)
        
        self.require_symbols = QCheckBox("Require symbols")
        password_layout.addRow(self.require_symbols)
        
        layout.addWidget(password_group)
        
        # Security Settings Group
        security_group = QGroupBox("Security Settings")
        security_layout = QFormLayout(security_group)
        
        self.default_security_level = QComboBox()
        self.default_security_level.addItems(["basic", "standard", "high", "maximum"])
        security_layout.addRow("Default Security Level:", self.default_security_level)
        
        self.enable_keyfile_auth = QCheckBox("Enable keyfile authentication by default")
        security_layout.addRow(self.enable_keyfile_auth)
        
        self.secure_delete_passes = QSpinBox()
        self.secure_delete_passes.setRange(1, 10)
        self.secure_delete_passes.setSuffix(" passes")
        security_layout.addRow("Secure Delete Passes:", self.secure_delete_passes)
        
        self.session_timeout = QSpinBox()
        self.session_timeout.setRange(5, 120)
        self.session_timeout.setSuffix(" minutes")
        security_layout.addRow("Session Timeout:", self.session_timeout)
        
        layout.addWidget(security_group)
        
        layout.addStretch()
        tab.setWidget(content)
        self.tab_widget.addTab(tab, "🔒 Security")
    
    def _create_interface_tab(self):
        """Create the interface settings tab."""
        tab = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Appearance Group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout(appearance_group)
        
        self.theme = QComboBox()
        self.theme.addItems(["dark", "light", "auto"])
        appearance_layout.addRow("Theme:", self.theme)
        
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 24)
        self.font_size.setSuffix(" pt")
        appearance_layout.addRow("Font Size:", self.font_size)
        
        self.enable_animations = QCheckBox("Enable animations")
        appearance_layout.addRow(self.enable_animations)
        
        layout.addWidget(appearance_group)
        
        # Window Settings Group
        window_group = QGroupBox("Window Settings")
        window_layout = QFormLayout(window_group)
        
        self.remember_window_state = QCheckBox("Remember window size and position")
        window_layout.addRow(self.remember_window_state)
        
        self.window_width = QSpinBox()
        self.window_width.setRange(800, 3840)
        self.window_width.setSuffix(" px")
        window_layout.addRow("Default Width:", self.window_width)
        
        self.window_height = QSpinBox()
        self.window_height.setRange(600, 2160)
        self.window_height.setSuffix(" px")
        window_layout.addRow("Default Height:", self.window_height)
        
        layout.addWidget(window_group)
        
        # User Experience Group
        ux_group = QGroupBox("User Experience")
        ux_layout = QFormLayout(ux_group)
        
        self.show_progress_details = QCheckBox("Show detailed progress information")
        ux_layout.addRow(self.show_progress_details)
        
        self.enable_notifications = QCheckBox("Enable system notifications")
        ux_layout.addRow(self.enable_notifications)
        
        self.language = QComboBox()
        self.language.addItems(["en", "es", "fr", "de", "zh"])
        ux_layout.addRow("Language:", self.language)
        
        layout.addWidget(ux_group)
        
        layout.addStretch()
        tab.setWidget(content)
        self.tab_widget.addTab(tab, "🎨 Interface")
    
    def _create_performance_tab(self):
        """Create the performance settings tab."""
        tab = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Processing Group
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QFormLayout(processing_group)
        
        self.max_threads = QSpinBox()
        self.max_threads.setRange(1, 16)
        processing_layout.addRow("Maximum Threads:", self.max_threads)
        
        self.memory_limit = QSpinBox()
        self.memory_limit.setRange(100, 8192)
        self.memory_limit.setSuffix(" MB")
        processing_layout.addRow("Memory Limit:", self.memory_limit)
        
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(1, 100)
        self.chunk_size.setSuffix(" MB")
        processing_layout.addRow("Chunk Size:", self.chunk_size)
        
        layout.addWidget(processing_group)
        
        # Compression Group
        compression_group = QGroupBox("Compression Settings")
        compression_layout = QFormLayout(compression_group)
        
        self.compression_level = QSlider(Qt.Orientation.Horizontal)
        self.compression_level.setRange(1, 9)
        self.compression_level.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.compression_level.setTickInterval(1)
        
        compression_label = QLabel()
        self.compression_level.valueChanged.connect(
            lambda v: compression_label.setText(f"Level {v} ({'Fast' if v < 4 else 'Balanced' if v < 7 else 'Best'})"))
        
        compression_layout.addRow("Compression Level:", self.compression_level)
        compression_layout.addRow("", compression_label)
        
        self.analysis_quality = QComboBox()
        self.analysis_quality.addItems(["fast", "balanced", "thorough"])
        compression_layout.addRow("Analysis Quality:", self.analysis_quality)
        
        layout.addWidget(compression_group)
        
        # Hardware Group
        hardware_group = QGroupBox("Hardware Acceleration")
        hardware_layout = QFormLayout(hardware_group)
        
        self.enable_gpu_acceleration = QCheckBox("Enable GPU acceleration (experimental)")
        hardware_layout.addRow(self.enable_gpu_acceleration)
        
        layout.addWidget(hardware_group)
        
        layout.addStretch()
        tab.setWidget(content)
        self.tab_widget.addTab(tab, "⚡ Performance")
    
    def _create_logging_tab(self):
        """Create the logging settings tab."""
        tab = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Logging Settings Group
        logging_group = QGroupBox("Logging Configuration")
        logging_layout = QFormLayout(logging_group)
        
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        logging_layout.addRow("Log Level:", self.log_level)
        
        self.max_log_size = QSpinBox()
        self.max_log_size.setRange(1, 100)
        self.max_log_size.setSuffix(" MB")
        logging_layout.addRow("Maximum Log Size:", self.max_log_size)
        
        self.log_retention_days = QSpinBox()
        self.log_retention_days.setRange(1, 365)
        self.log_retention_days.setSuffix(" days")
        logging_layout.addRow("Log Retention:", self.log_retention_days)
        
        self.enable_debug_logging = QCheckBox("Enable debug logging")
        logging_layout.addRow(self.enable_debug_logging)
        
        self.log_performance_metrics = QCheckBox("Log performance metrics")
        logging_layout.addRow(self.log_performance_metrics)
        
        layout.addWidget(logging_group)
        
        # Log Actions Group
        actions_group = QGroupBox("Log Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        log_actions_layout = QHBoxLayout()
        
        open_log_btn = QPushButton("Open Log Directory")
        open_log_btn.clicked.connect(self.open_log_directory)
        log_actions_layout.addWidget(open_log_btn)
        
        clear_logs_btn = QPushButton("Clear All Logs")
        clear_logs_btn.clicked.connect(self.clear_logs)
        log_actions_layout.addWidget(clear_logs_btn)
        
        log_actions_layout.addStretch()
        actions_layout.addLayout(log_actions_layout)
        
        layout.addWidget(actions_group)
        
        layout.addStretch()
        tab.setWidget(content)
        self.tab_widget.addTab(tab, "📝 Logging")
    
    def _create_backup_tab(self):
        """Create the backup settings tab."""
        tab = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Backup Settings Group
        backup_group = QGroupBox("Backup Configuration")
        backup_layout = QFormLayout(backup_group)
        
        self.auto_backup_enabled = QCheckBox("Enable automatic backups")
        backup_layout.addRow(self.auto_backup_enabled)
        
        self.backup_before_operations = QCheckBox("Backup before operations")
        backup_layout.addRow(self.backup_before_operations)
        
        self.backup_retention_count = QSpinBox()
        self.backup_retention_count.setRange(1, 20)
        self.backup_retention_count.setSuffix(" backups")
        backup_layout.addRow("Retention Count:", self.backup_retention_count)
        
        self.compress_backups = QCheckBox("Compress backup files")
        backup_layout.addRow(self.compress_backups)
        
        layout.addWidget(backup_group)
        
        # Backup Location Group
        location_group = QGroupBox("Backup Location")
        location_layout = QVBoxLayout(location_group)
        
        location_input_layout = QHBoxLayout()
        
        self.backup_location = QLineEdit()
        self.backup_location.setPlaceholderText("Leave empty for default location")
        location_input_layout.addWidget(self.backup_location)
        
        browse_backup_btn = QPushButton("Browse...")
        browse_backup_btn.clicked.connect(self.browse_backup_location)
        location_input_layout.addWidget(browse_backup_btn)
        
        location_layout.addLayout(location_input_layout)
        layout.addWidget(location_group)
        
        layout.addStretch()
        tab.setWidget(content)
        self.tab_widget.addTab(tab, "💾 Backup")
    
    def _create_advanced_tab(self):
        """Create the advanced settings tab."""
        tab = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Developer Settings Group
        dev_group = QGroupBox("Developer Settings")
        dev_layout = QFormLayout(dev_group)
        
        self.developer_mode = QCheckBox("Enable developer mode")
        dev_layout.addRow(self.developer_mode)
        
        self.experimental_features = QCheckBox("Enable experimental features")
        dev_layout.addRow(self.experimental_features)
        
        layout.addWidget(dev_group)
        
        # Privacy Group
        privacy_group = QGroupBox("Privacy Settings")
        privacy_layout = QFormLayout(privacy_group)
        
        self.enable_telemetry = QCheckBox("Enable anonymous usage statistics")
        privacy_layout.addRow(self.enable_telemetry)
        
        self.check_for_updates = QCheckBox("Check for updates automatically")
        privacy_layout.addRow(self.check_for_updates)
        
        layout.addWidget(privacy_group)
        
        # Configuration Actions Group
        config_group = QGroupBox("Configuration Actions")
        config_layout = QVBoxLayout(config_group)
        
        config_actions_layout = QHBoxLayout()
        
        export_config_btn = QPushButton("Export Configuration")
        export_config_btn.clicked.connect(self.export_configuration)
        config_actions_layout.addWidget(export_config_btn)
        
        import_config_btn = QPushButton("Import Configuration")
        import_config_btn.clicked.connect(self.import_configuration)
        config_actions_layout.addWidget(import_config_btn)
        
        config_actions_layout.addStretch()
        config_layout.addLayout(config_actions_layout)
        
        layout.addWidget(config_group)
        
        layout.addStretch()
        tab.setWidget(content)
        self.tab_widget.addTab(tab, "🔧 Advanced")
    
    def _backup_current_config(self):
        """Backup current configuration for reset functionality."""
        for section in ConfigSection:
            self.original_config[section.value] = {}
            section_config = self.config.config.get(section.value, {})
            for key, value in section_config.items():
                self.original_config[section.value][key] = value
    
    def load_current_settings(self):
        """Load current configuration values into UI components."""
        try:
            # Security settings
            self.password_min_length.setValue(self.config.get(ConfigSection.SECURITY, "password_min_length", 8))
            self.require_uppercase.setChecked(self.config.get(ConfigSection.SECURITY, "password_require_uppercase", True))
            self.require_lowercase.setChecked(self.config.get(ConfigSection.SECURITY, "password_require_lowercase", True))
            self.require_numbers.setChecked(self.config.get(ConfigSection.SECURITY, "password_require_numbers", True))
            self.require_symbols.setChecked(self.config.get(ConfigSection.SECURITY, "password_require_symbols", True))
            self.default_security_level.setCurrentText(self.config.get(ConfigSection.SECURITY, "default_security_level", "standard"))
            self.enable_keyfile_auth.setChecked(self.config.get(ConfigSection.SECURITY, "enable_keyfile_auth", False))
            self.secure_delete_passes.setValue(self.config.get(ConfigSection.SECURITY, "secure_delete_passes", 3))
            self.session_timeout.setValue(self.config.get(ConfigSection.SECURITY, "session_timeout_minutes", 30))
            
            # Interface settings
            self.theme.setCurrentText(self.config.get(ConfigSection.INTERFACE, "theme", "dark"))
            self.font_size.setValue(self.config.get(ConfigSection.INTERFACE, "font_size", 12))
            self.enable_animations.setChecked(self.config.get(ConfigSection.INTERFACE, "enable_animations", True))
            self.remember_window_state.setChecked(self.config.get(ConfigSection.INTERFACE, "remember_window_state", True))
            self.window_width.setValue(self.config.get(ConfigSection.INTERFACE, "window_width", 1200))
            self.window_height.setValue(self.config.get(ConfigSection.INTERFACE, "window_height", 800))
            self.show_progress_details.setChecked(self.config.get(ConfigSection.INTERFACE, "show_progress_details", True))
            self.enable_notifications.setChecked(self.config.get(ConfigSection.INTERFACE, "enable_notifications", True))
            self.language.setCurrentText(self.config.get(ConfigSection.INTERFACE, "language", "en"))
            
            # Performance settings
            self.max_threads.setValue(self.config.get(ConfigSection.PERFORMANCE, "max_threads", 4))
            self.memory_limit.setValue(self.config.get(ConfigSection.PERFORMANCE, "memory_limit_mb", 500))
            self.chunk_size.setValue(self.config.get(ConfigSection.PERFORMANCE, "chunk_size_mb", 10))
            self.compression_level.setValue(self.config.get(ConfigSection.PERFORMANCE, "compression_level", 6))
            self.analysis_quality.setCurrentText(self.config.get(ConfigSection.PERFORMANCE, "analysis_quality", "balanced"))
            self.enable_gpu_acceleration.setChecked(self.config.get(ConfigSection.PERFORMANCE, "enable_gpu_acceleration", False))
            
            # Logging settings
            self.log_level.setCurrentText(self.config.get(ConfigSection.LOGGING, "log_level", "INFO"))
            self.max_log_size.setValue(self.config.get(ConfigSection.LOGGING, "max_log_size_mb", 10))
            self.log_retention_days.setValue(self.config.get(ConfigSection.LOGGING, "log_retention_days", 30))
            self.enable_debug_logging.setChecked(self.config.get(ConfigSection.LOGGING, "enable_debug_logging", False))
            self.log_performance_metrics.setChecked(self.config.get(ConfigSection.LOGGING, "log_performance_metrics", False))
            
            # Backup settings
            self.auto_backup_enabled.setChecked(self.config.get(ConfigSection.BACKUP, "auto_backup_enabled", True))
            self.backup_before_operations.setChecked(self.config.get(ConfigSection.BACKUP, "backup_before_operations", True))
            self.backup_retention_count.setValue(self.config.get(ConfigSection.BACKUP, "backup_retention_count", 5))
            self.compress_backups.setChecked(self.config.get(ConfigSection.BACKUP, "compress_backups", True))
            self.backup_location.setText(self.config.get(ConfigSection.BACKUP, "backup_location", ""))
            
            # Advanced settings
            self.developer_mode.setChecked(self.config.get(ConfigSection.ADVANCED, "developer_mode", False))
            self.experimental_features.setChecked(self.config.get(ConfigSection.ADVANCED, "experimental_features", False))
            self.enable_telemetry.setChecked(self.config.get(ConfigSection.ADVANCED, "enable_telemetry", False))
            self.check_for_updates.setChecked(self.config.get(ConfigSection.ADVANCED, "check_for_updates", True))
            
            # Trigger compression level label update
            self.compression_level.valueChanged.emit(self.compression_level.value())
            
            self.logger.debug("Settings loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            QMessageBox.warning(self, "Settings Error", f"Error loading current settings: {str(e)}")
    
    def apply_changes(self):
        """Apply configuration changes without closing dialog."""
        try:
            self._save_settings()
            QMessageBox.information(self, "Settings Applied", "Settings have been applied successfully.")
            self.settings_changed.emit()
        except Exception as e:
            self.logger.error(f"Error applying settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {str(e)}")
    
    def accept_changes(self):
        """Apply changes and close dialog."""
        try:
            self._save_settings()
            self.settings_changed.emit()
            self.accept()
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")
    
    def reject_changes(self):
        """Discard changes and close dialog."""
        self.reject()
    
    def restore_defaults(self):
        """Restore all settings to default values."""
        reply = QMessageBox.question(
            self, "Restore Defaults",
            "Are you sure you want to restore all settings to their default values?\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Reset config to defaults
                self.config.config = self.config.DEFAULT_CONFIG.copy()
                self.config.save_config()
                
                # Reload UI with default values
                self.load_current_settings()
                
                QMessageBox.information(self, "Defaults Restored", "All settings have been restored to default values.")
                self.settings_changed.emit()
                
            except Exception as e:
                self.logger.error(f"Error restoring defaults: {e}")
                QMessageBox.critical(self, "Error", f"Failed to restore defaults: {str(e)}")
    
    def _save_settings(self):
        """Save all settings to configuration."""
        # Security settings
        self.config.set(ConfigSection.SECURITY, "password_min_length", self.password_min_length.value())
        self.config.set(ConfigSection.SECURITY, "password_require_uppercase", self.require_uppercase.isChecked())
        self.config.set(ConfigSection.SECURITY, "password_require_lowercase", self.require_lowercase.isChecked())
        self.config.set(ConfigSection.SECURITY, "password_require_numbers", self.require_numbers.isChecked())
        self.config.set(ConfigSection.SECURITY, "password_require_symbols", self.require_symbols.isChecked())
        self.config.set(ConfigSection.SECURITY, "default_security_level", self.default_security_level.currentText())
        self.config.set(ConfigSection.SECURITY, "enable_keyfile_auth", self.enable_keyfile_auth.isChecked())
        self.config.set(ConfigSection.SECURITY, "secure_delete_passes", self.secure_delete_passes.value())
        self.config.set(ConfigSection.SECURITY, "session_timeout_minutes", self.session_timeout.value())
        
        # Interface settings
        self.config.set(ConfigSection.INTERFACE, "theme", self.theme.currentText())
        self.config.set(ConfigSection.INTERFACE, "font_size", self.font_size.value())
        self.config.set(ConfigSection.INTERFACE, "enable_animations", self.enable_animations.isChecked())
        self.config.set(ConfigSection.INTERFACE, "remember_window_state", self.remember_window_state.isChecked())
        self.config.set(ConfigSection.INTERFACE, "window_width", self.window_width.value())
        self.config.set(ConfigSection.INTERFACE, "window_height", self.window_height.value())
        self.config.set(ConfigSection.INTERFACE, "show_progress_details", self.show_progress_details.isChecked())
        self.config.set(ConfigSection.INTERFACE, "enable_notifications", self.enable_notifications.isChecked())
        self.config.set(ConfigSection.INTERFACE, "language", self.language.currentText())
        
        # Performance settings
        self.config.set(ConfigSection.PERFORMANCE, "max_threads", self.max_threads.value())
        self.config.set(ConfigSection.PERFORMANCE, "memory_limit_mb", self.memory_limit.value())
        self.config.set(ConfigSection.PERFORMANCE, "chunk_size_mb", self.chunk_size.value())
        self.config.set(ConfigSection.PERFORMANCE, "compression_level", self.compression_level.value())
        self.config.set(ConfigSection.PERFORMANCE, "analysis_quality", self.analysis_quality.currentText())
        self.config.set(ConfigSection.PERFORMANCE, "enable_gpu_acceleration", self.enable_gpu_acceleration.isChecked())
        
        # Logging settings
        self.config.set(ConfigSection.LOGGING, "log_level", self.log_level.currentText())
        self.config.set(ConfigSection.LOGGING, "max_log_size_mb", self.max_log_size.value())
        self.config.set(ConfigSection.LOGGING, "log_retention_days", self.log_retention_days.value())
        self.config.set(ConfigSection.LOGGING, "enable_debug_logging", self.enable_debug_logging.isChecked())
        self.config.set(ConfigSection.LOGGING, "log_performance_metrics", self.log_performance_metrics.isChecked())
        
        # Backup settings
        self.config.set(ConfigSection.BACKUP, "auto_backup_enabled", self.auto_backup_enabled.isChecked())
        self.config.set(ConfigSection.BACKUP, "backup_before_operations", self.backup_before_operations.isChecked())
        self.config.set(ConfigSection.BACKUP, "backup_retention_count", self.backup_retention_count.value())
        self.config.set(ConfigSection.BACKUP, "compress_backups", self.compress_backups.isChecked())
        self.config.set(ConfigSection.BACKUP, "backup_location", self.backup_location.text().strip())
        
        # Advanced settings
        self.config.set(ConfigSection.ADVANCED, "developer_mode", self.developer_mode.isChecked())
        self.config.set(ConfigSection.ADVANCED, "experimental_features", self.experimental_features.isChecked())
        self.config.set(ConfigSection.ADVANCED, "enable_telemetry", self.enable_telemetry.isChecked())
        self.config.set(ConfigSection.ADVANCED, "check_for_updates", self.check_for_updates.isChecked())
        
        # Save configuration
        self.config.save_config()
        self.logger.info("Configuration settings saved successfully")
    
    def browse_backup_location(self):
        """Browse for backup location directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Backup Directory",
            self.backup_location.text() or str(Path.home())
        )
        
        if directory:
            self.backup_location.setText(directory)
    
    def open_log_directory(self):
        """Open the log directory in file explorer."""
        try:
            log_dir = Path.home() / '.invisiovault' / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Open directory in file explorer
            import subprocess
            import sys
            
            if sys.platform == 'win32':
                subprocess.run(['explorer', str(log_dir)])
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(log_dir)])
            else:
                subprocess.run(['xdg-open', str(log_dir)])
                
        except Exception as e:
            self.logger.error(f"Error opening log directory: {e}")
            QMessageBox.warning(self, "Error", f"Could not open log directory: {str(e)}")
    
    def clear_logs(self):
        """Clear all log files."""
        reply = QMessageBox.question(
            self, "Clear Logs",
            "Are you sure you want to clear all log files?\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                log_dir = Path.home() / '.invisiovault' / 'logs'
                if log_dir.exists():
                    for log_file in log_dir.glob('*.log'):
                        log_file.unlink()
                    
                    QMessageBox.information(self, "Logs Cleared", "All log files have been cleared.")
                else:
                    QMessageBox.information(self, "No Logs", "No log files found to clear.")
                    
            except Exception as e:
                self.logger.error(f"Error clearing logs: {e}")
                QMessageBox.critical(self, "Error", f"Failed to clear logs: {str(e)}")
    
    def export_configuration(self):
        """Export current configuration to a file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Configuration",
                str(Path.home() / "invisiovault_config.json"),
                "JSON files (*.json)"
            )
            
            if file_path:
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config.config, f, indent=2)
                
                QMessageBox.information(self, "Export Complete", f"Configuration exported to:\n{file_path}")
                
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export configuration: {str(e)}")
    
    def import_configuration(self):
        """Import configuration from a file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Import Configuration",
                str(Path.home()),
                "JSON files (*.json)"
            )
            
            if file_path:
                reply = QMessageBox.question(
                    self, "Import Configuration",
                    "Importing configuration will overwrite current settings.\n"
                    "Are you sure you want to continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        imported_config = json.load(f)
                    
                    # Merge with current config
                    self.config.config = self.config._merge_configs(self.config.DEFAULT_CONFIG, imported_config)
                    self.config.save_config()
                    
                    # Reload UI
                    self.load_current_settings()
                    
                    QMessageBox.information(self, "Import Complete", "Configuration imported successfully.")
                    self.settings_changed.emit()
                    
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            QMessageBox.critical(self, "Import Error", f"Failed to import configuration: {str(e)}")
