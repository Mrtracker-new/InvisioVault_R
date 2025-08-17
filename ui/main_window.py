"""
Main Application Window
Primary interface for InvisioVault with navigation and operation panels.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QToolBar, QStatusBar, QLabel, QFrame,
    QPushButton, QStackedWidget, QListWidget, QListWidgetItem,
    QMessageBox, QApplication, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QAction, QIcon, QFont, QPixmap

from utils.logger import Logger
from utils.config_manager import ConfigManager, ConfigSection
from utils.error_handler import ErrorHandler
from core.security_service import SecurityService


class MainWindow(QMainWindow):
    """Main application window with navigation and operation panels."""
    
    def __init__(self, security_service: SecurityService = None):
        super().__init__()
        
        # Initialize components
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        self.security_service = security_service or SecurityService()
        
        # Window state
        self.current_operation = None
        
        # Initialize UI
        self.init_ui()
        self.init_menu_bar()
        self.init_toolbar()
        self.init_status_bar()
        
        # Load settings
        self.load_window_state()
        
        # Update initial security status
        self.update_security_status()
        
        self.logger.info("Main window initialized")
    
    def init_ui(self):
        """Initialize the user interface."""
        
        # Set window properties
        self.setWindowTitle("InvisioVault - Advanced Steganography Suite")
        self.setMinimumSize(800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Navigation panel
        self.navigation_panel = self.create_navigation_panel()
        splitter.addWidget(self.navigation_panel)
        
        # Main content area
        self.content_area = self.create_content_area()
        splitter.addWidget(self.content_area)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 0)  # Navigation panel - fixed width
        splitter.setStretchFactor(1, 1)  # Content area - expandable
        splitter.setSizes([250, 800])
        
        # Connect navigation after UI is fully initialized
        self.nav_list.currentItemChanged.connect(self.on_navigation_changed)
        
        # Select first item by default
        self.nav_list.setCurrentRow(0)
    
    def create_navigation_panel(self) -> QWidget:
        """Create the navigation panel with operation categories."""
        
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(280)
        panel.setMinimumWidth(220)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Navigation title
        title_label = QLabel("Operations")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Navigation list
        self.nav_list = QListWidget()
        
        # Navigation items
        nav_items = [
            ("🏠 Basic Operations", "basic", "Hide and extract files with standard encryption"),
            ("🔐 Keyfile Mode", "keyfile", "Two-factor authentication with keyfiles"),
            ("🛡️ Two-Factor", "twofactor", "Distribute data across multiple images"),
            ("👻 Decoy Mode", "decoy", "Hide multiple datasets with plausible deniability"),
            ("🚀 Self-Executing", "self_executing", "Create images that execute embedded code when triggered"),
            ("🔍 Image Analysis", "analysis", "Analyze image capacity and suitability"),
            ("⚙️ Settings", "settings", "Configure application preferences")
        ]
        
        for text, key, tooltip in nav_items:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(tooltip)
            self.nav_list.addItem(item)
        
        layout.addWidget(self.nav_list)
        
        return panel
    
    def create_content_area(self) -> QWidget:
        """Create the main content area with stacked widgets."""
        
        # Stacked widget for different operations
        self.content_stack = QStackedWidget()
        
        # Create functional widgets for each operation
        self.basic_widget = self.create_basic_operations_widget()
        self.keyfile_widget = self.create_keyfile_operations_widget()
        self.twofactor_widget = self.create_twofactor_operations_widget()
        self.decoy_widget = self.create_decoy_operations_widget()
        self.self_executing_widget = self.create_self_executing_operations_widget()
        self.analysis_widget = self.create_analysis_operations_widget()
        self.settings_widget = self.create_settings_widget()
        
        # Add widgets to stack
        self.content_stack.addWidget(self.basic_widget)      # Index 0
        self.content_stack.addWidget(self.keyfile_widget)    # Index 1
        self.content_stack.addWidget(self.twofactor_widget)  # Index 2
        self.content_stack.addWidget(self.decoy_widget)      # Index 3
        self.content_stack.addWidget(self.self_executing_widget)  # Index 4
        self.content_stack.addWidget(self.analysis_widget)   # Index 5
        self.content_stack.addWidget(self.settings_widget)   # Index 6
        
        return self.content_stack
    
    def create_basic_operations_widget(self) -> QWidget:
        """Create the basic operations widget."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Basic Steganography Operations")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Hide and extract files using enhanced LSB steganography with AES-256 encryption and advanced anti-detection features.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(desc)
        
        # Operation buttons
        button_layout = QHBoxLayout()
        
        hide_btn = QPushButton("🔒 Hide Files")
        hide_btn.setMinimumHeight(60)
        hide_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        hide_btn.clicked.connect(self.show_hide_dialog)
        
        extract_btn = QPushButton("🔓 Extract Files")
        extract_btn.setMinimumHeight(60)
        extract_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        extract_btn.clicked.connect(self.show_extract_dialog)
        
        button_layout.addWidget(hide_btn)
        button_layout.addWidget(extract_btn)
        layout.addLayout(button_layout)
        
        # Feature list
        features_label = QLabel("Features:")
        features_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(features_label)
        
        features = [
            "• AES-256 encryption with PBKDF2 key derivation",
            "• Advanced anti-detection steganography algorithms",
            "• Real-time steganalysis resistance testing",
            "• Support for PNG, BMP, and TIFF images",
            "• Randomized LSB positioning for enhanced security",
            "• Image capacity analysis and validation",
            "• Secure file compression and integrity checking"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("margin-left: 10px; color: #444;")
            layout.addWidget(feature_label)
        
        # Spacer to push content to top
        layout.addStretch()
        
        return widget
    
    def create_keyfile_operations_widget(self) -> QWidget:
        """Create the keyfile operations widget."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Keyfile Operations - Two-Factor Authentication")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Enhanced security using password + keyfile combination for two-factor authentication.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(desc)
        
        # Operation buttons
        button_layout = QHBoxLayout()
        
        keyfile_btn = QPushButton("🔐 Open Keyfile Dialog")
        keyfile_btn.setMinimumHeight(60)
        keyfile_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:pressed {
                background-color: #6A1B9A;
            }
        """)
        keyfile_btn.clicked.connect(self.show_keyfile_dialog)
        
        button_layout.addWidget(keyfile_btn)
        layout.addLayout(button_layout)
        
        # Feature list
        features_label = QLabel("Features:")
        features_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(features_label)
        
        features = [
            "• Two-factor authentication (password + keyfile)",
            "• Cryptographically secure keyfile generation",
            "• Enhanced protection against brute force attacks",
            "• Keyfile-specific randomization patterns",
            "• Support for custom keyfile sizes"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("margin-left: 10px; color: #444;")
            layout.addWidget(feature_label)
        
        # Spacer to push content to top
        layout.addStretch()
        
        return widget
    
    def create_twofactor_operations_widget(self) -> QWidget:
        """Create the two-factor operations widget."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Multi-Image Distribution")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Distribute your data across multiple images for enhanced security and redundancy.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(desc)
        
        # Operation buttons
        button_layout = QHBoxLayout()
        
        twofactor_btn = QPushButton("🛡️ Open Multi-Image Dialog")
        twofactor_btn.setMinimumHeight(60)
        twofactor_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
        """)
        twofactor_btn.clicked.connect(self.show_twofactor_dialog)
        
        button_layout.addWidget(twofactor_btn)
        layout.addLayout(button_layout)
        
        # Feature list
        features_label = QLabel("Features:")
        features_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(features_label)
        
        features = [
            "• Split data across multiple carrier images",
            "• Configurable redundancy levels",
            "• Automatic reconstruction from partial sets",
            "• Enhanced security through distribution",
            "• Support for different image formats per fragment"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("margin-left: 10px; color: #444;")
            layout.addWidget(feature_label)
        
        # Spacer to push content to top
        layout.addStretch()
        
        return widget
    
    def create_decoy_operations_widget(self) -> QWidget:
        """Create the decoy operations widget."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Decoy Mode - Plausible Deniability")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Hide multiple datasets with different passwords, providing plausible deniability.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(desc)
        
        # Operation buttons
        button_layout = QHBoxLayout()
        
        decoy_btn = QPushButton("👻 Open Decoy Dialog")
        decoy_btn.setMinimumHeight(60)
        decoy_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
            QPushButton:pressed {
                background-color: #455A64;
            }
        """)
        decoy_btn.clicked.connect(self.show_decoy_dialog)
        
        button_layout.addWidget(decoy_btn)
        layout.addLayout(button_layout)
        
        # Feature list
        features_label = QLabel("Features:")
        features_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(features_label)
        
        features = [
            "• Multiple hidden datasets in single image",
            "• Different passwords reveal different content",
            "• Plausible deniability protection",
            "• Layered security architecture",
            "• Configurable decoy data complexity"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("margin-left: 10px; color: #444;")
            layout.addWidget(feature_label)
        
        # Spacer to push content to top
        layout.addStretch()
        
        return widget
    
    def create_self_executing_operations_widget(self) -> QWidget:
        """Create the self-executing operations widget."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Self-Executing Images")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Create images that can execute embedded code when triggered. Advanced technique for educational steganography research.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(desc)
        
        # Warning notice
        warning_label = QLabel("⚠️ WARNING: This feature is for educational purposes only. Always exercise caution with executable content!")
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("""
            background-color: #FFF3CD;
            color: #856404;
            border: 2px solid #FFE69C;
            border-radius: 8px;
            padding: 10px;
            font-weight: bold;
            margin: 5px 0;
        """)
        layout.addWidget(warning_label)
        
        # Operation buttons
        button_layout = QHBoxLayout()
        
        self_exec_btn = QPushButton("🚀 Open Self-Executing Dialog")
        self_exec_btn.setMinimumHeight(60)
        self_exec_btn.setStyleSheet("""
            QPushButton {
                background-color: #E91E63;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #C2185B;
            }
            QPushButton:pressed {
                background-color: #AD1457;
            }
        """)
        self_exec_btn.clicked.connect(self.show_self_executing_dialog)
        
        button_layout.addWidget(self_exec_btn)
        layout.addLayout(button_layout)
        
        # Feature list
        features_label = QLabel("Capabilities:")
        features_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(features_label)
        
        features = [
            "• Create polyglot files (image + executable)",
            "• Embed executable scripts in images",
            "• Support Python, JavaScript, PowerShell, Batch",
            "• Safe analysis mode for detection",
            "• Custom viewer for self-executing images",
            "• Encrypted payload support",
            "• Educational malware research techniques"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("margin-left: 10px; color: #444;")
            layout.addWidget(feature_label)
        
        # Spacer to push content to top
        layout.addStretch()
        
        return widget
    
    def create_analysis_operations_widget(self) -> QWidget:
        """Create the analysis operations widget."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Image Analysis & Capacity Assessment")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Analyze images for steganographic capacity and detect existing hidden data.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(desc)
        
        # Operation buttons
        button_layout = QHBoxLayout()
        
        analysis_btn = QPushButton("🔍 Open Analysis Dialog")
        analysis_btn.setMinimumHeight(60)
        analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #795548;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #6D4C41;
            }
            QPushButton:pressed {
                background-color: #5D4037;
            }
        """)
        analysis_btn.clicked.connect(self.show_analysis_dialog)
        
        button_layout.addWidget(analysis_btn)
        layout.addLayout(button_layout)
        
        # Feature list
        features_label = QLabel("Features:")
        features_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(features_label)
        
        features = [
            "• Calculate maximum data capacity",
            "• Assess image quality and suitability",
            "• Detect existing steganographic content",
            "• Statistical analysis and entropy measurement",
            "• Visual quality impact assessment"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("margin-left: 10px; color: #444;")
            layout.addWidget(feature_label)
        
        # Spacer to push content to top
        layout.addStretch()
        
        return widget
    
    def create_settings_widget(self) -> QWidget:
        """Create the settings widget."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Application Settings")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Configure application preferences and security settings.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(desc)
        
        # Operation buttons
        button_layout = QHBoxLayout()
        
        settings_btn = QPushButton("⚙️ Open Settings Panel")
        settings_btn.setMinimumHeight(60)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #424242;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
            QPushButton:pressed {
                background-color: #212121;
            }
        """)
        settings_btn.clicked.connect(self.show_settings_panel)
        
        button_layout.addWidget(settings_btn)
        layout.addLayout(button_layout)
        
        # Feature list
        features_label = QLabel("Configuration Options:")
        features_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(features_label)
        
        features = [
            "• Security and encryption preferences",
            "• Default file locations and formats",
            "• User interface themes and language",
            "• Logging and debugging options",
            "• Performance and memory settings"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setStyleSheet("margin-left: 10px; color: #444;")
            layout.addWidget(feature_label)
        
        # Spacer to push content to top
        layout.addStretch()
        
        return widget
    
    def create_placeholder_widget(self, title: str, description: str) -> QWidget:
        """Create a placeholder widget for operations not yet implemented."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Title
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #666; font-size: 14px; margin: 20px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Coming soon label
        coming_soon = QLabel("Coming Soon")
        coming_soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        coming_soon.setStyleSheet("""
            background-color: #FFF3CD;
            color: #856404;
            border: 2px solid #FFE69C;
            border-radius: 8px;
            padding: 15px;
            font-size: 16px;
            font-weight: bold;
            margin: 20px;
        """)
        layout.addWidget(coming_soon)
        
        return widget
    
    def init_menu_bar(self):
        """Initialize the menu bar."""
        
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # Quick Hide action
        quick_hide_action = QAction("&Quick Hide", self)
        quick_hide_action.setShortcut("Ctrl+H")
        quick_hide_action.setStatusTip("Quickly hide files in an image")
        quick_hide_action.triggered.connect(self.show_hide_dialog)
        file_menu.addAction(quick_hide_action)
        
        # Quick Extract action
        quick_extract_action = QAction("&Quick Extract", self)
        quick_extract_action.setShortcut("Ctrl+E")
        quick_extract_action.setStatusTip("Quickly extract files from an image")
        quick_extract_action.triggered.connect(self.show_extract_dialog)
        file_menu.addAction(quick_extract_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        # Unicode RTL Polyglot action
        unicode_polyglot_action = QAction("🎭 &Unicode RTL Polyglot", self)
        unicode_polyglot_action.setShortcut("Ctrl+U")
        unicode_polyglot_action.setStatusTip("Create executables disguised as PNG images")
        unicode_polyglot_action.triggered.connect(self.show_unicode_polyglot_dialog)
        tools_menu.addAction(unicode_polyglot_action)
        
        tools_menu.addSeparator()
        
        # Image Analysis action
        analysis_action = QAction("&Image Analysis", self)
        analysis_action.setStatusTip("Analyze image suitability")
        analysis_action.triggered.connect(lambda: self.nav_list.setCurrentRow(4))
        tools_menu.addAction(analysis_action)
        
        # Settings action
        settings_action = QAction("&Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.setStatusTip("Open application settings")
        settings_action.triggered.connect(lambda: self.nav_list.setCurrentRow(5))
        tools_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def init_toolbar(self):
        """Initialize the toolbar."""
        
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        
        # Quick Hide button
        hide_action = QAction("🔒 Quick Hide", self)
        hide_action.setStatusTip("Hide files in an image")
        hide_action.triggered.connect(self.show_hide_dialog)
        toolbar.addAction(hide_action)
        
        # Quick Extract button
        extract_action = QAction("🔓 Quick Extract", self)
        extract_action.setStatusTip("Extract files from an image")
        extract_action.triggered.connect(self.show_extract_dialog)
        toolbar.addAction(extract_action)
        
        toolbar.addSeparator()
        
        # Unicode RTL Polyglot button
        unicode_polyglot_toolbar_action = QAction("🎭 Unicode RTL", self)
        unicode_polyglot_toolbar_action.setStatusTip("Create executables disguised as PNG images")
        unicode_polyglot_toolbar_action.triggered.connect(self.show_unicode_polyglot_dialog)
        toolbar.addAction(unicode_polyglot_toolbar_action)
        
        # Settings button
        settings_action = QAction("⚙️ Settings", self)
        settings_action.setStatusTip("Open settings")
        settings_action.triggered.connect(lambda: self.nav_list.setCurrentRow(5))
        toolbar.addAction(settings_action)
    
    def init_status_bar(self):
        """Initialize the status bar."""
        
        self.statusBar().showMessage("Ready")
        
        # Security status label
        self.security_status_label = QLabel("🔓 No Authentication")
        self.security_status_label.setStyleSheet("""
            QLabel {
                color: #d32f2f;
                font-weight: bold;
                padding: 4px 8px;
                border: 1px solid #d32f2f;
                border-radius: 4px;
                background-color: #ffebee;
            }
        """)
        self.statusBar().addPermanentWidget(self.security_status_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def on_navigation_changed(self, current, previous):
        """Handle navigation item change."""
        
        if current is None:
            return
        
        # Get the operation key
        operation_key = current.data(Qt.ItemDataRole.UserRole)
        
        # Map operation keys to stack indices
        operation_map = {
            "basic": 0,
            "keyfile": 1,
            "twofactor": 2,
            "decoy": 3,
            "self_executing": 4,
            "analysis": 5,
            "settings": 6
        }
        
        if operation_key in operation_map:
            self.content_stack.setCurrentIndex(operation_map[operation_key])
            self.current_operation = operation_key
            self.logger.debug(f"Switched to operation: {operation_key}")
    
    def show_hide_dialog(self):
        """Show the enhanced file hiding dialog with anti-detection capabilities."""
        try:
            from ui.dialogs.enhanced_hide_files_dialog import EnhancedHideFilesDialog
            dialog = EnhancedHideFilesDialog(self.security_service, self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open hide files dialog:\n{str(e)}")
    
    def show_extract_dialog(self):
        """Show the file extraction dialog."""
        try:
            from ui.dialogs.extract_files_dialog import ExtractFilesDialog
            dialog = ExtractFilesDialog(self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open extract files dialog:\n{str(e)}")
    
    def show_keyfile_dialog(self):
        """Show the keyfile operations dialog."""
        try:
            from ui.dialogs.keyfile_dialog import KeyfileDialog
            dialog = KeyfileDialog(self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open keyfile dialog:\n{str(e)}")
    
    def show_twofactor_dialog(self):
        """Show the two-factor operations dialog."""
        try:
            from ui.dialogs.two_factor_dialog import TwoFactorDialog
            dialog = TwoFactorDialog(self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open two-factor dialog:\n{str(e)}")
    
    def show_decoy_dialog(self):
        """Show the decoy mode operations dialog."""
        try:
            from ui.dialogs.decoy_dialog import DecoyDialog
            dialog = DecoyDialog(self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open decoy dialog:\n{str(e)}")
    
    def show_self_executing_dialog(self):
        """Show the self-executing images dialog."""
        try:
            from ui.dialogs.self_executing_dialog import SelfExecutingDialog
            dialog = SelfExecutingDialog(self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open self-executing dialog:\n{str(e)}")
    
    def show_unicode_polyglot_dialog(self):
        """Show the Unicode RTL polyglot dialog."""
        try:
            from ui.dialogs.unicode_polyglot_dialog import UnicodePolyglotDialog
            dialog = UnicodePolyglotDialog(self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Unicode RTL polyglot dialog:\n{str(e)}")
    
    def show_analysis_dialog(self):
        """Show the image analysis dialog."""
        try:
            from ui.dialogs.analysis_dialog import AnalysisDialog
            dialog = AnalysisDialog(self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open analysis dialog:\n{str(e)}")
    
    def show_settings_panel(self):
        """Show the settings panel dialog."""
        try:
            from ui.components.settings_panel import SettingsPanel
            dialog = SettingsPanel(self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open settings panel:\n{str(e)}")
    
    def show_about(self):
        """Show the about dialog."""
        
        about_text = """
        <h2>InvisioVault v1.0.0</h2>
        <p>Advanced Steganography Suite</p>
        <p>Professional-grade steganography application with AES-256 encryption,<br>
        keyfile authentication, decoy mode, and multi-image distribution.</p>
        <p><b>Author:</b> Rolan (RNR)</p>
        <p><b>Purpose:</b> Educational project for learning security technologies</p>
        <p><b>Features:</b></p>
        <ul>
        <li>LSB (Least Significant Bit) steganography with ultra-fast extraction</li>
        <li>AES-256 encryption with PBKDF2 key derivation</li>
        <li>Keyfile-based two-factor authentication</li>
        <li>Decoy mode with plausible deniability</li>
        <li>Multi-image data distribution</li>
        <li>Comprehensive image analysis and preview</li>
        </ul>
        <p><b>Disclaimer:</b> For educational and legitimate privacy purposes only.</p>
        <p>© 2025 Rolan (RNR). Educational project for learning security technologies.</p>
        """
        
        QMessageBox.about(self, "About InvisioVault", about_text)
    
    def load_window_state(self):
        """Load window state from configuration."""
        
        # Load window size
        width = self.config.get(ConfigSection.INTERFACE, "window_width", 1200)
        height = self.config.get(ConfigSection.INTERFACE, "window_height", 800)
        self.resize(width, height)
        
        # Center window on screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        self.move(x, y)
    
    def save_window_state(self):
        """Save window state to configuration."""
        
        if self.config.get(ConfigSection.INTERFACE, "remember_window_state", True):
            self.config.set(ConfigSection.INTERFACE, "window_width", self.width())
            self.config.set(ConfigSection.INTERFACE, "window_height", self.height())
            self.config.save_config()
    
    def show_progress(self, message: str, maximum: int = 0):
        """Show progress bar with message."""
        
        self.statusBar().showMessage(message)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
    
    def update_progress(self, value: int):
        """Update progress bar value."""
        
        self.progress_bar.setValue(value)
    
    def hide_progress(self):
        """Hide progress bar and reset status."""
        
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Ready")
    
    def update_security_status(self):
        """Update the security status display - simplified for offline application."""
        try:
            # For offline application, show simple ready status
            self.security_status_label.setText("🔓 Local Mode - Ready")
            self.security_status_label.setStyleSheet("""
                QLabel {
                    color: #388e3c;
                    font-weight: bold;
                    padding: 4px 8px;
                    border: 1px solid #388e3c;
                    border-radius: 4px;
                    background-color: #e8f5e8;
                }
            """)
        except Exception as e:
            self.logger.error(f"Failed to update security status: {e}")
    
    
    def closeEvent(self, event):
        """Handle application close event."""
        
        # Save window state
        self.save_window_state()
        
        # Log shutdown
        self.logger.info("Application closing")
        
        # Accept the close event
        event.accept()
