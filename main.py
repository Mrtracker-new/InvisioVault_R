#!/usr/bin/env python3
"""
InvisioVault - Advanced Steganography Suite
Main application entry point

Author: Rolan (RNR)
Version: 1.0.0
License: MIT
Purpose: Educational project for learning security technologies

DISCLAIMER: This software is developed for educational and research purposes.
Users are responsible for compliance with local laws and regulations.
"""

import sys
import os
import warnings
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress Qt warnings about unhandled schemes (common Windows issue)
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.mime=false'

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QDir, qInstallMessageHandler, QtMsgType
from PySide6.QtGui import QIcon

from ui.main_window import MainWindow
from utils.config_manager import ConfigManager
from utils.logger import Logger
from utils.error_handler import ErrorHandler
from core.security.security_service import SecurityService


def qt_message_handler(msg_type, context, message):
    """Custom Qt message handler to filter unwanted warnings."""
    # Filter out known harmless Qt warnings
    if "Unhandled scheme" in message:
        return  # Suppress this specific warning
    if "QWindowsNativeFileDialogBase::shellItem" in message:
        return  # Suppress file dialog warnings
    if "data" in message and "scheme" in message:
        return  # Suppress scheme-related warnings
    
    # Allow other Qt messages to be printed (optional)
    if msg_type == QtMsgType.QtWarningMsg:
        print(f"Qt Warning: {message}")
    elif msg_type == QtMsgType.QtCriticalMsg:
        print(f"Qt Critical: {message}")
    elif msg_type == QtMsgType.QtFatalMsg:
        print(f"Qt Fatal: {message}")


def setup_application():
    """Initialize application settings and resources."""
    # Install Qt message handler to filter warnings
    qInstallMessageHandler(qt_message_handler)
    
    # Initialize logging
    logger = Logger()
    logger.info("Starting InvisioVault application")
    
    # Initialize configuration
    config = ConfigManager()
    
    # Initialize error handling
    error_handler = ErrorHandler()
    
    # Initialize security service
    security_service = SecurityService()
    logger.info("Security service initialized")
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("InvisioVault")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Rolan (RNR) - Educational Project")
    app.setOrganizationDomain("github.com/Mrtracker-new")
    
    # Set application icon
    icon_path = project_root / "assets" / "icons" / "app.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # Note: High DPI scaling is automatically enabled in Qt6/PySide6
    # The AA_EnableHighDpiScaling and AA_UseHighDpiPixmaps attributes are deprecated
    # and no longer needed as high DPI support is built-in
    
    return app, logger, config, error_handler, security_service


def main():
    """Main application function."""
    try:
        # Setup application
        app, logger, config, error_handler, security_service = setup_application()
        
        # Create main window
        main_window = MainWindow(security_service)
        main_window.show()
        
        logger.info("Application started successfully")
        
        # Run application
        return app.exec()
        
    except Exception as e:
        print(f"Fatal error starting InvisioVault: {e}")
        if 'error_handler' in locals():
            error_handler.handle_exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
