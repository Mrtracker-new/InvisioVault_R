#!/usr/bin/env python3
"""
InvisioVault - Advanced Steganography Suite
Main application entry point

Author: InvisioVault Team
Version: 1.0.0
License: MIT
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QDir
from PySide6.QtGui import QIcon

from ui.main_window import MainWindow
from utils.config_manager import ConfigManager
from utils.logger import Logger
from utils.error_handler import ErrorHandler


def setup_application():
    """Initialize application settings and resources."""
    # Initialize logging
    logger = Logger()
    logger.info("Starting InvisioVault application")
    
    # Initialize configuration
    config = ConfigManager()
    
    # Initialize error handling
    error_handler = ErrorHandler()
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("InvisioVault")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("InvisioVault Team")
    app.setOrganizationDomain("invisiovault.com")
    
    # Set application icon
    icon_path = project_root / "assets" / "icons" / "app.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # Note: High DPI scaling is automatically enabled in Qt6/PySide6
    # The AA_EnableHighDpiScaling and AA_UseHighDpiPixmaps attributes are deprecated
    # and no longer needed as high DPI support is built-in
    
    return app, logger, config, error_handler


def main():
    """Main application function."""
    try:
        # Setup application
        app, logger, config, error_handler = setup_application()
        
        # Create main window
        main_window = MainWindow()
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
