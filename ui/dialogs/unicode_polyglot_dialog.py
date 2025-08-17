#!/usr/bin/env python3
"""
Unicode RTL Polyglot Dialog for InVisioVault
==============================================

Specialized dialog for creating PNG/EXE polyglot files using the Unicode RTL 
extension spoofing method integrated with InVisioVault.

This dialog asks the user for:
1. PNG image to use as disguise reference
2. EXE file to disguise
3. Output settings
Then creates a disguised executable that appears as a PNG in Windows Explorer.

Author: InVisioVault Integration Team
"""

import os
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QGroupBox,
    QFormLayout, QProgressBar, QMessageBox, QFrame, QTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont, QPixmap, QIcon

from core.unicode_polyglot_engine import UnicodePolyglotEngine
from utils.logger import Logger
from utils.error_handler import ErrorHandler


class UnicodePolyglotWorker(QObject):
    """Background worker for Unicode RTL polyglot creation."""
    
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished = Signal(bool, str, str)  # success, message, output_path
    
    def __init__(self, engine: UnicodePolyglotEngine, **kwargs):
        super().__init__()
        self.engine = engine
        self.kwargs = kwargs
    
    def run(self):
        """Execute the Unicode RTL polyglot creation process."""
        try:
            self.status_updated.emit("🔍 Analyzing input files...")
            self.progress_updated.emit(10)
            
            self.status_updated.emit("🎭 Creating disguised filename...")
            self.progress_updated.emit(25)
            
            self.status_updated.emit("🖼️ Generating PNG-style icon...")
            self.progress_updated.emit(40)
            
            self.status_updated.emit("⚡ Creating Unicode RTL polyglot...")
            self.progress_updated.emit(60)
            
            # Create the Unicode RTL polyglot
            success = self.engine.create_unicode_polyglot(
                png_image_path=self.kwargs['image_path'],
                executable_path=self.kwargs['executable_path'],
                output_path=self.kwargs['output_path'],
                disguise_name=self.kwargs.get('disguise_name')
            )
            
            if success:
                self.status_updated.emit("📁 Creating companion files...")
                self.progress_updated.emit(85)
                
                self.status_updated.emit("✅ Finalizing polyglot creation...")
                self.progress_updated.emit(100)
                
                self.finished.emit(True, "Unicode RTL polyglot created successfully!", self.kwargs['output_path'])
            else:
                status = self.engine.get_last_operation_status()
                error_msg = status.get('error', 'Unknown error occurred')
                self.finished.emit(False, f"Failed to create polyglot: {error_msg}", "")
                
        except Exception as e:
            self.finished.emit(False, f"Error during creation: {str(e)}", "")


class UnicodePolyglotDialog(QDialog):
    """Dialog for creating Unicode RTL PNG/EXE polyglots."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Window setup
        self.setWindowTitle("Unicode RTL Polyglot Creator - InVisioVault")
        self.setMinimumSize(700, 500)
        self.setModal(True)
        
        # Initialize components
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.engine = UnicodePolyglotEngine()
        
        # Thread management
        self.worker_thread = None
        self.worker = None
        
        # Initialize UI
        self.setup_ui()
        
        self.logger.info("Unicode RTL Polyglot Dialog initialized")
    
    def setup_ui(self):
        """Setup the dialog user interface."""
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header section
        self.create_header(main_layout)
        
        # Input section
        self.create_input_section(main_layout)
        
        # Button section
        self.create_button_section(main_layout)
        
        # Progress section (initially hidden)
        self.create_progress_section(main_layout)
        
        # Results section
        self.create_results_section(main_layout)
        
    def create_header(self, layout):
        """Create header with title and description."""
        
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #2196F3, stop:1 #1976D2);
                border-radius: 10px;
                border: 2px solid #1565C0;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        # Title
        title_label = QLabel("🎭 Unicode RTL Polyglot Creator")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: white; font-weight: bold;")
        
        # Description
        desc_label = QLabel(
            "Create executables that appear as PNG images in Windows Explorer\n"
            "using Unicode Right-to-Left Override extension spoofing."
        )
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #E3F2FD; font-size: 14px; margin-top: 5px;")
        desc_label.setWordWrap(True)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(desc_label)
        
        layout.addWidget(header_frame)
    
    def create_input_section(self, layout):
        """Create input section for file selection."""
        
        input_group = QGroupBox("File Selection")
        input_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ccc;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        input_layout = QFormLayout(input_group)
        input_layout.setSpacing(15)
        
        # PNG image selection
        png_layout = QHBoxLayout()
        self.png_input = QLineEdit()
        self.png_input.setPlaceholderText("Select PNG image for disguise reference...")
        self.png_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        png_browse_btn = QPushButton("Browse")
        png_browse_btn.clicked.connect(self.browse_png_file)
        png_browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        png_layout.addWidget(self.png_input)
        png_layout.addWidget(png_browse_btn)
        input_layout.addRow("PNG Image:", png_layout)
        
        # Executable selection
        exe_layout = QHBoxLayout()
        self.exe_input = QLineEdit()
        self.exe_input.setPlaceholderText("Select executable file to disguise...")
        self.exe_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        exe_browse_btn = QPushButton("Browse")
        exe_browse_btn.clicked.connect(self.browse_exe_file)
        exe_browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        exe_layout.addWidget(self.exe_input)
        exe_layout.addWidget(exe_browse_btn)
        input_layout.addRow("Executable:", exe_layout)
        
        # Output directory selection
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select output directory...")
        self.output_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        output_browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_browse_btn)
        input_layout.addRow("Output Dir:", output_layout)
        
        # Disguise name (optional)
        self.disguise_name_input = QLineEdit()
        self.disguise_name_input.setPlaceholderText("Custom disguise name (optional)")
        self.disguise_name_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        input_layout.addRow("Disguise Name:", self.disguise_name_input)
        
        layout.addWidget(input_group)
    
    def create_button_section(self, layout):
        """Create button section for actions."""
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Create polyglot button
        self.create_btn = QPushButton("🎭 Create Unicode RTL Polyglot")
        self.create_btn.setMinimumHeight(50)
        self.create_btn.setStyleSheet("""
            QPushButton {
                background-color: #E91E63;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: #C2185B;
                transform: translateY(-2px);
            }
            QPushButton:pressed {
                background-color: #AD1457;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        self.create_btn.clicked.connect(self.create_polyglot)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(50)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 15px 25px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.create_btn)
        
        layout.addLayout(button_layout)
    
    def create_progress_section(self, layout):
        """Create progress section."""
        
        self.progress_frame = QFrame()
        self.progress_frame.setVisible(False)
        self.progress_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        progress_layout = QVBoxLayout(self.progress_frame)
        
        self.status_label = QLabel("Ready to create polyglot...")
        self.status_label.setStyleSheet("color: #495057; font-weight: bold;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 6px;
            }
        """)
        
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(self.progress_frame)
    
    def create_results_section(self, layout):
        """Create results section."""
        
        self.results_frame = QFrame()
        self.results_frame.setVisible(False)
        results_layout = QVBoxLayout(self.results_frame)
        
        results_group = QGroupBox("Creation Results")
        results_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #28a745;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
        """)
        results_group_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)
        
        results_group_layout.addWidget(self.results_text)
        results_layout.addWidget(results_group)
        
        layout.addWidget(self.results_frame)
    
    def browse_png_file(self):
        """Browse for PNG image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PNG Image for Disguise",
            "",
            "PNG Images (*.png);;All Files (*)"
        )
        if file_path:
            self.png_input.setText(file_path)
    
    def browse_exe_file(self):
        """Browse for executable file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Executable to Disguise",
            "",
            "Executables (*.exe);;All Files (*)"
        )
        if file_path:
            self.exe_input.setText(file_path)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        if dir_path:
            self.output_input.setText(dir_path)
    
    def create_polyglot(self):
        """Create the Unicode RTL polyglot."""
        
        try:
            # Validate inputs
            png_path = self.png_input.text().strip()
            exe_path = self.exe_input.text().strip()
            output_dir = self.output_input.text().strip()
            disguise_name = self.disguise_name_input.text().strip()
            
            # Validation
            if not exe_path or not os.path.exists(exe_path):
                QMessageBox.warning(self, "Input Error", 
                    "Please select a valid executable file to disguise.")
                return
            
            if not output_dir or not os.path.exists(output_dir):
                QMessageBox.warning(self, "Input Error", 
                    "Please select a valid output directory.")
                return
            
            # PNG is optional - we can work without it
            if png_path and not os.path.exists(png_path):
                reply = QMessageBox.question(self, "PNG Not Found", 
                    "The selected PNG image was not found. Continue with generic icon?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return
                png_path = ""  # Use generic icon
            
            # Generate output path
            exe_name = Path(exe_path).stem
            if not disguise_name:
                disguise_name = f"{exe_name}_disguised"
            
            output_path = os.path.join(output_dir, f"{disguise_name}.exe")
            
            # Show progress section
            self.progress_frame.setVisible(True)
            self.create_btn.setEnabled(False)
            
            # Create worker thread
            self.worker_thread = QThread()
            self.worker = UnicodePolyglotWorker(
                self.engine,
                image_path=png_path if png_path else None,
                executable_path=exe_path,
                output_path=output_path,
                disguise_name=disguise_name
            )
            
            # Move worker to thread
            self.worker.moveToThread(self.worker_thread)
            
            # Connect signals
            self.worker_thread.started.connect(self.worker.run)
            self.worker.progress_updated.connect(self.progress_bar.setValue)
            self.worker.status_updated.connect(self.status_label.setText)
            self.worker.finished.connect(self.on_creation_finished)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker_thread.finished.connect(self.worker.deleteLater)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
            
            # Start the thread
            self.worker_thread.start()
            
        except Exception as e:
            self.error_handler.handle_exception(e)
            QMessageBox.critical(self, "Error", f"Failed to start polyglot creation: {str(e)}")
            self.create_btn.setEnabled(True)
    
    def on_creation_finished(self, success: bool, message: str, output_path: str):
        """Handle creation completion."""
        
        # Re-enable create button
        self.create_btn.setEnabled(True)
        
        if success:
            # Show results
            self.results_frame.setVisible(True)
            
            result_text = f"""🎉 SUCCESS! Unicode RTL Polyglot Created Successfully!

📁 Output Location: {output_path}

✅ Features:
• File appears as PNG image in Windows Explorer
• Double-click to execute directly  
• Custom PNG-style icon generated
• Realistic file properties applied
• Companion files created for authenticity

🎯 Usage Instructions:
1. Navigate to the output directory
2. The disguised file will appear as a PNG image
3. Double-click to execute the hidden program
4. Use launcher scripts for detailed execution info

⚠️  Important Notes:
• Antivirus software may flag the file (false positive)
• Windows SmartScreen may show protection warnings
• For educational and research purposes only

🛠️  Additional Files Created:
• Custom PNG-style icon (.ico)
• Hidden metadata files
• Launcher script (.bat)
• Documentation (README.txt)
"""
            
            self.results_text.setPlainText(result_text)
            
            # Show success dialog
            QMessageBox.information(self, "Success", 
                f"{message}\n\nOutput created at:\n{output_path}")
            
        else:
            # Show error dialog
            QMessageBox.critical(self, "Creation Failed", message)
            
            # Hide progress
            self.progress_frame.setVisible(False)
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        # Clean up thread if running
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        
        event.accept()


def main():
    """Test the Unicode RTL Polyglot Dialog."""
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    dialog = UnicodePolyglotDialog()
    result = dialog.exec()
    
    sys.exit(result)


if __name__ == "__main__":
    main()
