"""
Extract Files Dialog
Professional dialog for extracting files from steganographic images.
"""

from pathlib import Path
import zipfile
import tempfile
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QLineEdit, QProgressBar, 
    QTextEdit, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPixmap

from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler
from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel


class ExtractWorkerThread(QThread):
    """Worker thread for file extraction operations."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished_successfully = Signal(list)  # List of extracted file paths
    error_occurred = Signal(str)
    
    def __init__(self, stego_path, password, output_dir, randomize):
        super().__init__()
        self.stego_path = Path(stego_path)
        self.password = password
        self.output_dir = Path(output_dir)
        self.randomize = randomize
        
        # Initialize engines
        self.stego_engine = SteganographyEngine()
        self.logger = Logger()
    
    def run(self):
        """Execute the extraction operation."""
        try:
            self.status_updated.emit("Analyzing steganographic image...")
            self.progress_updated.emit(10)
            
            # Extract encrypted data from the image
            seed = None
            if self.randomize:
                # Generate deterministic seed from password for reproducible randomization
                import hashlib
                seed_hash = hashlib.sha256(self.password.encode('utf-8')).digest()
                seed = int.from_bytes(seed_hash[:4], byteorder='big') % (2**32)
            
            encrypted_data = self.stego_engine.extract_data(
                self.stego_path,
                randomize=self.randomize,
                seed=seed
            )
            
            if not encrypted_data:
                raise Exception("No hidden data found in the image")
            
            self.status_updated.emit("Decrypting data...")
            self.progress_updated.emit(40)
            
            # Try different security levels for decryption
            archive_data = None
            for security_level in [SecurityLevel.HIGH, SecurityLevel.STANDARD, SecurityLevel.MAXIMUM]:
                try:
                    encryption_engine = EncryptionEngine(security_level)
                    archive_data = encryption_engine.decrypt_with_metadata(encrypted_data, self.password)
                    break
                except Exception:
                    continue
            
            if not archive_data:
                raise Exception("Failed to decrypt data - incorrect password or corrupted data")
            
            self.status_updated.emit("Extracting files...")
            self.progress_updated.emit(70)
            
            # Create temporary file for the archive
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                temp_zip_path = Path(temp_file.name)
                temp_file.write(archive_data)
            
            # Extract files from archive
            extracted_files = []
            with zipfile.ZipFile(temp_zip_path, 'r') as archive:
                for file_info in archive.filelist:
                    # Extract to output directory
                    output_path = self.output_dir / file_info.filename
                    
                    # Ensure output directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Extract the file
                    with archive.open(file_info) as source, open(output_path, 'wb') as dest:
                        dest.write(source.read())
                    
                    extracted_files.append(output_path)
            
            # Clean up temp file
            temp_zip_path.unlink()
            
            self.status_updated.emit("Extraction completed successfully!")
            self.progress_updated.emit(100)
            
            self.finished_successfully.emit(extracted_files)
            
        except Exception as e:
            self.logger.error(f"Extract operation failed: {e}")
            self.error_occurred.emit(str(e))


class ExtractFilesDialog(QDialog):
    """Dialog for extracting files from steganographic images."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extract Files from Image")
        self.setModal(True)
        self.resize(600, 500)
        
        # Initialize components
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        self.stego_engine = SteganographyEngine()
        
        # State variables
        self.stego_image_path = None
        self.output_directory = None
        self.worker_thread = None
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🔓 Extract Files from Image")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Select a steganographic image to extract hidden files.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Steganographic image selection
        image_group = QGroupBox("📸 Steganographic Image")
        image_layout = QVBoxLayout(image_group)
        
        self.image_label = QLabel("No image selected")
        self.image_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.image_button = QPushButton("Select Steganographic Image")
        
        # Image preview
        self.image_preview = QLabel()
        self.image_preview.setMaximumHeight(150)
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setStyleSheet("border: 1px solid #ccc; border-radius: 5px;")
        self.image_preview.hide()
        
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.image_button)
        image_layout.addWidget(self.image_preview)
        layout.addWidget(image_group)
        
        # Settings group
        settings_group = QGroupBox("⚙️ Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Password
        password_layout = QHBoxLayout()
        password_layout.addWidget(QLabel("Password:"))
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Enter the password used for hiding")
        password_layout.addWidget(self.password_input)
        settings_layout.addLayout(password_layout)
        
        # Options
        self.randomize_checkbox = QCheckBox("Use randomized LSB positioning")
        self.randomize_checkbox.setChecked(True)
        settings_layout.addWidget(self.randomize_checkbox)
        
        layout.addWidget(settings_group)
        
        # Output directory
        output_group = QGroupBox("📁 Output Directory")
        output_layout = QVBoxLayout(output_group)
        
        self.output_label = QLabel("No output directory selected")
        self.output_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.output_button = QPushButton("Select Output Directory")
        
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_button)
        layout.addWidget(output_group)
        
        # Progress section
        self.progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(self.progress_group)
        
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_group)
        self.progress_group.hide()
        
        # Buttons
        button_layout = QHBoxLayout()
        self.extract_button = QPushButton("🔓 Extract Files")
        self.extract_button.setEnabled(False)
        self.extract_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(self.extract_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        self.image_button.clicked.connect(self.select_stego_image)
        self.output_button.clicked.connect(self.select_output_directory)
        self.extract_button.clicked.connect(self.extract_files)
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.password_input.textChanged.connect(self.check_ready_state)
    
    def select_stego_image(self):
        """Select steganographic image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Steganographic Image",
            "",
            "Image Files (*.png *.bmp *.tiff *.tif);;PNG Files (*.png);;BMP Files (*.bmp);;TIFF Files (*.tiff *.tif)"
        )
        
        if file_path:
            self.stego_image_path = file_path
            self.image_label.setText(f"✅ {Path(file_path).name}")
            
            # Show image preview
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.image_preview.setPixmap(scaled_pixmap)
                self.image_preview.show()
            
            self.check_ready_state()
    
    def select_output_directory(self):
        """Select output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        
        if dir_path:
            self.output_directory = dir_path
            self.output_label.setText(f"📁 Output: {Path(dir_path).name}")
            self.check_ready_state()
    
    def check_ready_state(self):
        """Check if all requirements are met to enable extract button."""
        password_ok = len(self.password_input.text().strip()) >= 6
        ready = bool(
            self.stego_image_path and 
            self.output_directory and
            password_ok
        )
        self.extract_button.setEnabled(ready)
    
    def extract_files(self):
        """Start the file extraction operation."""
        try:
            # Validate inputs
            password = self.password_input.text().strip()
            if len(password) < 6:
                raise ValueError("Password must be at least 6 characters long")
            
            # Show progress
            self.progress_group.show()
            self.extract_button.setEnabled(False)
            self.status_label.setText("Starting extraction...")
            self.progress_bar.setValue(0)
            
            # Start worker thread
            self.worker_thread = ExtractWorkerThread(
                self.stego_image_path,
                password,
                self.output_directory,
                self.randomize_checkbox.isChecked()
            )
            
            # Connect signals
            self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
            self.worker_thread.status_updated.connect(self.status_label.setText)
            self.worker_thread.finished_successfully.connect(self.on_success)
            self.worker_thread.error_occurred.connect(self.on_error)
            
            # Start the operation
            self.worker_thread.start()
            
        except Exception as e:
            self.on_error(str(e))
    
    def on_success(self, extracted_files):
        """Handle successful completion."""
        files_text = "\n".join([f"• {Path(f).name}" for f in extracted_files])
        
        QMessageBox.information(
            self,
            "Success! 🎉",
            f"Successfully extracted {len(extracted_files)} file(s):\n\n"
            f"{files_text}\n\n"
            f"Files saved to: {self.output_directory}"
        )
        self.accept()
    
    def on_error(self, error_message):
        """Handle operation error."""
        QMessageBox.critical(
            self,
            "Extraction Failed ❌",
            f"Failed to extract files:\n\n{error_message}"
        )
        self.progress_group.hide()
        self.extract_button.setEnabled(True)
    
    def cancel_operation(self):
        """Cancel the current operation."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()
        
        self.reject()
