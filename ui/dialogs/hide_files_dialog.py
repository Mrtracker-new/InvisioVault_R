"""
Hide Files Dialog
Professional dialog for hiding files in images using steganography.
"""

from pathlib import Path
import json
import zipfile
import tempfile
from typing import List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QLineEdit, QSpinBox, QCheckBox,
    QProgressBar, QTextEdit, QComboBox, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap

from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler
from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel


class HideWorkerThread(QThread):
    """Worker thread for file hiding operations."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished_successfully = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, carrier_path, files_to_hide, output_path, password, security_level, randomize):
        super().__init__()
        self.carrier_path = Path(carrier_path)
        self.files_to_hide = [Path(f) for f in files_to_hide]
        self.output_path = Path(output_path)
        self.password = password
        self.security_level = security_level
        self.randomize = randomize
        
        # Initialize engines
        self.stego_engine = SteganographyEngine()
        self.encryption_engine = EncryptionEngine(security_level)
        self.logger = Logger()
    
    def run(self):
        """Execute the hiding operation."""
        try:
            self.status_updated.emit("Preparing files...")
            self.progress_updated.emit(10)
            
            # Create temporary archive with all files
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                temp_zip_path = Path(temp_file.name)
            
            # Create archive containing all files
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
                for file_path in self.files_to_hide:
                    if file_path.exists():
                        archive.write(file_path, file_path.name)
            
            self.status_updated.emit("Reading archive data...")
            self.progress_updated.emit(30)
            
            # Read the archive data
            with open(temp_zip_path, 'rb') as f:
                archive_data = f.read()
            
            # Clean up temp file
            temp_zip_path.unlink()
            
            self.status_updated.emit("Encrypting data...")
            self.progress_updated.emit(50)
            
            # Encrypt the archive data
            encrypted_data = self.encryption_engine.encrypt_with_metadata(archive_data, self.password)
            
            self.status_updated.emit("Hiding data in image...")
            self.progress_updated.emit(70)
            
            # Hide encrypted data in the carrier image
            seed = None
            if self.randomize:
                # Generate deterministic seed from password for reproducible randomization
                import hashlib
                seed_hash = hashlib.sha256(self.password.encode('utf-8')).digest()
                seed = int.from_bytes(seed_hash[:4], byteorder='big') % (2**32)
            
            success = self.stego_engine.hide_data(
                self.carrier_path, 
                encrypted_data, 
                self.output_path,
                randomize=self.randomize,
                seed=seed
            )
            
            if not success:
                raise Exception("Failed to hide data in image")
            
            self.status_updated.emit("Operation completed successfully!")
            self.progress_updated.emit(100)
            
            self.finished_successfully.emit()
            
        except Exception as e:
            self.logger.error(f"Hide operation failed: {e}")
            self.error_occurred.emit(str(e))


class HideFilesDialog(QDialog):
    """Dialog for hiding files in images using steganography."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hide Files in Image")
        self.setModal(True)
        self.resize(700, 600)
        
        # Initialize components
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        self.stego_engine = SteganographyEngine()
        
        # State variables
        self.carrier_image_path = None
        self.files_to_hide = []
        self.output_path = None
        self.worker_thread = None
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🔒 Hide Files in Image")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Select files to hide in a carrier image using advanced steganography and encryption.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Carrier image selection
        carrier_group = QGroupBox("📁 Carrier Image")
        carrier_layout = QVBoxLayout(carrier_group)
        
        self.carrier_label = QLabel("No carrier image selected")
        self.carrier_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.carrier_button = QPushButton("Select Carrier Image (PNG, BMP, TIFF)")
        
        # Image preview
        self.image_preview = QLabel()
        self.image_preview.setMaximumHeight(150)
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setStyleSheet("border: 1px solid #ccc; border-radius: 5px;")
        self.image_preview.hide()
        
        carrier_layout.addWidget(self.carrier_label)
        carrier_layout.addWidget(self.carrier_button)
        carrier_layout.addWidget(self.image_preview)
        layout.addWidget(carrier_group)
        
        # Files to hide selection
        files_group = QGroupBox("📂 Files to Hide")
        files_layout = QVBoxLayout(files_group)
        
        self.files_list = QTextEdit()
        self.files_list.setMaximumHeight(120)
        self.files_list.setPlaceholderText("No files selected")
        self.files_list.setReadOnly(True)
        self.files_button = QPushButton("Select Files to Hide")
        
        files_layout.addWidget(self.files_list)
        files_layout.addWidget(self.files_button)
        layout.addWidget(files_group)
        
        # Settings group
        settings_group = QGroupBox("⚙️ Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Password
        password_layout = QHBoxLayout()
        password_layout.addWidget(QLabel("Password:"))
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Enter a strong password")
        password_layout.addWidget(self.password_input)
        settings_layout.addLayout(password_layout)
        
        # Security level
        security_layout = QHBoxLayout()
        security_layout.addWidget(QLabel("Security Level:"))
        self.security_combo = QComboBox()
        self.security_combo.addItems(["Standard (100K iterations)", "High (500K iterations)", "Maximum (1M iterations)"])
        self.security_combo.setCurrentIndex(1)  # Default to High
        security_layout.addWidget(self.security_combo)
        settings_layout.addLayout(security_layout)
        
        # Options
        self.randomize_checkbox = QCheckBox("Use randomized LSB positioning (recommended)")
        self.randomize_checkbox.setChecked(True)
        settings_layout.addWidget(self.randomize_checkbox)
        
        layout.addWidget(settings_group)
        
        # Output location
        output_group = QGroupBox("💾 Output Location")
        output_layout = QVBoxLayout(output_group)
        
        self.output_label = QLabel("No output location selected")
        self.output_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.output_button = QPushButton("Select Output Location")
        
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
        self.hide_button = QPushButton("🔒 Hide Files")
        self.hide_button.setEnabled(False)
        self.hide_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
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
        button_layout.addWidget(self.hide_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        self.carrier_button.clicked.connect(self.select_carrier_image)
        self.files_button.clicked.connect(self.select_files)
        self.output_button.clicked.connect(self.select_output)
        self.hide_button.clicked.connect(self.hide_files)
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.password_input.textChanged.connect(self.check_ready_state)
    
    def select_carrier_image(self):
        """Select carrier image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Carrier Image",
            "",
            "Image Files (*.png *.bmp *.tiff *.tif);;PNG Files (*.png);;BMP Files (*.bmp);;TIFF Files (*.tiff *.tif)"
        )
        
        if file_path:
            self.carrier_image_path = file_path
            
            # Analyze image capacity
            try:
                capacity = self.stego_engine.calculate_capacity(Path(file_path))
                analysis = self.stego_engine.analyze_image_suitability(Path(file_path))
                
                capacity_mb = capacity / (1024 * 1024)
                suitability = analysis.get('suitability_score', 0)
                
                self.carrier_label.setText(
                    f"✅ {Path(file_path).name}\n"
                    f"📊 Capacity: {capacity_mb:.2f} MB\n"
                    f"⭐ Suitability: {suitability}/10"
                )
                
                # Show image preview
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.image_preview.setPixmap(scaled_pixmap)
                    self.image_preview.show()
                
            except Exception as e:
                self.carrier_label.setText(f"❌ Error analyzing image: {str(e)}")
            
            self.check_ready_state()
    
    def select_files(self):
        """Select files to hide."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files to Hide",
            "",
            "All Files (*.*);;Text Files (*.txt);;Document Files (*.pdf *.doc *.docx);;Image Files (*.jpg *.png *.gif)"
        )
        
        if file_paths:
            self.files_to_hide = file_paths
            
            # Calculate total size
            total_size = 0
            files_info = []
            
            for path in file_paths:
                try:
                    file_path = Path(path)
                    size = file_path.stat().st_size
                    total_size += size
                    size_mb = size / (1024 * 1024)
                    files_info.append(f"📄 {file_path.name} ({size_mb:.2f} MB)")
                except Exception:
                    files_info.append(f"❌ {Path(path).name} (Error reading)")
            
            total_mb = total_size / (1024 * 1024)
            files_text = "\n".join(files_info)
            files_text += f"\n\n📊 Total size: {total_mb:.2f} MB"
            
            self.files_list.setText(files_text)
            self.check_ready_state()
    
    def select_output(self):
        """Select output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Steganographic Image As",
            "",
            "PNG Files (*.png);;BMP Files (*.bmp);;TIFF Files (*.tiff)"
        )
        
        if file_path:
            self.output_path = file_path
            self.output_label.setText(f"💾 Output: {Path(file_path).name}")
            self.check_ready_state()
    
    def check_ready_state(self):
        """Check if all requirements are met to enable hide button."""
        password_ok = len(self.password_input.text().strip()) >= 6
        ready = bool(
            self.carrier_image_path and 
            self.files_to_hide and 
            self.output_path and
            password_ok
        )
        self.hide_button.setEnabled(ready)
    
    def hide_files(self):
        """Start the file hiding operation."""
        try:
            # Validate inputs
            password = self.password_input.text().strip()
            if len(password) < 6:
                raise ValueError("Password must be at least 6 characters long")
            
            # Get security level
            security_levels = [SecurityLevel.STANDARD, SecurityLevel.HIGH, SecurityLevel.MAXIMUM]
            security_level = security_levels[self.security_combo.currentIndex()]
            
            # Show progress
            self.progress_group.show()
            self.hide_button.setEnabled(False)
            self.status_label.setText("Starting operation...")
            self.progress_bar.setValue(0)
            
            # Start worker thread
            self.worker_thread = HideWorkerThread(
                self.carrier_image_path,
                self.files_to_hide,
                self.output_path,
                password,
                security_level,
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
    
    def on_success(self):
        """Handle successful completion."""
        QMessageBox.information(
            self,
            "Success! 🎉",
            f"Files successfully hidden in:\n{self.output_path}\n\n"
            "Remember your password - you'll need it to extract the files!"
        )
        self.accept()
    
    def on_error(self, error_message):
        """Handle operation error."""
        QMessageBox.critical(
            self,
            "Operation Failed ❌",
            f"Failed to hide files:\n\n{error_message}"
        )
        self.progress_group.hide()
        self.hide_button.setEnabled(True)
    
    def cancel_operation(self):
        """Cancel the current operation."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()
        
        self.reject()
