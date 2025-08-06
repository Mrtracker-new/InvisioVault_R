"""
Keyfile Dialog
Two-factor authentication dialog for keyfile-based steganography operations.
"""

import secrets
import hashlib
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QLineEdit, QCheckBox, QTextEdit,
    QProgressBar, QComboBox, QMessageBox, QTabWidget, QWidget
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler
from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel


class KeyfileWorkerThread(QThread):
    """Worker thread for keyfile operations."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished_successfully = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, operation, **kwargs):
        super().__init__()
        self.operation = operation
        self.params = kwargs
        
        # Initialize engines
        self.stego_engine = SteganographyEngine()
        self.encryption_engine = EncryptionEngine(SecurityLevel.MAXIMUM)
        self.logger = Logger()
    
    def run(self):
        """Execute the keyfile operation."""
        try:
            if self.operation == "hide":
                self._execute_hide_operation()
            elif self.operation == "extract":
                self._execute_extract_operation()
            elif self.operation == "generate_keyfile":
                self._execute_generate_keyfile()
            
        except Exception as e:
            self.logger.error(f"Keyfile operation failed: {e}")
            self.error_occurred.emit(str(e))
    
    def _execute_hide_operation(self):
        """Execute hide operation with keyfile."""
        carrier_path = self.params['carrier_path']
        files_to_hide = self.params['files_to_hide']
        output_path = self.params['output_path']
        password = self.params['password']
        keyfile_path = self.params['keyfile_path']
        
        self.status_updated.emit("Reading keyfile...")
        self.progress_updated.emit(20)
        
        # Read and hash keyfile
        with open(keyfile_path, 'rb') as f:
            keyfile_data = f.read()
        keyfile_hash = hashlib.sha256(keyfile_data).hexdigest()
        
        # Combine password with keyfile hash
        combined_key = f"{password}:{keyfile_hash}"
        
        self.status_updated.emit("Preparing files...")
        self.progress_updated.emit(40)
        
        # Create file archive (simplified)
        import tempfile
        import zipfile
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_zip_path = Path(temp_file.name)
        
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file_path in files_to_hide:
                archive.write(file_path, file_path.name)
        
        with open(temp_zip_path, 'rb') as f:
            archive_data = f.read()
        temp_zip_path.unlink()
        
        self.status_updated.emit("Encrypting with keyfile...")
        self.progress_updated.emit(60)
        
        # Encrypt with combined key
        encrypted_data = self.encryption_engine.encrypt_with_metadata(archive_data, combined_key)
        
        self.status_updated.emit("Hiding data in image...")
        self.progress_updated.emit(80)
        
        # Hide encrypted data
        seed = hash(combined_key) % (2**32)
        success = self.stego_engine.hide_data(
            carrier_path, encrypted_data, output_path,
            randomize=True, seed=seed
        )
        
        if not success:
            raise Exception("Failed to hide data in image")
        
        self.status_updated.emit("Operation completed successfully!")
        self.progress_updated.emit(100)
        self.finished_successfully.emit()
    
    def _execute_extract_operation(self):
        """Execute extract operation with keyfile."""
        stego_path = self.params['stego_path']
        password = self.params['password']
        keyfile_path = self.params['keyfile_path']
        output_dir = self.params['output_dir']
        
        self.status_updated.emit("Reading keyfile...")
        self.progress_updated.emit(20)
        
        # Read and hash keyfile
        with open(keyfile_path, 'rb') as f:
            keyfile_data = f.read()
        keyfile_hash = hashlib.sha256(keyfile_data).hexdigest()
        
        # Combine password with keyfile hash
        combined_key = f"{password}:{keyfile_hash}"
        
        self.status_updated.emit("Extracting data from image...")
        self.progress_updated.emit(40)
        
        # Extract encrypted data
        encrypted_data = self.stego_engine.extract_data(stego_path, randomize=True)
        if not encrypted_data:
            raise Exception("No data found in image")
        
        self.status_updated.emit("Decrypting with keyfile...")
        self.progress_updated.emit(60)
        
        # Decrypt with combined key
        try:
            archive_data = self.encryption_engine.decrypt_with_metadata(encrypted_data, combined_key)
        except Exception:
            raise Exception("Decryption failed - invalid password or keyfile")
        
        self.status_updated.emit("Extracting files...")
        self.progress_updated.emit(80)
        
        # Extract files from archive
        import tempfile
        import zipfile
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_zip_path = Path(temp_file.name)
        
        with open(temp_zip_path, 'wb') as f:
            f.write(archive_data)
        
        with zipfile.ZipFile(temp_zip_path, 'r') as archive:
            archive.extractall(output_dir)
        
        temp_zip_path.unlink()
        
        self.status_updated.emit("Files extracted successfully!")
        self.progress_updated.emit(100)
        self.finished_successfully.emit()
    
    def _execute_generate_keyfile(self):
        """Generate a new keyfile."""
        keyfile_path = self.params['keyfile_path']
        keyfile_size = self.params.get('keyfile_size', 1024)
        
        self.status_updated.emit("Generating random keyfile...")
        self.progress_updated.emit(50)
        
        # Generate random data
        keyfile_data = secrets.token_bytes(keyfile_size)
        
        # Write to file
        with open(keyfile_path, 'wb') as f:
            f.write(keyfile_data)
        
        self.status_updated.emit("Keyfile generated successfully!")
        self.progress_updated.emit(100)
        self.finished_successfully.emit()


class KeyfileDialog(QDialog):
    """Dialog for keyfile-based steganography operations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyfile Operations - Two-Factor Authentication")
        self.setModal(True)
        self.resize(800, 700)
        
        # Initialize components
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        
        # State variables
        self.worker_thread = None
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🔐 Keyfile Operations")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Two-factor authentication using password + keyfile for enhanced security.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Tab widget for different operations
        self.tabs = QTabWidget()
        
        # Hide tab
        hide_tab = self.create_hide_tab()
        self.tabs.addTab(hide_tab, "🔒 Hide Files")
        
        # Extract tab
        extract_tab = self.create_extract_tab()
        self.tabs.addTab(extract_tab, "🔓 Extract Files")
        
        # Generate keyfile tab
        generate_tab = self.create_generate_tab()
        self.tabs.addTab(generate_tab, "🗝️ Generate Keyfile")
        
        layout.addWidget(self.tabs)
        
        # Progress section
        progress_group = QGroupBox("Operation Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.execute_button = QPushButton("Execute Operation")
        self.execute_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover { background-color: #da190b; }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(self.execute_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
    
    def create_hide_tab(self) -> QWidget:
        """Create the hide files tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Carrier image
        carrier_group = QGroupBox("📁 Carrier Image")
        carrier_layout = QVBoxLayout(carrier_group)
        
        self.hide_carrier_label = QLabel("No carrier image selected")
        self.hide_carrier_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.hide_carrier_button = QPushButton("Select Carrier Image")
        
        carrier_layout.addWidget(self.hide_carrier_label)
        carrier_layout.addWidget(self.hide_carrier_button)
        layout.addWidget(carrier_group)
        
        # Files to hide
        files_group = QGroupBox("📂 Files to Hide")
        files_layout = QVBoxLayout(files_group)
        
        self.hide_files_list = QTextEdit()
        self.hide_files_list.setMaximumHeight(100)
        self.hide_files_list.setPlaceholderText("No files selected")
        self.hide_files_list.setReadOnly(True)
        self.hide_files_button = QPushButton("Select Files")
        
        files_layout.addWidget(self.hide_files_list)
        files_layout.addWidget(self.hide_files_button)
        layout.addWidget(files_group)
        
        # Authentication
        auth_group = QGroupBox("🔐 Two-Factor Authentication")
        auth_layout = QVBoxLayout(auth_group)
        
        # Password
        password_layout = QHBoxLayout()
        password_layout.addWidget(QLabel("Password:"))
        self.hide_password_input = QLineEdit()
        self.hide_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hide_password_input.setPlaceholderText("Enter password")
        password_layout.addWidget(self.hide_password_input)
        auth_layout.addLayout(password_layout)
        
        # Keyfile
        keyfile_layout = QHBoxLayout()
        keyfile_layout.addWidget(QLabel("Keyfile:"))
        self.hide_keyfile_input = QLineEdit()
        self.hide_keyfile_input.setPlaceholderText("Select keyfile")
        self.hide_keyfile_input.setReadOnly(True)
        self.hide_keyfile_button = QPushButton("Browse")
        keyfile_layout.addWidget(self.hide_keyfile_input)
        keyfile_layout.addWidget(self.hide_keyfile_button)
        auth_layout.addLayout(keyfile_layout)
        
        layout.addWidget(auth_group)
        
        # Output
        output_group = QGroupBox("💾 Output")
        output_layout = QVBoxLayout(output_group)
        
        output_file_layout = QHBoxLayout()
        output_file_layout.addWidget(QLabel("Output Image:"))
        self.hide_output_input = QLineEdit()
        self.hide_output_input.setPlaceholderText("Choose output location")
        self.hide_output_button = QPushButton("Browse")
        output_file_layout.addWidget(self.hide_output_input)
        output_file_layout.addWidget(self.hide_output_button)
        output_layout.addLayout(output_file_layout)
        
        layout.addWidget(output_group)
        
        return widget
    
    def create_extract_tab(self) -> QWidget:
        """Create the extract files tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Steganographic image
        stego_group = QGroupBox("🖼️ Steganographic Image")
        stego_layout = QVBoxLayout(stego_group)
        
        self.extract_stego_label = QLabel("No image selected")
        self.extract_stego_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.extract_stego_button = QPushButton("Select Steganographic Image")
        
        stego_layout.addWidget(self.extract_stego_label)
        stego_layout.addWidget(self.extract_stego_button)
        layout.addWidget(stego_group)
        
        # Authentication
        auth_group = QGroupBox("🔐 Two-Factor Authentication")
        auth_layout = QVBoxLayout(auth_group)
        
        # Password
        password_layout = QHBoxLayout()
        password_layout.addWidget(QLabel("Password:"))
        self.extract_password_input = QLineEdit()
        self.extract_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.extract_password_input.setPlaceholderText("Enter password")
        password_layout.addWidget(self.extract_password_input)
        auth_layout.addLayout(password_layout)
        
        # Keyfile
        keyfile_layout = QHBoxLayout()
        keyfile_layout.addWidget(QLabel("Keyfile:"))
        self.extract_keyfile_input = QLineEdit()
        self.extract_keyfile_input.setPlaceholderText("Select keyfile")
        self.extract_keyfile_input.setReadOnly(True)
        self.extract_keyfile_button = QPushButton("Browse")
        keyfile_layout.addWidget(self.extract_keyfile_input)
        keyfile_layout.addWidget(self.extract_keyfile_button)
        auth_layout.addLayout(keyfile_layout)
        
        layout.addWidget(auth_group)
        
        # Output directory
        output_group = QGroupBox("📁 Output Directory")
        output_layout = QVBoxLayout(output_group)
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Extract to:"))
        self.extract_output_input = QLineEdit()
        self.extract_output_input.setPlaceholderText("Choose extraction directory")
        self.extract_output_button = QPushButton("Browse")
        output_dir_layout.addWidget(self.extract_output_input)
        output_dir_layout.addWidget(self.extract_output_button)
        output_layout.addLayout(output_dir_layout)
        
        layout.addWidget(output_group)
        
        return widget
    
    def create_generate_tab(self) -> QWidget:
        """Create the generate keyfile tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        info_label = QLabel(
            "Generate a secure keyfile for two-factor authentication. "
            "Store the keyfile separately from your password for maximum security."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 12px; margin: 10px; padding: 10px; background-color: #f0f8ff; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # Keyfile settings
        settings_group = QGroupBox("⚙️ Keyfile Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Keyfile Size (bytes):"))
        self.keyfile_size_combo = QComboBox()
        self.keyfile_size_combo.addItems(["512", "1024", "2048", "4096"])
        self.keyfile_size_combo.setCurrentText("1024")
        size_layout.addWidget(self.keyfile_size_combo)
        size_layout.addStretch()
        settings_layout.addLayout(size_layout)
        
        layout.addWidget(settings_group)
        
        # Output keyfile
        output_group = QGroupBox("💾 Output Keyfile")
        output_layout = QVBoxLayout(output_group)
        
        output_file_layout = QHBoxLayout()
        output_file_layout.addWidget(QLabel("Save as:"))
        self.generate_keyfile_input = QLineEdit()
        self.generate_keyfile_input.setPlaceholderText("Choose keyfile location")
        self.generate_keyfile_button = QPushButton("Browse")
        output_file_layout.addWidget(self.generate_keyfile_input)
        output_file_layout.addWidget(self.generate_keyfile_button)
        output_layout.addLayout(output_file_layout)
        
        layout.addWidget(output_group)
        
        # Security warning
        warning_label = QLabel(
            "⚠️ WARNING: Keep your keyfile secure and make backups! "
            "Without both the password and keyfile, your data cannot be recovered."
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet(
            "color: #d63031; font-weight: bold; margin: 10px; padding: 10px; "
            "background-color: #ffe6e6; border: 1px solid #ff7675; border-radius: 5px;"
        )
        layout.addWidget(warning_label)
        
        return widget
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        # Hide tab signals
        self.hide_carrier_button.clicked.connect(self.select_hide_carrier)
        self.hide_files_button.clicked.connect(self.select_hide_files)
        self.hide_keyfile_button.clicked.connect(self.select_hide_keyfile)
        self.hide_output_button.clicked.connect(self.select_hide_output)
        
        # Extract tab signals
        self.extract_stego_button.clicked.connect(self.select_extract_stego)
        self.extract_keyfile_button.clicked.connect(self.select_extract_keyfile)
        self.extract_output_button.clicked.connect(self.select_extract_output)
        
        # Generate tab signals
        self.generate_keyfile_button.clicked.connect(self.select_generate_keyfile)
        
        # Main buttons
        self.execute_button.clicked.connect(self.execute_operation)
        self.cancel_button.clicked.connect(self.reject)
    
    def select_hide_carrier(self):
        """Select carrier image for hiding."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Carrier Image", "",
            "Image Files (*.png *.bmp *.tiff *.tif);;All Files (*)"
        )
        if file_path:
            self.hide_carrier_label.setText(f"Selected: {Path(file_path).name}")
            self.hide_carrier_path = file_path
    
    def select_hide_files(self):
        """Select files to hide."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files to Hide", "",
            "All Files (*);;Text Files (*.txt);;Documents (*.pdf *.doc *.docx)"
        )
        if file_paths:
            self.hide_files_paths = file_paths
            file_list = "\n".join([Path(f).name for f in file_paths])
            self.hide_files_list.setText(file_list)
    
    def select_hide_keyfile(self):
        """Select keyfile for hiding."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Keyfile", "",
            "Keyfile (*.key);;All Files (*)"
        )
        if file_path:
            self.hide_keyfile_input.setText(file_path)
    
    def select_hide_output(self):
        """Select output location for hiding."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Steganographic Image", "",
            "PNG Files (*.png);;BMP Files (*.bmp);;TIFF Files (*.tiff)"
        )
        if file_path:
            self.hide_output_input.setText(file_path)
    
    def select_extract_stego(self):
        """Select steganographic image for extraction."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Steganographic Image", "",
            "Image Files (*.png *.bmp *.tiff *.tif);;All Files (*)"
        )
        if file_path:
            self.extract_stego_label.setText(f"Selected: {Path(file_path).name}")
            self.extract_stego_path = file_path
    
    def select_extract_keyfile(self):
        """Select keyfile for extraction."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Keyfile", "",
            "Keyfile (*.key);;All Files (*)"
        )
        if file_path:
            self.extract_keyfile_input.setText(file_path)
    
    def select_extract_output(self):
        """Select output directory for extraction."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.extract_output_input.setText(dir_path)
    
    def select_generate_keyfile(self):
        """Select location to save generated keyfile."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Keyfile", "secure.key",
            "Keyfile (*.key);;All Files (*)"
        )
        if file_path:
            self.generate_keyfile_input.setText(file_path)
    
    def execute_operation(self):
        """Execute the selected operation."""
        current_tab = self.tabs.currentIndex()
        
        try:
            if current_tab == 0:  # Hide
                self.execute_hide_operation()
            elif current_tab == 1:  # Extract
                self.execute_extract_operation()
            elif current_tab == 2:  # Generate
                self.execute_generate_operation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Operation failed: {str(e)}")
    
    def execute_hide_operation(self):
        """Execute hide operation with keyfile."""
        # Validate inputs
        if not hasattr(self, 'hide_carrier_path'):
            raise ValueError("Please select a carrier image")
        if not hasattr(self, 'hide_files_paths'):
            raise ValueError("Please select files to hide")
        if not self.hide_password_input.text():
            raise ValueError("Please enter a password")
        if not self.hide_keyfile_input.text():
            raise ValueError("Please select a keyfile")
        if not self.hide_output_input.text():
            raise ValueError("Please specify output location")
        
        # Start worker thread
        self.worker_thread = KeyfileWorkerThread(
            "hide",
            carrier_path=self.hide_carrier_path,
            files_to_hide=self.hide_files_paths,
            output_path=self.hide_output_input.text(),
            password=self.hide_password_input.text(),
            keyfile_path=self.hide_keyfile_input.text()
        )
        self.start_operation()
    
    def execute_extract_operation(self):
        """Execute extract operation with keyfile."""
        # Validate inputs
        if not hasattr(self, 'extract_stego_path'):
            raise ValueError("Please select a steganographic image")
        if not self.extract_password_input.text():
            raise ValueError("Please enter a password")
        if not self.extract_keyfile_input.text():
            raise ValueError("Please select a keyfile")
        if not self.extract_output_input.text():
            raise ValueError("Please specify output directory")
        
        # Start worker thread
        self.worker_thread = KeyfileWorkerThread(
            "extract",
            stego_path=self.extract_stego_path,
            password=self.extract_password_input.text(),
            keyfile_path=self.extract_keyfile_input.text(),
            output_dir=self.extract_output_input.text()
        )
        self.start_operation()
    
    def execute_generate_operation(self):
        """Execute keyfile generation."""
        # Validate inputs
        if not self.generate_keyfile_input.text():
            raise ValueError("Please specify keyfile location")
        
        # Start worker thread
        keyfile_size = int(self.keyfile_size_combo.currentText())
        self.worker_thread = KeyfileWorkerThread(
            "generate_keyfile",
            keyfile_path=self.generate_keyfile_input.text(),
            keyfile_size=keyfile_size
        )
        self.start_operation()
    
    def start_operation(self):
        """Start the worker thread operation."""
        # Connect worker signals
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.status_label.setText)
        self.worker_thread.finished_successfully.connect(self.on_operation_success)
        self.worker_thread.error_occurred.connect(self.on_operation_error)
        
        # Update UI state
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.execute_button.setEnabled(False)
        
        # Start operation
        self.worker_thread.start()
    
    def on_operation_success(self):
        """Handle successful operation completion."""
        self.progress_bar.setVisible(False)
        self.execute_button.setEnabled(True)
        self.status_label.setText("Operation completed successfully!")
        
        QMessageBox.information(self, "Success", "Operation completed successfully!")
    
    def on_operation_error(self, error_message):
        """Handle operation error."""
        self.progress_bar.setVisible(False)
        self.execute_button.setEnabled(True)
        self.status_label.setText(f"Error: {error_message}")
        
        QMessageBox.critical(self, "Error", f"Operation failed:\n{error_message}")
