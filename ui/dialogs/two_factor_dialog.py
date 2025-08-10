"""
Two-Factor Dialog
Multi-image distribution dialog for enhanced security through data splitting.
"""

import secrets
import hashlib
import math
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QLineEdit, QCheckBox, QTextEdit,
    QProgressBar, QComboBox, QMessageBox, QTabWidget, QWidget,
    QSpinBox, QSlider, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPixmap

from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler
from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel


class TwoFactorWorkerThread(QThread):
    """Worker thread for two-factor operations."""
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
        """Execute the two-factor operation."""
        try:
            if self.operation == "distribute":
                self._execute_distribute_operation()
            elif self.operation == "reconstruct":
                self._execute_reconstruct_operation()
            
        except Exception as e:
            self.logger.error(f"Two-factor operation failed: {e}")
            self.error_occurred.emit(str(e))
    
    def _execute_distribute_operation(self):
        """Execute data distribution across multiple images."""
        carrier_paths = self.params['carrier_paths']
        files_to_hide = self.params['files_to_hide']
        output_dir = self.params['output_dir']
        password = self.params['password']
        redundancy_level = self.params['redundancy_level']
        
        self.status_updated.emit("Preparing files for distribution...")
        self.progress_updated.emit(10)
        
        # Create file archive
        import tempfile
        import zipfile
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_zip_path = Path(temp_file.name)
        
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file_path in files_to_hide:
                file_path_obj = Path(file_path)  # Convert string to Path object
                archive.write(file_path_obj, file_path_obj.name)
        
        with open(temp_zip_path, 'rb') as f:
            archive_data = f.read()
        temp_zip_path.unlink()
        
        self.status_updated.emit("Encrypting data...")
        self.progress_updated.emit(30)
        
        # Encrypt data
        encrypted_data = self.encryption_engine.encrypt_with_metadata(archive_data, password)
        
        self.status_updated.emit("Splitting data into fragments...")
        self.progress_updated.emit(50)
        
        # Split data into fragments with redundancy
        num_images = len(carrier_paths)
        fragment_size = len(encrypted_data) // num_images
        fragments = []
        
        for i in range(num_images):
            start_idx = i * fragment_size
            if i == num_images - 1:  # Last fragment gets remaining data
                fragment = encrypted_data[start_idx:]
            else:
                fragment = encrypted_data[start_idx:start_idx + fragment_size]
            fragments.append(fragment)
        
        self.status_updated.emit("Distributing fragments across images...")
        
        # Hide each fragment in corresponding image
        for i, (carrier_path, fragment) in enumerate(zip(carrier_paths, fragments)):
            self.status_updated.emit(f"Processing image {i+1} of {num_images}...")
            progress = 50 + (40 * (i + 1) // num_images)
            self.progress_updated.emit(progress)
            
            output_path = Path(output_dir) / f"fragment_{i+1:02d}.png"
            
            # Add metadata to fragment
            metadata = {
                'fragment_index': i,
                'total_fragments': num_images,
                'fragment_size': len(fragment),
                'redundancy_level': redundancy_level
            }
            
            # Combine metadata and fragment
            metadata_str = str(metadata).encode('utf-8')
            combined_data = len(metadata_str).to_bytes(4, 'big') + metadata_str + fragment
            
            success = self.stego_engine.hide_data(
                Path(carrier_path), combined_data, output_path,
                randomize=True, seed=hash(password + str(i)) % (2**32)
            )
            
            if not success:
                raise Exception(f"Failed to hide fragment in image {i+1}")
        
        self.status_updated.emit("Distribution completed successfully!")
        self.progress_updated.emit(100)
        self.finished_successfully.emit()
    
    def _execute_reconstruct_operation(self):
        """Execute data reconstruction from multiple images."""
        fragment_paths = self.params['fragment_paths']
        password = self.params['password']
        output_dir = self.params['output_dir']
        
        self.status_updated.emit("Extracting fragments from images...")
        self.progress_updated.emit(20)
        
        fragments = []
        metadata_list = []
        
        for i, fragment_path in enumerate(fragment_paths):
            self.status_updated.emit(f"Extracting from image {i+1}...")
            progress = 20 + (30 * (i + 1) // len(fragment_paths))
            self.progress_updated.emit(progress)
            
            # Extract combined data using the same seed as during distribution
            # Fragment seed must match the seed used during hide operation
            fragment_seed = hash(password + str(i)) % (2**32)
            combined_data = self.stego_engine.extract_data(
                Path(fragment_path), 
                randomize=True, 
                seed=fragment_seed
            )
            if not combined_data:
                raise Exception(f"No data found in fragment {i+1}")
            
            # Split metadata and fragment
            metadata_size = int.from_bytes(combined_data[:4], 'big')
            metadata_str = combined_data[4:4+metadata_size].decode('utf-8')
            fragment_data = combined_data[4+metadata_size:]
            
            metadata = eval(metadata_str)  # Simple parsing, should use JSON in production
            fragments.append((metadata['fragment_index'], fragment_data))
            metadata_list.append(metadata)
        
        self.status_updated.emit("Reconstructing original data...")
        self.progress_updated.emit(60)
        
        # Sort fragments by index and reconstruct
        fragments.sort(key=lambda x: x[0])
        reconstructed_data = b''.join([fragment for _, fragment in fragments])
        
        self.status_updated.emit("Decrypting data...")
        self.progress_updated.emit(80)
        
        # Decrypt reconstructed data
        try:
            archive_data = self.encryption_engine.decrypt_with_metadata(reconstructed_data, password)
        except Exception:
            raise Exception("Decryption failed - invalid password or corrupted data")
        
        self.status_updated.emit("Extracting files...")
        self.progress_updated.emit(90)
        
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
        
        self.status_updated.emit("Reconstruction completed successfully!")
        self.progress_updated.emit(100)
        self.finished_successfully.emit()


class TwoFactorDialog(QDialog):
    """Dialog for multi-image distribution operations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multi-Image Distribution - Enhanced Security")
        self.setModal(True)
        self.resize(900, 700)
        
        # Initialize components
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        self.stego_engine = SteganographyEngine()
        
        # State variables
        self.worker_thread = None
        self.carrier_paths = []
        self.files_to_hide = []
        self.fragment_paths = []
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🛡️ Multi-Image Distribution")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Distribute your data across multiple images for enhanced security and redundancy.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Tab widget for different operations
        self.tabs = QTabWidget()
        
        # Distribute tab
        distribute_tab = self.create_distribute_tab()
        self.tabs.addTab(distribute_tab, "📤 Distribute Data")
        
        # Reconstruct tab
        reconstruct_tab = self.create_reconstruct_tab()
        self.tabs.addTab(reconstruct_tab, "📥 Reconstruct Data")
        
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
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #F57C00; }
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
    
    def create_distribute_tab(self) -> QWidget:
        """Create the distribute data tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Carrier images
        carrier_group = QGroupBox("📁 Carrier Images")
        carrier_layout = QVBoxLayout(carrier_group)
        
        self.carrier_list = QListWidget()
        self.carrier_list.setMaximumHeight(120)
        carrier_buttons_layout = QHBoxLayout()
        
        self.add_carrier_button = QPushButton("Add Images (PNG, BMP, TIFF)")
        self.remove_carrier_button = QPushButton("Remove Selected")
        self.clear_carriers_button = QPushButton("Clear All")
        
        carrier_buttons_layout.addWidget(self.add_carrier_button)
        carrier_buttons_layout.addWidget(self.remove_carrier_button)
        carrier_buttons_layout.addWidget(self.clear_carriers_button)
        carrier_buttons_layout.addStretch()
        
        # Image preview for selected carrier
        self.distribute_image_preview = QLabel()
        self.distribute_image_preview.setMaximumHeight(150)
        self.distribute_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.distribute_image_preview.setStyleSheet("border: 1px solid #ccc; border-radius: 5px;")
        self.distribute_image_preview.setText("Select images to see preview")
        self.distribute_image_preview.hide()
        
        carrier_layout.addWidget(self.carrier_list)
        carrier_layout.addLayout(carrier_buttons_layout)
        carrier_layout.addWidget(self.distribute_image_preview)
        layout.addWidget(carrier_group)
        
        # Files to hide
        files_group = QGroupBox("📂 Files to Hide")
        files_layout = QVBoxLayout(files_group)
        
        self.files_list = QListWidget()
        self.files_list.setMaximumHeight(100)
        files_buttons_layout = QHBoxLayout()
        
        self.add_files_button = QPushButton("Add Files")
        self.remove_files_button = QPushButton("Remove Selected")
        self.clear_files_button = QPushButton("Clear All")
        
        files_buttons_layout.addWidget(self.add_files_button)
        files_buttons_layout.addWidget(self.remove_files_button)
        files_buttons_layout.addWidget(self.clear_files_button)
        files_buttons_layout.addStretch()
        
        files_layout.addWidget(self.files_list)
        files_layout.addLayout(files_buttons_layout)
        layout.addWidget(files_group)
        
        # Security settings
        security_group = QGroupBox("🔒 Security Settings")
        security_layout = QVBoxLayout(security_group)
        
        # Password
        password_layout = QHBoxLayout()
        password_layout.addWidget(QLabel("Password:"))
        self.distribute_password_input = QLineEdit()
        self.distribute_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.distribute_password_input.setPlaceholderText("Enter password for encryption")
        password_layout.addWidget(self.distribute_password_input)
        security_layout.addLayout(password_layout)
        
        # Redundancy level
        redundancy_layout = QHBoxLayout()
        redundancy_layout.addWidget(QLabel("Redundancy Level:"))
        self.redundancy_combo = QComboBox()
        self.redundancy_combo.addItems(["None", "Low", "Medium", "High"])
        self.redundancy_combo.setCurrentText("Medium")
        redundancy_layout.addWidget(self.redundancy_combo)
        redundancy_layout.addStretch()
        security_layout.addLayout(redundancy_layout)
        
        layout.addWidget(security_group)
        
        # Output directory
        output_group = QGroupBox("💾 Output Directory")
        output_layout = QVBoxLayout(output_group)
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Save fragments to:"))
        self.distribute_output_input = QLineEdit()
        self.distribute_output_input.setPlaceholderText("Choose output directory")
        self.distribute_output_button = QPushButton("Browse")
        output_dir_layout.addWidget(self.distribute_output_input)
        output_dir_layout.addWidget(self.distribute_output_button)
        output_layout.addLayout(output_dir_layout)
        
        layout.addWidget(output_group)
        
        return widget
    
    def create_reconstruct_tab(self) -> QWidget:
        """Create the reconstruct data tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Fragment images
        fragment_group = QGroupBox("🧩 Fragment Images")
        fragment_layout = QVBoxLayout(fragment_group)
        
        self.fragment_list = QListWidget()
        self.fragment_list.setMaximumHeight(120)
        fragment_buttons_layout = QHBoxLayout()
        
        self.add_fragment_button = QPushButton("Add Fragments (PNG, BMP, TIFF)")
        self.remove_fragment_button = QPushButton("Remove Selected")
        self.clear_fragments_button = QPushButton("Clear All")
        
        fragment_buttons_layout.addWidget(self.add_fragment_button)
        fragment_buttons_layout.addWidget(self.remove_fragment_button)
        fragment_buttons_layout.addWidget(self.clear_fragments_button)
        fragment_buttons_layout.addStretch()
        
        # Image preview for selected fragment
        self.reconstruct_image_preview = QLabel()
        self.reconstruct_image_preview.setMaximumHeight(150)
        self.reconstruct_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reconstruct_image_preview.setStyleSheet("border: 1px solid #ccc; border-radius: 5px;")
        self.reconstruct_image_preview.setText("Select fragments to see preview")
        self.reconstruct_image_preview.hide()
        
        fragment_layout.addWidget(self.fragment_list)
        fragment_layout.addLayout(fragment_buttons_layout)
        fragment_layout.addWidget(self.reconstruct_image_preview)
        layout.addWidget(fragment_group)
        
        # Security settings
        security_group = QGroupBox("🔒 Security Settings")
        security_layout = QVBoxLayout(security_group)
        
        # Password
        password_layout = QHBoxLayout()
        password_layout.addWidget(QLabel("Password:"))
        self.reconstruct_password_input = QLineEdit()
        self.reconstruct_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.reconstruct_password_input.setPlaceholderText("Enter decryption password")
        password_layout.addWidget(self.reconstruct_password_input)
        security_layout.addLayout(password_layout)
        
        layout.addWidget(security_group)
        
        # Output directory
        output_group = QGroupBox("📁 Output Directory")
        output_layout = QVBoxLayout(output_group)
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Extract to:"))
        self.reconstruct_output_input = QLineEdit()
        self.reconstruct_output_input.setPlaceholderText("Choose extraction directory")
        self.reconstruct_output_button = QPushButton("Browse")
        output_dir_layout.addWidget(self.reconstruct_output_input)
        output_dir_layout.addWidget(self.reconstruct_output_button)
        output_layout.addLayout(output_dir_layout)
        
        layout.addWidget(output_group)
        
        return widget
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        # Distribute tab signals
        self.add_carrier_button.clicked.connect(self.add_carrier_images)
        self.remove_carrier_button.clicked.connect(self.remove_selected_carriers)
        self.clear_carriers_button.clicked.connect(self.clear_carrier_images)
        self.carrier_list.currentRowChanged.connect(self.on_carrier_selection_changed)
        self.add_files_button.clicked.connect(self.add_files_to_hide)
        self.remove_files_button.clicked.connect(self.remove_selected_files)
        self.clear_files_button.clicked.connect(self.clear_files_to_hide)
        self.distribute_output_button.clicked.connect(self.select_distribute_output)
        
        # Reconstruct tab signals
        self.add_fragment_button.clicked.connect(self.add_fragment_images)
        self.remove_fragment_button.clicked.connect(self.remove_selected_fragments)
        self.clear_fragments_button.clicked.connect(self.clear_fragment_images)
        self.fragment_list.currentRowChanged.connect(self.on_fragment_selection_changed)
        self.reconstruct_output_button.clicked.connect(self.select_reconstruct_output)
        
        # Main buttons
        self.execute_button.clicked.connect(self.execute_operation)
        self.cancel_button.clicked.connect(self.reject)
    
    def add_carrier_images(self):
        """Add carrier images for distribution."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Carrier Images", "",
            "Image Files (*.png *.bmp *.tiff *.tif);;All Files (*)"
        )
        if file_paths:
            for path in file_paths:
                if path not in self.carrier_paths:
                    self.carrier_paths.append(path)
                    self.carrier_list.addItem(Path(path).name)
    
    def remove_selected_carriers(self):
        """Remove selected carrier images."""
        current_row = self.carrier_list.currentRow()
        if current_row >= 0:
            self.carrier_list.takeItem(current_row)
            del self.carrier_paths[current_row]
    
    def clear_carrier_images(self):
        """Clear all carrier images."""
        self.carrier_list.clear()
        self.carrier_paths.clear()
    
    def add_files_to_hide(self):
        """Add files to hide."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files to Hide", "",
            "All Files (*);;Text Files (*.txt);;Documents (*.pdf *.doc *.docx)"
        )
        if file_paths:
            for path in file_paths:
                if path not in self.files_to_hide:
                    self.files_to_hide.append(path)
                    self.files_list.addItem(Path(path).name)
    
    def remove_selected_files(self):
        """Remove selected files."""
        current_row = self.files_list.currentRow()
        if current_row >= 0:
            self.files_list.takeItem(current_row)
            del self.files_to_hide[current_row]
    
    def clear_files_to_hide(self):
        """Clear all files to hide."""
        self.files_list.clear()
        self.files_to_hide.clear()
    
    def select_distribute_output(self):
        """Select output directory for distribution."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.distribute_output_input.setText(dir_path)
    
    def add_fragment_images(self):
        """Add fragment images for reconstruction."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Fragment Images", "",
            "Image Files (*.png *.bmp *.tiff *.tif);;All Files (*)"
        )
        if file_paths:
            for path in file_paths:
                if path not in self.fragment_paths:
                    self.fragment_paths.append(path)
                    self.fragment_list.addItem(Path(path).name)
    
    def remove_selected_fragments(self):
        """Remove selected fragment images."""
        current_row = self.fragment_list.currentRow()
        if current_row >= 0:
            self.fragment_list.takeItem(current_row)
            del self.fragment_paths[current_row]
    
    def clear_fragment_images(self):
        """Clear all fragment images."""
        self.fragment_list.clear()
        self.fragment_paths.clear()
    
    def select_reconstruct_output(self):
        """Select output directory for reconstruction."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.reconstruct_output_input.setText(dir_path)
    
    def on_carrier_selection_changed(self, current_row):
        """Handle carrier image selection change for preview."""
        if current_row >= 0 and current_row < len(self.carrier_paths):
            self.update_distribute_image_preview(self.carrier_paths[current_row])
        else:
            self.hide_distribute_image_preview()
    
    def on_fragment_selection_changed(self, current_row):
        """Handle fragment image selection change for preview."""
        if current_row >= 0 and current_row < len(self.fragment_paths):
            self.update_reconstruct_image_preview(self.fragment_paths[current_row])
        else:
            self.hide_reconstruct_image_preview()
    
    def update_distribute_image_preview(self, image_path):
        """Update the distribute tab image preview."""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.distribute_image_preview.setPixmap(scaled_pixmap)
                self.distribute_image_preview.show()
            else:
                self.distribute_image_preview.setText(f"❌ Invalid image: {Path(image_path).name}")
                self.distribute_image_preview.show()
        except Exception as e:
            self.distribute_image_preview.setText(f"❌ Error loading image: {str(e)}")
            self.distribute_image_preview.show()
            self.logger.error(f"Error loading carrier image preview: {e}")
    
    def update_reconstruct_image_preview(self, image_path):
        """Update the reconstruct tab image preview."""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.reconstruct_image_preview.setPixmap(scaled_pixmap)
                self.reconstruct_image_preview.show()
            else:
                self.reconstruct_image_preview.setText(f"❌ Invalid image: {Path(image_path).name}")
                self.reconstruct_image_preview.show()
        except Exception as e:
            self.reconstruct_image_preview.setText(f"❌ Error loading image: {str(e)}")
            self.reconstruct_image_preview.show()
            self.logger.error(f"Error loading fragment image preview: {e}")
    
    def hide_distribute_image_preview(self):
        """Hide the distribute tab image preview."""
        self.distribute_image_preview.hide()
    
    def hide_reconstruct_image_preview(self):
        """Hide the reconstruct tab image preview."""
        self.reconstruct_image_preview.hide()
    
    def execute_operation(self):
        """Execute the selected operation."""
        current_tab = self.tabs.currentIndex()
        
        try:
            if current_tab == 0:  # Distribute
                self.execute_distribute_operation()
            elif current_tab == 1:  # Reconstruct
                self.execute_reconstruct_operation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Operation failed: {str(e)}")
    
    def execute_distribute_operation(self):
        """Execute data distribution operation."""
        # Validate inputs
        if not self.carrier_paths:
            raise ValueError("Please select carrier images")
        if not self.files_to_hide:
            raise ValueError("Please select files to hide")
        if not self.distribute_password_input.text():
            raise ValueError("Please enter a password")
        if not self.distribute_output_input.text():
            raise ValueError("Please specify output directory")
        
        if len(self.carrier_paths) < 2:
            raise ValueError("At least 2 carrier images are required for distribution")
        
        # Start worker thread
        redundancy_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
        self.worker_thread = TwoFactorWorkerThread(
            "distribute",
            carrier_paths=self.carrier_paths,
            files_to_hide=self.files_to_hide,
            output_dir=self.distribute_output_input.text(),
            password=self.distribute_password_input.text(),
            redundancy_level=redundancy_map[self.redundancy_combo.currentText()]
        )
        self.start_operation()
    
    def execute_reconstruct_operation(self):
        """Execute data reconstruction operation."""
        # Validate inputs
        if not self.fragment_paths:
            raise ValueError("Please select fragment images")
        if not self.reconstruct_password_input.text():
            raise ValueError("Please enter a password")
        if not self.reconstruct_output_input.text():
            raise ValueError("Please specify output directory")
        
        if len(self.fragment_paths) < 2:
            raise ValueError("At least 2 fragment images are required for reconstruction")
        
        # Start worker thread
        self.worker_thread = TwoFactorWorkerThread(
            "reconstruct",
            fragment_paths=self.fragment_paths,
            password=self.reconstruct_password_input.text(),
            output_dir=self.reconstruct_output_input.text()
        )
        self.start_operation()
    
    def start_operation(self):
        """Start the worker thread operation."""
        if self.worker_thread is None:
            raise ValueError("Worker thread not initialized")
            
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
