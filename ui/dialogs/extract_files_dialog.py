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
from core.enhanced_steganography_engine import EnhancedSteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel
from core.multi_decoy_engine import MultiDecoyEngine


class ExtractWorkerThread(QThread):
    """Worker thread for file extraction operations."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished_successfully = Signal(list, dict)  # List of extracted file paths and extraction info
    error_occurred = Signal(str)
    
    def __init__(self, stego_path, password, output_dir, randomize):
        super().__init__()
        self.stego_path = Path(stego_path)
        self.password = password
        self.output_dir = Path(output_dir)
        self.randomize = randomize
        
        # Initialize engines
        self.stego_engine = SteganographyEngine()
        self.enhanced_engine = EnhancedSteganographyEngine(use_anti_detection=True)
        self.multi_decoy_engine = MultiDecoyEngine(SecurityLevel.MAXIMUM)
        self.logger = Logger()
    
    def run(self):
        """Execute the extraction operation with transparent decoy mode support."""
        try:
            self.status_updated.emit("Analyzing steganographic image for hidden datasets...")
            self.progress_updated.emit(10)
            
            # First try to extract using multi-decoy engine (new format)
            self.status_updated.emit("Attempting advanced format extraction...")
            self.progress_updated.emit(30)
            
            try:
                # Try to extract dataset using the provided password
                metadata = self.multi_decoy_engine.extract_dataset(
                    stego_path=self.stego_path,
                    password=self.password,
                    output_dir=self.output_dir
                )
                
                if metadata:
                    # Successfully extracted using multi-decoy format
                    self.status_updated.emit(f"Successfully extracted dataset: {metadata.get('dataset_id', 'UserFiles')}")
                    self.progress_updated.emit(100)
                    
                    # Get list of actually extracted files from metadata
                    extracted_files = []
                    if 'extracted_files' in metadata:
                        # Use the extracted files list from metadata
                        for file_info in metadata['extracted_files']:
                            if isinstance(file_info, dict) and 'path' in file_info:
                                extracted_files.append(Path(file_info['path']))
                            elif isinstance(file_info, (str, Path)):
                                extracted_files.append(Path(file_info))
                    else:
                        # Fallback: look for files with recent timestamps
                        import time
                        current_time = time.time()
                        for file_path in self.output_dir.iterdir():
                            if file_path.is_file():
                                # Only include files modified in the last 10 seconds (recently extracted)
                                if abs(file_path.stat().st_mtime - current_time) < 10:
                                    extracted_files.append(file_path)
                    
                    if not extracted_files:
                        # Last fallback: create a representative entry
                        extracted_files = [self.output_dir / f"dataset_{metadata.get('dataset_id', 'files')}"]            
                    
                    # Multi-decoy extraction info
                    multi_decoy_info = {
                        'extraction_method': 'multi_decoy',
                        'format': 'decoy_protected',
                        'dataset_id': metadata.get('dataset_id', 'Unknown')
                    }
                    self.finished_successfully.emit(extracted_files, multi_decoy_info)
                    return
                    
            except Exception as e:
                self.logger.debug(f"Multi-decoy extraction failed, trying legacy format: {e}")
                # Continue to legacy extraction method
            
            # Try enhanced steganography extraction (supports both anti-detection and regular modes)
            self.status_updated.emit("Trying enhanced steganography extraction...")
            self.progress_updated.emit(50)
            
            # Generate seed from password if randomization is enabled
            seed = None
            if self.randomize:
                import hashlib
                seed = int(hashlib.sha256(self.password.encode()).hexdigest()[:8], 16)
            
            # Since we know this is likely a fallback image, try regular extraction first for speed
            encrypted_data, extraction_info = self.enhanced_engine.extract_data_enhanced(
                stego_path=self.stego_path,
                password=self.password,
                randomize=self.randomize,
                seed=seed,
                use_anti_detection=False  # Try regular extraction first (faster)
            )
            
            # Log extraction details
            if extraction_info:
                method = extraction_info.get('successful_method', 'unknown')
                data_size = extraction_info.get('data_size', 0)
                compatibility = extraction_info.get('compatibility_note', '')
                self.logger.info(f"First extraction attempt - Method: {method}, Size: {data_size} bytes")
                if compatibility:
                    self.logger.info(f"Compatibility note: {compatibility}")
            
            if not encrypted_data:
                # If regular extraction failed, try anti-detection (slower)
                self.status_updated.emit("Trying anti-detection extraction...")
                self.progress_updated.emit(60)
                
                try:
                    # Set a reasonable timeout to avoid hanging
                    import threading
                    import time
                    from typing import List, Optional
                    
                    result: List[Optional[bytes]] = [None]
                    exception: List[Optional[Exception]] = [None]
                    
                    def extract_with_timeout():
                        try:
                            data, info = self.enhanced_engine.extract_data_enhanced(
                                stego_path=self.stego_path,
                                password=self.password,
                                randomize=self.randomize,
                                seed=seed,
                                use_anti_detection=True  # Try anti-detection
                            )
                            result[0] = data
                            # Log anti-detection extraction details
                            if info:
                                method = info.get('successful_method', 'unknown')
                                data_size = info.get('data_size', 0) if data else 0
                                compatibility = info.get('compatibility_note', '')
                                self.logger.info(f"Anti-detection extraction - Method: {method}, Size: {data_size} bytes")
                                if compatibility:
                                    self.logger.info(f"Compatibility note: {compatibility}")
                        except Exception as e:
                            exception[0] = e
                    
                    thread = threading.Thread(target=extract_with_timeout)
                    thread.start()
                    thread.join(timeout=30)  # 30 second timeout
                    
                    if thread.is_alive():
                        self.logger.warning("Anti-detection extraction timed out after 30 seconds")
                        encrypted_data = None
                    else:
                        encrypted_data = result[0]
                        if exception[0]:
                            raise exception[0]
                            
                except Exception as e:
                    self.logger.debug(f"Anti-detection extraction failed: {e}")
                    encrypted_data = None
            
            if not encrypted_data:
                raise Exception("No hidden data found in the image")
            
            self.status_updated.emit("Decrypting extracted data...")
            self.progress_updated.emit(70)
            
            # Try different security levels for decryption
            archive_data = None
            for security_level in [SecurityLevel.MAXIMUM, SecurityLevel.HIGH, SecurityLevel.STANDARD]:
                try:
                    encryption_engine = EncryptionEngine(security_level)
                    archive_data = encryption_engine.decrypt_with_metadata(encrypted_data, self.password)
                    self.logger.info(f"Successfully decrypted with {security_level} security level")
                    break
                except Exception as e:
                    self.logger.debug(f"Decryption failed with {security_level}: {e}")
                    continue
            
            if not archive_data:
                raise Exception("Failed to decrypt data - incorrect password or corrupted data")
            
            self.status_updated.emit("Extracting files from archive...")
            self.progress_updated.emit(85)
            
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
            
            # Pass extraction info if we have it
            final_info = extraction_info if extraction_info else {}
            self.finished_successfully.emit(extracted_files, final_info)
            
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
        desc = QLabel("Select a steganographic image to extract hidden files. Works with both basic and decoy-protected images.")
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
    
    def on_success(self, extracted_files, extraction_info=None):
        """Handle successful completion."""
        files_text = "\n".join([f"• {Path(f).name}" for f in extracted_files])
        
        # Create base message
        message = f"Successfully extracted {len(extracted_files)} file(s):\n\n{files_text}\n\n"
        
        # Security-conscious extraction info display
        if extraction_info:
            method = extraction_info.get('successful_method', extraction_info.get('extraction_method', 'unknown'))
            data_size = extraction_info.get('data_size', 0)
            compatibility_note = extraction_info.get('compatibility_note', '')
            
            # SECURITY: Don't reveal decoy protection details in UI
            # This prevents attackers from learning about decoy systems under coercion
            if method == 'multi_decoy':
                # Show generic success message for decoy-protected files
                # Log the real method details but don't display them
                self.logger.info(f"Multi-decoy extraction successful - dataset: {extraction_info.get('dataset_id', 'Unknown')}")
                # Just show basic extraction success
                pass  # No additional details shown to user
            else:
                # For non-decoy methods, show limited technical details
                # This helps with troubleshooting without revealing security architecture
                
                # Only show compatibility notes for fallback scenarios (helpful for user understanding)
                if compatibility_note and ('FALLBACK' in compatibility_note.upper() or 'WARNING' in compatibility_note.upper()):
                    # Clean up the note to remove technical method details
                    if 'anti-detection' in compatibility_note.lower():
                        simplified_note = "Note: Used standard extraction method as a fallback."
                    elif 'randomized' in compatibility_note.lower() and 'sequential' in compatibility_note.lower():
                        simplified_note = "Note: Image was created with basic sequential method."
                    else:
                        simplified_note = "Note: Used alternative extraction approach."
                    
                    message += f"\n💡 {simplified_note}\n"
                
                # Show data size only for non-sensitive extractions
                if data_size > 0 and method != 'multi_decoy':
                    message += f"\n📊 Data Size: {data_size:,} bytes\n"
        
        message += f"\nFiles saved to: {self.output_directory}"
        
        QMessageBox.information(
            self,
            "Success! 🎉",
            message
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
