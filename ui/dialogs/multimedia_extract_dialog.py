"""
Multimedia Extract Dialog
Professional dialog for extracting files from video and audio files using steganography.
"""

from pathlib import Path
import zipfile
import tempfile
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QLineEdit, QTextEdit, QComboBox, 
    QMessageBox, QCheckBox, QGridLayout, QProgressBar, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler
from core.video_steganography_engine import VideoSteganographyEngine
from core.audio_steganography_engine import AudioSteganographyEngine
from core.multimedia_analyzer import MultimediaAnalyzer
from core.encryption_engine import SecurityLevel
from ui.components.file_drop_zone import FileDropZone
from ui.components.progress_dialog import ProgressDialog


class MultimediaExtractWorkerThread(QThread):
    """Worker thread for multimedia extraction operations."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished_successfully = Signal(str)  # Pass extracted files info
    error_occurred = Signal(str)
    
    def __init__(self, multimedia_path, password, output_dir, media_type, 
                 technique, security_level):
        super().__init__()
        self.multimedia_path = Path(multimedia_path)
        self.password = password
        self.output_dir = Path(output_dir)
        self.media_type = media_type
        self.technique = technique
        self.security_level = security_level
        
        # Initialize appropriate engine based on media type
        self.video_engine = None
        self.audio_engine = None
        
        if media_type == 'video':
            self.video_engine = VideoSteganographyEngine(security_level)
        else:  # audio
            self.audio_engine = AudioSteganographyEngine(security_level)
        
        self.logger = Logger()
    
    def run(self):
        """Execute the multimedia extraction operation."""
        try:
            self.status_updated.emit("Analyzing multimedia file...")
            self.progress_updated.emit(10)
            
            # Extract data using appropriate engine
            if self.media_type == 'video':
                if self.video_engine is None:
                    raise Exception("Video engine not initialized")
                self.status_updated.emit("Extracting data from video frames...")
                extracted_data = self.video_engine.extract_data_from_video(
                    self.multimedia_path, self.password
                )
            else:  # audio
                if self.audio_engine is None:
                    raise Exception("Audio engine not initialized")
                self.status_updated.emit("Extracting data from audio samples...")
                extracted_data = self.audio_engine.extract_data_from_audio(
                    self.multimedia_path, self.password, technique=self.technique
                )
            
            self.progress_updated.emit(60)
            
            if not extracted_data:
                raise Exception("No hidden data found or incorrect password")
            
            self.status_updated.emit("Decompressing extracted files...")
            self.progress_updated.emit(80)
            
            # Extract files from archive
            extracted_files = self._extract_files_from_archive(extracted_data)
            
            if not extracted_files:
                raise Exception("Failed to extract files from archive")
            
            self.status_updated.emit("Extraction completed successfully!")
            self.progress_updated.emit(100)
            
            # Create summary of extracted files
            file_summary = f"Extracted {len(extracted_files)} files:\n"
            for file_path in extracted_files:
                file_size = file_path.stat().st_size if file_path.exists() else 0
                file_summary += f"• {file_path.name} ({self._format_file_size(file_size)})\n"
            
            self.finished_successfully.emit(file_summary)
            
        except Exception as e:
            self.logger.error(f"Multimedia extract operation failed: {e}")
            self.error_occurred.emit(str(e))
    
    def _extract_files_from_archive(self, archive_data: bytes) -> list:
        """Extract files from ZIP archive data."""
        try:
            import io
            
            # Create temp file for archive
            archive_buffer = io.BytesIO(archive_data)
            extracted_files = []
            
            with zipfile.ZipFile(archive_buffer, 'r') as zip_file:
                # Extract all files
                for file_info in zip_file.infolist():
                    if not file_info.is_dir():
                        # Extract file to output directory
                        output_path = self.output_dir / file_info.filename
                        
                        # Ensure output directory exists
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Extract file
                        with zip_file.open(file_info) as source:
                            with open(output_path, 'wb') as target:
                                target.write(source.read())
                        
                        extracted_files.append(output_path)
                        self.logger.info(f"Extracted: {output_path}")
            
            return extracted_files
            
        except Exception as e:
            self.logger.error(f"Failed to extract files from archive: {e}")
            raise
    
    def _format_file_size(self, size: float) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


class MultimediaExtractDialog(QDialog):
    """Professional dialog for extracting files from multimedia."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        self.analyzer = MultimediaAnalyzer()
        
        # Dialog state
        self.multimedia_file = None
        self.extract_worker = None
        
        # Initialize UI
        self.init_ui()
        self.setup_connections()
        
        self.logger.info("Multimedia extract dialog initialized")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Extract Files from Multimedia")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Multimedia Steganography - Extract Files")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(16)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "Extract hidden files from multimedia files that were created with "
            "InvisioVault's multimedia steganography feature."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # File selection
        file_group = self.create_file_selection_group()
        layout.addWidget(file_group)
        
        # Settings
        settings_group = self.create_settings_group()
        layout.addWidget(settings_group)
        
        # Output settings
        output_group = self.create_output_group()
        layout.addWidget(output_group)
        
        # Info display
        info_group = self.create_info_group()
        layout.addWidget(info_group)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.extract_button = QPushButton("🔓 Extract Files")
        self.extract_button.setMinimumHeight(40)
        self.extract_button.setEnabled(False)
        self.extract_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumHeight(40)
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(self.extract_button)
        
        layout.addLayout(button_layout)
        
        # Connect button signals
        self.extract_button.clicked.connect(self.start_extraction)
        cancel_button.clicked.connect(self.reject)
    
    def create_file_selection_group(self) -> QGroupBox:
        """Create the file selection group."""
        group = QGroupBox("Multimedia File Selection")
        layout = QVBoxLayout(group)
        
        # Drop zone without internal browse button
        self.file_drop_zone = FileDropZone(
            title="Drop multimedia file here\n(Video: MP4, AVI, MKV, MOV)\n(Audio: MP3, WAV, FLAC, AAC)",
            show_browse_button=False
        )
        self.file_drop_zone.setMinimumHeight(100)
        self.file_drop_zone.files_dropped.connect(self.on_files_dropped)
        layout.addWidget(self.file_drop_zone)
        
        # File info
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.file_info_label)
        
        # Browse button
        browse_button = QPushButton("Browse Multimedia File...")
        browse_button.clicked.connect(self.browse_multimedia_file)
        layout.addWidget(browse_button)
        
        return group
    
    def create_settings_group(self) -> QGroupBox:
        """Create the settings group."""
        group = QGroupBox("Extraction Settings")
        layout = QGridLayout(group)
        
        # Password
        layout.addWidget(QLabel("Password:"), 0, 0)
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Enter the password used for hiding...")
        layout.addWidget(self.password_input, 0, 1)
        
        # Show password checkbox
        self.show_password_check = QCheckBox("Show password")
        self.show_password_check.toggled.connect(self.toggle_password_visibility)
        layout.addWidget(self.show_password_check, 0, 2)
        
        # Security level
        layout.addWidget(QLabel("Security Level:"), 1, 0)
        self.security_combo = QComboBox()
        self.security_combo.addItems(["MAXIMUM", "HIGH", "STANDARD"])
        self.security_combo.setCurrentText("MAXIMUM")
        layout.addWidget(self.security_combo, 1, 1)
        
        # Audio technique (for audio files)
        layout.addWidget(QLabel("Audio Technique:"), 2, 0)
        self.technique_combo = QComboBox()
        self.technique_combo.addItems(["lsb", "spread_spectrum", "phase_coding"])
        self.technique_combo.setCurrentText("lsb")
        self.technique_combo.setToolTip("LSB: Least Significant Bit (fastest, most compatible)\nSpread Spectrum: Advanced frequency domain hiding\nPhase Coding: Phase manipulation technique")
        layout.addWidget(self.technique_combo, 2, 1)
        
        # Video technique (for video files)
        self.video_technique_label = QLabel("Video Technique:")
        layout.addWidget(self.video_technique_label, 3, 0)
        self.video_technique_combo = QComboBox()
        self.video_technique_combo.addItems(["frame_lsb", "dct_embedding", "motion_vector"])
        self.video_technique_combo.setCurrentText("frame_lsb")
        self.video_technique_combo.setToolTip("Frame LSB: Hide data in video frame pixels (current implementation)\nDCT Embedding: DCT coefficient modification (future)\nMotion Vector: Hide in motion vector data (future)")
        # Initially disabled - will be enabled based on multimedia file type
        self.video_technique_combo.setEnabled(False)
        layout.addWidget(self.video_technique_combo, 3, 1)
        
        # Note about technique
        note_label = QLabel("Note: Use the same technique and settings that were used for hiding the data.")
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        layout.addWidget(note_label, 4, 0, 1, 2)
        
        return group
    
    def create_output_group(self) -> QGroupBox:
        """Create the output settings group."""
        group = QGroupBox("Output Settings")
        layout = QGridLayout(group)
        
        # Output directory
        layout.addWidget(QLabel("Extract to:"), 0, 0)
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setPlaceholderText("Select output directory...")
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_output_directory)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(browse_output_btn)
        layout.addLayout(output_layout, 0, 1)
        
        # Overwrite files checkbox
        self.overwrite_check = QCheckBox("Overwrite existing files")
        layout.addWidget(self.overwrite_check, 1, 0, 1, 2)
        
        return group
    
    def create_info_group(self) -> QGroupBox:
        """Create the information display group."""
        group = QGroupBox("File Information")
        layout = QVBoxLayout(group)
        
        # Info text area
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        self.info_text.setPlainText("Select a multimedia file to see information...")
        layout.addWidget(self.info_text)
        
        return group
    
    def setup_connections(self):
        """Set up signal connections."""
        # Update extract button state when inputs change
        self.password_input.textChanged.connect(self.update_extract_button_state)
        self.output_dir_input.textChanged.connect(self.update_extract_button_state)
    
    def toggle_password_visibility(self, show: bool):
        """Toggle password field visibility."""
        if show:
            self.password_input.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
    
    def on_files_dropped(self, file_paths):
        """Handle file drop."""
        if file_paths:
            file_path = Path(file_paths[0])
            if self.analyzer.is_multimedia_file(file_path):
                self.set_multimedia_file(file_path)
            else:
                QMessageBox.warning(
                    self, "Invalid File",
                    "Please select a supported multimedia file (video or audio)."
                )
    
    def browse_multimedia_file(self):
        """Browse for multimedia file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Multimedia File with Hidden Data",
            "",
            "Multimedia Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.mp3 *.wav *.flac *.aac *.ogg *.m4a);;All Files (*)"
        )
        
        if file_path:
            multimedia_path = Path(file_path)
            if self.analyzer.is_multimedia_file(multimedia_path):
                self.set_multimedia_file(multimedia_path)
            else:
                QMessageBox.warning(
                    self, "Invalid File",
                    "Please select a supported multimedia file."
                )
    
    def browse_output_directory(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            ""
        )
        
        if directory:
            self.output_dir_input.setText(directory)
            self.update_extract_button_state()
    
    def set_multimedia_file(self, file_path: Path):
        """Set the multimedia file and update info."""
        self.multimedia_file = file_path
        self.file_info_label.setText(f"Selected: {file_path.name}")
        
        # Suggest output directory
        if not self.output_dir_input.text():
            suggested_dir = file_path.parent / f"{file_path.stem}_extracted"
            self.output_dir_input.setText(str(suggested_dir))
        
        # Update technique controls based on media type
        self.update_technique_controls()
        
        # Update file information
        self.update_file_info()
        self.update_extract_button_state()
    
    def update_file_info(self):
        """Update file information display."""
        if not self.multimedia_file:
            self.info_text.setPlainText("No file selected.")
            return
        
        try:
            # Basic file info
            file_size = self.multimedia_file.stat().st_size
            media_type = 'Video' if self.analyzer.is_video_file(self.multimedia_file) else 'Audio'
            
            info_text = f"File: {self.multimedia_file.name}\n"
            info_text += f"Type: {media_type}\n"
            info_text += f"Size: {self.format_file_size(file_size)}\n"
            info_text += f"Format: {self.multimedia_file.suffix.upper()}\n\n"
            
            info_text += "This file may contain hidden data if it was created with InvisioVault.\n"
            info_text += "Enter the correct password to extract the hidden files."
            
            self.info_text.setPlainText(info_text)
            
        except Exception as e:
            self.info_text.setPlainText(f"Error reading file information: {e}")
    
    def update_technique_controls(self):
        """Enable/disable technique controls based on multimedia file type."""
        if not self.multimedia_file:
            # No file selected - disable both techniques
            self.technique_combo.setEnabled(False)
            self.video_technique_combo.setEnabled(False)
            self.video_technique_label.setStyleSheet("color: #999;")
            return
        
        is_video = self.analyzer.is_video_file(self.multimedia_file)
        
        if is_video:
            # Video file - enable video technique, disable audio technique
            self.technique_combo.setEnabled(False)
            self.video_technique_combo.setEnabled(True)
            self.video_technique_label.setStyleSheet("")
        else:
            # Audio file - enable audio technique, disable video technique
            self.technique_combo.setEnabled(True)
            self.video_technique_combo.setEnabled(False)
            self.video_technique_label.setStyleSheet("color: #999;")
    
    def update_extract_button_state(self, text=None):
        """Update the state of the extract button."""
        can_extract = (
            self.multimedia_file is not None and
            bool(self.password_input.text().strip()) and
            bool(self.output_dir_input.text().strip())
        )
        
        self.extract_button.setEnabled(bool(can_extract))
    
    def start_extraction(self):
        """Start the extraction operation."""
        try:
            # Validate inputs
            if not self.multimedia_file or not self.multimedia_file.exists():
                QMessageBox.warning(self, "Error", "Please select a valid multimedia file.")
                return
            
            password = self.password_input.text().strip()
            if not password:
                QMessageBox.warning(self, "Error", "Please enter the password.")
                return
            
            output_dir = Path(self.output_dir_input.text().strip())
            if not output_dir.parent.exists():
                QMessageBox.warning(self, "Error", "Output directory parent does not exist.")
                return
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if directory is empty or if overwrite is enabled
            if output_dir.exists() and list(output_dir.iterdir()) and not self.overwrite_check.isChecked():
                reply = QMessageBox.question(
                    self, "Directory Not Empty",
                    f"The output directory '{output_dir}' is not empty. "
                    "Files may be overwritten. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply != QMessageBox.StandardButton.Yes:
                    return
            
            # Determine media type and settings
            media_type = 'video' if self.analyzer.is_video_file(self.multimedia_file) else 'audio'
            
            # Select the appropriate technique based on media type
            if media_type == 'video':
                technique = self.video_technique_combo.currentText()
            else:  # audio
                technique = self.technique_combo.currentText()
            
            security_level = getattr(SecurityLevel, self.security_combo.currentText())
            
            # Create and show progress dialog
            self.progress_dialog = ProgressDialog(
                title="Extracting Files from Multimedia",
                message="Initializing extraction process...",
                can_cancel=True,
                parent=self
            )
            
            # Start extraction worker
            self.extract_worker = MultimediaExtractWorkerThread(
                str(self.multimedia_file),
                password,
                str(output_dir),
                media_type,
                technique,
                security_level
            )
            
            # Connect worker signals
            self.extract_worker.progress_updated.connect(self.progress_dialog.update_progress)
            self.extract_worker.status_updated.connect(self.progress_dialog.set_status)
            self.extract_worker.finished_successfully.connect(self.on_extraction_finished)
            self.extract_worker.error_occurred.connect(self.on_extraction_error)
            
            # Start worker and show progress
            self.extract_worker.start()
            self.progress_dialog.show()
            
        except Exception as e:
            self.error_handler.handle_exception(e)
            QMessageBox.critical(self, "Error", f"Failed to start extraction: {e}")
    
    def on_extraction_finished(self, file_summary: str):
        """Handle successful extraction completion."""
        self.progress_dialog.close()
        
        # Show success message with file summary
        QMessageBox.information(
            self, "Extraction Successful",
            f"Files successfully extracted!\n\nOutput directory: {self.output_dir_input.text()}\n\n{file_summary}"
        )
        
        # Ask if user wants to open the output directory
        reply = QMessageBox.question(
            self, "Open Directory",
            "Would you like to open the output directory?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            import os
            import subprocess
            import platform
            
            output_dir = self.output_dir_input.text()
            
            try:
                if platform.system() == "Windows":
                    os.startfile(output_dir)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", output_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", output_dir])
            except Exception as e:
                self.logger.warning(f"Failed to open directory: {e}")
        
        self.accept()
    
    def on_extraction_error(self, error_msg: str):
        """Handle extraction operation error."""
        self.progress_dialog.close()
        
        # Show appropriate error message
        if "password" in error_msg.lower() or "decrypt" in error_msg.lower():
            QMessageBox.critical(
                self, "Incorrect Password",
                "Failed to extract files. This could be due to:\n\n"
                "• Incorrect password\n"
                "• Wrong security level\n"
                "• Wrong audio technique (for audio files)\n"
                "• File doesn't contain hidden data\n\n"
                f"Technical details: {error_msg}"
            )
        else:
            QMessageBox.critical(
                self, "Extraction Failed",
                f"Failed to extract files from multimedia:\n\n{error_msg}"
            )
    
    def format_file_size(self, size: float) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        # Stop any running workers
        if self.extract_worker and self.extract_worker.isRunning():
            self.extract_worker.terminate()
            self.extract_worker.wait()
        
        super().closeEvent(event)
