"""
Multimedia Hide Dialog
Professional dialog for hiding files in video and audio files using steganography.
"""

from pathlib import Path
import json
import zipfile
import tempfile
from typing import List, Optional, Dict, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QLineEdit, QSpinBox, QCheckBox,
    QProgressBar, QTextEdit, QComboBox, QMessageBox, QTabWidget,
    QWidget, QGridLayout, QSlider, QListWidget, QListWidgetItem,
    QFrame, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent

from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler
from utils.performance_profiler import PerformanceProfiler
from core.video_steganography_engine import VideoSteganographyEngine
from core.audio_steganography_engine import AudioSteganographyEngine
from core.multimedia_analyzer import MultimediaAnalyzer
from core.encryption_engine import SecurityLevel
from ui.components.file_drop_zone import FileDropZone
from ui.components.progress_dialog import ProgressDialog


class MultimediaAnalysisWorker(QThread):
    """Worker thread for analyzing multimedia files without blocking UI."""
    analysis_completed = Signal(dict)  # Analysis results
    analysis_failed = Signal(str)  # Error message
    
    def __init__(self, file_path, analyzer):
        super().__init__()
        self.file_path = Path(file_path)
        self.analyzer = analyzer
    
    def run(self):
        """Analyze multimedia file in background thread."""
        try:
            if self.analyzer.is_video_file(self.file_path):
                analysis = self.analyzer.analyze_video_file(self.file_path)
                analysis['media_type'] = 'video'
            elif self.analyzer.is_audio_file(self.file_path):
                analysis = self.analyzer.analyze_audio_file(self.file_path)
                analysis['media_type'] = 'audio'
            else:
                raise ValueError("Unsupported multimedia format")
            
            analysis['filename'] = self.file_path.name
            self.analysis_completed.emit(analysis)
            
        except Exception as e:
            self.analysis_failed.emit(str(e))


class MultimediaHideWorkerThread(QThread):
    """Worker thread for multimedia hiding operations."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished_successfully = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, carrier_path, files_to_hide, output_path, password, 
                 media_type, technique, quality, security_level):
        super().__init__()
        self.carrier_path = Path(carrier_path)
        self.files_to_hide = [Path(f) for f in files_to_hide]
        self.output_path = Path(output_path)
        self.password = password
        self.media_type = media_type
        self.technique = technique
        self.quality = quality
        self.security_level = security_level
        
        # Initialize engines
        if media_type == 'video':
            self.engine = VideoSteganographyEngine(security_level)
        else:  # audio
            self.engine = AudioSteganographyEngine(security_level)
        
        self.logger = Logger()
    
    def run(self):
        """Execute the multimedia hiding operation."""
        try:
            self.status_updated.emit("Preparing files for hiding...")
            self.progress_updated.emit(10)
            
            # Create file archive
            archive_data = self._create_file_archive()
            
            self.status_updated.emit("Analyzing multimedia capacity...")
            self.progress_updated.emit(30)
            
            # Hide data using appropriate engine
            if self.media_type == 'video':
                self.status_updated.emit("Hiding data in video frames...")
                success = self.engine.hide_data_in_video(
                    self.carrier_path, archive_data, self.output_path,
                    self.password, compression_quality=self.quality
                )
            else:  # audio
                self.status_updated.emit("Hiding data in audio samples...")
                success = self.engine.hide_data_in_audio(
                    self.carrier_path, archive_data, self.output_path,
                    self.password, technique=self.technique, quality=self.quality
                )
            
            self.progress_updated.emit(90)
            
            if not success:
                raise Exception("Failed to hide data in multimedia file")
            
            self.status_updated.emit("Operation completed successfully!")
            self.progress_updated.emit(100)
            
            self.finished_successfully.emit()
            
        except Exception as e:
            self.logger.error(f"Multimedia hide operation failed: {e}")
            self.error_occurred.emit(str(e))
    
    def _create_file_archive(self) -> bytes:
        """Create a ZIP archive from the files to be hidden."""
        try:
            import io
            
            # Create in-memory ZIP archive
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in self.files_to_hide:
                    if file_path.exists():
                        zip_file.write(file_path, file_path.name)
                    else:
                        self.logger.warning(f"File not found: {file_path}")
            
            zip_data = zip_buffer.getvalue()
            self.logger.info(f"Created ZIP archive: {len(zip_data)} bytes")
            
            return zip_data
            
        except Exception as e:
            self.logger.error(f"Failed to create file archive: {e}")
            raise


class MultimediaHideDialog(QDialog):
    """Professional dialog for hiding files in multimedia."""
    
    def __init__(self, parent=None):
        profiler = PerformanceProfiler()
        
        with profiler.timer("multimedia_hide_dialog_total_init"):
            super().__init__(parent)
            
            with profiler.timer("multimedia_hide_dialog_core_init"):
                self.logger = Logger()
                self.config = ConfigManager()
                self.error_handler = ErrorHandler()
                self.analyzer = MultimediaAnalyzer()
            
            # Dialog state
            self.carrier_file = None
            self.files_to_hide = []
            self.analysis_worker = None
            self.hide_worker = None
            
            # Initialize UI
            with profiler.timer("multimedia_hide_dialog_ui_init"):
                self.init_ui()
            
            with profiler.timer("multimedia_hide_dialog_connections"):
                self.setup_connections()
            
            self.logger.info("Multimedia hide dialog initialized")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Hide Files in Multimedia")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Multimedia Steganography - Hide Files")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(16)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - File selection
        left_panel = self.create_file_selection_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Configuration and analysis
        right_panel = self.create_configuration_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.hide_button = QPushButton("🔒 Hide Files")
        self.hide_button.setMinimumHeight(40)
        self.hide_button.setEnabled(False)
        self.hide_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumHeight(40)
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(self.hide_button)
        
        layout.addLayout(button_layout)
        
        # Connect button signals
        self.hide_button.clicked.connect(self.start_hiding)
        cancel_button.clicked.connect(self.reject)
    
    def create_file_selection_panel(self) -> QWidget:
        """Create the file selection panel."""
        panel = QGroupBox("File Selection")
        layout = QVBoxLayout(panel)
        
        # Carrier file section
        carrier_group = QGroupBox("Multimedia Carrier File")
        carrier_layout = QVBoxLayout(carrier_group)
        
        # Drop zone for carrier file
        self.carrier_drop_zone = FileDropZone(
            "Drop multimedia file here", 
            file_types=['.mp4', '.avi', '.mkv', '.mov', '.mp3', '.wav', '.flac', '.aac']
        )
        self.carrier_drop_zone.setMinimumHeight(180)
        self.carrier_drop_zone.files_dropped.connect(self.on_carrier_files_dropped)
        carrier_layout.addWidget(self.carrier_drop_zone)
        
        # Carrier file info
        self.carrier_info_label = QLabel("No carrier file selected")
        self.carrier_info_label.setStyleSheet("color: #666; font-style: italic;")
        carrier_layout.addWidget(self.carrier_info_label)
        
        # Browse button for carrier
        browse_carrier_btn = QPushButton("Browse Multimedia File...")
        browse_carrier_btn.clicked.connect(self.browse_carrier_file)
        carrier_layout.addWidget(browse_carrier_btn)
        
        layout.addWidget(carrier_group)
        
        # Files to hide section
        files_group = QGroupBox("Files to Hide")
        files_layout = QVBoxLayout(files_group)
        
        # Drop zone for files to hide
        self.files_drop_zone = FileDropZone("Drop files to hide here")
        self.files_drop_zone.setMinimumHeight(180)
        self.files_drop_zone.files_dropped.connect(self.on_files_to_hide_dropped)
        files_layout.addWidget(self.files_drop_zone)
        
        # Files list
        self.files_list = QListWidget()
        self.files_list.setMaximumHeight(100)
        files_layout.addWidget(self.files_list)
        
        # Files info
        self.files_info_label = QLabel("No files selected")
        self.files_info_label.setStyleSheet("color: #666; font-style: italic;")
        files_layout.addWidget(self.files_info_label)
        
        # Browse button for files
        browse_files_btn = QPushButton("Browse Files to Hide...")
        browse_files_btn.clicked.connect(self.browse_files_to_hide)
        files_layout.addWidget(browse_files_btn)
        
        # Clear files button
        clear_files_btn = QPushButton("Clear Files")
        clear_files_btn.clicked.connect(self.clear_files_to_hide)
        files_layout.addWidget(clear_files_btn)
        
        layout.addWidget(files_group)
        
        return panel
    
    def create_configuration_panel(self) -> QWidget:
        """Create the configuration and analysis panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Analysis tab
        analysis_tab = self.create_analysis_tab()
        tab_widget.addTab(analysis_tab, "📊 Analysis")
        
        # Configuration tab
        config_tab = self.create_configuration_tab()
        tab_widget.addTab(config_tab, "⚙️ Settings")
        
        return panel
    
    def create_analysis_tab(self) -> QWidget:
        """Create the analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Analysis results
        analysis_group = QGroupBox("Multimedia Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(200)
        self.analysis_text.setPlainText("Select a multimedia file then click 'Analyze File' to see detailed analysis...")
        analysis_layout.addWidget(self.analysis_text)
        
        # Add analyze button for on-demand analysis
        self.analyze_button = QPushButton("📊 Analyze File")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.start_analysis)
        analysis_layout.addWidget(self.analyze_button)
        
        layout.addWidget(analysis_group)
        
        # Capacity information
        capacity_group = QGroupBox("Capacity Information")
        capacity_layout = QGridLayout(capacity_group)
        
        self.capacity_label = QLabel("Capacity: Unknown")
        self.files_size_label = QLabel("Files Size: 0 bytes")
        self.remaining_label = QLabel("Remaining: Unknown")
        
        capacity_layout.addWidget(QLabel("Available Capacity:"), 0, 0)
        capacity_layout.addWidget(self.capacity_label, 0, 1)
        capacity_layout.addWidget(QLabel("Selected Files Size:"), 1, 0)
        capacity_layout.addWidget(self.files_size_label, 1, 1)
        capacity_layout.addWidget(QLabel("Remaining Space:"), 2, 0)
        capacity_layout.addWidget(self.remaining_label, 2, 1)
        
        layout.addWidget(capacity_group)
        
        # Recommendations
        recommendations_group = QGroupBox("Recommendations")
        recommendations_layout = QVBoxLayout(recommendations_group)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        self.recommendations_text.setMaximumHeight(100)
        self.recommendations_text.setPlainText("Analysis recommendations will appear here...")
        recommendations_layout.addWidget(self.recommendations_text)
        
        layout.addWidget(recommendations_group)
        
        layout.addStretch()
        
        return widget
    
    def create_configuration_tab(self) -> QWidget:
        """Create the configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Security settings
        security_group = QGroupBox("Security Settings")
        security_layout = QGridLayout(security_group)
        
        # Password
        security_layout.addWidget(QLabel("Password:"), 0, 0)
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Enter strong password...")
        security_layout.addWidget(self.password_input, 0, 1)
        
        # Security level
        security_layout.addWidget(QLabel("Security Level:"), 1, 0)
        self.security_combo = QComboBox()
        self.security_combo.addItems(["MAXIMUM", "HIGH", "STANDARD"])
        self.security_combo.setCurrentText("MAXIMUM")
        security_layout.addWidget(self.security_combo, 1, 1)
        
        layout.addWidget(security_group)
        
        # Multimedia settings
        multimedia_group = QGroupBox("Multimedia Settings")
        multimedia_layout = QGridLayout(multimedia_group)
        
        # Technique (for audio)
        multimedia_layout.addWidget(QLabel("Audio Technique:"), 0, 0)
        self.technique_combo = QComboBox()
        self.technique_combo.addItems(["lsb", "spread_spectrum", "phase_coding"])
        self.technique_combo.setCurrentText("lsb")
        multimedia_layout.addWidget(self.technique_combo, 0, 1)
        
        # Quality
        multimedia_layout.addWidget(QLabel("Output Quality:"), 1, 0)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["high", "medium", "low"])
        self.quality_combo.setCurrentText("high")
        multimedia_layout.addWidget(self.quality_combo, 1, 1)
        
        # Video quality (for video files)
        multimedia_layout.addWidget(QLabel("Video Quality (CRF):"), 2, 0)
        self.video_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_quality_slider.setRange(18, 28)
        self.video_quality_slider.setValue(23)
        self.video_quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.video_quality_slider.setTickInterval(2)
        self.video_quality_label = QLabel("23 (Good)")
        
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(self.video_quality_slider)
        quality_layout.addWidget(self.video_quality_label)
        multimedia_layout.addLayout(quality_layout, 2, 1)
        
        layout.addWidget(multimedia_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout(output_group)
        
        # Output file
        output_layout.addWidget(QLabel("Output File:"), 0, 0)
        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText("Select output location...")
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_output_file)
        
        output_file_layout = QHBoxLayout()
        output_file_layout.addWidget(self.output_path_input)
        output_file_layout.addWidget(browse_output_btn)
        output_layout.addLayout(output_file_layout, 0, 1)
        
        layout.addWidget(output_group)
        
        layout.addStretch()
        
        return widget
    
    def setup_connections(self):
        """Set up signal connections."""
        # Update video quality label when slider changes
        self.video_quality_slider.valueChanged.connect(self.update_video_quality_label)
        
        # Update UI when inputs change
        self.password_input.textChanged.connect(self.update_hide_button_state)
        self.output_path_input.textChanged.connect(self.update_hide_button_state)
    
    def update_video_quality_label(self, value):
        """Update video quality label based on slider value."""
        if value <= 20:
            quality_text = f"{value} (Excellent)"
        elif value <= 23:
            quality_text = f"{value} (Good)"
        elif value <= 26:
            quality_text = f"{value} (Fair)"
        else:
            quality_text = f"{value} (Low)"
        
        self.video_quality_label.setText(quality_text)
    
    def on_carrier_files_dropped(self, file_paths):
        """Handle carrier file drop."""
        if file_paths:
            carrier_path = Path(file_paths[0])
            if self.analyzer.is_multimedia_file(carrier_path):
                self.set_carrier_file(carrier_path)
            else:
                QMessageBox.warning(
                    self, "Invalid File",
                    "Please select a supported multimedia file (video or audio)."
                )
    
    def on_files_to_hide_dropped(self, file_paths):
        """Handle files to hide drop."""
        self.add_files_to_hide([Path(p) for p in file_paths])
    
    def browse_carrier_file(self):
        """Browse for carrier multimedia file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Multimedia Carrier File",
            "",
            "Lossless Audio (*.wav *.flac *.aiff *.au);;Lossy Audio (*.mp3 *.aac *.ogg *.m4a);;Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;All Files (*)"
        )
        
        if file_path:
            carrier_path = Path(file_path)
            if self.analyzer.is_multimedia_file(carrier_path):
                # Check if it's a lossy audio format and warn
                if self.analyzer.is_audio_file(carrier_path) and self._is_lossy_audio(carrier_path):
                    reply = QMessageBox.warning(
                        self, "Lossy Audio Format Warning",
                        f"You've selected a {carrier_path.suffix.upper()} file as a carrier, which is a lossy format.\n\n"
                        "This is NOT recommended for steganography. When using lossy formats:\n\n"
                        "1. The hidden data may already be damaged in the source file\n"
                        "2. You MUST choose a lossless format like WAV or FLAC for the output file\n"
                        "3. Extraction may still fail due to the lossy compression artifacts\n\n"
                        "For best results, please use WAV or FLAC files as carriers.\n\n"
                        "Do you want to continue with this file anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.set_carrier_file(carrier_path)
                else:
                    self.set_carrier_file(carrier_path)
            else:
                QMessageBox.warning(
                    self, "Invalid File",
                    "Please select a supported multimedia file."
                )
    
    def browse_files_to_hide(self):
        """Browse for files to hide."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files to Hide",
            "",
            "All Files (*)"
        )
        
        if file_paths:
            self.add_files_to_hide([Path(p) for p in file_paths])
    
    def browse_output_file(self):
        """Browse for output file location."""
        if self.carrier_file:
            # Suggest output name based on carrier file
            suggested_name = self.carrier_file.stem + "_hidden" + self.carrier_file.suffix
            suggested_path = self.carrier_file.parent / suggested_name
            
            # Determine if this is audio or video
            is_audio = self.analyzer.is_audio_file(self.carrier_file)
            
            if is_audio:
                # For audio files, show format warning and recommendations
                file_path, selected_filter = QFileDialog.getSaveFileName(
                    self, "Save Hidden Audio File",
                    str(suggested_path),
                    "Lossless Audio (*.wav *.flac *.aiff);;Lossy Audio (*.mp3 *.aac *.ogg *.m4a);;All Files (*)"
                )
            else:
                # For video files
                file_path, selected_filter = QFileDialog.getSaveFileName(
                    self, "Save Hidden Video File",
                    str(suggested_path),
                    f"Video Files (*{self.carrier_file.suffix});;All Files (*)"
                )
            
            if file_path:
                output_path = Path(file_path)
                
                # Show format warning for audio files
                if is_audio and self._show_audio_format_warning(output_path):
                    self.output_path_input.setText(file_path)
                    self.update_hide_button_state()
                elif not is_audio:
                    self.output_path_input.setText(file_path)
                    self.update_hide_button_state()
    
    def set_carrier_file(self, file_path: Path):
        """Set the carrier file and analyze it."""
        self.carrier_file = file_path
        self.carrier_info_label.setText(f"Selected: {file_path.name}")
        
        # Suggest output path
        if not self.output_path_input.text():
            suggested_name = file_path.stem + "_hidden" + file_path.suffix
            suggested_path = file_path.parent / suggested_name
            self.output_path_input.setText(str(suggested_path))
        
        # Don't start analysis automatically - make it on-demand for better performance
        # User can trigger analysis manually if needed
        self.current_analysis = None  # Reset analysis
        self.analyze_button.setEnabled(True)  # Enable analyze button
        self.update_capacity_info()  # Update with basic info
        self.update_hide_button_state()
    
    def add_files_to_hide(self, file_paths: List[Path]):
        """Add files to the list of files to hide."""
        for file_path in file_paths:
            if file_path.exists() and file_path not in self.files_to_hide:
                self.files_to_hide.append(file_path)
        
        self.update_files_list()
        self.update_capacity_info()
        self.update_hide_button_state()
    
    def clear_files_to_hide(self):
        """Clear the list of files to hide."""
        self.files_to_hide.clear()
        self.update_files_list()
        self.update_capacity_info()
        self.update_hide_button_state()
    
    def update_files_list(self):
        """Update the files list widget."""
        self.files_list.clear()
        
        if not self.files_to_hide:
            self.files_info_label.setText("No files selected")
            return
        
        total_size = 0
        for file_path in self.files_to_hide:
            try:
                size = file_path.stat().st_size
                total_size += size
                size_str = self.format_file_size(size)
                item_text = f"{file_path.name} ({size_str})"
                self.files_list.addItem(item_text)
            except:
                self.files_list.addItem(f"{file_path.name} (size unknown)")
        
        self.files_info_label.setText(
            f"{len(self.files_to_hide)} files selected ({self.format_file_size(total_size)})"
        )
    
    def start_analysis(self):
        """Start multimedia file analysis."""
        if not self.carrier_file:
            return
        
        self.analysis_text.setPlainText("Analyzing multimedia file...")
        
        # Start analysis worker
        self.analysis_worker = MultimediaAnalysisWorker(self.carrier_file, self.analyzer)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_worker.analysis_failed.connect(self.on_analysis_failed)
        self.analysis_worker.start()
    
    def on_analysis_completed(self, analysis):
        """Handle completed analysis."""
        try:
            media_type = analysis.get('media_type', 'unknown')
            
            # Update analysis text
            analysis_text = f"File: {analysis.get('filename', 'Unknown')}\n"
            analysis_text += f"Type: {media_type.title()}\n"
            analysis_text += f"Format: {analysis.get('format', 'Unknown')}\n"
            analysis_text += f"Size: {self.format_file_size(analysis.get('file_size', 0))}\n"
            
            if media_type == 'video':
                analysis_text += f"Resolution: {analysis.get('width', 0)}x{analysis.get('height', 0)}\n"
                analysis_text += f"Duration: {analysis.get('duration_formatted', 'Unknown')}\n"
                analysis_text += f"FPS: {analysis.get('fps', 0):.1f}\n"
                analysis_text += f"Frames: {analysis.get('frame_count', 0):,}\n"
            elif media_type == 'audio':
                analysis_text += f"Duration: {analysis.get('duration_formatted', 'Unknown')}\n"
                analysis_text += f"Sample Rate: {analysis.get('sample_rate', 0):,} Hz\n"
                analysis_text += f"Channels: {analysis.get('channels', 0)}\n"
            
            analysis_text += f"Capacity: {self.format_file_size(analysis.get('capacity_bytes', 0))}\n"
            analysis_text += f"Suitability: {analysis.get('suitability_score', 0)}/10"
            
            self.analysis_text.setPlainText(analysis_text)
            
            # Update recommendations
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                self.recommendations_text.setPlainText('\n'.join(f"• {rec}" for rec in recommendations))
            else:
                self.recommendations_text.setPlainText("No specific recommendations.")
            
            # Store analysis for capacity calculations
            self.current_analysis = analysis
            self.update_capacity_info()
            
        except Exception as e:
            self.logger.error(f"Error processing analysis results: {e}")
            self.analysis_text.setPlainText(f"Error processing analysis: {e}")
    
    def on_analysis_failed(self, error_msg):
        """Handle failed analysis."""
        self.analysis_text.setPlainText(f"Analysis failed: {error_msg}")
        self.current_analysis = None
    
    def update_capacity_info(self):
        """Update capacity information display."""
        if not hasattr(self, 'current_analysis') or not self.current_analysis:
            self.capacity_label.setText("Unknown")
            self.files_size_label.setText("0 bytes")
            self.remaining_label.setText("Unknown")
            return
        
        # Calculate capacity
        capacity = self.current_analysis.get('capacity_bytes', 0)
        self.capacity_label.setText(self.format_file_size(capacity))
        
        # Calculate files size
        files_size = sum(f.stat().st_size for f in self.files_to_hide if f.exists())
        self.files_size_label.setText(self.format_file_size(files_size))
        
        # Calculate remaining
        remaining = capacity - files_size
        if remaining >= 0:
            self.remaining_label.setText(self.format_file_size(remaining))
            self.remaining_label.setStyleSheet("color: green;")
        else:
            self.remaining_label.setText(f"-{self.format_file_size(-remaining)}")
            self.remaining_label.setStyleSheet("color: red;")
    
    def update_hide_button_state(self):
        """Update the state of the hide button."""
        can_hide = (self.carrier_file is not None and
                    bool(self.files_to_hide) and
                    bool(self.password_input.text().strip()) and
                    bool(self.output_path_input.text().strip()))
        
        self.hide_button.setEnabled(can_hide)
    
    def start_hiding(self):
        """Start the hiding operation."""
        try:
            # Validate inputs
            if not self.carrier_file or not self.carrier_file.exists():
                QMessageBox.warning(self, "Error", "Please select a valid carrier file.")
                return
            
            if not self.files_to_hide:
                QMessageBox.warning(self, "Error", "Please select files to hide.")
                return
            
            password = self.password_input.text().strip()
            if len(password) < 6:
                QMessageBox.warning(self, "Error", "Password must be at least 6 characters long.")
                return
            
            output_path = Path(self.output_path_input.text().strip())
            if not output_path.parent.exists():
                QMessageBox.warning(self, "Error", "Output directory does not exist.")
                return
            
            # Check capacity
            if hasattr(self, 'current_analysis') and self.current_analysis:
                capacity = self.current_analysis.get('capacity_bytes', 0)
                files_size = sum(f.stat().st_size for f in self.files_to_hide if f.exists())
                
                if files_size > capacity:
                    QMessageBox.warning(
                        self, "Insufficient Capacity",
                        f"Selected files ({self.format_file_size(files_size)}) exceed "
                        f"carrier capacity ({self.format_file_size(capacity)})."
                    )
                    return
            
            # Determine media type and settings
            media_type = 'video' if self.analyzer.is_video_file(self.carrier_file) else 'audio'
            technique = self.technique_combo.currentText()
            
            if media_type == 'video':
                quality = self.video_quality_slider.value()
            else:
                quality = self.quality_combo.currentText()
            
            security_level = getattr(SecurityLevel, self.security_combo.currentText())
            
            # Create and show progress dialog
            self.progress_dialog = ProgressDialog("Hiding Files in Multimedia", self)
            
            # Start hiding worker
            self.hide_worker = MultimediaHideWorkerThread(
                self.carrier_file,
                [str(f) for f in self.files_to_hide],
                str(output_path),
                password,
                media_type,
                technique,
                quality,
                security_level
            )
            
            # Connect worker signals
            self.hide_worker.progress_updated.connect(self.progress_dialog.update_progress)
            self.hide_worker.status_updated.connect(self.progress_dialog.set_status)
            self.hide_worker.finished_successfully.connect(self.on_hiding_finished)
            self.hide_worker.error_occurred.connect(self.on_hiding_error)
            
            # Start worker and show progress
            self.hide_worker.start()
            self.progress_dialog.show()
            
        except Exception as e:
            self.error_handler.handle_exception(e)
            QMessageBox.critical(self, "Error", f"Failed to start hiding operation: {e}")
    
    def on_hiding_finished(self):
        """Handle successful hiding completion."""
        self.progress_dialog.close()
        
        QMessageBox.information(
            self, "Success",
            f"Files successfully hidden in multimedia file!\n\nOutput: {self.output_path_input.text()}"
        )
        
        self.accept()
    
    def on_hiding_error(self, error_msg):
        """Handle hiding operation error."""
        self.progress_dialog.close()
        
        QMessageBox.critical(
            self, "Hiding Failed",
            f"Failed to hide files in multimedia:\n\n{error_msg}"
        )
    
    def _is_lossy_audio(self, file_path: Path) -> bool:
        """Check if the audio file is in a lossy format."""
        lossy_formats = {'.mp3', '.aac', '.ogg', '.m4a', '.wma'}
        return file_path.suffix.lower() in lossy_formats
    
    def _show_audio_format_warning(self, output_path: Path) -> bool:
        """Show warning about audio format choice and return True if user wants to proceed."""
        output_format = output_path.suffix.lower()
        
        # Define format categories
        lossless_formats = {'.wav', '.flac', '.aiff', '.au'}
        lossy_formats = {'.mp3', '.aac', '.ogg', '.m4a', '.wma'}
        
        if output_format in lossless_formats:
            # Good format - show positive message
            QMessageBox.information(
                self, "Excellent Format Choice!",
                f"You've selected {output_format.upper()} format - a lossless audio format.\n\n"
                "This is the ideal choice for audio steganography as it preserves all \n"
                "audio data without compression, ensuring your hidden files can be \n"
                "extracted successfully."
            )
            return True
            
        elif output_format in lossy_formats:
            # Lossy format - show warning
            reply = QMessageBox.warning(
                self, "Lossy Format Warning",
                f"Warning: You've selected {output_format.upper()} format - a lossy audio format.\n\n"
                "Lossy compression may corrupt or destroy the hidden steganographic data, \n"
                "making it impossible to extract your files later.\n\n"
                "Recommended lossless formats: WAV, FLAC, AIFF\n\n"
                "Do you want to continue with this format anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No  # Default to No for safety
            )
            return reply == QMessageBox.StandardButton.Yes
            
        else:
            # Unknown format - show warning and recommend alternatives
            reply = QMessageBox.warning(
                self, "Unknown Audio Format",
                f"Unknown audio format: {output_format.upper()}\n\n"
                "For audio steganography to work reliably, we recommend using \n"
                "lossless formats that preserve all audio data:\n\n"
                "• WAV - Uncompressed, universally supported\n"
                "• FLAC - Lossless compression, smaller files\n"
                "• AIFF - Apple's lossless format\n\n"
                "Do you want to continue with this format anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No  # Default to No for safety
            )
            return reply == QMessageBox.StandardButton.Yes
    
    def format_file_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        # Stop any running workers
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.terminate()
            self.analysis_worker.wait()
        
        if self.hide_worker and self.hide_worker.isRunning():
            self.hide_worker.terminate()
            self.hide_worker.wait()
        
        super().closeEvent(event)
