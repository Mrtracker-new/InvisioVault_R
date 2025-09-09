"""
Enhanced Hide Files Dialog with Anti-Detection Capabilities
Extends the original dialog with advanced anti-detection features and steganalysis testing.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QTextEdit, QLineEdit, QComboBox, QCheckBox, QProgressBar, QFileDialog,
    QMessageBox, QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
    QScrollArea, QFrame, QSlider, QSpinBox
)
from PySide6.QtGui import QFont, QPixmap

from core.steganography.enhanced_steganography_engine import EnhancedSteganographyEngine
from core.security.encryption_engine import EncryptionEngine, SecurityLevel
from core.security.security_service import SecurityService
from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler


class CarrierAnalysisWorker(QThread):
    """Worker thread for carrier image analysis to prevent UI blocking."""
    
    analysis_completed = Signal(dict)
    analysis_failed = Signal(str)
    progress_updated = Signal(int)
    status_updated = Signal(str)
    
    def __init__(self, enhanced_engine, carrier_path, analysis_type="basic"):
        super().__init__()
        self.enhanced_engine = enhanced_engine
        self.carrier_path = carrier_path
        self.analysis_type = analysis_type  # "basic" or "detailed"
        self.logger = Logger()
    
    def run(self):
        """Perform image analysis in background thread."""
        try:
            self.status_updated.emit("Analyzing carrier image...")
            self.progress_updated.emit(20)
            
            if self.analysis_type == "basic":
                # Basic analysis for initial carrier selection
                analysis = self.enhanced_engine.analyze_image_suitability(self.carrier_path)
                self.progress_updated.emit(80)
                
                results = {
                    'type': 'basic',
                    'capacity_mb': analysis.get('capacity_mb', 0),
                    'suitability_score': analysis.get('suitability_score', 0),
                    'width': analysis.get('width', 0),
                    'height': analysis.get('height', 0),
                    'channels': analysis.get('channels', 0),
                    'full_analysis': analysis
                }
                
            else:  # detailed analysis
                self.status_updated.emit("Performing detailed carrier analysis...")
                analysis = self.enhanced_engine.analyze_carrier_suitability(self.carrier_path)
                self.progress_updated.emit(60)
                
                self.status_updated.emit("Generating recommendations...")
                results = {
                    'type': 'detailed',
                    'full_analysis': analysis,
                    'analysis_text': self._format_detailed_analysis(analysis)
                }
                self.progress_updated.emit(80)
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Analysis complete!")
            self.analysis_completed.emit(results)
            
        except Exception as e:
            self.logger.error(f"Carrier analysis failed: {e}")
            self.analysis_failed.emit(str(e))
    
    def _format_detailed_analysis(self, analysis):
        """Format detailed analysis results for display."""
        analysis_text = f"""
🎯 ENHANCED CARRIER ANALYSIS

📊 Basic Metrics:
• Image Size: {analysis.get('width', 0)} x {analysis.get('height', 0)} pixels
• Channels: {analysis.get('channels', 0)}
• Total Capacity: {analysis.get('capacity_mb', 0):.2f} MB
• Suitability Score: {analysis.get('suitability_score', 0)}/10

🛡️ Anti-Detection Metrics:
• Complexity Score: {analysis.get('complexity_score', 0):.3f}
• Secure Capacity: {analysis.get('secure_capacity_bytes', 0):,} bytes
• Texture Regions: {analysis.get('texture_regions_percent', 0):.1f}%
• Smooth Regions: {analysis.get('smooth_regions_percent', 0):.1f}%
• Anti-Detection Score: {analysis.get('anti_detection_score', 0)}/10

💡 Recommendations:
"""
        
        for rec in analysis.get('recommendations', []):
            analysis_text += f"• {rec}\n"
        
        return analysis_text.strip()


class EnhancedHideWorkerThread(QThread):
    """Worker thread for enhanced hiding operations with anti-detection."""
    
    progress_updated = Signal(float)
    status_updated = Signal(str)
    risk_analysis_updated = Signal(dict)
    finished = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self, carrier_path: Path, files_to_hide: List[str], output_path: Path,
                 password: str, security_level: SecurityLevel, use_anti_detection: bool,
                 target_risk_level: str = "LOW", randomize_lsb: bool = True, 
                 custom_seed: Optional[int] = None):
        super().__init__()
        
        self.carrier_path = carrier_path
        self.files_to_hide = files_to_hide
        self.output_path = output_path
        self.password = password
        self.security_level = security_level
        self.use_anti_detection = use_anti_detection
        self.target_risk_level = target_risk_level
        self.randomize_lsb = randomize_lsb
        self.custom_seed = custom_seed
        
        self.enhanced_engine = EnhancedSteganographyEngine(use_anti_detection=use_anti_detection)
        self.encryption_engine = EncryptionEngine(security_level)
        self.logger = Logger()
        
        self._cancelled = False
    
    def run(self):
        """Execute the enhanced hiding operation."""
        try:
            self.status_updated.emit("Preparing files for enhanced hiding...")
            self.progress_updated.emit(0.1)
            
            # Prepare files data
            files_data = self._prepare_files_data()
            if not files_data:
                self.error_occurred.emit("Failed to prepare files data")
                return
            
            self.progress_updated.emit(0.3)
            self.status_updated.emit("Encrypting data with military-grade security...")
            
            # Encrypt the data
            encrypted_data = self.encryption_engine.encrypt_with_metadata(files_data, self.password)
            
            self.progress_updated.emit(0.5)
            
            if self.use_anti_detection:
                self.status_updated.emit("Creating undetectable steganographic image...")
                
                # Use enhanced anti-detection hiding
                result = self.enhanced_engine.create_undetectable_stego(
                    carrier_path=self.carrier_path,
                    data=encrypted_data,
                    output_path=self.output_path,
                    password=self.password,
                    target_risk_level=self.target_risk_level
                )
                
                self.progress_updated.emit(0.8)
                
                if result['success']:
                    # Perform steganalysis testing
                    self.status_updated.emit("Testing against steganalysis tools...")
                    steganalysis_results = self.enhanced_engine.test_against_steganalysis(self.output_path)
                    
                    self.risk_analysis_updated.emit({
                        'creation_result': result,
                        'steganalysis_test': steganalysis_results
                    })
                    
                    self.progress_updated.emit(1.0)
                    self.status_updated.emit("Enhanced hiding completed successfully!")
                    
                    final_result = {
                        'success': True,
                        'output_path': str(self.output_path),
                        'files_hidden': len(self.files_to_hide),
                        'data_size': len(encrypted_data),
                        'anti_detection_used': True,
                        'risk_level': result['risk_level'],
                        'risk_score': result['risk_score'],
                        'attempts_needed': result['attempts'],
                        'steganalysis_test': steganalysis_results
                    }
                    
                    self.finished.emit(final_result)
                else:
                    # Anti-detection failed, try fallback with randomized LSB if enabled
                    error_msg = result.get('error', 'Unknown error')
                    if self.randomize_lsb and (
                        'capacity' in error_msg.lower() or 
                        'secure' in error_msg.lower() or 
                        'risk level' in error_msg.lower() or
                        'target risk' in error_msg.lower() or
                        'constraint' in error_msg.lower()
                    ):
                        # Extract detailed constraint information
                        constraint_details = result.get('constraint_details', [])
                        detailed_reason = "\n".join(constraint_details) if constraint_details else "Target risk level could not be achieved"
                        
                        self.status_updated.emit(f"Anti-detection failed: {detailed_reason}. Falling back to randomized LSB...")
                        
                        # Use randomized LSB as fallback
                        # Generate seed from password for randomization
                        import hashlib
                        seed = int(hashlib.sha256(self.password.encode()).hexdigest()[:8], 16) if self.password else None
                        
                        success = self.enhanced_engine.hide_data_enhanced(
                            carrier_path=self.carrier_path,
                            data=encrypted_data,
                            output_path=self.output_path,
                            password=self.password,
                            use_anti_detection=False,
                            randomize=True,
                            seed=seed
                        )
                        
                        self.progress_updated.emit(1.0)
                        
                        if success:
                            self.status_updated.emit("Fallback to randomized LSB completed successfully!")
                            
                            final_result = {
                                'success': True,
                                'output_path': str(self.output_path),
                                'files_hidden': len(self.files_to_hide),
                                'data_size': len(encrypted_data),
                                'anti_detection_used': False,
                                'randomized_lsb': True,
                                'fallback_used': True,
                                'fallback_reason': 'Anti-detection constraints not met',
                                'constraint_details': constraint_details,
                                'anti_detection_error': error_msg
                            }
                            
                            self.finished.emit(final_result)
                        else:
                            self.error_occurred.emit("Both anti-detection and randomized LSB fallback failed")
                    else:
                        self.error_occurred.emit(f"Anti-detection hiding failed: {error_msg}")
            
            else:
                self.status_updated.emit("Using high-performance steganography...")
                
                # Use standard hiding with optional randomization
                if self.randomize_lsb:
                    self.status_updated.emit("Applying randomized LSB positioning...")
                    
                    # Use randomized hiding via enhanced engine
                    success = self.enhanced_engine.hide_data_enhanced(
                        carrier_path=self.carrier_path,
                        data=encrypted_data,
                        output_path=self.output_path,
                        password=self.password,
                        use_anti_detection=False,
                        randomize=True
                    )
                else:
                    # Use basic sequential hiding
                    success = self.enhanced_engine.hide_data_enhanced(
                        carrier_path=self.carrier_path,
                        data=encrypted_data,
                        output_path=self.output_path,
                        password=self.password,
                        use_anti_detection=False,
                        randomize=False
                    )
                
                self.progress_updated.emit(1.0)
                
                if success:
                    mode_desc = "randomized LSB" if self.randomize_lsb else "sequential LSB"
                    self.status_updated.emit(f"Standard hiding ({mode_desc}) completed successfully!")
                    
                    final_result = {
                        'success': True,
                        'output_path': str(self.output_path),
                        'files_hidden': len(self.files_to_hide),
                        'data_size': len(encrypted_data),
                        'anti_detection_used': False,
                        'randomized_lsb': self.randomize_lsb
                    }
                    
                    self.finished.emit(final_result)
                else:
                    self.error_occurred.emit("Standard hiding failed")
            
        except Exception as e:
            self.error_occurred.emit(f"Hide operation failed: {str(e)}")
    
    def _prepare_files_data(self) -> Optional[bytes]:
        """Prepare files data for hiding."""
        try:
            import zipfile
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                temp_zip_path = Path(temp_file.name)
            
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for file_path in self.files_to_hide:
                    if Path(file_path).exists():
                        archive.write(file_path, Path(file_path).name)
            
            with open(temp_zip_path, 'rb') as f:
                data = f.read()
            
            temp_zip_path.unlink()  # Clean up
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare files data: {e}")
            return None
    
    def cancel(self):
        """Cancel the operation."""
        self._cancelled = True
        self.terminate()


class EnhancedHideFilesDialog(QDialog):
    """Enhanced dialog for hiding files with anti-detection capabilities."""
    
    def __init__(self, security_service: Optional[SecurityService] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enhanced Hide Files - Anti-Detection Steganography")
        self.setModal(True)
        self.resize(900, 800)
        
        # Initialize components
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        self.enhanced_engine = EnhancedSteganographyEngine(use_anti_detection=True)
        self.security_service = security_service or SecurityService()
        
        # State variables
        self.carrier_image_path = None
        self.files_to_hide = []
        self.output_path = None
        self.worker_thread = None
        self.analysis_worker = None  # For carrier analysis
        
        self.init_ui()
        self.connect_signals()
        
        # Auto-analyze timer
        self.analysis_timer = QTimer()
        self.analysis_timer.setSingleShot(True)
        self.analysis_timer.timeout.connect(self.start_auto_analysis)
    
    def init_ui(self):
        """Initialize the enhanced user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("🕵️‍♂️ Enhanced Anti-Detection Steganography")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(20)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Advanced steganography designed to evade tools like StegExpose, zsteg, StegSeek, and other steganalysis methods.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Create tab widget for organized interface
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Setup tabs
        self._init_basic_tab()
        self._init_advanced_tab()
        self._init_analysis_tab()
        
        # Progress section
        self.progress_group = QGroupBox("Operation Progress")
        progress_layout = QVBoxLayout(self.progress_group)
        
        self.status_label = QLabel("Ready for enhanced steganography")
        self.progress_bar = QProgressBar()
        
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_group)
        self.progress_group.hide()
        
        # Risk Analysis section
        self._init_risk_analysis_section(layout)
        
        # Action buttons
        self._init_action_buttons(layout)
    
    def _init_basic_tab(self):
        """Initialize basic settings tab."""
        basic_tab = QWidget()
        layout = QVBoxLayout(basic_tab)
        
        # Carrier image selection
        carrier_group = QGroupBox("📁 Carrier Image")
        carrier_layout = QVBoxLayout(carrier_group)
        
        self.carrier_label = QLabel("No carrier image selected")
        self.carrier_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.carrier_button = QPushButton("Select Carrier Image (PNG, BMP, TIFF)")
        
        # Image preview and analysis
        self.image_preview = QLabel()
        self.image_preview.setMaximumHeight(150)
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setStyleSheet("border: 1px solid #ccc; border-radius: 5px;")
        self.image_preview.hide()
        
        # Analysis progress indicator
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                height: 20px;
                background-color: #3a3a3a;
            }
            QProgressBar::chunk {
                background-color: #4fc3f7;
                border-radius: 3px;
            }
        """)
        self.analysis_progress.hide()
        
        self.carrier_analysis_label = QLabel()
        self.carrier_analysis_label.setWordWrap(True)
        self.carrier_analysis_label.hide()
        
        carrier_layout.addWidget(self.carrier_label)
        carrier_layout.addWidget(self.carrier_button)
        carrier_layout.addWidget(self.image_preview)
        carrier_layout.addWidget(self.analysis_progress)
        carrier_layout.addWidget(self.carrier_analysis_label)
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
        
        # Basic settings
        settings_group = QGroupBox("⚙️ Basic Settings")
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
        self.security_combo.setCurrentIndex(2)  # Default to Maximum
        security_layout.addWidget(self.security_combo)
        settings_layout.addLayout(security_layout)
        
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
        
        self.tab_widget.addTab(basic_tab, "📋 Basic Settings")
    
    def _init_advanced_tab(self):
        """Initialize advanced anti-detection tab."""
        advanced_tab = QWidget()
        layout = QVBoxLayout(advanced_tab)
        
        # Anti-detection settings
        anti_detection_group = QGroupBox("🛡️ Anti-Detection Settings")
        anti_detection_layout = QVBoxLayout(anti_detection_group)
        
        self.use_anti_detection = QCheckBox("Enable Anti-Detection Mode (Recommended)")
        self.use_anti_detection.setChecked(True)
        self.use_anti_detection.setStyleSheet("font-weight: bold; color: #2E8B57;")
        anti_detection_layout.addWidget(self.use_anti_detection)
        
        # Target risk level
        risk_layout = QHBoxLayout()
        risk_layout.addWidget(QLabel("Target Detectability Risk:"))
        self.risk_level_combo = QComboBox()
        self.risk_level_combo.addItems(["LOW (Maximum Protection)", "MEDIUM (Balanced)", "HIGH (Minimum Protection)"])
        self.risk_level_combo.setCurrentIndex(0)  # Default to LOW
        risk_layout.addWidget(self.risk_level_combo)
        anti_detection_layout.addLayout(risk_layout)
        
        # Advanced options
        self.histogram_preservation = QCheckBox("Preserve Original Histogram")
        self.histogram_preservation.setChecked(True)
        anti_detection_layout.addWidget(self.histogram_preservation)
        
        self.adaptive_positioning = QCheckBox("Use Adaptive Positioning")
        self.adaptive_positioning.setChecked(True)
        anti_detection_layout.addWidget(self.adaptive_positioning)
        
        self.edge_filtering = QCheckBox("Apply Edge-Aware Filtering")
        self.edge_filtering.setChecked(True)
        anti_detection_layout.addWidget(self.edge_filtering)
        
        layout.addWidget(anti_detection_group)
        
        # LSB Positioning Settings
        lsb_group = QGroupBox("📍 LSB Positioning Strategy")
        lsb_layout = QVBoxLayout(lsb_group)
        
        self.randomize_lsb = QCheckBox("Enable Randomized LSB Positioning")
        self.randomize_lsb.setChecked(True)
        self.randomize_lsb.setStyleSheet("font-weight: bold; color: #1E90FF;")
        lsb_layout.addWidget(self.randomize_lsb)
        
        lsb_info = QLabel("Randomizes the position of hidden bits to make detection more difficult. Can be used independently or combined with anti-detection mode.")
        lsb_info.setWordWrap(True)
        lsb_info.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 10px;")
        lsb_layout.addWidget(lsb_info)
        
        # Seed option
        seed_layout = QHBoxLayout()
        self.custom_seed = QCheckBox("Use Custom Seed:")
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Leave empty for password-derived seed")
        self.seed_input.setEnabled(False)
        seed_layout.addWidget(self.custom_seed)
        seed_layout.addWidget(self.seed_input)
        lsb_layout.addLayout(seed_layout)
        
        # Connect seed checkbox to enable/disable input
        self.custom_seed.toggled.connect(self.seed_input.setEnabled)
        
        layout.addWidget(lsb_group)
        
        # Steganalysis testing
        testing_group = QGroupBox("🔍 Steganalysis Testing")
        testing_layout = QVBoxLayout(testing_group)
        
        self.auto_test = QCheckBox("Automatically Test Against Common Steganalysis Tools")
        self.auto_test.setChecked(True)
        testing_layout.addWidget(self.auto_test)
        
        test_info = QLabel("This will simulate detection by StegExpose, zsteg, StegSeek, and other common tools.")
        test_info.setWordWrap(True)
        test_info.setStyleSheet("color: #666; font-size: 11px;")
        testing_layout.addWidget(test_info)
        
        layout.addWidget(testing_group)
        
        # Performance settings
        performance_group = QGroupBox("⚡ Performance Settings")
        performance_layout = QVBoxLayout(performance_group)
        
        self.fallback_mode = QCheckBox("Enable High-Speed Fallback Mode")
        self.fallback_mode.setChecked(True)
        performance_layout.addWidget(self.fallback_mode)
        
        fallback_info = QLabel("Falls back to ultra-fast hiding if anti-detection mode fails.")
        fallback_info.setWordWrap(True)
        fallback_info.setStyleSheet("color: #666; font-size: 11px;")
        performance_layout.addWidget(fallback_info)
        
        layout.addWidget(performance_group)
        
        self.tab_widget.addTab(advanced_tab, "🛡️ Anti-Detection")
    
    def _init_analysis_tab(self):
        """Initialize analysis and testing tab."""
        analysis_tab = QWidget()
        layout = QVBoxLayout(analysis_tab)
        
        # Carrier analysis
        analysis_group = QGroupBox("📊 Carrier Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setMaximumHeight(200)
        self.analysis_text.setPlaceholderText("Select a carrier image to see detailed analysis...")
        self.analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_text)
        
        self.analyze_button = QPushButton("🔍 Analyze Current Carrier")
        self.analyze_button.setEnabled(False)
        analysis_layout.addWidget(self.analyze_button)
        
        layout.addWidget(analysis_group)
        
        # Optimal settings
        optimal_group = QGroupBox("⚙️ Optimal Settings Recommendations")
        optimal_layout = QVBoxLayout(optimal_group)
        
        self.optimal_text = QTextEdit()
        self.optimal_text.setMaximumHeight(150)
        self.optimal_text.setPlaceholderText("Analysis will provide optimal settings recommendations...")
        self.optimal_text.setReadOnly(True)
        optimal_layout.addWidget(self.optimal_text)
        
        layout.addWidget(optimal_group)
        
        self.tab_widget.addTab(analysis_tab, "📊 Analysis")
    
    def _init_risk_analysis_section(self, layout):
        """Initialize risk analysis results section."""
        self.risk_group = QGroupBox("🛡️ Security Risk Analysis")
        risk_layout = QVBoxLayout(self.risk_group)
        
        # Risk level display
        self.risk_level_label = QLabel()
        self.risk_level_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.risk_level_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        risk_layout.addWidget(self.risk_level_label)
        
        # Risk details
        self.risk_details = QTextEdit()
        self.risk_details.setMaximumHeight(150)
        self.risk_details.setReadOnly(True)
        risk_layout.addWidget(self.risk_details)
        
        layout.addWidget(self.risk_group)
        self.risk_group.hide()
    
    def _init_action_buttons(self, layout):
        """Initialize action buttons."""
        button_layout = QHBoxLayout()
        
        self.hide_button = QPushButton("🛡️ Create Undetectable Stego")
        self.hide_button.setEnabled(False)
        self.hide_button.setStyleSheet("""
            QPushButton {
                background-color: #2E8B57;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #228B22;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        self.test_button = QPushButton("🔍 Test Existing Image")
        self.test_button.setStyleSheet("""
            QPushButton {
                background-color: #4682B4;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4169E1;
            }
        """)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #DC143C;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #B22222;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(self.hide_button)
        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        self.carrier_button.clicked.connect(self.select_carrier_image)
        self.files_button.clicked.connect(self.select_files)
        self.output_button.clicked.connect(self.select_output)
        self.analyze_button.clicked.connect(self.analyze_carrier)
        self.hide_button.clicked.connect(self.create_undetectable_stego)
        self.test_button.clicked.connect(self.test_existing_image)
        self.cancel_button.clicked.connect(self.cancel_operation)
        
        # Auto-update on changes
        self.password_input.textChanged.connect(self.check_ready_state)
        self.use_anti_detection.toggled.connect(self.update_ui_mode)
        
        # Auto-analyze carrier when selected (delayed to allow UI to update)
        self.carrier_button.clicked.connect(lambda: self.analysis_timer.start(1000))
    
    def select_carrier_image(self):
        """Select carrier image file with enhanced analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Carrier Image",
            "",
            "Image Files (*.png *.bmp *.tiff *.tif);;PNG Files (*.png);;BMP Files (*.bmp);;TIFF Files (*.tiff *.tif)"
        )
        
        if file_path:
            self.carrier_image_path = Path(file_path)
            
            # Show immediate feedback
            self.carrier_label.setText(
                f"🔍 Analyzing {self.carrier_image_path.name}..."
            )
            
            # Show image preview immediately (this is lightweight)
            try:
                pixmap = QPixmap(str(self.carrier_image_path))
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.image_preview.setPixmap(scaled_pixmap)
                    self.image_preview.show()
            except Exception as e:
                self.logger.warning(f"Could not load image preview: {e}")
            
            # Start asynchronous basic analysis
            self.start_basic_analysis()
            
            self.check_ready_state()
    
    def start_auto_analysis(self):
        """Start automatic analysis with delay to avoid blocking UI."""
        if self.carrier_image_path:
            self.start_detailed_analysis()
    
    def start_basic_analysis(self):
        """Start basic carrier analysis in background thread."""
        if not self.carrier_image_path or self.analysis_worker and self.analysis_worker.isRunning():
            return
        
        # Clean up any previous analysis worker
        if self.analysis_worker:
            self.analysis_worker.quit()
            self.analysis_worker.wait()
        
        # Show progress indicator
        self.analysis_progress.setValue(0)
        self.analysis_progress.show()
        
        # Start basic analysis worker
        self.analysis_worker = CarrierAnalysisWorker(
            self.enhanced_engine, self.carrier_image_path, "basic"
        )
        self.analysis_worker.analysis_completed.connect(self.on_basic_analysis_completed)
        self.analysis_worker.analysis_failed.connect(self.on_analysis_failed)
        self.analysis_worker.progress_updated.connect(self.analysis_progress.setValue)
        self.analysis_worker.status_updated.connect(self.update_carrier_status)
        self.analysis_worker.start()
    
    def start_detailed_analysis(self):
        """Start detailed carrier analysis in background thread."""
        if not self.carrier_image_path:
            return
        
        # Disable the analyze button during analysis
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("🔍 Analyzing...")
        
        # Clean up any previous analysis worker
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.quit()
            self.analysis_worker.wait()
        
        # Show progress indicator 
        self.analysis_progress.setValue(0)
        self.analysis_progress.show()
        
        # Start detailed analysis worker
        self.analysis_worker = CarrierAnalysisWorker(
            self.enhanced_engine, self.carrier_image_path, "detailed"
        )
        self.analysis_worker.analysis_completed.connect(self.on_detailed_analysis_completed)
        self.analysis_worker.analysis_failed.connect(self.on_analysis_failed)
        self.analysis_worker.progress_updated.connect(self.analysis_progress.setValue)
        self.analysis_worker.status_updated.connect(self.update_carrier_status)
        self.analysis_worker.finished.connect(self.on_analysis_worker_finished)
        self.analysis_worker.start()
    
    def analyze_carrier(self):
        """Perform detailed carrier analysis (public method for button click)."""
        self.start_detailed_analysis()
    
    def update_carrier_status(self, status):
        """Update carrier status text during analysis."""
        if self.carrier_image_path:
            self.carrier_label.setText(f"🔍 {status}...")
    
    def on_basic_analysis_completed(self, results):
        """Handle completed basic analysis."""
        try:
            # Hide progress indicator
            self.analysis_progress.hide()
            
            capacity_mb = results.get('capacity_mb', 0)
            suitability = results.get('suitability_score', 0)
            
            self.carrier_label.setText(
                f"✅ {self.carrier_image_path.name}\n"
                f"📊 Capacity: {capacity_mb:.2f} MB\n"
                f"⭐ Suitability: {suitability}/10"
            )
            
            self.analyze_button.setEnabled(True)
            
        except Exception as e:
            self.logger.error(f"Error processing basic analysis results: {e}")
            self.on_analysis_failed(str(e))
    
    def on_detailed_analysis_completed(self, results):
        """Handle completed detailed analysis."""
        try:
            # Hide progress indicator
            self.analysis_progress.hide()
            
            analysis = results.get('full_analysis', {})
            analysis_text = results.get('analysis_text', "Analysis completed")
            
            # Display analysis results
            self.analysis_text.setText(analysis_text)
            
            # Get optimal settings if files are selected
            if self.files_to_hide:
                total_size = sum(Path(f).stat().st_size for f in self.files_to_hide if Path(f).exists())
                optimal = self.enhanced_engine.get_optimal_settings(self.carrier_image_path, total_size)
                
                optimal_text = f"""
⚙️ OPTIMAL SETTINGS

🎯 Recommendations:
• Anti-Detection: {'✅ Enabled' if optimal.get('use_anti_detection') else '❌ Disabled'}
• Randomization: {'✅ Enabled' if optimal.get('randomize') else '❌ Disabled'}

"""
                
                if 'warning' in optimal:
                    optimal_text += f"⚠️ Warning: {optimal['warning']}\n"
                
                if 'recommendation' in optimal:
                    optimal_text += f"💡 Suggestion: {optimal['recommendation']}\n"
                
                if 'image_recommendation' in optimal:
                    optimal_text += f"🖼️ Image: {optimal['image_recommendation']}\n"
                
                self.optimal_text.setText(optimal_text.strip())
            
            # Switch to analysis tab
            self.tab_widget.setCurrentIndex(2)
            
        except Exception as e:
            self.logger.error(f"Error processing detailed analysis results: {e}")
            self.on_analysis_failed(str(e))
    
    def on_analysis_failed(self, error_message):
        """Handle analysis failure."""
        # Hide progress indicator
        self.analysis_progress.hide()
        
        self.carrier_label.setText(f"❌ Error analyzing image: {error_message}")
        self.analyze_button.setEnabled(True)
        
    def on_analysis_worker_finished(self):
        """Handle analysis worker completion."""
        # Re-enable the analyze button
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("🔍 Analyze Current Carrier")
    
    def create_undetectable_stego(self):
        """Create undetectable steganographic image."""
        if not self._validate_inputs():
            return
        
        # Check authentication first
        if not self._check_authentication():
            return
        
        # Get settings
        password = self.password_input.text().strip()
        security_levels = [SecurityLevel.STANDARD, SecurityLevel.HIGH, SecurityLevel.MAXIMUM]
        security_level = security_levels[self.security_combo.currentIndex()]
        use_anti_detection = self.use_anti_detection.isChecked()
        
        risk_levels = ["LOW", "MEDIUM", "HIGH"]
        target_risk_level = risk_levels[self.risk_level_combo.currentIndex()]
        
        # LSB randomization settings
        randomize_lsb = self.randomize_lsb.isChecked()
        custom_seed = None
        if self.custom_seed.isChecked() and self.seed_input.text().strip():
            try:
                custom_seed = int(self.seed_input.text().strip())
            except ValueError:
                QMessageBox.warning(self, "Invalid Seed", "Custom seed must be a valid integer.")
                return
        
        # Setup UI for operation
        self.progress_group.show()
        self.hide_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Start worker thread
        # Assert that paths are not None after validation
        assert self.carrier_image_path is not None, "Carrier image path must be set"
        assert self.output_path is not None, "Output path must be set"
        
        self.worker_thread = EnhancedHideWorkerThread(
            carrier_path=self.carrier_image_path,
            files_to_hide=self.files_to_hide,
            output_path=self.output_path,
            password=password,
            security_level=security_level,
            use_anti_detection=use_anti_detection,
            target_risk_level=target_risk_level,
            randomize_lsb=randomize_lsb,
            custom_seed=custom_seed
        )
        
        # Connect signals
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.status_label.setText)
        self.worker_thread.risk_analysis_updated.connect(self.display_risk_analysis)
        self.worker_thread.finished.connect(self.on_hiding_finished)
        self.worker_thread.error_occurred.connect(self.on_hiding_error)
        
        self.worker_thread.start()
    
    def display_risk_analysis(self, analysis: Dict[str, Any]):
        """Display risk analysis results."""
        self.risk_group.show()
        
        creation_result = analysis.get('creation_result', {})
        steganalysis_test = analysis.get('steganalysis_test', {})
        
        # Risk level display
        risk_level = creation_result.get('risk_level', 'UNKNOWN')
        risk_score = creation_result.get('risk_score', 1.0)
        
        if risk_level == 'LOW':
            color = "#2E8B57"
            icon = "🛡️"
        elif risk_level == 'MEDIUM':
            color = "#FF8C00"
            icon = "⚠️"
        else:
            color = "#DC143C"
            icon = "🚨"
        
        self.risk_level_label.setText(f"{icon} Risk Level: {risk_level}")
        self.risk_level_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                margin: 5px;
            }}
        """)
        
        # Risk details
        overall_assessment = steganalysis_test.get('overall_assessment', {})
        tool_simulation = steganalysis_test.get('tool_simulation', {})
        
        details_text = f"""
🛡️ SECURITY ANALYSIS RESULTS

📊 Overall Risk Score: {risk_score:.3f}/1.0
🎯 Creation Attempts: {creation_result.get('attempts', 1)}
🔍 Safety Level: {overall_assessment.get('safety_level', 'UNKNOWN')}

🔍 STEGANALYSIS TOOL SIMULATION:

StegExpose-like Detection:
• LSB Evenness: {tool_simulation.get('stegexpose_risk', {}).get('lsb_evenness', 0):.3f}
• Detection Risk: {'🔴 HIGH' if tool_simulation.get('stegexpose_risk', {}).get('likely_detected', False) else '🟢 LOW'}

Chi-Square Test:
• Risk Score: {tool_simulation.get('chi_square_test', {}).get('risk_score', 0):.3f}
• Detection Risk: {'🔴 HIGH' if tool_simulation.get('chi_square_test', {}).get('likely_detected', False) else '🟢 LOW'}

Histogram Analysis:
• Anomaly Score: {tool_simulation.get('histogram_analysis', {}).get('anomaly_score', 0):.3f}
• Detection Risk: {'🔴 HIGH' if tool_simulation.get('histogram_analysis', {}).get('likely_detected', False) else '🟢 LOW'}

Noise Pattern Analysis:
• Pattern Risk: {tool_simulation.get('noise_analysis', {}).get('artificial_pattern_risk', 0):.3f}
• Detection Risk: {'🔴 HIGH' if tool_simulation.get('noise_analysis', {}).get('likely_detected', False) else '🟢 LOW'}

🎯 OVERALL ASSESSMENT:
Detection by any common tool: {'🔴 LIKELY' if overall_assessment.get('likely_detected_by_any_tool', False) else '🟢 UNLIKELY'}
Average Detection Risk: {overall_assessment.get('average_detection_risk', 0):.3f}
"""
        
        self.risk_details.setText(details_text.strip())
    
    def on_hiding_finished(self, result: Dict[str, Any]):
        """Handle successful hiding completion."""
        self.progress_group.hide()
        self.hide_button.setEnabled(True)
        
        # Show success message
        message = f"""
🎉 Enhanced Steganography Successful!

✅ Files Hidden: {result['files_hidden']}
📊 Data Size: {result['data_size']:,} bytes
🛡️ Anti-Detection: {'✅ Enabled' if result['anti_detection_used'] else '❌ Disabled'}
"""
        
        if result['anti_detection_used']:
            message += f"""
🎯 Risk Level: {result['risk_level']}
📊 Risk Score: {result['risk_score']:.3f}
🔄 Attempts Needed: {result['attempts_needed']}
"""
        else:
            # Show LSB randomization status for non-anti-detection mode
            if result.get('randomized_lsb', False):
                message += "\n📍 LSB Positioning: Randomized"
            else:
                message += "\n📍 LSB Positioning: Sequential"
            
            # Show fallback information if used
            if result.get('fallback_used', False):
                message += f"\n🔄 Fallback Used: {result.get('fallback_reason', 'Unknown reason')}"
                message += "\n⚠️ Note: Anti-detection failed, but data was successfully hidden using randomized LSB"
                
                # Show detailed constraint failure reasons
                constraint_details = result.get('constraint_details', [])
                if constraint_details:
                    message += "\n\n🔍 WHY ANTI-DETECTION FAILED:"
                    for constraint in constraint_details:
                        message += f"\n• {constraint}"
        
        message += f"\n💾 Saved to: {result['output_path']}"
        
        QMessageBox.information(self, "Success!", message)
        
        # Optional: Close dialog or reset for another operation
        # self.accept()
    
    def on_hiding_error(self, error_message: str):
        """Handle hiding operation error."""
        self.progress_group.hide()
        self.hide_button.setEnabled(True)
        
        QMessageBox.critical(self, "Operation Failed", f"Enhanced hiding failed:\n\n{error_message}")
    
    def test_existing_image(self):
        """Test an existing steganographic image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Steganographic Image to Test",
            "",
            "Image Files (*.png *.bmp *.tiff *.tif);;All Files (*.*)"
        )
        
        if file_path:
            try:
                test_results = self.enhanced_engine.test_against_steganalysis(Path(file_path))
                
                # Display results
                self.display_risk_analysis({'steganalysis_test': test_results, 'creation_result': {}})
                
                # Switch to show results
                self.risk_group.show()
                
                QMessageBox.information(
                    self, 
                    "Steganalysis Test Complete", 
                    f"Test completed for {Path(file_path).name}.\nCheck the Risk Analysis section for detailed results."
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Test Failed", f"Failed to test image:\n\n{str(e)}")
    
    def _validate_inputs(self) -> bool:
        """Validate all inputs before operation."""
        if not self.carrier_image_path:
            QMessageBox.warning(self, "Missing Input", "Please select a carrier image.")
            return False
        
        if not self.files_to_hide:
            QMessageBox.warning(self, "Missing Input", "Please select files to hide.")
            return False
        
        if not self.output_path:
            QMessageBox.warning(self, "Missing Input", "Please select an output location.")
            return False
        
        password = self.password_input.text().strip()
        if len(password) < 6:
            QMessageBox.warning(self, "Weak Password", "Password must be at least 6 characters long.")
            return False
        
        return True
    
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
            
            # Calculate total size and display info
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
            files_text += f"\n\n📊 Total size: {total_mb:.2f} MB ({total_size:,} bytes)"
            
            self.files_list.setText(files_text)
            self.check_ready_state()
    
    def select_output(self):
        """Select output file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Enhanced Steganographic Image As",
            "",
            "PNG Files (*.png);;BMP Files (*.bmp);;TIFF Files (*.tiff)"
        )
        
        if file_path:
            self.output_path = Path(file_path)
            self.output_label.setText(f"💾 Output: {self.output_path.name}")
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
    
    def update_ui_mode(self):
        """Update UI based on anti-detection mode."""
        if self.use_anti_detection.isChecked():
            self.hide_button.setText("🛡️ Create Undetectable Stego")
        else:
            self.hide_button.setText("⚡ Create Fast Stego")
    
    def _check_authentication(self) -> bool:
        """Check if user is authenticated - simplified for offline application."""
        # For offline steganography application, no authentication required
        # Just return True to allow all operations
        return True
    
    def cancel_operation(self):
        """Cancel current operation or close dialog."""
        # Cancel hide operation if running
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.progress_group.hide()
            self.hide_button.setEnabled(True)
        
        # Cancel analysis operation if running
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.quit()
            self.analysis_worker.wait(1000)  # Wait up to 1 second
            self.analyze_button.setEnabled(True)
            self.analyze_button.setText("🔍 Analyze Current Carrier")
        
        # Close dialog if no operations are running
        if not ((self.worker_thread and self.worker_thread.isRunning()) or 
                (self.analysis_worker and self.analysis_worker.isRunning())):
            self.reject()
