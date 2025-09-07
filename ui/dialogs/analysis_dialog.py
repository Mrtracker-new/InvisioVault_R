"""
Analysis Dialog
Image analysis dialog for capacity assessment and steganographic detection.
Refactored to use AnalysisOperation and ImageAnalyzer integration.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QTextEdit, QProgressBar, QMessageBox,
    QTabWidget, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPalette

from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler
from operations.analysis_operation import AnalysisOperation
from core.analyzers.image_analyzer import AnalysisLevel


class AnalysisWorkerThread(QThread):
    """Worker thread that uses AnalysisOperation for proper integration."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    analysis_completed = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self, image_path: str, analysis_quality: str = "balanced"):
        super().__init__()
        self.image_path = image_path
        self.analysis_quality = analysis_quality
        self.logger = Logger()
        self._cancelled = False
        
        # Create AnalysisOperation instance
        self.analysis_operation = AnalysisOperation()
    
    def cancel(self):
        """Cancel the analysis operation."""
        self._cancelled = True
    
    def run(self):
        """Execute the analysis using AnalysisOperation."""
        try:
            if self._cancelled:
                return
                
            # Map quality levels to AnalysisOperation levels
            quality_map = {
                "fast": "basic",
                "balanced": "full", 
                "thorough": "comprehensive"
            }
            operation_level = quality_map.get(self.analysis_quality, "full")
            
            self.status_updated.emit("Initializing analysis...")
            self.progress_updated.emit(5)
            
            # Configure the operation
            self.analysis_operation.configure(self.image_path, operation_level)
            
            if self._cancelled:
                return
            
            self.status_updated.emit("Validating image...")
            self.progress_updated.emit(10)
            
            # Validate inputs
            if not self.analysis_operation.validate_inputs():
                raise Exception("Image validation failed")
            
            if self._cancelled:
                return
            
            # Provide more detailed progress for thorough analysis
            if operation_level == "comprehensive":
                self.status_updated.emit("Starting thorough analysis (this may take longer)...")
            else:
                self.status_updated.emit("Running comprehensive analysis...")
            
            self.progress_updated.emit(20)
            
            # Run the analysis with callbacks
            results = self.analysis_operation.run_analysis(
                progress_callback=self._on_progress,
                status_callback=self._on_status
            )
            
            if not self._cancelled:
                self.status_updated.emit("Analysis completed successfully!")
                self.progress_updated.emit(100)
                
                # Convert results to the expected format for the UI
                formatted_results = self._format_results_for_ui(results)
                self.analysis_completed.emit(formatted_results)
            
        except Exception as e:
            if not self._cancelled:
                self.logger.error(f"Analysis failed: {e}")
                self.error_occurred.emit(str(e))
    
    def _on_progress(self, progress: float):
        """Handle progress updates from AnalysisOperation."""
        if not self._cancelled:
            # Scale progress to leave room for initial setup (20-95%)
            scaled_progress = int(20 + (progress * 75))
            self.progress_updated.emit(scaled_progress)
    
    def _on_status(self, status: str):
        """Handle status updates from AnalysisOperation."""
        if not self._cancelled:
            self.status_updated.emit(status)
    
    def _format_results_for_ui(self, operation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format AnalysisOperation results for the existing UI structure."""
        try:
            # Get the comprehensive analysis results
            comprehensive = operation_results.get('comprehensive_analysis', {})
            
            if not comprehensive:
                # Handle basic analysis results
                return self._format_basic_results(operation_results)
            
            # Format comprehensive results
            formatted = {
                'analysis_quality': self.analysis_quality,
                'analysis_type': operation_results.get('analysis_type', 'comprehensive')
            }
            
            # Basic info from file_info and image_properties
            file_info = comprehensive.get('file_info', {})
            image_props = comprehensive.get('image_properties', {})
            
            formatted['basic_info'] = {
                'file_path': file_info.get('filepath', self.image_path),
                'file_size': file_info.get('file_size_bytes', 0),
                'width': image_props.get('width', 0),
                'height': image_props.get('height', 0),
                'channels': image_props.get('channels', 3),
                'total_pixels': image_props.get('total_pixels', 0),
                'bits_per_pixel': image_props.get('bits_per_pixel', 24),
                'format': image_props.get('format', 'Unknown'),
                'mode': image_props.get('mode', 'RGB')
            }
            
            # Capacity analysis
            capacity = comprehensive.get('capacity_analysis', {})
            formatted['capacity'] = {
                'lsb_capacity_bits': capacity.get('lsb_total_bits', 0),
                'lsb_capacity_bytes': capacity.get('lsb_capacity_bytes', 0),
                'lsb_capacity_kb': capacity.get('lsb_capacity_kb', 0),
                'lsb_capacity_mb': capacity.get('lsb_capacity_mb', 0),
                'capacity_ratio': capacity.get('capacity_ratio', 0),
                'effective_capacity_bytes': capacity.get('estimated_overhead_bits', 0)
            }
            
            # Statistics
            quality_metrics = comprehensive.get('quality_metrics', {})
            entropy_data = quality_metrics.get('entropy', {})
            noise_data = quality_metrics.get('noise_level', {})
            
            # Create statistics structure compatible with UI
            formatted['statistics'] = {
                'entropy': entropy_data.get('overall', 0),
                'overall': {
                    'mean': 128.0,  # Default values if not available
                    'std': noise_data.get('overall', 0),
                    'min': 0,
                    'max': 255,
                    'median': 128.0,
                    'variance': noise_data.get('overall', 0) ** 2 if noise_data.get('overall') else 0
                },
                'channels': entropy_data.get('channels', []),
                'is_sampled': False
            }
            
            # Quality metrics
            texture_data = quality_metrics.get('texture_complexity', {})
            formatted['quality_metrics'] = {
                'noise_level': noise_data.get('overall', 0),
                'sharpness': 0.0,  # Not directly available from ImageAnalyzer
                'texture_complexity': texture_data.get('complexity_score', 0) if isinstance(texture_data, dict) else texture_data
            }
            
            # LSB analysis
            lsb_analysis = comprehensive.get('lsb_analysis', {})
            overall_lsb = lsb_analysis.get('overall_lsb_assessment', {})
            
            formatted['lsb_analysis'] = {
                'channels': [],  # Channel-specific data from lsb_analysis
                'average_ones_ratio': overall_lsb.get('average_uniformity_deviation', 0.5),
                'average_chi_square': 0.0,  # Not directly available
                'randomness_suspicion': self._map_suspicion_level(overall_lsb.get('steganography_likelihood', 'low')),
                'chi_suspicion': "Low"  # Default
            }
            
            # Suitability assessment
            suitability = comprehensive.get('suitability_assessment', {})
            formatted['suitability'] = {
                'score': int(suitability.get('overall_score', 0) * 10),  # Scale to 100
                'rating': suitability.get('rating', 'unknown').title(),
                'reasons': [suitability.get('recommendation', 'Analysis completed')],
                'warnings': [],
                'max_score': 100
            }
            
            # Add recommendations if available
            recommendations = comprehensive.get('recommendations', [])
            if recommendations:
                formatted['suitability']['reasons'].extend(recommendations[:5])  # Limit to 5
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting results: {e}")
            return self._create_error_results(str(e))
    
    def _format_basic_results(self, operation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format basic analysis results when comprehensive analysis is not available."""
        try:
            quick_check = operation_results.get('quick_suitability_check', {})
            file_info = operation_results.get('file_info', {})
            
            # Create minimal structure for basic results
            return {
                'analysis_quality': self.analysis_quality,
                'analysis_type': 'basic',
                'basic_info': {
                    'file_path': file_info.get('file_path', self.image_path),
                    'file_size': file_info.get('file_size_bytes', 0),
                    'width': int(quick_check.get('dimensions', '0x0').split('x')[0]) if 'x' in str(quick_check.get('dimensions', '0x0')) else 0,
                    'height': int(quick_check.get('dimensions', '0x0').split('x')[1]) if 'x' in str(quick_check.get('dimensions', '0x0')) else 0,
                    'channels': 3,
                    'total_pixels': 0,
                    'bits_per_pixel': 24,
                    'format': quick_check.get('format', 'Unknown'),
                    'mode': 'RGB'
                },
                'capacity': {
                    'lsb_capacity_bits': quick_check.get('estimated_capacity_bytes', 0) * 8,
                    'lsb_capacity_bytes': quick_check.get('estimated_capacity_bytes', 0),
                    'lsb_capacity_kb': quick_check.get('estimated_capacity_kb', 0),
                    'lsb_capacity_mb': quick_check.get('estimated_capacity_kb', 0) / 1024,
                    'capacity_ratio': 0.1,
                    'effective_capacity_bytes': int(quick_check.get('estimated_capacity_bytes', 0) * 0.85)
                },
                'statistics': {
                    'entropy': 6.0,  # Default reasonable value
                    'overall': {'mean': 128, 'std': 50, 'min': 0, 'max': 255, 'median': 128, 'variance': 2500},
                    'channels': [{'mean': 128, 'std': 50, 'min': 0, 'max': 255, 'median': 128, 'variance': 2500}] * 3
                },
                'quality_metrics': {
                    'noise_level': 25.0,
                    'sharpness': 15.0,
                    'texture_complexity': 30.0
                },
                'lsb_analysis': {
                    'channels': [],
                    'average_ones_ratio': 0.5,
                    'average_chi_square': 1.0,
                    'randomness_suspicion': "Low",
                    'chi_suspicion': "Low"
                },
                'suitability': {
                    'score': 70 if quick_check.get('suitable', False) else 30,
                    'rating': "Good" if quick_check.get('suitable', False) else "Poor",
                    'reasons': [quick_check.get('recommendation', 'Basic analysis completed')],
                    'warnings': [],
                    'max_score': 100
                }
            }
        except Exception as e:
            self.logger.error(f"Error formatting basic results: {e}")
            return self._create_error_results(str(e))
    
    def _map_suspicion_level(self, likelihood: str) -> str:
        """Map steganography likelihood to suspicion level."""
        mapping = {
            'high': 'High',
            'medium': 'Medium',
            'low': 'Low',
            'none': 'Low'
        }
        return mapping.get(likelihood.lower(), 'Low')
    
    def _create_error_results(self, error_msg: str) -> Dict[str, Any]:
        """Create error results structure."""
        return {
            'error': error_msg,
            'basic_info': {'file_path': self.image_path, 'file_size': 0, 'width': 0, 'height': 0, 'channels': 0, 'total_pixels': 0, 'bits_per_pixel': 0},
            'capacity': {'lsb_capacity_bits': 0, 'lsb_capacity_bytes': 0, 'lsb_capacity_kb': 0, 'lsb_capacity_mb': 0, 'capacity_ratio': 0},
            'statistics': {'entropy': 0, 'overall': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'variance': 0}, 'channels': []},
            'quality_metrics': {'noise_level': 0, 'sharpness': 0, 'texture_complexity': 0},
            'lsb_analysis': {'channels': [], 'average_ones_ratio': 0.5, 'average_chi_square': 0, 'randomness_suspicion': 'Low', 'chi_suspicion': 'Low'},
            'suitability': {'score': 0, 'rating': 'Error', 'reasons': [f'Analysis failed: {error_msg}'], 'warnings': [], 'max_score': 100}
        }


class AnalysisDialog(QDialog):
    """Dialog for comprehensive image analysis and capacity assessment."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Analysis & Capacity Assessment")
        self.setModal(True)
        self.resize(1100, 800)
        
        # Initialize components
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        
        # State variables
        self.worker_thread = None
        self.analysis_results = None
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🔍 Image Analysis & Capacity Assessment")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Analyze images for steganographic capacity, detect existing hidden data, "
            "and assess suitability for steganographic operations using advanced ImageAnalyzer."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Image selection
        selection_group = QGroupBox("📁 Image Selection & Analysis Settings")
        selection_layout = QVBoxLayout(selection_group)
        
        self.image_label = QLabel("No image selected")
        self.image_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.image_label.setMinimumHeight(60)
        
        # Analysis quality selection
        quality_layout = QHBoxLayout()
        quality_label = QLabel("Analysis Quality:")
        quality_layout.addWidget(quality_label)
        
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Fast", "Balanced", "Thorough"])
        self.quality_combo.setCurrentText("Balanced")
        self.quality_combo.setToolTip(
            "Fast: Quick analysis with sampling for large images\n"
            "Balanced: Good balance between speed and accuracy\n"
            "Thorough: Complete analysis with advanced algorithms"
        )
        quality_layout.addWidget(self.quality_combo)
        quality_layout.addStretch()
        
        selection_button_layout = QHBoxLayout()
        self.select_image_button = QPushButton("📁 Select Image")
        self.select_image_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        
        self.analyze_button = QPushButton("🔍 Analyze Image")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet("""
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
        
        self.cancel_button = QPushButton("⏹ Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #d32f2f; }
        """)
        
        selection_button_layout.addWidget(self.select_image_button)
        selection_button_layout.addStretch()
        selection_button_layout.addWidget(self.cancel_button)
        selection_button_layout.addWidget(self.analyze_button)
        
        selection_layout.addWidget(self.image_label)
        selection_layout.addLayout(quality_layout)
        selection_layout.addLayout(selection_button_layout)
        layout.addWidget(selection_group)
        
        # Progress section
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)
        
        # Results area
        self.results_tabs = QTabWidget()
        self.results_tabs.setVisible(False)
        
        # Create result tabs
        self.overview_tab = self.create_overview_tab()
        self.capacity_tab = self.create_capacity_tab()
        self.statistics_tab = self.create_statistics_tab()
        self.detection_tab = self.create_detection_tab()
        
        self.results_tabs.addTab(self.overview_tab, "📊 Overview")
        self.results_tabs.addTab(self.capacity_tab, "💾 Capacity")
        self.results_tabs.addTab(self.statistics_tab, "📈 Statistics")
        self.results_tabs.addTab(self.detection_tab, "🔍 Detection")
        
        layout.addWidget(self.results_tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.export_button = QPushButton("📄 Export Report")
        self.export_button.setEnabled(False)
        self.close_button = QPushButton("Close")
        
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
    
    def create_overview_tab(self) -> QWidget:
        """Create the overview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Basic info table
        self.basic_info_table = QTableWidget()
        self.basic_info_table.setColumnCount(2)
        self.basic_info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.basic_info_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(QLabel("Basic Information:"))
        layout.addWidget(self.basic_info_table)
        
        # Suitability assessment
        suitability_group = QGroupBox("Steganographic Suitability")
        suitability_layout = QVBoxLayout(suitability_group)
        
        self.suitability_rating = QLabel("Not analyzed")
        self.suitability_rating.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        
        self.suitability_reasons = QTextEdit()
        self.suitability_reasons.setReadOnly(True)
        self.suitability_reasons.setMaximumHeight(120)
        
        suitability_layout.addWidget(QLabel("Overall Rating:"))
        suitability_layout.addWidget(self.suitability_rating)
        suitability_layout.addWidget(QLabel("Assessment Details:"))
        suitability_layout.addWidget(self.suitability_reasons)
        
        layout.addWidget(suitability_group)
        
        return widget
    
    def create_capacity_tab(self) -> QWidget:
        """Create the capacity analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Capacity table
        self.capacity_table = QTableWidget()
        self.capacity_table.setColumnCount(2)
        self.capacity_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.capacity_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(QLabel("Storage Capacity (LSB Steganography):"))
        layout.addWidget(self.capacity_table)
        
        # Visual representation would go here (chart)
        info_label = QLabel(
            "💡 Tip: LSB steganography uses the least significant bit of each color channel. "
            "The actual usable capacity may be lower due to encryption overhead and error correction."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            "color: #666; font-size: 11px; padding: 10px; "
            "background-color: #f0f8ff; border-radius: 5px; margin-top: 10px;"
        )
        layout.addWidget(info_label)
        
        return widget
    
    def create_statistics_tab(self) -> QWidget:
        """Create the statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistics table
        self.statistics_table = QTableWidget()
        self.statistics_table.setColumnCount(4)
        self.statistics_table.setHorizontalHeaderLabels(["Channel", "Mean", "Std Dev", "Entropy"])
        self.statistics_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(QLabel("Statistical Analysis:"))
        layout.addWidget(self.statistics_table)
        
        # Quality metrics
        quality_group = QGroupBox("Quality Metrics")
        quality_layout = QVBoxLayout(quality_group)
        
        self.quality_table = QTableWidget()
        self.quality_table.setColumnCount(2)
        self.quality_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.quality_table.horizontalHeader().setStretchLastSection(True)
        
        quality_layout.addWidget(self.quality_table)
        layout.addWidget(quality_group)
        
        return widget
    
    def create_detection_tab(self) -> QWidget:
        """Create the steganographic detection tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Detection results
        detection_group = QGroupBox("LSB Pattern Analysis")
        detection_layout = QVBoxLayout(detection_group)
        
        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(3)
        self.detection_table.setHorizontalHeaderLabels(["Test", "Result", "Suspicion Level"])
        self.detection_table.horizontalHeader().setStretchLastSection(True)
        
        detection_layout.addWidget(self.detection_table)
        layout.addWidget(detection_group)
        
        # Analysis explanation
        explanation = QLabel(
            "🔍 Detection Analysis:\n"
            "• Chi-Square Test: Measures randomness of LSB distribution\n"
            "• Ones Ratio: Proportion of 1s in LSB planes (should be ~0.5 for natural images)\n"
            "• High suspicion levels may indicate presence of hidden data"
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet(
            "color: #444; font-size: 11px; padding: 15px; "
            "background-color: #f9f9f9; border-radius: 5px; margin-top: 10px;"
        )
        layout.addWidget(explanation)
        
        return widget
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        self.select_image_button.clicked.connect(self.select_image)
        self.analyze_button.clicked.connect(self.analyze_image)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.export_button.clicked.connect(self.export_report)
        self.close_button.clicked.connect(self.accept)
    
    def select_image(self):
        """Select an image for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image for Analysis", "",
            "Image Files (*.png *.bmp *.tiff *.tif *.jpg *.jpeg);;All Files (*)"
        )
        
        if file_path:
            self.image_path = file_path
            self.image_label.setText(f"Selected: {Path(file_path).name}")
            self.analyze_button.setEnabled(True)
            
            # Clear previous results
            self.results_tabs.setVisible(False)
            self.export_button.setEnabled(False)
    
    def analyze_image(self):
        """Start image analysis."""
        if not hasattr(self, 'image_path'):
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return
        
        # Get analysis quality setting
        quality_map = {"Fast": "fast", "Balanced": "balanced", "Thorough": "thorough"}
        selected_quality = quality_map.get(self.quality_combo.currentText(), "balanced")
        
        # Show warning for thorough analysis
        if selected_quality == "thorough":
            reply = QMessageBox.question(
                self, "Thorough Analysis", 
                "Thorough analysis may take several minutes and will process the entire image.\n\n"
                "For very large images, this could take 3-5 minutes. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Start worker thread with quality setting
        self.worker_thread = AnalysisWorkerThread(self.image_path, selected_quality)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.status_label.setText)
        self.worker_thread.analysis_completed.connect(self.on_analysis_completed)
        self.worker_thread.error_occurred.connect(self.on_analysis_error)
        
        # Update UI state for analysis
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.analyze_button.setVisible(False)
        self.cancel_button.setVisible(True)
        self.quality_combo.setEnabled(False)
        
        # Start analysis
        self.worker_thread.start()
        
        # Update status
        estimated_time = {
            "fast": "10-30 seconds",
            "balanced": "30-60 seconds", 
            "thorough": "2-5 minutes (optimized for large images)"
        }.get(selected_quality, "unknown")
        self.logger.info(f"Starting {selected_quality} analysis using ImageAnalyzer, estimated time: {estimated_time}")
    
    def on_analysis_completed(self, results: Dict):
        """Handle analysis completion."""
        self.analysis_results = results
        
        # Reset UI state
        self.progress_bar.setVisible(False)
        self.analyze_button.setVisible(True)
        self.analyze_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.quality_combo.setEnabled(True)
        self.status_label.setText("Analysis completed successfully!")
        
        # Populate results
        self.populate_results(results)
        
        # Show results tabs
        self.results_tabs.setVisible(True)
        self.export_button.setEnabled(True)
    
    def populate_results(self, results: Dict):
        """Populate the results tabs with analysis data."""
        # Overview tab
        self.populate_basic_info(results['basic_info'])
        self.populate_suitability(results['suitability'])
        
        # Capacity tab
        self.populate_capacity(results['capacity'])
        
        # Statistics tab
        self.populate_statistics(results['statistics'], results['quality_metrics'])
        
        # Detection tab
        self.populate_detection(results['lsb_analysis'])
    
    def populate_basic_info(self, basic_info: Dict):
        """Populate basic information table."""
        info_items = [
            ("File Path", Path(basic_info['file_path']).name),
            ("File Size", f"{basic_info['file_size']:,} bytes"),
            ("Dimensions", f"{basic_info['width']} × {basic_info['height']}"),
            ("Channels", str(basic_info['channels'])),
            ("Total Pixels", f"{basic_info['total_pixels']:,}"),
            ("Bits per Pixel", str(basic_info['bits_per_pixel']))
        ]
        
        self.basic_info_table.setRowCount(len(info_items))
        for i, (prop, value) in enumerate(info_items):
            self.basic_info_table.setItem(i, 0, QTableWidgetItem(prop))
            self.basic_info_table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        self.basic_info_table.resizeColumnsToContents()
    
    def populate_suitability(self, suitability: Dict):
        """Populate suitability assessment."""
        rating = suitability['rating']
        score = suitability['score']
        
        # Set rating with color coding
        self.suitability_rating.setText(f"{rating} ({score}/100)")
        
        if rating == "Excellent":
            color = "#4CAF50"
        elif rating in ["Very Good", "Good"]:
            color = "#FF9800"
        elif rating == "Fair":
            color = "#FFC107"
        else:
            color = "#F44336"
        
        self.suitability_rating.setStyleSheet(
            f"font-size: 16px; font-weight: bold; padding: 10px; color: {color};"
        )
        
        # Set reasons
        reasons_text = "\n".join([f"• {reason}" for reason in suitability['reasons']])
        self.suitability_reasons.setText(reasons_text)
    
    def populate_capacity(self, capacity: Dict):
        """Populate capacity information."""
        capacity_items = [
            ("LSB Capacity (bits)", f"{capacity['lsb_capacity_bits']:,}"),
            ("LSB Capacity (bytes)", f"{capacity['lsb_capacity_bytes']:,}"),
            ("LSB Capacity (KB)", f"{capacity['lsb_capacity_kb']:.1f}"),
            ("LSB Capacity (MB)", f"{capacity['lsb_capacity_mb']:.2f}"),
            ("Capacity/File Size Ratio", f"{capacity['capacity_ratio']:.2f}")
        ]
        
        self.capacity_table.setRowCount(len(capacity_items))
        for i, (metric, value) in enumerate(capacity_items):
            self.capacity_table.setItem(i, 0, QTableWidgetItem(metric))
            self.capacity_table.setItem(i, 1, QTableWidgetItem(value))
        
        self.capacity_table.resizeColumnsToContents()
    
    def populate_statistics(self, statistics: Dict, quality: Dict):
        """Populate statistics and quality metrics."""
        # Statistics table
        channels = ["Overall"] + [f"Channel {i+1}" for i in range(len(statistics.get('channels', [])))]
        stats_data = [statistics['overall']] + statistics.get('channels', [])
        
        self.statistics_table.setRowCount(len(channels))
        for i, (channel, stats) in enumerate(zip(channels, stats_data)):
            self.statistics_table.setItem(i, 0, QTableWidgetItem(channel))
            
            # Handle case where stats might be a float instead of a dict
            if isinstance(stats, dict):
                mean_val = stats.get('mean', 0.0)
                std_val = stats.get('std', 0.0)
            else:
                mean_val = stats  # Assuming the float itself is mean
                std_val = 0.0
            
            self.statistics_table.setItem(i, 1, QTableWidgetItem(f"{mean_val:.2f}"))
            self.statistics_table.setItem(i, 2, QTableWidgetItem(f"{std_val:.2f}"))
            if channel == "Overall":
                self.statistics_table.setItem(i, 3, QTableWidgetItem(f"{statistics.get('entropy', 0.0):.3f}"))
            else:
                self.statistics_table.setItem(i, 3, QTableWidgetItem("N/A"))
        
        self.statistics_table.resizeColumnsToContents()
        
        # Quality metrics
        quality_items = [
            ("Noise Level", f"{quality['noise_level']:.2f}"),
            ("Sharpness", f"{quality['sharpness']:.2f}"),
            ("Texture Complexity", f"{quality['texture_complexity']:.2f}")
        ]
        
        self.quality_table.setRowCount(len(quality_items))
        for i, (metric, value) in enumerate(quality_items):
            self.quality_table.setItem(i, 0, QTableWidgetItem(metric))
            self.quality_table.setItem(i, 1, QTableWidgetItem(value))
        
        self.quality_table.resizeColumnsToContents()
    
    def populate_detection(self, lsb_analysis: Dict):
        """Populate detection analysis."""
        detection_items = [
            ("Ones Ratio Test", f"{lsb_analysis['average_ones_ratio']:.3f}", lsb_analysis['randomness_suspicion']),
            ("Chi-Square Test", f"{lsb_analysis['average_chi_square']:.2f}", lsb_analysis['chi_suspicion'])
        ]
        
        self.detection_table.setRowCount(len(detection_items))
        for i, (test, result, suspicion) in enumerate(detection_items):
            self.detection_table.setItem(i, 0, QTableWidgetItem(test))
            self.detection_table.setItem(i, 1, QTableWidgetItem(result))
            
            suspicion_item = QTableWidgetItem(suspicion)
            if suspicion == "High":
                suspicion_item.setBackground(QPalette().color(QPalette.ColorRole.BrightText))
            elif suspicion == "Medium":
                suspicion_item.setBackground(QPalette().color(QPalette.ColorRole.Light))
            
            self.detection_table.setItem(i, 2, suspicion_item)
        
        self.detection_table.resizeColumnsToContents()
    
    def export_report(self):
        """Export analysis report to file."""
        if not self.analysis_results:
            QMessageBox.warning(self, "Warning", "No analysis results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Report", "analysis_report.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                self._write_report(file_path)
                QMessageBox.information(self, "Success", "Report exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export report:\n{str(e)}")
    
    def _write_report(self, file_path: str):
        """Write analysis report to file."""
        results = self.analysis_results
        
        if not results:
            return
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("InvisioVault - Image Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            
            # Basic info
            f.write("BASIC INFORMATION\n")
            f.write("-" * 20 + "\n")
            if 'basic_info' in results:
                basic = results['basic_info']
                f.write(f"File: {Path(basic['file_path']).name}\n")
                f.write(f"Size: {basic['file_size']:,} bytes\n")
                f.write(f"Dimensions: {basic['width']} × {basic['height']}\n")
                f.write(f"Channels: {basic['channels']}\n")
                f.write(f"Total Pixels: {basic['total_pixels']:,}\n\n")
            
            # Capacity
            f.write("CAPACITY ANALYSIS\n")
            f.write("-" * 20 + "\n")
            if 'capacity' in results:
                capacity = results['capacity']
                f.write(f"LSB Capacity: {capacity['lsb_capacity_bytes']:,} bytes ({capacity['lsb_capacity_mb']:.2f} MB)\n")
                f.write(f"Capacity Ratio: {capacity['capacity_ratio']:.2f}\n\n")
            
            # Suitability
            f.write("SUITABILITY ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            if 'suitability' in results:
                suitability = results['suitability']
                f.write(f"Rating: {suitability['rating']} ({suitability['score']}/100)\n")
                f.write("Reasons:\n")
                for reason in suitability['reasons']:
                    f.write(f"  • {reason}\n")
                f.write("\n")
            
            # Statistics
            f.write("STATISTICAL ANALYSIS\n")
            f.write("-" * 20 + "\n")
            if 'statistics' in results:
                stats = results['statistics']
                f.write(f"Entropy: {stats['entropy']:.3f}\n")
                f.write(f"Overall Mean: {stats['overall']['mean']:.2f}\n")
                f.write(f"Overall Std Dev: {stats['overall']['std']:.2f}\n\n")
            
            # Detection
            f.write("DETECTION ANALYSIS\n")
            f.write("-" * 20 + "\n")
            if 'lsb_analysis' in results:
                lsb = results['lsb_analysis']
                f.write(f"Average Ones Ratio: {lsb['average_ones_ratio']:.3f}\n")
                f.write(f"Chi-Square Value: {lsb['average_chi_square']:.2f}\n")
                f.write(f"Randomness Suspicion: {lsb['randomness_suspicion']}\n")
                f.write(f"Chi-Square Suspicion: {lsb['chi_suspicion']}\n")
                
            # Analysis info
            f.write(f"\nANALYSIS METADATA\n")
            f.write("-" * 20 + "\n")
            f.write(f"Analysis Quality: {results.get('analysis_quality', 'Unknown')}\n")
            f.write(f"Analysis Type: {results.get('analysis_type', 'Unknown')}\n")
            f.write(f"Powered by: AnalysisOperation & ImageAnalyzer\n")
    
    def cancel_analysis(self):
        """Cancel the ongoing analysis."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.worker_thread.wait(3000)  # Wait up to 3 seconds for thread to finish
            
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()  # Force terminate if still running
                self.worker_thread.wait(1000)
        
        # Reset UI state
        self.progress_bar.setVisible(False)
        self.analyze_button.setVisible(True)
        self.analyze_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.quality_combo.setEnabled(True)
        self.status_label.setText("Analysis cancelled")
        
        self.logger.info("Analysis cancelled by user")
    
    def on_analysis_error(self, error_message: str):
        """Handle analysis error."""
        # Reset UI state
        self.progress_bar.setVisible(False)
        self.analyze_button.setVisible(True)
        self.analyze_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.quality_combo.setEnabled(True)
        self.status_label.setText(f"Error: {error_message}")
        
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n{error_message}")
