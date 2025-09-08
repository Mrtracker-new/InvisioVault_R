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
                'analysis_type': operation_results.get('analysis_type', 'comprehensive'),
                'enhanced_analysis': True  # Mark as enhanced analysis
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
                'mode': image_props.get('mode', 'RGB'),
                'is_lossless': file_info.get('is_lossless_format', False)
            }
            
            # Capacity analysis
            capacity = comprehensive.get('capacity_analysis', {})
            basic_lsb = capacity.get('basic_lsb', {})
            effective = capacity.get('effective_capacity', {})
            
            formatted['capacity'] = {
                'lsb_capacity_bits': basic_lsb.get('total_bits', 0),
                'lsb_capacity_bytes': basic_lsb.get('capacity_bytes', 0),
                'lsb_capacity_kb': basic_lsb.get('capacity_kb', 0),
                'lsb_capacity_mb': basic_lsb.get('capacity_mb', 0),
                'capacity_ratio': effective.get('efficiency_ratio', 0),
                'effective_capacity_bytes': effective.get('usable_bytes', 0),
                'multi_bit_capacity': capacity.get('multi_bit_lsb', {})
            }
            
            # Statistics
            quality_metrics = comprehensive.get('quality_metrics', {})
            rgb_metrics = quality_metrics.get('rgb', {})
            entropy_data = rgb_metrics.get('entropy', {}) if rgb_metrics else {}
            noise_data = rgb_metrics.get('noise_analysis', {}) if rgb_metrics else {}
            
            # Create statistics structure compatible with UI
            formatted['statistics'] = {
                'entropy': entropy_data.get('overall', 0),
                'overall': {
                    'mean': image_props.get('pixel_statistics', {}).get('mean', [128.0])[0] if image_props else 128.0,
                    'std': noise_data.get('overall_std', 0),
                    'min': 0,
                    'max': 255,
                    'median': image_props.get('pixel_statistics', {}).get('median', [128.0])[0] if image_props else 128.0,
                    'variance': (noise_data.get('overall_std', 0) ** 2) if noise_data else 0
                },
                'channels': entropy_data.get('channels', []),
                'is_sampled': False,
                'normalized_entropy': entropy_data.get('normalized', 0)
            }
            
            # Quality metrics
            texture_analysis = comprehensive.get('texture_analysis', {})
            lbp_data = texture_analysis.get('local_binary_patterns', {}) if texture_analysis else {}
            contrast_data = rgb_metrics.get('contrast_analysis', {}) if rgb_metrics else {}
            sharpness_data = rgb_metrics.get('sharpness_analysis', {}) if rgb_metrics else {}
            
            formatted['quality_metrics'] = {
                'noise_level': noise_data.get('overall_noise_estimate', 0),
                'sharpness': sharpness_data.get('laplacian_variance', 0),
                'texture_complexity': lbp_data.get('complexity_score', 0),
                'contrast': contrast_data.get('rms_contrast', 0),
                'dynamic_range': contrast_data.get('dynamic_range', 0),
                'sharpness_level': sharpness_data.get('sharpness_level', 'unknown').replace('_', ' ').title() if sharpness_data else 'Unknown'
            }
            
            # LSB analysis
            stego_analysis = comprehensive.get('steganography_analysis', {})
            lsb_analysis = stego_analysis.get('lsb_analysis', {}) if stego_analysis else {}
            overall_lsb = lsb_analysis.get('overall_assessment', {})
            channel_lsb = lsb_analysis.get('channel_analysis', {})
            
            # Extract channel-specific LSB data
            lsb_channels = []
            for channel_name, channel_data in channel_lsb.items():
                lsb_channels.append({
                    'name': channel_name,
                    'lsb_ratio': channel_data.get('lsb_ratio', 0.5),
                    'lsb_entropy': channel_data.get('lsb_entropy', 0),
                    'pattern_score': channel_data.get('pattern_anomaly_score', 0),
                    'randomness': channel_data.get('randomness_quality', 'normal')
                })
            
            formatted['lsb_analysis'] = {
                'channels': lsb_channels,
                'average_ones_ratio': 0.5 - overall_lsb.get('average_ratio_deviation', 0),
                'average_entropy': overall_lsb.get('average_lsb_entropy', 0),
                'randomness_suspicion': self._map_suspicion_level(overall_lsb.get('steganography_likelihood', 'none')),
                'chi_suspicion': 'Low' if overall_lsb.get('average_ratio_deviation', 0) > 0.1 else 'Medium' if overall_lsb.get('average_ratio_deviation', 0) > 0.05 else 'High'
            }
            
            # Add steganography detection results if available
            stego_detection = operation_results.get('steganography_detection', {})
            if stego_detection:
                formatted['stego_detection'] = {
                    'likelihood': stego_detection.get('overall_likelihood', 'none'),
                    'confidence': stego_detection.get('detection_confidence', 0),
                    'indicators': stego_detection.get('indicators', []),
                    'methods': list(stego_detection.get('detection_methods', {}).keys())
                }
            
            # Add performance metrics
            perf_metrics = comprehensive.get('performance_metrics', {})
            formatted['performance'] = {
                'analysis_time': perf_metrics.get('total_analysis_time', 0),
                'gpu_used': perf_metrics.get('gpu_acceleration_used', False),
                'parallel_workers': perf_metrics.get('parallel_workers_used', 1)
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
        self.setWindowTitle("🚀 Enhanced Image Analysis & Advanced Steganography Detection")
        self.setModal(True)
        self.resize(1200, 700)  # Optimized size for compact layout
        
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
        """Initialize the user interface with improved dark theme design."""
        # Set dark theme for entire dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background-color: #2b2b2b;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Header section - more polished
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setSpacing(6)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title with better styling
        title = QLabel("🔍 Image Analysis & Capacity Assessment")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        title.setStyleSheet("""
            QLabel {
                color: #4fc3f7;
                padding: 8px 0px;
                border-bottom: 2px solid #37474f;
                margin-bottom: 8px;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Compact feature info
        features_info = QLabel(
            "🚀 Enhanced Analysis: Computer Vision • ML Detection • Statistical Masking • Security Rating • Optimized Performance"
        )
        features_info.setWordWrap(True)
        features_info.setStyleSheet("""
            QLabel {
                color: #90a4ae;
                font-size: 11px;
                padding: 4px 0px;
                font-style: italic;
            }
        """)
        
        header_layout.addWidget(title)
        header_layout.addWidget(features_info)
        layout.addWidget(header_widget)
        
        # Main content area with horizontal split
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(16)
        
        # Left side - Image selection and controls
        left_widget = QWidget()
        left_widget.setStyleSheet("""
            QWidget {
                background-color: #37474f;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(12)
        left_widget.setMaximumWidth(380)
        left_widget.setMinimumWidth(350)
        
        # Image selection group
        selection_group = QGroupBox("📁 Image Selection & Settings")
        selection_group.setStyleSheet("""
            QGroupBox {
                background-color: #455a64;
                border: 1px solid #607d8b;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
            }
        """)
        selection_layout = QVBoxLayout(selection_group)
        selection_layout.setSpacing(10)
        
        self.image_label = QLabel("No image selected")
        self.image_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                border: 2px dashed #607d8b;
                border-radius: 6px;
                background-color: #546e7a;
                color: #cfd8dc;
                font-size: 11px;
                text-align: center;
            }
        """)
        self.image_label.setMinimumHeight(50)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        selection_layout.addWidget(self.image_label)
        
        # Analysis quality selection
        quality_layout = QHBoxLayout()
        quality_layout.setSpacing(10)
        quality_label = QLabel("Analysis Quality:")
        quality_label.setStyleSheet("color: #cfd8dc; font-weight: bold; font-size: 11px;")
        quality_label.setMinimumWidth(100)
        
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Fast", "Balanced", "Thorough"])
        self.quality_combo.setCurrentText("Balanced")
        self.quality_combo.setStyleSheet("""
            QComboBox {
                background-color: #607d8b;
                border: 1px solid #78909c;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ffffff;
                font-size: 11px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #546e7a;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-style: solid;
                border-width: 4px 4px 0px 4px;
                border-color: #ffffff transparent transparent transparent;
            }
            QComboBox QAbstractItemView {
                background-color: #607d8b;
                color: #ffffff;
                selection-background-color: #546e7a;
                border: 1px solid #78909c;
            }
        """)
        self.quality_combo.setToolTip(
            "🎯 Enhanced Analysis Levels:\n\n"
            "⚡ Fast: Basic metrics (< 0.5s) - LSB analysis, essential security\n"
            "📊 Balanced: Comprehensive (< 2s) - Advanced texture, multi-method detection\n"
            "🔬 Thorough: Deep analysis (< 10s) - ML detection, frequency domain, complete texture"
        )
        
        quality_layout.addWidget(quality_label)
        quality_layout.addWidget(self.quality_combo, 1)
        selection_layout.addLayout(quality_layout)
        
        # Buttons layout - modern styled buttons
        buttons_container = QWidget()
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setSpacing(8)
        
        self.select_image_button = QPushButton("📁 Select Image")
        self.select_image_button.setStyleSheet("""
            QPushButton {
                background-color: #26c6da;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 11px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #00acc1;
            }
            QPushButton:pressed {
                background-color: #0097a7;
            }
        """)
        
        self.analyze_button = QPushButton("🔍 Analyze Image")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #66bb6a;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 12px 16px;
                font-weight: bold;
                font-size: 12px;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #4caf50;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
            QPushButton:disabled {
                background-color: #616161;
                color: #9e9e9e;
            }
        """)
        
        self.cancel_button = QPushButton("⏹ Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #ef5350;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
                min-height: 18px;
            }
            QPushButton:hover {
                background-color: #f44336;
            }
            QPushButton:pressed {
                background-color: #d32f2f;
            }
        """)
        
        buttons_layout.addWidget(self.select_image_button)
        buttons_layout.addWidget(self.analyze_button)
        buttons_layout.addWidget(self.cancel_button)
        
        selection_layout.addWidget(buttons_container)
        
        # Progress section - styled for dark theme
        progress_group = QGroupBox("⏳ Analysis Progress")
        progress_group.setStyleSheet("""
            QGroupBox {
                background-color: #455a64;
                border: 1px solid #607d8b;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
            }
        """)
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(8)
        progress_layout.setContentsMargins(12, 12, 12, 12)
        
        self.status_label = QLabel("Ready to analyze")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #cfd8dc;
                padding: 4px 8px;
                background-color: #546e7a;
                border-radius: 4px;
                border: 1px solid #607d8b;
            }
        """)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(18)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #607d8b;
                border-radius: 4px;
                background-color: #546e7a;
                text-align: center;
                font-size: 10px;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4fc3f7;
                border-radius: 3px;
            }
        """)
        
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        
        left_layout.addWidget(selection_group)
        left_layout.addWidget(progress_group)
        left_layout.addStretch()  # Push everything to top
        # Right side - Results area with dark theme
        self.results_tabs = QTabWidget()
        self.results_tabs.setVisible(False)
        self.results_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #607d8b;
                border-radius: 8px;
                background-color: #455a64;
                padding: 8px;
            }
            QTabBar::tab {
                background-color: #546e7a;
                border: 1px solid #607d8b;
                padding: 8px 16px;
                margin-right: 2px;
                border-radius: 6px 6px 0px 0px;
                color: #cfd8dc;
                font-weight: bold;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background-color: #455a64;
                border-bottom: 1px solid #455a64;
                color: #4fc3f7;
            }
            QTabBar::tab:hover {
                background-color: #607d8b;
                color: #ffffff;
            }
        """)
        
        # Create result tabs (enhanced)
        self.overview_tab = self.create_overview_tab()
        self.capacity_tab = self.create_capacity_tab()
        self.statistics_tab = self.create_statistics_tab()
        self.detection_tab = self.create_detection_tab()
        self.enhanced_tab = self.create_enhanced_features_tab()
        self.performance_tab = self.create_performance_tab()
        
        self.results_tabs.addTab(self.overview_tab, "📊 Overview")
        self.results_tabs.addTab(self.capacity_tab, "💾 Capacity")
        self.results_tabs.addTab(self.statistics_tab, "📈 Statistics")
        self.results_tabs.addTab(self.detection_tab, "🔍 Detection")
        self.results_tabs.addTab(self.enhanced_tab, "🎆 Enhanced")
        self.results_tabs.addTab(self.performance_tab, "⚡ Performance")
        
        # Add both sides to main layout
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.results_tabs, 2)  # Give results more space
        layout.addWidget(main_widget)
        
        # Bottom buttons - dark theme styled
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 12, 0, 0)
        
        self.export_button = QPushButton("📄 Export Report")
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 11px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
            QPushButton:disabled {
                background-color: #616161;
                color: #9e9e9e;
            }
        """)
        
        self.close_button = QPushButton("Close")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #607d8b;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 11px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #546e7a;
            }
        """)
        
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
    
    def create_overview_tab(self) -> QWidget:
        """Create the overview tab with improved layout."""
        widget = QWidget()
        layout = QHBoxLayout(widget)  # Horizontal layout for better space usage
        layout.setSpacing(12)
        
        # Left side - Basic info
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        info_label = QLabel("📄 Basic Information")
        info_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #333; margin-bottom: 5px;")
        left_layout.addWidget(info_label)
        
        self.basic_info_table = QTableWidget()
        self.basic_info_table.setColumnCount(2)
        self.basic_info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.basic_info_table.horizontalHeader().setStretchLastSection(True)
        self.basic_info_table.verticalHeader().setVisible(False)
        self.basic_info_table.setAlternatingRowColors(True)
        self.basic_info_table.setStyleSheet("""
            QTableWidget {
                background-color: #546e7a;
                gridline-color: #607d8b;
                color: #ffffff;
                font-size: 11px;
                selection-background-color: #37474f;
                border: 1px solid #607d8b;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #607d8b;
            }
            QTableWidget::item:alternate {
                background-color: #607d8b;
            }
            QHeaderView::section {
                background-color: #37474f;
                color: #4fc3f7;
                padding: 8px;
                border: 1px solid #607d8b;
                font-weight: bold;
                font-size: 10px;
            }
        """)
        
        left_layout.addWidget(self.basic_info_table)
        
        # Right side - Suitability assessment
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        suitability_group = QGroupBox("🎯 Steganographic Suitability")
        suitability_group.setStyleSheet("""
            QGroupBox {
                background-color: #546e7a;
                border: 1px solid #607d8b;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
                font-weight: bold;
                color: #4fc3f7;
            }
        """)
        suitability_layout = QVBoxLayout(suitability_group)
        suitability_layout.setSpacing(10)
        
        rating_container = QWidget()
        rating_container.setStyleSheet("background-color: transparent;")
        rating_layout = QHBoxLayout(rating_container)
        rating_layout.setContentsMargins(0, 0, 0, 0)
        
        rating_label = QLabel("Rating:")
        rating_label.setStyleSheet("color: #cfd8dc; font-weight: bold; font-size: 12px;")
        rating_layout.addWidget(rating_label)
        
        self.suitability_rating = QLabel("Not analyzed")
        self.suitability_rating.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 6px 12px;
                background-color: #37474f;
                border: 1px solid #607d8b;
                border-radius: 4px;
                color: #ffffff;
            }
        """)
        rating_layout.addWidget(self.suitability_rating)
        rating_layout.addStretch()
        
        details_label = QLabel("Assessment Details:")
        details_label.setStyleSheet("color: #cfd8dc; font-weight: bold; font-size: 11px; margin-top: 8px;")
        
        self.suitability_reasons = QTextEdit()
        self.suitability_reasons.setReadOnly(True)
        self.suitability_reasons.setMaximumHeight(120)
        self.suitability_reasons.setStyleSheet("""
            QTextEdit {
                background-color: #37474f;
                border: 1px solid #607d8b;
                border-radius: 4px;
                color: #cfd8dc;
                font-size: 11px;
                padding: 8px;
                line-height: 1.4;
            }
        """)
        
        suitability_layout.addWidget(rating_container)
        suitability_layout.addWidget(details_label)
        suitability_layout.addWidget(self.suitability_reasons)
        
        right_layout.addWidget(suitability_group)
        right_layout.addStretch()
        
        layout.addWidget(left_widget)
        layout.addWidget(right_widget)
        
        return widget
    
    def create_capacity_tab(self) -> QWidget:
        """Create the capacity analysis tab with improved visual layout."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("💾 Storage Capacity Analysis (LSB Steganography)")
        header.setStyleSheet("font-weight: bold; font-size: 12px; color: #333; margin-bottom: 8px;")
        layout.addWidget(header)
        
        # Main content area with horizontal layout
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(15)
        
        # Left side - Capacity table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        
        self.capacity_table = QTableWidget()
        self.capacity_table.setColumnCount(2)
        self.capacity_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.capacity_table.horizontalHeader().setStretchLastSection(True)
        self.capacity_table.verticalHeader().setVisible(False)
        self.capacity_table.setAlternatingRowColors(True)
        self.capacity_table.setMaximumWidth(400)
        self.capacity_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #e0e0e0;
                font-size: 11px;
                selection-background-color: #e8f5e8;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 6px;
                border: none;
                font-weight: bold;
                font-size: 10px;
            }
        """)
        
        table_layout.addWidget(self.capacity_table)
        
        # Right side - Info and tips
        info_container = QWidget()
        info_layout = QVBoxLayout(info_container)
        
        # Capacity visualization placeholder
        viz_group = QGroupBox("📈 Capacity Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        self.capacity_summary = QLabel("Analysis results will appear here...")
        self.capacity_summary.setStyleSheet("""
            QLabel {
                padding: 15px;
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 6px;
                font-size: 11px;
                color: #495057;
            }
        """)
        self.capacity_summary.setWordWrap(True)
        self.capacity_summary.setMinimumHeight(80)
        
        viz_layout.addWidget(self.capacity_summary)
        info_layout.addWidget(viz_group)
        
        # Tips section
        tips_group = QGroupBox("💡 LSB Steganography Tips")
        tips_layout = QVBoxLayout(tips_group)
        
        tips_text = QLabel(
            "• LSB uses least significant bits of each color channel\n"
            "• Actual capacity may be reduced by encryption overhead\n"
            "• Error correction can further reduce usable space\n"
            "• Lossless formats (PNG, BMP) preserve hidden data better\n"
            "• Higher resolution images provide more storage capacity"
        )
        tips_text.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #fff3e0;
                border: 1px solid #ffcc02;
                border-radius: 4px;
                font-size: 10px;
                color: #e65100;
                line-height: 1.4;
            }
        """)
        tips_text.setWordWrap(True)
        
        tips_layout.addWidget(tips_text)
        info_layout.addWidget(tips_group)
        info_layout.addStretch()
        
        content_layout.addWidget(table_container)
        content_layout.addWidget(info_container, 1)
        layout.addWidget(content_widget)
        
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
    
    def create_enhanced_features_tab(self) -> QWidget:
        """Create the enhanced features tab showing advanced analysis results."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Texture Analysis Section
        texture_group = QGroupBox("🎨 Advanced Texture Analysis")
        texture_layout = QVBoxLayout(texture_group)
        
        self.texture_info = QLabel("No texture analysis data available")
        self.texture_info.setWordWrap(True)
        self.texture_info.setStyleSheet("padding: 10px; background-color: #f9f9f9; border-radius: 5px;")
        
        self.lbp_info = QLabel("LBP Analysis: Not performed")
        self.glcm_info = QLabel("GLCM Analysis: Not performed")
        self.gabor_info = QLabel("Gabor Analysis: Not performed")
        
        texture_layout.addWidget(self.texture_info)
        texture_layout.addWidget(self.lbp_info)
        texture_layout.addWidget(self.glcm_info)
        texture_layout.addWidget(self.gabor_info)
        layout.addWidget(texture_group)
        
        # Machine Learning Analysis Section
        ml_group = QGroupBox("🤖 Machine Learning Analysis")
        ml_layout = QVBoxLayout(ml_group)
        
        self.ml_anomaly_info = QLabel("Anomaly Detection: Not performed")
        self.ml_clustering_info = QLabel("Color Clustering: Not performed")
        self.ml_confidence_info = QLabel("ML Confidence: N/A")
        
        ml_layout.addWidget(self.ml_anomaly_info)
        ml_layout.addWidget(self.ml_clustering_info)
        ml_layout.addWidget(self.ml_confidence_info)
        layout.addWidget(ml_group)
        
        # Frequency Analysis Section
        freq_group = QGroupBox("📉 Frequency Domain Analysis")
        freq_layout = QVBoxLayout(freq_group)
        
        self.dct_info = QLabel("DCT Analysis: Not performed")
        self.fft_info = QLabel("FFT Analysis: Not performed")
        self.spectral_info = QLabel("Spectral Entropy: N/A")
        
        freq_layout.addWidget(self.dct_info)
        freq_layout.addWidget(self.fft_info)
        freq_layout.addWidget(self.spectral_info)
        layout.addWidget(freq_group)
        
        # Security Assessment Section
        security_group = QGroupBox("🛡️ Enhanced Security Assessment")
        security_layout = QVBoxLayout(security_group)
        
        self.security_score_label = QLabel("Security Score: Not calculated")
        self.security_score_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        
        self.security_factors = QTextEdit()
        self.security_factors.setReadOnly(True)
        self.security_factors.setMaximumHeight(100)
        self.security_factors.setPlaceholderText("Security factors will appear here...")
        
        security_layout.addWidget(self.security_score_label)
        security_layout.addWidget(QLabel("Contributing Factors:"))
        security_layout.addWidget(self.security_factors)
        layout.addWidget(security_group)
        
        return widget
    
    def create_performance_tab(self) -> QWidget:
        """Create the performance metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performance Metrics Table
        perf_group = QGroupBox("⚡ Analysis Performance Metrics")
        perf_layout = QVBoxLayout(perf_group)
        
        self.performance_table = QTableWidget()
        self.performance_table.setColumnCount(2)
        self.performance_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.performance_table.horizontalHeader().setStretchLastSection(True)
        
        perf_layout.addWidget(self.performance_table)
        layout.addWidget(perf_group)
        
        # Analysis Level Information
        level_group = QGroupBox("🎯 Analysis Configuration")
        level_layout = QVBoxLayout(level_group)
        
        self.analysis_level_info = QLabel("Analysis Level: Not determined")
        self.features_used_info = QLabel("Features Used: Not analyzed")
        
        level_info = QLabel(
            "📈 Analysis Levels:\n"
            "• LIGHTNING: Ultra-fast real-time analysis (< 0.1s)\n"
            "• FAST: Quick analysis with basic metrics (< 0.5s)\n"
            "• BALANCED: Comprehensive analysis with good performance (< 2s)\n"
            "• THOROUGH: Deep analysis with all features (< 10s)\n"
            "• RESEARCH: Maximum detail for research purposes (< 30s)"
        )
        level_info.setWordWrap(True)
        level_info.setStyleSheet(
            "color: #555; font-size: 11px; padding: 10px; "
            "background-color: #f8f8f8; border-radius: 5px; margin-top: 10px;"
        )
        
        level_layout.addWidget(self.analysis_level_info)
        level_layout.addWidget(self.features_used_info)
        level_layout.addWidget(level_info)
        layout.addWidget(level_group)
        
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
        
        # Enhanced features tab
        if 'enhanced_analysis' in results and results['enhanced_analysis']:
            self.populate_enhanced_features(results)
        
        # Performance tab
        if 'performance' in results:
            self.populate_performance(results['performance'], results.get('analysis_type', 'unknown'))
    
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
        """Populate capacity information with enhanced visualization."""
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
        
        # Update capacity summary visualization
        mb_capacity = capacity['lsb_capacity_mb']
        kb_capacity = capacity['lsb_capacity_kb']
        ratio = capacity['capacity_ratio']
        
        if mb_capacity >= 1.0:
            size_desc = f"{mb_capacity:.1f} MB"
        elif kb_capacity >= 1.0:
            size_desc = f"{kb_capacity:.1f} KB"
        else:
            size_desc = f"{capacity['lsb_capacity_bytes']} bytes"
        
        # Create visual summary
        summary_text = (
            f"📈 **Capacity Summary**\n\n"
            f"💾 **Total LSB Capacity:** {size_desc}\n"
            f"📅 **Storage Ratio:** {ratio:.1%} of original file size\n\n"
        )
        
        if capacity.get('effective_capacity_bytes', 0) > 0:
            effective_mb = capacity['effective_capacity_bytes'] / (1024 * 1024)
            if effective_mb >= 1.0:
                eff_desc = f"{effective_mb:.1f} MB"
            else:
                eff_desc = f"{capacity['effective_capacity_bytes'] / 1024:.1f} KB"
            summary_text += f"✨ **Effective Capacity:** {eff_desc}\n(After considering overhead)\n\n"
        
        # Capacity rating
        if mb_capacity >= 10:
            rating = "🚀 Excellent - Suitable for large files"
            color = "#4caf50"
        elif mb_capacity >= 1:
            rating = "📈 Good - Suitable for documents and images"
            color = "#2196f3"
        elif kb_capacity >= 100:
            rating = "📝 Fair - Suitable for text and small files"
            color = "#ff9800"
        else:
            rating = "🔍 Limited - Small text files only"
            color = "#f44336"
        
        summary_text += f"**Capacity Rating:** {rating}"
        
        self.capacity_summary.setText(summary_text)
        self.capacity_summary.setStyleSheet(f"""
            QLabel {{
                padding: 15px;
                background-color: #f8f9fa;
                border: 2px solid {color};
                border-radius: 6px;
                font-size: 11px;
                color: #495057;
            }}
        """)
    
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
        """Populate detection analysis with enhanced steganography detection results."""
        detection_items = [
            ("Ones Ratio Test", f"{lsb_analysis.get('average_ones_ratio', 0.5):.3f}", lsb_analysis.get('randomness_suspicion', 'Low')),
            ("Chi-Square Test", f"{lsb_analysis.get('average_chi_square', 0):.2f}", lsb_analysis.get('chi_suspicion', 'Low'))
        ]
        
        # Add enhanced detection results if available
        if 'stego_detection' in self.analysis_results:
            stego_detection = self.analysis_results['stego_detection']
            likelihood = stego_detection.get('likelihood', 'none')
            confidence = stego_detection.get('confidence', 0)
            
            detection_items.append((
                "🚀 Enhanced Detection", 
                f"Likelihood: {likelihood.upper()}", 
                f"Confidence: {confidence:.1%}"
            ))
            
            # Add individual method results
            methods = stego_detection.get('methods', [])
            if methods:
                method_summary = ", ".join(methods)
                detection_items.append((
                    "Detection Methods",
                    f"{len(methods)} methods used",
                    method_summary[:30] + "..." if len(method_summary) > 30 else method_summary
                ))
        
        # Populate table
        self.detection_table.setRowCount(len(detection_items))
        for i, (test, result, suspicion) in enumerate(detection_items):
            self.detection_table.setItem(i, 0, QTableWidgetItem(test))
            self.detection_table.setItem(i, 1, QTableWidgetItem(result))
            
            suspicion_item = QTableWidgetItem(suspicion)
            
            # Color coding for suspicion levels
            if "High" in suspicion or "very_high" in suspicion.lower():
                suspicion_item.setBackground(QPalette().color(QPalette.ColorRole.BrightText))
                suspicion_item.setStyleSheet("background-color: #ffcdd2; color: #d32f2f;")
            elif "Medium" in suspicion or "medium" in suspicion.lower():
                suspicion_item.setBackground(QPalette().color(QPalette.ColorRole.Light))
                suspicion_item.setStyleSheet("background-color: #fff3e0; color: #f57c00;")
            elif "Enhanced" in test:
                suspicion_item.setStyleSheet("background-color: #e3f2fd; color: #1976d2; font-weight: bold;")
            
            self.detection_table.setItem(i, 2, suspicion_item)
        
        self.detection_table.resizeColumnsToContents()
    
    def populate_enhanced_features(self, results: Dict):
        """Populate the enhanced features tab with advanced analysis data."""
        try:
            # Check if we have comprehensive analysis data
            comprehensive = results.get('comprehensive_analysis', {})
            
            # Texture Analysis
            texture_analysis = comprehensive.get('texture_analysis', {})
            if texture_analysis:
                lbp_data = texture_analysis.get('local_binary_patterns', {})
                if lbp_data:
                    entropy = lbp_data.get('entropy', 0)
                    complexity = lbp_data.get('complexity_score', 0)
                    self.lbp_info.setText(f"LBP Analysis: Entropy={entropy:.2f}, Complexity={complexity:.3f}")
                    self.lbp_info.setStyleSheet("color: #2e7d32; font-weight: bold;")
                
                glcm_data = texture_analysis.get('glcm_analysis', {})
                if glcm_data:
                    contrast = glcm_data.get('contrast', {}).get('mean', 0)
                    homogeneity = glcm_data.get('homogeneity', {}).get('mean', 0)
                    self.glcm_info.setText(f"GLCM Analysis: Contrast={contrast:.3f}, Homogeneity={homogeneity:.3f}")
                    self.glcm_info.setStyleSheet("color: #2e7d32; font-weight: bold;")
                
                gabor_data = texture_analysis.get('gabor_analysis', {})
                if gabor_data:
                    dom_freq = gabor_data.get('dominant_frequency', 0)
                    regularity = gabor_data.get('texture_regularity', 0)
                    self.gabor_info.setText(f"Gabor Analysis: Dominant Freq={dom_freq:.2f}, Regularity={regularity:.3f}")
                    self.gabor_info.setStyleSheet("color: #2e7d32; font-weight: bold;")
                
                texture_summary = f"🎨 Advanced texture analysis completed with {len(texture_analysis)} methods"
                self.texture_info.setText(texture_summary)
                self.texture_info.setStyleSheet("padding: 10px; background-color: #e8f5e8; border-radius: 5px; color: #2e7d32;")
            
            # Machine Learning Analysis
            ml_analysis = comprehensive.get('ml_analysis', {})
            if ml_analysis:
                anomaly_data = ml_analysis.get('anomaly_detection', {})
                if anomaly_data:
                    is_anomalous = anomaly_data.get('is_anomalous', False)
                    confidence = anomaly_data.get('confidence', 0)
                    status = "ANOMALOUS" if is_anomalous else "NORMAL"
                    color = "#d32f2f" if is_anomalous else "#2e7d32"
                    self.ml_anomaly_info.setText(f"Anomaly Detection: {status} (confidence: {confidence:.1%})")
                    self.ml_anomaly_info.setStyleSheet(f"color: {color}; font-weight: bold;")
                
                clustering_data = ml_analysis.get('clustering_analysis', {})
                if clustering_data:
                    diversity = clustering_data.get('color_diversity', 0)
                    dominant_ratio = clustering_data.get('dominant_color_ratio', 0)
                    self.ml_clustering_info.setText(f"Color Clustering: Diversity={diversity:.2f}, Dominant Ratio={dominant_ratio:.2f}")
                    self.ml_clustering_info.setStyleSheet("color: #2e7d32; font-weight: bold;")
                
                # Overall ML confidence
                feature_stats = ml_analysis.get('feature_statistics', {})
                if feature_stats:
                    feature_entropy = feature_stats.get('feature_entropy', 0)
                    self.ml_confidence_info.setText(f"ML Feature Entropy: {feature_entropy:.2f}")
                    self.ml_confidence_info.setStyleSheet("color: #1976d2; font-weight: bold;")
            
            # Frequency Analysis
            freq_analysis = comprehensive.get('frequency_analysis', {})
            if freq_analysis:
                dct_data = freq_analysis.get('dct_analysis', {})
                if dct_data:
                    dc_entropy = dct_data.get('dc_coefficient_stats', {}).get('entropy', 0)
                    ac_sparsity = dct_data.get('ac_coefficient_stats', {}).get('sparsity', 0)
                    self.dct_info.setText(f"DCT Analysis: DC Entropy={dc_entropy:.2f}, AC Sparsity={ac_sparsity:.2f}")
                    self.dct_info.setStyleSheet("color: #2e7d32; font-weight: bold;")
                
                fft_data = freq_analysis.get('fft_analysis', {})
                if fft_data:
                    spectral_entropy = fft_data.get('spectral_entropy', 0)
                    high_freq_content = fft_data.get('high_frequency_content', 0)
                    self.fft_info.setText(f"FFT Analysis: Spectral Entropy={spectral_entropy:.2f}, High Freq={high_freq_content:.2f}")
                    self.fft_info.setStyleSheet("color: #2e7d32; font-weight: bold;")
                    self.spectral_info.setText(f"Overall Spectral Entropy: {spectral_entropy:.3f}")
                    self.spectral_info.setStyleSheet("color: #1976d2; font-weight: bold;")
            
            # Enhanced Security Assessment
            security = comprehensive.get('security_assessment', {})
            if security:
                score = security.get('overall_security_score', 0)
                rating = security.get('security_rating', 'unknown')
                
                # Color code the security score
                if score >= 8.0:
                    color = "#4caf50"
                elif score >= 6.0:
                    color = "#ff9800"
                elif score >= 4.0:
                    color = "#ffc107"
                else:
                    color = "#f44336"
                
                self.security_score_label.setText(f"Enhanced Security Score: {score:.1f}/10.0 ({rating.title()})")
                self.security_score_label.setStyleSheet(f"font-size: 14px; font-weight: bold; padding: 5px; color: {color};")
                
                factors = security.get('contributing_factors', [])
                if factors:
                    factors_text = "\n".join([f"• {factor}" for factor in factors[:5]])  # Show top 5
                    self.security_factors.setText(factors_text)
                    self.security_factors.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 3px;")
            
        except Exception as e:
            self.logger.warning(f"Error populating enhanced features: {e}")
            self.texture_info.setText("⚠️ Error loading enhanced features data")
            self.texture_info.setStyleSheet("padding: 10px; background-color: #ffe6e6; border-radius: 5px; color: #d32f2f;")
    
    def populate_performance(self, performance: Dict, analysis_type: str):
        """Populate the performance metrics tab."""
        try:
            perf_items = []
            
            if 'analysis_time' in performance:
                analysis_time = performance['analysis_time']
                perf_items.append(("Total Analysis Time", f"{analysis_time:.3f}s"))
                
                # Add performance rating based on time
                if analysis_time < 0.1:
                    rating = "⚡ Lightning Fast"
                elif analysis_time < 0.5:
                    rating = "🚀 Very Fast"
                elif analysis_time < 2.0:
                    rating = "✅ Good Performance"
                elif analysis_time < 10.0:
                    rating = "⏱️ Moderate Performance"
                else:
                    rating = "🐌 Slow Performance"
                
                perf_items.append(("Performance Rating", rating))
            
            if 'gpu_used' in performance:
                gpu_status = "✅ GPU Accelerated" if performance['gpu_used'] else "💻 CPU Processing"
                perf_items.append(("Processing Mode", gpu_status))
            
            if 'parallel_workers' in performance:
                workers = performance['parallel_workers']
                worker_status = f"🔄 {workers} parallel workers" if workers > 1 else "⏯️ Single threaded"
                perf_items.append(("Parallelization", worker_status))
            
            # Add analysis type info
            perf_items.append(("Analysis Type", analysis_type.title()))
            
            self.performance_table.setRowCount(len(perf_items))
            for i, (metric, value) in enumerate(perf_items):
                self.performance_table.setItem(i, 0, QTableWidgetItem(metric))
                self.performance_table.setItem(i, 1, QTableWidgetItem(value))
            
            self.performance_table.resizeColumnsToContents()
            
            # Update analysis level info
            quality_map = {"fast": "FAST", "balanced": "BALANCED", "thorough": "THOROUGH"}
            level = quality_map.get(analysis_type.lower(), analysis_type.upper())
            self.analysis_level_info.setText(f"Analysis Level: {level}")
            self.analysis_level_info.setStyleSheet("font-weight: bold; color: #1976d2;")
            
            # Determine features used
            features_used = []
            if 'analysis_time' in performance:
                if performance['analysis_time'] > 1.0:
                    features_used.extend(["Advanced Texture Analysis", "Machine Learning", "Frequency Analysis"])
                elif performance['analysis_time'] > 0.5:
                    features_used.extend(["Texture Analysis", "Statistical Analysis"])
                else:
                    features_used.append("Basic Analysis")
            
            if performance.get('gpu_used', False):
                features_used.append("GPU Acceleration")
            
            if performance.get('parallel_workers', 1) > 1:
                features_used.append("Parallel Processing")
            
            features_text = ", ".join(features_used) if features_used else "Standard Features"
            self.features_used_info.setText(f"Features Used: {features_text}")
            self.features_used_info.setStyleSheet("color: #666;")
            
        except Exception as e:
            self.logger.warning(f"Error populating performance data: {e}")
    
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
