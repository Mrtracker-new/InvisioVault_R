"""
Multimedia Analysis Dialog
Professional dialog for analyzing multimedia files for steganography capacity and suitability.
"""

from pathlib import Path
from typing import List, Dict, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QWidget, QProgressBar, QMessageBox, QSplitter,
    QListWidget, QListWidgetItem, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPixmap

from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler
from core.multimedia_analyzer import MultimediaAnalyzer
from ui.components.file_drop_zone import FileDropZone


class MultimediaBatchAnalysisWorker(QThread):
    """Worker thread for batch analysis of multimedia files."""
    progress_updated = Signal(int)
    file_analyzed = Signal(str, dict)  # filename, analysis results
    analysis_completed = Signal()
    analysis_failed = Signal(str)
    
    def __init__(self, file_paths, analyzer):
        super().__init__()
        self.file_paths = [Path(p) for p in file_paths]
        self.analyzer = analyzer
    
    def run(self):
        """Analyze all files in background thread."""
        try:
            total_files = len(self.file_paths)
            
            for i, file_path in enumerate(self.file_paths):
                try:
                    if self.analyzer.is_video_file(file_path):
                        analysis = self.analyzer.analyze_video_file(file_path)
                        analysis['media_type'] = 'video'
                    elif self.analyzer.is_audio_file(file_path):
                        analysis = self.analyzer.analyze_audio_file(file_path)
                        analysis['media_type'] = 'audio'
                    else:
                        analysis = {'error': 'Unsupported file format'}
                    
                    analysis['filename'] = file_path.name
                    self.file_analyzed.emit(file_path.name, analysis)
                    
                except Exception as e:
                    analysis = {'error': str(e), 'filename': file_path.name}
                    self.file_analyzed.emit(file_path.name, analysis)
                
                # Update progress
                progress = int(((i + 1) / total_files) * 100)
                self.progress_updated.emit(progress)
            
            self.analysis_completed.emit()
            
        except Exception as e:
            self.analysis_failed.emit(str(e))


class MultimediaAnalysisDialog(QDialog):
    """Professional dialog for multimedia analysis."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        self.analyzer = MultimediaAnalyzer()
        
        # Dialog state
        self.multimedia_files = []
        self.analysis_results = {}
        self.analysis_worker = None
        
        # Initialize UI
        self.init_ui()
        self.setup_connections()
        
        self.logger.info("Multimedia analysis dialog initialized")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Multimedia Analysis")
        self.setMinimumSize(900, 700)
        self.resize(1200, 800)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Multimedia Steganography Analysis")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(16)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "Analyze video and audio files for steganography capacity, "
            "quality metrics, and suitability recommendations."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - File selection and list
        left_panel = self.create_file_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Analysis results
        right_panel = self.create_analysis_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 800])
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("🔍 Analyze Files")
        self.analyze_button.setMinimumHeight(40)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.export_button = QPushButton("📊 Export Results")
        self.export_button.setMinimumHeight(40)
        self.export_button.setEnabled(False)
        
        clear_button = QPushButton("Clear All")
        clear_button.setMinimumHeight(40)
        
        close_button = QPushButton("Close")
        close_button.setMinimumHeight(40)
        
        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        button_layout.addWidget(clear_button)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # Connect button signals
        self.analyze_button.clicked.connect(self.start_analysis)
        self.export_button.clicked.connect(self.export_results)
        clear_button.clicked.connect(self.clear_files)
        close_button.clicked.connect(self.accept)
    
    def create_file_panel(self) -> QWidget:
        """Create the file selection and list panel."""
        panel = QGroupBox("Multimedia Files")
        layout = QVBoxLayout(panel)
        
        # Drop zone
        self.file_drop_zone = FileDropZone(
            "Drop multimedia files here\n(Video: MP4, AVI, MKV, MOV)\n(Audio: MP3, WAV, FLAC, AAC)"
        )
        self.file_drop_zone.setMinimumHeight(120)
        self.file_drop_zone.files_dropped.connect(self.on_files_dropped)
        layout.addWidget(self.file_drop_zone)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(200)
        self.file_list.currentItemChanged.connect(self.on_file_selection_changed)
        layout.addWidget(self.file_list)
        
        # File info
        self.file_count_label = QLabel("No files selected")
        self.file_count_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.file_count_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        browse_button = QPushButton("Browse Files...")
        browse_button.clicked.connect(self.browse_files)
        
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected_file)
        
        button_layout.addWidget(browse_button)
        button_layout.addWidget(remove_button)
        layout.addLayout(button_layout)
        
        return panel
    
    def create_analysis_panel(self) -> QWidget:
        """Create the analysis results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different views
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Summary tab
        summary_tab = self.create_summary_tab()
        tab_widget.addTab(summary_tab, "📋 Summary")
        
        # Details tab
        details_tab = self.create_details_tab()
        tab_widget.addTab(details_tab, "📊 Details")
        
        # Comparison tab
        comparison_tab = self.create_comparison_tab()
        tab_widget.addTab(comparison_tab, "⚖️ Comparison")
        
        return panel
    
    def create_summary_tab(self) -> QWidget:
        """Create the summary analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Overall statistics
        stats_group = QGroupBox("Overall Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setPlainText("Analyze files to see overall statistics...")
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        
        # Best candidates
        candidates_group = QGroupBox("Best Candidates for Steganography")
        candidates_layout = QVBoxLayout(candidates_group)
        
        self.candidates_text = QTextEdit()
        self.candidates_text.setReadOnly(True)
        self.candidates_text.setMaximumHeight(150)
        self.candidates_text.setPlainText("Analysis results will show the best files for steganography...")
        candidates_layout.addWidget(self.candidates_text)
        
        layout.addWidget(candidates_group)
        
        # Recommendations
        recommendations_group = QGroupBox("General Recommendations")
        recommendations_layout = QVBoxLayout(recommendations_group)
        
        self.general_recommendations_text = QTextEdit()
        self.general_recommendations_text.setReadOnly(True)
        self.general_recommendations_text.setPlainText("General recommendations will appear here...")
        recommendations_layout.addWidget(self.general_recommendations_text)
        
        layout.addWidget(recommendations_group)
        
        return widget
    
    def create_details_tab(self) -> QWidget:
        """Create the detailed analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Details table
        self.details_table = QTableWidget()
        self.details_table.setAlternatingRowColors(True)
        self.details_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        # Set up table headers
        headers = [
            "Filename", "Type", "Format", "Size", "Duration", 
            "Capacity", "Suitability", "Recommendations"
        ]
        self.details_table.setColumnCount(len(headers))
        self.details_table.setHorizontalHeaderLabels(headers)
        
        # Configure table
        header = self.details_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Filename
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Type
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Format
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Size
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Duration
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # Capacity
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)  # Suitability
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)  # Recommendations
        
        layout.addWidget(self.details_table)
        
        return widget
    
    def create_comparison_tab(self) -> QWidget:
        """Create the comparison analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Comparison text
        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        self.comparison_text.setPlainText("File comparison analysis will appear here after analyzing multiple files...")
        layout.addWidget(self.comparison_text)
        
        return widget
    
    def setup_connections(self):
        """Set up signal connections."""
        pass  # Connections are set up in init_ui
    
    def on_files_dropped(self, file_paths):
        """Handle file drop."""
        valid_files = []
        invalid_files = []
        
        for file_path in file_paths:
            path = Path(file_path)
            if self.analyzer.is_multimedia_file(path):
                valid_files.append(path)
            else:
                invalid_files.append(path.name)
        
        if valid_files:
            self.add_files(valid_files)
        
        if invalid_files:
            QMessageBox.warning(
                self, "Invalid Files",
                f"The following files are not supported multimedia formats:\n\n" +
                "\n".join(invalid_files)
            )
    
    def browse_files(self):
        """Browse for multimedia files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Multimedia Files",
            "",
            "Multimedia Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.mp3 *.wav *.flac *.aac *.ogg *.m4a);;All Files (*)"
        )
        
        if file_paths:
            valid_files = []
            invalid_files = []
            
            for file_path in file_paths:
                path = Path(file_path)
                if self.analyzer.is_multimedia_file(path):
                    valid_files.append(path)
                else:
                    invalid_files.append(path.name)
            
            if valid_files:
                self.add_files(valid_files)
            
            if invalid_files:
                QMessageBox.warning(
                    self, "Invalid Files",
                    f"The following files are not supported:\n\n" +
                    "\n".join(invalid_files)
                )
    
    def add_files(self, file_paths: List[Path]):
        """Add files to the analysis list."""
        for file_path in file_paths:
            # Check if file is already in the list
            if file_path not in self.multimedia_files:
                self.multimedia_files.append(file_path)
                
                # Add to list widget
                item = QListWidgetItem(file_path.name)
                item.setData(Qt.ItemDataRole.UserRole, str(file_path))
                
                # Set icon based on file type
                if self.analyzer.is_video_file(file_path):
                    item.setText(f"🎬 {file_path.name}")
                else:
                    item.setText(f"🎵 {file_path.name}")
                
                self.file_list.addItem(item)
        
        self.update_file_count()
        self.update_button_states()
    
    def remove_selected_file(self):
        """Remove selected file from the list."""
        current_item = self.file_list.currentItem()
        if current_item:
            file_path = Path(current_item.data(Qt.ItemDataRole.UserRole))
            
            # Remove from list
            self.multimedia_files.remove(file_path)
            
            # Remove from UI
            row = self.file_list.row(current_item)
            self.file_list.takeItem(row)
            
            # Remove from analysis results
            if file_path.name in self.analysis_results:
                del self.analysis_results[file_path.name]
            
            self.update_file_count()
            self.update_button_states()
            self.update_analysis_displays()
    
    def clear_files(self):
        """Clear all files."""
        self.multimedia_files.clear()
        self.analysis_results.clear()
        self.file_list.clear()
        self.update_file_count()
        self.update_button_states()
        self.clear_analysis_displays()
    
    def update_file_count(self):
        """Update the file count label."""
        count = len(self.multimedia_files)
        if count == 0:
            self.file_count_label.setText("No files selected")
        elif count == 1:
            self.file_count_label.setText("1 file selected")
        else:
            self.file_count_label.setText(f"{count} files selected")
    
    def update_button_states(self):
        """Update button enabled states."""
        has_files = len(self.multimedia_files) > 0
        has_results = len(self.analysis_results) > 0
        
        self.analyze_button.setEnabled(has_files)
        self.export_button.setEnabled(has_results)
    
    def on_file_selection_changed(self, current, previous):
        """Handle file selection change in the list."""
        # Could update a detailed view for the selected file
        pass
    
    def start_analysis(self):
        """Start the analysis process."""
        if not self.multimedia_files:
            return
        
        # Clear previous results
        self.analysis_results.clear()
        self.clear_analysis_displays()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable analyze button
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("Analyzing...")
        
        # Start analysis worker
        self.analysis_worker = MultimediaBatchAnalysisWorker(
            [str(f) for f in self.multimedia_files], 
            self.analyzer
        )
        self.analysis_worker.progress_updated.connect(self.progress_bar.setValue)
        self.analysis_worker.file_analyzed.connect(self.on_file_analyzed)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_worker.analysis_failed.connect(self.on_analysis_failed)
        self.analysis_worker.start()
    
    def on_file_analyzed(self, filename: str, analysis: Dict[str, Any]):
        """Handle individual file analysis completion."""
        self.analysis_results[filename] = analysis
        self.update_analysis_displays()
    
    def on_analysis_completed(self):
        """Handle analysis completion."""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Re-enable analyze button
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("🔍 Analyze Files")
        
        # Enable export button
        self.export_button.setEnabled(True)
        
        # Final update of displays
        self.update_analysis_displays()
        
        self.logger.info(f"Analysis completed for {len(self.analysis_results)} files")
    
    def on_analysis_failed(self, error_msg: str):
        """Handle analysis failure."""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Re-enable analyze button
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("🔍 Analyze Files")
        
        QMessageBox.critical(
            self, "Analysis Failed",
            f"Failed to analyze files:\n\n{error_msg}"
        )
    
    def update_analysis_displays(self):
        """Update all analysis display components."""
        self.update_summary_tab()
        self.update_details_table()
        self.update_comparison_tab()
    
    def update_summary_tab(self):
        """Update the summary tab content."""
        if not self.analysis_results:
            return
        
        # Calculate overall statistics
        total_files = len(self.analysis_results)
        video_files = sum(1 for r in self.analysis_results.values() 
                         if r.get('media_type') == 'video' and 'error' not in r)
        audio_files = sum(1 for r in self.analysis_results.values() 
                         if r.get('media_type') == 'audio' and 'error' not in r)
        error_files = sum(1 for r in self.analysis_results.values() if 'error' in r)
        
        total_capacity = sum(r.get('capacity_bytes', 0) for r in self.analysis_results.values() 
                           if 'error' not in r)
        
        avg_suitability = 0
        valid_results = [r for r in self.analysis_results.values() 
                        if 'error' not in r and 'suitability_score' in r]
        if valid_results:
            avg_suitability = sum(r['suitability_score'] for r in valid_results) / len(valid_results)
        
        # Update statistics text
        stats_text = f"Total Files: {total_files}\n"
        stats_text += f"Video Files: {video_files}\n"
        stats_text += f"Audio Files: {audio_files}\n"
        if error_files > 0:
            stats_text += f"Error Files: {error_files}\n"
        stats_text += f"Total Capacity: {self.format_file_size(total_capacity)}\n"
        stats_text += f"Average Suitability: {avg_suitability:.1f}/10"
        
        self.stats_text.setPlainText(stats_text)
        
        # Find best candidates
        candidates = []
        for filename, result in self.analysis_results.items():
            if 'error' not in result and 'suitability_score' in result:
                score = result['suitability_score']
                capacity = result.get('capacity_bytes', 0)
                if score >= 7:  # Good suitability
                    candidates.append((filename, score, capacity))
        
        # Sort by suitability score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            candidates_text = "Top candidates for steganography:\n\n"
            for i, (filename, score, capacity) in enumerate(candidates[:5]):
                candidates_text += f"{i+1}. {filename}\n"
                candidates_text += f"   Suitability: {score}/10\n"
                candidates_text += f"   Capacity: {self.format_file_size(capacity)}\n\n"
        else:
            candidates_text = "No files with high suitability scores found.\n"
            candidates_text += "Consider using files with more complexity, longer duration, or higher resolution."
        
        self.candidates_text.setPlainText(candidates_text)
        
        # General recommendations
        recommendations = []
        
        if video_files > audio_files:
            recommendations.append("• Video files generally offer higher capacity than audio files")
        
        if avg_suitability < 5:
            recommendations.append("• Consider using files with more visual/audio complexity")
            recommendations.append("• Longer files typically provide better security")
        
        if total_capacity > 100 * 1024 * 1024:  # > 100MB
            recommendations.append("• Excellent total capacity available for large data sets")
        elif total_capacity < 10 * 1024 * 1024:  # < 10MB
            recommendations.append("• Limited capacity - consider larger or more files")
        
        if not recommendations:
            recommendations.append("• Files show good characteristics for steganography")
            recommendations.append("• Remember to use strong passwords and secure deletion")
        
        self.general_recommendations_text.setPlainText('\n'.join(recommendations))
    
    def update_details_table(self):
        """Update the details table."""
        self.details_table.setRowCount(len(self.analysis_results))
        
        for row, (filename, result) in enumerate(self.analysis_results.items()):
            # Filename
            self.details_table.setItem(row, 0, QTableWidgetItem(filename))
            
            if 'error' in result:
                # Error case
                self.details_table.setItem(row, 1, QTableWidgetItem("Error"))
                self.details_table.setItem(row, 2, QTableWidgetItem("-"))
                self.details_table.setItem(row, 3, QTableWidgetItem("-"))
                self.details_table.setItem(row, 4, QTableWidgetItem("-"))
                self.details_table.setItem(row, 5, QTableWidgetItem("-"))
                self.details_table.setItem(row, 6, QTableWidgetItem("-"))
                self.details_table.setItem(row, 7, QTableWidgetItem(result['error']))
            else:
                # Normal case
                media_type = result.get('media_type', 'Unknown').title()
                file_format = result.get('format', 'Unknown').upper()
                file_size = self.format_file_size(result.get('file_size', 0))
                duration = result.get('duration_formatted', 'Unknown')
                capacity = self.format_file_size(result.get('capacity_bytes', 0))
                suitability = f"{result.get('suitability_score', 0)}/10"
                recommendations = '; '.join(result.get('recommendations', []))
                
                self.details_table.setItem(row, 1, QTableWidgetItem(media_type))
                self.details_table.setItem(row, 2, QTableWidgetItem(file_format))
                self.details_table.setItem(row, 3, QTableWidgetItem(file_size))
                self.details_table.setItem(row, 4, QTableWidgetItem(duration))
                self.details_table.setItem(row, 5, QTableWidgetItem(capacity))
                self.details_table.setItem(row, 6, QTableWidgetItem(suitability))
                self.details_table.setItem(row, 7, QTableWidgetItem(recommendations))
    
    def update_comparison_tab(self):
        """Update the comparison tab content."""
        if len(self.analysis_results) < 2:
            self.comparison_text.setPlainText("Add at least 2 files to see comparison analysis...")
            return
        
        # Separate by media type
        video_results = {k: v for k, v in self.analysis_results.items() 
                        if v.get('media_type') == 'video' and 'error' not in v}
        audio_results = {k: v for k, v in self.analysis_results.items() 
                        if v.get('media_type') == 'audio' and 'error' not in v}
        
        comparison_text = "File Comparison Analysis\n\n"
        
        # Video comparison
        if len(video_results) > 1:
            comparison_text += "VIDEO FILES COMPARISON:\n"
            comparison_text += "-" * 25 + "\n"
            
            # Find best video by capacity
            best_video = max(video_results.items(), key=lambda x: x[1].get('capacity_bytes', 0))
            comparison_text += f"Highest Capacity: {best_video[0]} ({self.format_file_size(best_video[1]['capacity_bytes'])})\n"
            
            # Find best video by suitability
            best_suit = max(video_results.items(), key=lambda x: x[1].get('suitability_score', 0))
            comparison_text += f"Best Suitability: {best_suit[0]} ({best_suit[1]['suitability_score']}/10)\n"
            
            # Resolution comparison
            resolutions = [(k, v.get('width', 0) * v.get('height', 0)) for k, v in video_results.items()]
            if resolutions:
                best_res = max(resolutions, key=lambda x: x[1])
                comparison_text += f"Highest Resolution: {best_res[0]}\n"
            
            comparison_text += "\n"
        
        # Audio comparison
        if len(audio_results) > 1:
            comparison_text += "AUDIO FILES COMPARISON:\n"
            comparison_text += "-" * 25 + "\n"
            
            # Find best audio by capacity
            best_audio = max(audio_results.items(), key=lambda x: x[1].get('capacity_bytes', 0))
            comparison_text += f"Highest Capacity: {best_audio[0]} ({self.format_file_size(best_audio[1]['capacity_bytes'])})\n"
            
            # Find best audio by suitability
            best_suit = max(audio_results.items(), key=lambda x: x[1].get('suitability_score', 0))
            comparison_text += f"Best Suitability: {best_suit[0]} ({best_suit[1]['suitability_score']}/10)\n"
            
            # Duration comparison
            longest = max(audio_results.items(), key=lambda x: x[1].get('duration_seconds', 0))
            comparison_text += f"Longest Duration: {longest[0]} ({longest[1].get('duration_formatted', 'Unknown')})\n"
            
            comparison_text += "\n"
        
        # Overall recommendations
        comparison_text += "RECOMMENDATIONS:\n"
        comparison_text += "-" * 15 + "\n"
        
        all_results = {k: v for k, v in self.analysis_results.items() if 'error' not in v}
        if all_results:
            # Best overall file
            best_overall = max(all_results.items(), 
                             key=lambda x: (x[1].get('suitability_score', 0) * 0.6 + 
                                          (x[1].get('capacity_bytes', 0) / 1024 / 1024) * 0.4))
            comparison_text += f"• Best Overall Choice: {best_overall[0]}\n"
            
            # Capacity recommendations
            total_capacity = sum(r.get('capacity_bytes', 0) for r in all_results.values())
            comparison_text += f"• Total Available Capacity: {self.format_file_size(total_capacity)}\n"
            
            if len(video_results) > 0 and len(audio_results) > 0:
                comparison_text += "• Mix of video and audio provides good diversity\n"
            
            avg_suitability = sum(r.get('suitability_score', 0) for r in all_results.values()) / len(all_results)
            if avg_suitability >= 7:
                comparison_text += "• Excellent overall file quality for steganography\n"
            elif avg_suitability >= 5:
                comparison_text += "• Good overall file quality for steganography\n"
            else:
                comparison_text += "• Consider higher quality files for better security\n"
        
        self.comparison_text.setPlainText(comparison_text)
    
    def clear_analysis_displays(self):
        """Clear all analysis displays."""
        self.stats_text.setPlainText("Analyze files to see overall statistics...")
        self.candidates_text.setPlainText("Analysis results will show the best files for steganography...")
        self.general_recommendations_text.setPlainText("General recommendations will appear here...")
        self.details_table.setRowCount(0)
        self.comparison_text.setPlainText("File comparison analysis will appear here after analyzing multiple files...")
    
    def export_results(self):
        """Export analysis results to a file."""
        if not self.analysis_results:
            QMessageBox.warning(self, "No Results", "No analysis results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Results",
            "multimedia_analysis_results.txt",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.export_to_csv(file_path)
                else:
                    self.export_to_text(file_path)
                
                QMessageBox.information(
                    self, "Export Successful",
                    f"Analysis results exported to:\n{file_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Failed",
                    f"Failed to export results:\n{e}"
                )
    
    def export_to_text(self, file_path: str):
        """Export results to text format."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("InvisioVault - Multimedia Analysis Results\n")
            f.write("=" * 45 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 18 + "\n")
            f.write(self.stats_text.toPlainText() + "\n\n")
            
            # Best candidates
            f.write("BEST CANDIDATES\n")
            f.write("-" * 15 + "\n")
            f.write(self.candidates_text.toPlainText() + "\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 16 + "\n")
            for filename, result in self.analysis_results.items():
                f.write(f"File: {filename}\n")
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")
                else:
                    f.write(f"Type: {result.get('media_type', 'Unknown').title()}\n")
                    f.write(f"Format: {result.get('format', 'Unknown')}\n")
                    f.write(f"Size: {self.format_file_size(result.get('file_size', 0))}\n")
                    f.write(f"Duration: {result.get('duration_formatted', 'Unknown')}\n")
                    f.write(f"Capacity: {self.format_file_size(result.get('capacity_bytes', 0))}\n")
                    f.write(f"Suitability: {result.get('suitability_score', 0)}/10\n")
                    recommendations = result.get('recommendations', [])
                    if recommendations:
                        f.write(f"Recommendations: {'; '.join(recommendations)}\n")
                f.write("\n")
            
            # General recommendations
            f.write("GENERAL RECOMMENDATIONS\n")
            f.write("-" * 23 + "\n")
            f.write(self.general_recommendations_text.toPlainText())
    
    def export_to_csv(self, file_path: str):
        """Export results to CSV format."""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write headers
            writer.writerow([
                "Filename", "Type", "Format", "Size (Bytes)", "Duration (Seconds)",
                "Capacity (Bytes)", "Suitability Score", "Recommendations"
            ])
            
            # Write data
            for filename, result in self.analysis_results.items():
                if 'error' in result:
                    writer.writerow([
                        filename, "Error", "", "", "", "", "", result['error']
                    ])
                else:
                    writer.writerow([
                        filename,
                        result.get('media_type', 'Unknown').title(),
                        result.get('format', 'Unknown'),
                        result.get('file_size', 0),
                        result.get('duration_seconds', 0),
                        result.get('capacity_bytes', 0),
                        result.get('suitability_score', 0),
                        '; '.join(result.get('recommendations', []))
                    ])
    
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
        
        super().closeEvent(event)
