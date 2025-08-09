"""
Decoy Dialog
Plausible deniability dialog for hiding multiple datasets with different passwords.
"""

import secrets
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QFileDialog, QLineEdit, QCheckBox, QTextEdit,
    QProgressBar, QComboBox, QMessageBox, QTabWidget, QWidget,
    QSpinBox, QSlider, QListWidget, QListWidgetItem, QTreeWidget,
    QTreeWidgetItem, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from utils.logger import Logger
from utils.config_manager import ConfigManager
from utils.error_handler import ErrorHandler
from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel
from core.multi_decoy_engine import MultiDecoyEngine


class DecoyWorkerThread(QThread):
    """Worker thread for decoy operations."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished_successfully = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, operation, **kwargs):
        super().__init__()
        self.operation = operation
        self.params = kwargs
        
        # Initialize engines
        self.multi_decoy_engine = MultiDecoyEngine(SecurityLevel.MAXIMUM)
        self.logger = Logger()
    
    def run(self):
        """Execute the decoy operation."""
        try:
            if self.operation == "hide":
                self._execute_hide_operation()
            elif self.operation == "extract":
                self._execute_extract_operation()
            
        except Exception as e:
            self.logger.error(f"Decoy operation failed: {e}")
            self.error_occurred.emit(str(e))
    
    def _execute_hide_operation(self):
        """Execute decoy hide operation with multiple datasets."""
        carrier_path = Path(self.params['carrier_path'])
        datasets = self.params['datasets']
        output_path = Path(self.params['output_path'])
        
        self.status_updated.emit("Checking image capacity...")
        self.progress_updated.emit(10)
        
        # Check capacity first
        capacity = self.multi_decoy_engine.calculate_multi_capacity(carrier_path, len(datasets))
        self.logger.info(f"Image capacity: {capacity}")
        
        if capacity['max_datasets'] < len(datasets):
            raise Exception(f"Image can only hold {capacity['max_datasets']} datasets, but {len(datasets)} provided")
        
        self.status_updated.emit(f"Hiding {len(datasets)} datasets...")
        self.progress_updated.emit(30)
        
        # Use MultiDecoyEngine to hide all datasets at once
        success = self.multi_decoy_engine.hide_multiple_datasets(
            carrier_path=carrier_path,
            datasets=datasets,
            output_path=output_path
        )
        
        if not success:
            raise Exception("Failed to hide datasets in image")
        
        self.status_updated.emit("Multi-decoy operation completed successfully!")
        self.progress_updated.emit(100)
        self.finished_successfully.emit()
    
    def _execute_extract_operation(self):
        """Execute decoy extract operation."""
        stego_path = Path(self.params['stego_path'])
        password = self.params['password']
        output_dir = Path(self.params['output_dir'])
        
        self.status_updated.emit("Extracting dataset with provided password...")
        self.progress_updated.emit(20)
        
        # Use MultiDecoyEngine to extract the dataset
        metadata = self.multi_decoy_engine.extract_dataset(
            stego_path=stego_path,
            password=password,
            output_dir=output_dir
        )
        
        if not metadata:
            raise Exception("No dataset found with the provided password")
        
        self.status_updated.emit(f"Dataset '{metadata.get('dataset_id', 'Unknown')}' extracted successfully!")
        self.progress_updated.emit(100)
        self.finished_successfully.emit()


class DecoyDialog(QDialog):
    """Dialog for decoy mode operations with plausible deniability."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Decoy Mode - Plausible Deniability")
        self.setModal(True)
        self.resize(1000, 700)
        
        # Initialize components
        self.logger = Logger()
        self.config = ConfigManager()
        self.error_handler = ErrorHandler()
        
        # State variables
        self.worker_thread = None
        self.datasets = []  # List of dataset configurations
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("👻 Decoy Mode - Plausible Deniability")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(18)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Hide multiple datasets with different passwords in a single image. "
            "Each password reveals only its corresponding dataset, providing plausible deniability."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Tab widget for different operations
        self.tabs = QTabWidget()
        
        # Hide tab
        hide_tab = self.create_hide_tab()
        self.tabs.addTab(hide_tab, "🔒 Hide Multiple Datasets")
        
        # Extract tab
        extract_tab = self.create_extract_tab()
        self.tabs.addTab(extract_tab, "🔓 Extract Dataset")
        
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
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #546E7A; }
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
        """Create the hide datasets tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Splitter for dataset management
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Dataset configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Carrier image
        carrier_group = QGroupBox("📁 Carrier Image")
        carrier_layout = QVBoxLayout(carrier_group)
        
        self.hide_carrier_label = QLabel("No carrier image selected")
        self.hide_carrier_label.setStyleSheet("padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.hide_carrier_button = QPushButton("Select Carrier Image")
        
        carrier_layout.addWidget(self.hide_carrier_label)
        carrier_layout.addWidget(self.hide_carrier_button)
        left_layout.addWidget(carrier_group)
        
        # Dataset configuration
        dataset_group = QGroupBox("📊 Dataset Configuration")
        dataset_layout = QVBoxLayout(dataset_group)
        
        # Dataset name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Dataset Name:"))
        self.dataset_name_input = QLineEdit()
        self.dataset_name_input.setPlaceholderText("Enter dataset name")
        name_layout.addWidget(self.dataset_name_input)
        dataset_layout.addLayout(name_layout)
        
        # Dataset password
        password_layout = QHBoxLayout()
        password_layout.addWidget(QLabel("Password:"))
        self.dataset_password_input = QLineEdit()
        self.dataset_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.dataset_password_input.setPlaceholderText("Enter password for this dataset")
        password_layout.addWidget(self.dataset_password_input)
        dataset_layout.addLayout(password_layout)
        
        # Priority level
        priority_layout = QHBoxLayout()
        priority_layout.addWidget(QLabel("Priority Level:"))
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["1 (Outer - Least Secure)", "2", "3", "4", "5 (Inner - Most Secure)"])
        priority_layout.addWidget(self.priority_combo)
        priority_layout.addStretch()
        dataset_layout.addLayout(priority_layout)
        
        # Decoy type
        decoy_type_layout = QHBoxLayout()
        decoy_type_layout.addWidget(QLabel("Decoy Type:"))
        self.decoy_type_combo = QComboBox()
        self.decoy_type_combo.addItems(["Standard", "Innocent", "Personal", "Business"])
        decoy_type_layout.addWidget(self.decoy_type_combo)
        decoy_type_layout.addStretch()
        dataset_layout.addLayout(decoy_type_layout)
        
        # Files selection
        files_layout = QVBoxLayout()
        files_layout.addWidget(QLabel("Files for this dataset:"))
        self.dataset_files_list = QListWidget()
        self.dataset_files_list.setMaximumHeight(100)
        
        files_button_layout = QHBoxLayout()
        self.add_dataset_files_button = QPushButton("Add Files")
        self.remove_dataset_file_button = QPushButton("Remove")
        self.clear_dataset_files_button = QPushButton("Clear")
        
        files_button_layout.addWidget(self.add_dataset_files_button)
        files_button_layout.addWidget(self.remove_dataset_file_button)
        files_button_layout.addWidget(self.clear_dataset_files_button)
        files_button_layout.addStretch()
        
        files_layout.addWidget(self.dataset_files_list)
        files_layout.addLayout(files_button_layout)
        dataset_layout.addLayout(files_layout)
        
        # Add dataset button
        self.add_dataset_button = QPushButton("➕ Add Dataset")
        self.add_dataset_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        dataset_layout.addWidget(self.add_dataset_button)
        
        left_layout.addWidget(dataset_group)
        
        # Output settings
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
        
        left_layout.addWidget(output_group)
        
        # Right panel - Datasets overview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        overview_group = QGroupBox("📋 Datasets Overview")
        overview_layout = QVBoxLayout(overview_group)
        
        self.datasets_tree = QTreeWidget()
        self.datasets_tree.setHeaderLabels(["Dataset", "Priority", "Files", "Type"])
        
        tree_button_layout = QHBoxLayout()
        self.edit_dataset_button = QPushButton("Edit Selected")
        self.remove_dataset_button = QPushButton("Remove Selected")
        self.clear_datasets_button = QPushButton("Clear All")
        
        tree_button_layout.addWidget(self.edit_dataset_button)
        tree_button_layout.addWidget(self.remove_dataset_button)
        tree_button_layout.addWidget(self.clear_datasets_button)
        tree_button_layout.addStretch()
        
        overview_layout.addWidget(self.datasets_tree)
        overview_layout.addLayout(tree_button_layout)
        right_layout.addWidget(overview_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 500])
        
        layout.addWidget(splitter)
        
        return widget
    
    def create_extract_tab(self) -> QWidget:
        """Create the extract dataset tab."""
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
        auth_group = QGroupBox("🔒 Authentication")
        auth_layout = QVBoxLayout(auth_group)
        
        password_layout = QHBoxLayout()
        password_layout.addWidget(QLabel("Password:"))
        self.extract_password_input = QLineEdit()
        self.extract_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.extract_password_input.setPlaceholderText("Enter password for dataset to extract")
        password_layout.addWidget(self.extract_password_input)
        auth_layout.addLayout(password_layout)
        
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
        
        # Information panel
        info_group = QGroupBox("ℹ️ Information")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Each password reveals only its corresponding dataset. This provides plausible "
            "deniability - you can reveal an innocent dataset while keeping sensitive data hidden "
            "with a different password."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #666; font-size: 11px; padding: 10px; background-color: #f0f8ff; border-radius: 5px;")
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        return widget
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        # Hide tab signals
        self.hide_carrier_button.clicked.connect(self.select_hide_carrier)
        self.add_dataset_files_button.clicked.connect(self.add_dataset_files)
        self.remove_dataset_file_button.clicked.connect(self.remove_dataset_file)
        self.clear_dataset_files_button.clicked.connect(self.clear_dataset_files)
        self.add_dataset_button.clicked.connect(self.add_dataset)
        self.hide_output_button.clicked.connect(self.select_hide_output)
        
        # Dataset overview signals
        self.edit_dataset_button.clicked.connect(self.edit_selected_dataset)
        self.remove_dataset_button.clicked.connect(self.remove_selected_dataset)
        self.clear_datasets_button.clicked.connect(self.clear_all_datasets)
        
        # Extract tab signals
        self.extract_stego_button.clicked.connect(self.select_extract_stego)
        self.extract_output_button.clicked.connect(self.select_extract_output)
        
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
    
    def add_dataset_files(self):
        """Add files to current dataset configuration."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files for Dataset", "",
            "All Files (*);;Text Files (*.txt);;Documents (*.pdf *.doc *.docx)"
        )
        if file_paths:
            for path in file_paths:
                self.dataset_files_list.addItem(Path(path).name)
                # Store full path as item data
                item = self.dataset_files_list.item(self.dataset_files_list.count() - 1)
                item.setData(Qt.ItemDataRole.UserRole, path)
    
    def remove_dataset_file(self):
        """Remove selected file from current dataset."""
        current_row = self.dataset_files_list.currentRow()
        if current_row >= 0:
            self.dataset_files_list.takeItem(current_row)
    
    def clear_dataset_files(self):
        """Clear all files from current dataset."""
        self.dataset_files_list.clear()
    
    def add_dataset(self):
        """Add configured dataset to the list."""
        # Validate dataset configuration
        if not self.dataset_name_input.text():
            QMessageBox.warning(self, "Warning", "Please enter a dataset name.")
            return
        
        if not self.dataset_password_input.text():
            QMessageBox.warning(self, "Warning", "Please enter a password for this dataset.")
            return
        
        if self.dataset_files_list.count() == 0:
            QMessageBox.warning(self, "Warning", "Please add files to this dataset.")
            return
        
        # Collect file paths
        file_paths = []
        for i in range(self.dataset_files_list.count()):
            item = self.dataset_files_list.item(i)
            file_paths.append(item.data(Qt.ItemDataRole.UserRole))
        
        # Create dataset configuration
        priority_text = self.priority_combo.currentText()
        priority = int(priority_text.split()[0])  # Extract number from "1 (Outer...)"
        
        dataset_config = {
            'name': self.dataset_name_input.text(),
            'password': self.dataset_password_input.text(),
            'priority': priority,
            'decoy_type': self.decoy_type_combo.currentText().lower(),
            'files': file_paths
        }
        
        # Check for duplicate names or priorities
        for existing in self.datasets:
            if existing['name'] == dataset_config['name']:
                QMessageBox.warning(self, "Warning", "A dataset with this name already exists.")
                return
            if existing['priority'] == dataset_config['priority']:
                QMessageBox.warning(self, "Warning", "A dataset with this priority level already exists.")
                return
        
        # Add to datasets list
        self.datasets.append(dataset_config)
        
        # Add to tree widget
        item = QTreeWidgetItem([
            dataset_config['name'],
            str(dataset_config['priority']),
            str(len(dataset_config['files'])),
            dataset_config['decoy_type'].title()
        ])
        self.datasets_tree.addTopLevelItem(item)
        
        # Clear form for next dataset
        self.dataset_name_input.clear()
        self.dataset_password_input.clear()
        self.dataset_files_list.clear()
        self.priority_combo.setCurrentIndex(0)
        self.decoy_type_combo.setCurrentIndex(0)
        
        QMessageBox.information(self, "Success", "Dataset added successfully!")
    
    def edit_selected_dataset(self):
        """Edit the selected dataset."""
        current_item = self.datasets_tree.currentItem()
        if not current_item:
            QMessageBox.information(self, "Info", "Please select a dataset to edit.")
            return
        
        # Find dataset by name
        dataset_name = current_item.text(0)
        dataset_config = None
        for config in self.datasets:
            if config['name'] == dataset_name:
                dataset_config = config
                break
        
        if dataset_config:
            # Populate form with existing data
            self.dataset_name_input.setText(dataset_config['name'])
            self.dataset_password_input.setText(dataset_config['password'])
            
            # Set priority
            for i in range(self.priority_combo.count()):
                if self.priority_combo.itemText(i).startswith(str(dataset_config['priority'])):
                    self.priority_combo.setCurrentIndex(i)
                    break
            
            # Set decoy type
            decoy_type = dataset_config['decoy_type'].title()
            index = self.decoy_type_combo.findText(decoy_type)
            if index >= 0:
                self.decoy_type_combo.setCurrentIndex(index)
            
            # Populate files list
            self.dataset_files_list.clear()
            for file_path in dataset_config['files']:
                item = QListWidgetItem(Path(file_path).name)
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                self.dataset_files_list.addItem(item)
            
            # Remove from datasets (will be re-added when user clicks Add Dataset)
            self.datasets.remove(dataset_config)
            
            # Remove from tree
            root = self.datasets_tree.invisibleRootItem()
            root.removeChild(current_item)
    
    def remove_selected_dataset(self):
        """Remove the selected dataset."""
        current_item = self.datasets_tree.currentItem()
        if not current_item:
            QMessageBox.information(self, "Info", "Please select a dataset to remove.")
            return
        
        dataset_name = current_item.text(0)
        
        # Remove from datasets list
        self.datasets = [d for d in self.datasets if d['name'] != dataset_name]
        
        # Remove from tree
        root = self.datasets_tree.invisibleRootItem()
        root.removeChild(current_item)
        
        QMessageBox.information(self, "Success", "Dataset removed successfully!")
    
    def clear_all_datasets(self):
        """Clear all datasets."""
        reply = QMessageBox.question(
            self, "Confirm", "Are you sure you want to clear all datasets?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.datasets.clear()
            self.datasets_tree.clear()
    
    def select_hide_output(self):
        """Select output location for hiding."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Decoy Image", "",
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
    
    def select_extract_output(self):
        """Select output directory for extraction."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.extract_output_input.setText(dir_path)
    
    def execute_operation(self):
        """Execute the selected operation."""
        current_tab = self.tabs.currentIndex()
        
        try:
            if current_tab == 0:  # Hide
                self.execute_hide_operation()
            elif current_tab == 1:  # Extract
                self.execute_extract_operation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Operation failed: {str(e)}")
    
    def execute_hide_operation(self):
        """Execute decoy hide operation."""
        # Validate inputs
        if not hasattr(self, 'hide_carrier_path'):
            raise ValueError("Please select a carrier image")
        if not self.datasets:
            raise ValueError("Please add at least one dataset")
        if not self.hide_output_input.text():
            raise ValueError("Please specify output location")
        
        # Start worker thread
        self.worker_thread = DecoyWorkerThread(
            "hide",
            carrier_path=self.hide_carrier_path,
            datasets=self.datasets,
            output_path=self.hide_output_input.text()
        )
        self.start_operation()
    
    def execute_extract_operation(self):
        """Execute decoy extract operation."""
        # Validate inputs
        if not hasattr(self, 'extract_stego_path'):
            raise ValueError("Please select a steganographic image")
        if not self.extract_password_input.text():
            raise ValueError("Please enter a password")
        if not self.extract_output_input.text():
            raise ValueError("Please specify output directory")
        
        # Start worker thread
        self.worker_thread = DecoyWorkerThread(
            "extract",
            stego_path=self.extract_stego_path,
            password=self.extract_password_input.text(),
            output_dir=self.extract_output_input.text()
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
