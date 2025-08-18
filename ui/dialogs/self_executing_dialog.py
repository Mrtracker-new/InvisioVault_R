"""
Self-Executing Image Dialog
UI interface for creating and managing self-executing images.

Author: Rolan (RNR)
Purpose: Educational demonstration of advanced steganography techniques
"""

import os
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, 
    QLineEdit, QTextEdit, QPushButton, QFileDialog, QComboBox,
    QCheckBox, QGroupBox, QFormLayout, QProgressBar, QMessageBox,
    QSplitter, QFrame, QGridLayout, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QPalette, QPixmap, QIcon

# from ui.components.password_input import PasswordInput  # Component not available
# from ui.components.progress_dialog import ProgressDialog  # Component not available
# from ui.themes.theme_manager import ThemeManager  # Component not available
from core.self_executing_engine import SelfExecutingEngine
from utils.logger import Logger
from utils.error_handler import ErrorHandler


class SelfExecutingCreationThread(QThread):
    """Background thread for creating self-executing images."""
    
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished = Signal(bool, str)
    
    def __init__(self, engine: SelfExecutingEngine, creation_type: str, **kwargs):
        super().__init__()
        self.engine = engine
        self.creation_type = creation_type
        self.kwargs = kwargs
    
    def run(self):
        """Execute the creation process."""
        try:
            self.status_updated.emit("Initializing creation process...")
            self.progress_updated.emit(10)
            
            if self.creation_type == 'polyglot':
                success = self._create_polyglot()
            elif self.creation_type == 'script':
                success = self._create_script_image()
            else:
                success = False
                
            if success:
                self.progress_updated.emit(100)
                self.finished.emit(True, "Self-executing image created successfully!")
            else:
                self.finished.emit(False, "Failed to create self-executing image")
                
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")
    
    def _create_polyglot(self) -> bool:
        """Create ICO/EXE polyglot executable."""
        self.status_updated.emit("Creating ICO/EXE polyglot...")
        self.progress_updated.emit(30)
        
        success = self.engine.create_ico_exe_polyglot(
            executable_path=self.kwargs['executable_path'],
            output_path=self.kwargs['output_path'],
            icon_sizes=self.kwargs.get('icon_sizes'),
            icon_colors=self.kwargs.get('icon_colors'),
            password=self.kwargs.get('password')
        )
        
        self.progress_updated.emit(80)
        return success
    
    def _create_script_image(self) -> bool:
        """Create script-executing image."""
        self.status_updated.emit("Embedding script in image...")
        self.progress_updated.emit(40)
        
        success = self.engine.create_script_executing_image(
            image_path=self.kwargs['image_path'],
            script_content=self.kwargs['script_content'],
            script_type=self.kwargs['script_type'],
            output_path=self.kwargs['output_path'],
            password=self.kwargs.get('password'),
            auto_execute=self.kwargs.get('auto_execute', False)
        )
        
        self.progress_updated.emit(80)
        return success


class SelfExecutingDialog(QDialog):
    """Dialog for creating and managing self-executing images."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Self-Executing Images - InVisioVault")
        self.setMinimumSize(800, 600)
        
        # Initialize components
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.engine = SelfExecutingEngine()
        
        # Progress tracking
        self.current_operation = None
        
        self.setup_ui()
        # self.apply_theme()  # Theme manager not available
        
        self.logger.info("Self-executing image dialog initialized")
    
    def setup_ui(self):
        """Setup the dialog UI."""
        main_layout = QVBoxLayout(self)
        
        # Header
        self.create_header(main_layout)
        
        # Tab widget for different creation methods
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_polyglot_tab()
        self.create_script_tab()
        self.create_analysis_tab()
        
        # Button bar
        self.create_button_bar(main_layout)
        
        # Status bar
        self.create_status_bar(main_layout)
    
    def create_header(self, layout):
        """Create dialog header."""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        header_layout = QVBoxLayout(header_frame)
        
        # Title
        title_label = QLabel("🚀 Self-Executing Images")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Subtitle
        subtitle_label = QLabel("Create images that can execute embedded code when triggered")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #666; font-style: italic;")
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        layout.addWidget(header_frame)
    
    def create_polyglot_tab(self):
        """Create tab for ICO/EXE polyglot files."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Enhanced description for ICO/EXE polyglot
        desc_label = QLabel(
            "Create revolutionary ICO/EXE polyglot files that work perfectly as both icons and executables.\n\n"
            "🚀 REVOLUTIONARY ICO/EXE POLYGLOT FEATURES:\n"
            "• True dual-format compatibility: works as .ico AND .exe\n"
            "• No extraction needed: just rename the file extension\n"
            "• Perfect Windows integration: displays as icon, runs as program\n"
            "• No corruption or compatibility issues\n"
            "• Customizable icon sizes and colors\n\n"
            "✅ Simply rename between .ico and .exe extensions to switch functionality!"
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("background: #e6f7ff; color: #0066cc; padding: 15px; border-radius: 5px; border: 2px solid #99ccff; font-weight: bold;")
        layout.addWidget(desc_label)
        
        # Form layout
        form_layout = QFormLayout()
        
        # Executable file selection (only input needed for ICO/EXE polyglot)
        exe_layout = QHBoxLayout()
        self.polyglot_exe_input = QLineEdit()
        self.polyglot_exe_input.setPlaceholderText("Select Windows executable (.exe) to convert")
        exe_browse_btn = QPushButton("Browse")
        exe_browse_btn.clicked.connect(lambda: self.browse_executable_file(self.polyglot_exe_input))
        exe_layout.addWidget(self.polyglot_exe_input)
        exe_layout.addWidget(exe_browse_btn)
        form_layout.addRow("Executable:", exe_layout)
        
        # Icon customization options
        icon_group = QGroupBox("Icon Customization")
        icon_layout = QFormLayout(icon_group)
        
        # Icon sizes
        sizes_layout = QHBoxLayout()
        self.icon_16_check = QCheckBox("16x16")
        self.icon_32_check = QCheckBox("32x32")
        self.icon_48_check = QCheckBox("48x48")
        self.icon_64_check = QCheckBox("64x64")
        # Set default selections
        self.icon_16_check.setChecked(True)
        self.icon_32_check.setChecked(True)
        self.icon_48_check.setChecked(True)
        sizes_layout.addWidget(self.icon_16_check)
        sizes_layout.addWidget(self.icon_32_check)
        sizes_layout.addWidget(self.icon_48_check)
        sizes_layout.addWidget(self.icon_64_check)
        sizes_layout.addStretch()
        icon_layout.addRow("Icon Sizes:", sizes_layout)
        
        # Icon color theme
        self.icon_color_combo = QComboBox()
        self.icon_color_combo.addItems([
            "Blue Theme (Default)",
            "Green Theme", 
            "Red Theme",
            "Purple Theme",
            "Orange Theme",
            "Custom..."
        ])
        icon_layout.addRow("Icon Color:", self.icon_color_combo)
        
        form_layout.addRow(icon_group)
        
        # Output path
        output_layout = QHBoxLayout()
        self.polyglot_output_input = QLineEdit()
        self.polyglot_output_input.setPlaceholderText("Output polyglot file path")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(lambda: self.browse_output_file(self.polyglot_output_input))
        output_layout.addWidget(self.polyglot_output_input)
        output_layout.addWidget(output_browse_btn)
        form_layout.addRow("Output File:", output_layout)
        
        # Password (optional)
        self.polyglot_password = QLineEdit()
        self.polyglot_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.polyglot_password.setPlaceholderText("Optional: Encrypt embedded executable")
        form_layout.addRow("Password:", self.polyglot_password)
        
        layout.addLayout(form_layout)
        
        # Create button
        create_polyglot_btn = QPushButton("🔨 Create Polyglot Executable")
        create_polyglot_btn.clicked.connect(self.create_polyglot)
        create_polyglot_btn.setMinimumHeight(40)
        layout.addWidget(create_polyglot_btn)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Polyglot Files")
    
    def create_script_tab(self):
        """Create tab for script-executing images."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Description
        desc_label = QLabel(
            "Embed executable scripts within images. Scripts can be extracted and executed "
            "when the image is processed with the appropriate tools."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("background: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(desc_label)
        
        # Form section
        form_group = QGroupBox("Script Configuration")
        form_layout = QFormLayout(form_group)
        
        # Image file selection
        script_image_layout = QHBoxLayout()
        self.script_image_input = QLineEdit()
        self.script_image_input.setPlaceholderText("Select carrier image (PNG, BMP, TIFF)")
        script_image_browse_btn = QPushButton("Browse")
        script_image_browse_btn.clicked.connect(lambda: self.browse_image_file(self.script_image_input))
        script_image_layout.addWidget(self.script_image_input)
        script_image_layout.addWidget(script_image_browse_btn)
        form_layout.addRow("Carrier Image:", script_image_layout)
        
        # Script type selection
        self.script_type_combo = QComboBox()
        self.script_type_combo.addItems([".py (Python)", ".js (JavaScript)", ".ps1 (PowerShell)", 
                                       ".bat (Batch)", ".sh (Bash)", ".vbs (VBScript)"])
        form_layout.addRow("Script Type:", self.script_type_combo)
        
        # Auto-execute option
        self.auto_execute_checkbox = QCheckBox("Auto-execute when extracted")
        form_layout.addRow("Options:", self.auto_execute_checkbox)
        
        # Password
        self.script_password = QLineEdit()
        self.script_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.script_password.setPlaceholderText("Password for script encryption")
        form_layout.addRow("Password:", self.script_password)
        
        # Output path
        output_layout = QHBoxLayout()
        self.script_output_input = QLineEdit()
        self.script_output_input.setPlaceholderText("Output image path")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(lambda: self.browse_output_file(self.script_output_input))
        output_layout.addWidget(self.script_output_input)
        output_layout.addWidget(output_browse_btn)
        form_layout.addRow("Output File:", output_layout)
        
        layout.addWidget(form_group)
        
        # Script content editor
        editor_group = QGroupBox("Script Content")
        editor_layout = QVBoxLayout(editor_group)
        
        self.script_editor = QTextEdit()
        self.script_editor.setPlaceholderText("Enter your script code here...")
        self.script_editor.setMinimumHeight(200)
        self.script_editor.setFont(QFont("Consolas", 10))
        editor_layout.addWidget(self.script_editor)
        
        # Script templates
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Templates:"))
        
        python_template_btn = QPushButton("Python Hello World")
        python_template_btn.clicked.connect(lambda: self.load_script_template("python"))
        template_layout.addWidget(python_template_btn)
        
        js_template_btn = QPushButton("JavaScript Alert")
        js_template_btn.clicked.connect(lambda: self.load_script_template("javascript"))
        template_layout.addWidget(js_template_btn)
        
        template_layout.addStretch()
        editor_layout.addLayout(template_layout)
        
        layout.addWidget(editor_group)
        
        # Create button
        create_script_btn = QPushButton("🎯 Create Script-Executing Image")
        create_script_btn.clicked.connect(self.create_script_image)
        create_script_btn.setMinimumHeight(40)
        layout.addWidget(create_script_btn)
        
        self.tab_widget.addTab(tab, "Script Images")
    
    def create_analysis_tab(self):
        """Create tab for analyzing self-executing images."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Description
        desc_label = QLabel(
            "Analyze images to detect embedded executable content and optionally execute it."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("background: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(desc_label)
        
        # Input section
        input_group = QGroupBox("Image Analysis")
        input_layout = QFormLayout(input_group)
        
        # Image file selection
        analysis_image_layout = QHBoxLayout()
        self.analysis_image_input = QLineEdit()
        self.analysis_image_input.setPlaceholderText("Select image to analyze")
        analysis_image_browse_btn = QPushButton("Browse")
        analysis_image_browse_btn.clicked.connect(lambda: self.browse_image_file(self.analysis_image_input))
        analysis_image_layout.addWidget(self.analysis_image_input)
        analysis_image_layout.addWidget(analysis_image_browse_btn)
        input_layout.addRow("Image File:", analysis_image_layout)
        
        # Password for encrypted content
        self.analysis_password = QLineEdit()
        self.analysis_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.analysis_password.setPlaceholderText("Password (if content is encrypted)")
        input_layout.addRow("Password:", self.analysis_password)
        
        layout.addWidget(input_group)
        
        # Analysis buttons
        button_layout = QHBoxLayout()
        
        analyze_btn = QPushButton("🔍 Analyze Image")
        analyze_btn.clicked.connect(self.analyze_image)
        button_layout.addWidget(analyze_btn)
        
        execute_btn = QPushButton("▶️ Execute Content")
        execute_btn.clicked.connect(self.execute_image_content)
        button_layout.addWidget(execute_btn)
        
        extract_btn = QPushButton("🖼️ Extract Image")
        extract_btn.clicked.connect(self.extract_image_from_polyglot)
        button_layout.addWidget(extract_btn)
        
        layout.addLayout(button_layout)
        
        # Results area
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.analysis_results = QTextEdit()
        self.analysis_results.setReadOnly(True)
        self.analysis_results.setMaximumHeight(200)
        results_layout.addWidget(self.analysis_results)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Analysis & Execution")
    
    def create_button_bar(self, layout):
        """Create bottom button bar."""
        button_layout = QHBoxLayout()
        
        # Advanced diagnostic button
        diagnostic_btn = QPushButton("🔍 Advanced Diagnostics")
        diagnostic_btn.clicked.connect(self.launch_advanced_diagnostics)
        diagnostic_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(diagnostic_btn)
        
        # Info button
        info_btn = QPushButton("ℹ️ Help")
        info_btn.clicked.connect(self.show_help)
        button_layout.addWidget(info_btn)
        
        # Viewer button
        viewer_btn = QPushButton("👁️ Launch Viewer")
        viewer_btn.clicked.connect(self.launch_viewer)
        button_layout.addWidget(viewer_btn)
        
        button_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_status_bar(self, layout):
        """Create status bar."""
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { padding: 5px; background: #f5f5f5; }")
        layout.addWidget(self.status_label)
    
    def browse_output_file(self, line_edit):
        """Browse for output file location - for polyglot files."""
        # Get the current tab to determine appropriate file filter
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # Polyglot Files tab
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save ICO/EXE Polyglot File",
                "polyglot_file.exe",
                "ICO/EXE Polyglot (*.exe);;Icon Files (*.ico);;Binary Files (*.bin);;All Files (*.*)"
            )
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Select Output File",
                "",
                "All Files (*.*)"
            )
        
        if file_path:
            line_edit.setText(file_path)
    
    def browse_image_file(self, line_edit):
        """Browse for image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.png *.bmp *.tiff *.tif);;PNG Files (*.png);;BMP Files (*.bmp);;TIFF Files (*.tiff *.tif);;All Files (*.*)"
        )
        
        if file_path:
            line_edit.setText(file_path)
    
    def browse_executable_file(self, line_edit):
        """Browse for executable file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Executable File",
            "",
            "Executable Files (*.exe *.bin);;Windows Executables (*.exe);;All Files (*.*)"
        )
        
        
        if file_path:
            line_edit.setText(file_path)
    
    def load_script_template(self, template_type):
        """Load a script template."""
        templates = {
            "python": '''#!/usr/bin/env python3
print("Hello from self-executing image!")
import sys
print(f"Python version: {sys.version}")

# Add your Python code here
input("Press Enter to continue...")
''',
            "javascript": '''console.log("Hello from self-executing image!");
console.log("Node.js version:", process.version);

// Add your JavaScript code here
process.stdin.setRawMode(true);
process.stdin.resume();
process.stdin.on('data', process.exit.bind(process, 0));
console.log("Press any key to exit...");
'''
        }
        
        if template_type in templates:
            self.script_editor.setPlainText(templates[template_type])
            # Update script type combo
            if template_type == "python":
                self.script_type_combo.setCurrentText(".py (Python)")
            elif template_type == "javascript":
                self.script_type_combo.setCurrentText(".js (JavaScript)")
    
    def create_polyglot(self):
        """Create ICO/EXE polyglot executable."""
        try:
            # Validate inputs
            exe_path = self.polyglot_exe_input.text().strip()
            output_path = self.polyglot_output_input.text().strip()
            password = self.polyglot_password.text().strip()
            
            if not exe_path or not output_path:
                QMessageBox.warning(self, "Input Required", 
                                  "Please provide executable file and output path.")
                return
                
            if not Path(exe_path).exists():
                QMessageBox.warning(self, "File Not Found", f"Executable not found: {exe_path}")
                return
            
            # Collect icon customization settings
            icon_sizes = []
            if self.icon_16_check.isChecked():
                icon_sizes.append(16)
            if self.icon_32_check.isChecked():
                icon_sizes.append(32)
            if self.icon_48_check.isChecked():
                icon_sizes.append(48)
            if self.icon_64_check.isChecked():
                icon_sizes.append(64)
            
            # Default to 32x32 if no sizes selected
            if not icon_sizes:
                icon_sizes = [32]
            
            # Get selected color theme and convert to RGB tuple
            color_theme = self.icon_color_combo.currentText()
            icon_colors = self._get_color_rgb_from_theme(color_theme)
            
            # Start creation in background thread
            self.status_label.setText("Creating ICO/EXE polyglot...")
            
            creation_thread = SelfExecutingCreationThread(
                engine=self.engine,
                creation_type='polyglot',
                executable_path=exe_path,
                output_path=output_path,
                icon_sizes=icon_sizes,
                icon_colors=icon_colors,
                password=password if password else None
            )
            
            # Connect thread signals and start (simplified for now)
            creation_thread.finished.connect(lambda success, msg: self.on_creation_finished(success, msg, None))
            creation_thread.start()
            creation_thread.wait()  # Wait for completion
            
        except Exception as e:
            self.error_handler.handle_exception(e)
            QMessageBox.critical(self, "Error", f"Failed to create polyglot: {e}")
    
    def create_script_image(self):
        """Create script-executing image."""
        try:
            # Validate inputs
            image_path = self.script_image_input.text().strip()
            script_content = self.script_editor.toPlainText().strip()
            script_type = self.script_type_combo.currentText().split()[0]  # Extract .py, .js, etc.
            output_path = self.script_output_input.text().strip()
            password = self.script_password.text().strip()
            auto_execute = self.auto_execute_checkbox.isChecked()
            
            if not image_path or not script_content or not output_path:
                QMessageBox.warning(self, "Input Required", 
                                  "Please provide carrier image, script content, and output path.")
                return
            
            if not Path(image_path).exists():
                QMessageBox.warning(self, "File Not Found", f"Carrier image not found: {image_path}")
                return
            
            # Start creation in background thread
            self.status_label.setText("Creating script-executing image...")
            
            creation_thread = SelfExecutingCreationThread(
                engine=self.engine,
                creation_type='script',
                image_path=image_path,
                script_content=script_content,
                script_type=script_type,
                output_path=output_path,
                password=password if password else None,
                auto_execute=auto_execute
            )
            
            # Connect thread signals and start (simplified for now)
            creation_thread.finished.connect(lambda success, msg: self.on_creation_finished(success, msg, None))
            creation_thread.start()
            creation_thread.wait()  # Wait for completion
            
        except Exception as e:
            self.error_handler.handle_exception(e)
            QMessageBox.critical(self, "Error", f"Failed to create script image: {e}")
    
    def analyze_image(self):
        """Analyze image for executable content."""
        try:
            image_path = self.analysis_image_input.text().strip()
            password = self.analysis_password.text().strip()
            
            if not image_path:
                QMessageBox.warning(self, "Input Required", "Please select an image to analyze.")
                return
            
            if not Path(image_path).exists():
                QMessageBox.warning(self, "File Not Found", f"Image not found: {image_path}")
                return
            
            self.status_label.setText("Analyzing image...")
            
            # Analyze in safe mode
            result = self.engine.extract_and_execute(
                image_path=image_path,
                password=password if password else None,
                execution_mode='safe'
            )
            
            # Display results
            if result.get('success'):
                analysis_text = f"""✅ Executable Content Detected

Type: {result.get('type', 'Unknown')}
Details: {result.get('message', 'No details available')}

Can Execute: {'Yes' if result.get('can_execute') else 'No'}
Auto-Execute: {'Yes' if result.get('auto_execute') else 'No'}
Script Type: {result.get('script_type', 'N/A')}
"""
            else:
                analysis_text = f"""❌ No Executable Content Found

Message: {result.get('message', 'Unknown error')}
Error: {result.get('error', 'None')}
"""
            
            self.analysis_results.setPlainText(analysis_text)
            self.status_label.setText("Analysis complete")
            
        except Exception as e:
            self.error_handler.handle_exception(e)
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze image: {e}")
    
    def execute_image_content(self):
        """Execute content from analyzed image."""
        try:
            image_path = self.analysis_image_input.text().strip()
            password = self.analysis_password.text().strip()
            
            if not image_path:
                QMessageBox.warning(self, "Input Required", "Please select an image to execute.")
                return
            
            # Confirm execution
            reply = QMessageBox.question(
                self, 
                "Confirm Execution",
                "Are you sure you want to execute the embedded content?\n\n"
                "⚠️ WARNING: This could be potentially dangerous!",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            self.status_label.setText("Executing content...")
            
            # Execute in auto mode
            result = self.engine.extract_and_execute(
                image_path=image_path,
                password=password if password else None,
                execution_mode='auto'
            )
            
            # Display results
            if result.get('success'):
                execution_text = f"""✅ Execution Completed

Return Code: {result.get('return_code', 'N/A')}
Output: {result.get('stdout', 'No output')}
Errors: {result.get('stderr', 'None')}
"""
            else:
                execution_text = f"""❌ Execution Failed

Error: {result.get('error', 'Unknown error')}
"""
            
            self.analysis_results.setPlainText(execution_text)
            self.status_label.setText("Execution complete")
            
        except Exception as e:
            self.error_handler.handle_exception(e)
            QMessageBox.critical(self, "Execution Error", f"Failed to execute content: {e}")
    
    def launch_advanced_diagnostics(self):
        """Launch the advanced polyglot diagnostic tool."""
        try:
            # Check if the diagnostic tool exists
            diagnostic_tool_path = Path(__file__).parent.parent.parent / "polyglot_advanced_diagnostic_fix.py"
            
            if not diagnostic_tool_path.exists():
                QMessageBox.warning(
                    self,
                    "Tool Not Found",
                    f"Advanced diagnostic tool not found at:\n{diagnostic_tool_path}\n\n"
                    "Please ensure the polyglot_advanced_diagnostic_fix.py file is in the application root directory."
                )
                return
            
            # Show information about the tool
            info_msg = QMessageBox(
                QMessageBox.Icon.Information,
                "Advanced Polyglot Diagnostics",
                "This tool provides comprehensive analysis and multiple solutions for ICO/EXE polyglot development:\n\n"
                "🔍 FEATURES:\n"
                "• Deep technical analysis of polyglot structures\n"
                "• ICO/EXE dual-format creation and testing\n"
                "• Parser compatibility verification\n"
                "• Icon format optimization\n"
                "• Automated quality assurance\n\n"
                "📁 TOOL LOCATION:\n"
                f"{diagnostic_tool_path}\n\n"
                "🚀 USAGE EXAMPLES:\n"
                "python polyglot_advanced_diagnostic_fix.py analyze file.exe\n"
                "python polyglot_advanced_diagnostic_fix.py create ico prog.exe out.ico\n\n"
                "Would you like to open the tool's directory?"
            )
            info_msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if info_msg.exec() == QMessageBox.StandardButton.Yes:
                import subprocess
                import sys
                
                # Open file explorer to the tool location
                if sys.platform.startswith('win'):
                    subprocess.run(['explorer', '/select,', str(diagnostic_tool_path)], check=False)
                elif sys.platform.startswith('darwin'):
                    subprocess.run(['open', '-R', str(diagnostic_tool_path)], check=False)
                else:
                    subprocess.run(['xdg-open', str(diagnostic_tool_path.parent)], check=False)
                    
        except Exception as e:
            self.error_handler.handle_exception(e)
            QMessageBox.critical(self, "Error", f"Failed to launch advanced diagnostics: {e}")
    
    def launch_viewer(self):
        """Launch the custom self-executing image viewer."""
        try:
            viewer_path = Path(__file__).parent.parent.parent / "self_executing_viewer.py"
            success = self.engine.create_custom_viewer(str(viewer_path))
            
            if success:
                QMessageBox.information(
                    self,
                    "Viewer Created",
                    f"Custom viewer created at:\n{viewer_path}\n\n"
                    "You can use this to open and analyze self-executing images."
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to create custom viewer.")
                
        except Exception as e:
            self.error_handler.handle_exception(e)
            QMessageBox.critical(self, "Error", f"Failed to create viewer: {e}")
    
    def show_help(self):
        """Show help information."""
        help_text = """🚀 Advanced Self-Executing Images Help - InVisioVault

🔧 ICO/EXE POLYGLOT SYSTEM:
• Revolutionary dual-format compatibility: works as .ico AND .exe
• No extraction needed: just rename file extension (.ico ↔ .exe)
• Perfect Windows integration: displays as icon, runs as program
• Icon customization: multiple sizes (16x16, 32x32, 48x48, 64x64)
• Color themes: blue, green, red, purple, orange, custom
• Password encryption support for embedded executables

✅ TESTED COMPATIBILITY:
• Windows Photo Viewer • Windows Explorer • Icon viewers
• Windows PE Loader • Antivirus scanners • File managers

🎯 SCRIPT IMAGES:
• Python, JavaScript, PowerShell, Batch, VBScript support
• Encrypted embedding with password protection
• Auto-execution capabilities
• Template library for common scripts
• Traditional steganographic hiding in carrier images

🔍 ADVANCED ANALYSIS:
• Deep polyglot structure analysis
• Content detection and extraction
• Safe execution environment
• Script type identification

📚 RESOURCES:
• ICO/EXE Polyglot: Revolutionary dual-format system
• Script Images: Traditional steganographic approach
• Analysis Tools: Content detection and execution
• Security Features: Password protection and safe mode

⚠️ SECURITY & LEGAL:
These techniques are for educational cybersecurity research only.
Users are responsible for compliance with applicable laws.
Always exercise extreme caution with executable content!
"""
        QMessageBox.information(self, "Help - Self-Executing Images", help_text)
    
    def on_creation_finished(self, success, message, progress_dialog):
        """Handle creation completion."""
        if progress_dialog:
            progress_dialog.close()
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.status_label.setText("Ready")
        else:
            QMessageBox.warning(self, "Failed", message)
            self.status_label.setText("Error occurred")
    
    def extract_image_from_polyglot(self):
        """Extract image data from a polyglot file for viewing."""
        try:
            image_path = self.analysis_image_input.text().strip()
            
            if not image_path:
                QMessageBox.warning(self, "Input Required", "Please select a polyglot file to extract image from.")
                return
            
            if not Path(image_path).exists():
                QMessageBox.warning(self, "File Not Found", f"File not found: {image_path}")
                return
            
            self.status_label.setText("Extracting image from polyglot...")
            
            # Extract the image
            success = self.engine.extract_image_from_polyglot(image_path)
            
            if success:
                # Get the extracted image path
                base_path = os.path.splitext(image_path)[0]
                extracted_images = [f"{base_path}_extracted.png", f"{base_path}_extracted.jpg", 
                                  f"{base_path}_extracted.bmp", f"{base_path}_extracted.gif",
                                  f"{base_path}_extracted.tiff"]
                
                # Find which one was created
                extracted_path = None
                for path in extracted_images:
                    if Path(path).exists():
                        extracted_path = path
                        break
                
                if extracted_path:
                    QMessageBox.information(
                        self,
                        "Image Extracted Successfully",
                        f"Image extracted from polyglot file!\n\n"
                        f"Extracted to: {extracted_path}\n\n"
                        f"You can now open this file with any image viewer."
                    )
                    
                    # Update analysis results
                    extract_text = f"""🖼️ Image Extraction Successful

Original polyglot: {image_path}
Extracted image: {extracted_path}

The extracted image can now be viewed with standard image viewers.
"""
                    self.analysis_results.setPlainText(extract_text)
                else:
                    QMessageBox.warning(self, "Extraction Failed", 
                                      "Image was processed but extracted file not found.")
            else:
                QMessageBox.warning(self, "Extraction Failed", 
                                  "Failed to extract image from polyglot file. "
                                  "Make sure the file contains embedded image data.")
            
            self.status_label.setText("Ready")
            
        except Exception as e:
            self.error_handler.handle_exception(e)
            QMessageBox.critical(self, "Extraction Error", f"Failed to extract image: {e}")
            self.status_label.setText("Error occurred")
    
    def _get_color_rgb_from_theme(self, theme_name: str) -> tuple:
        """Convert color theme name to RGB tuple."""
        color_map = {
            "Blue Theme (Default)": (70, 130, 180),      # Steel blue
            "Green Theme": (34, 139, 34),                 # Forest green
            "Red Theme": (220, 20, 60),                   # Crimson
            "Purple Theme": (138, 43, 226),               # Blue violet
            "Orange Theme": (255, 140, 0),                # Dark orange
            "Custom...": (100, 100, 100)                  # Gray default for custom
        }
        return color_map.get(theme_name, (70, 130, 180))  # Default to blue
    
    def apply_theme(self):
        """Apply current theme to dialog."""
        # Theme manager not available - using default theme
        pass
