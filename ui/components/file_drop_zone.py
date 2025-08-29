"""
File Drop Zone Component
Drag-and-drop widget for file selection with visual feedback.
"""

from pathlib import Path
from typing import List, Callable, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont, QPalette

from utils.logger import Logger


class FileDropZone(QWidget):
    """Drag-and-drop file selection widget with file list display."""
    
    files_changed = Signal(list)  # Emitted when file list changes
    files_dropped = Signal(list)  # Emitted specifically when files are dropped (compatibility)
    
    def __init__(self, 
                 title: str = "Drop Files Here",
                 file_types: Optional[List[str]] = None,
                 max_files: int = 0,  # 0 = unlimited
                 show_browse_button: bool = True,  # Option to hide internal browse button
                 parent=None):
        super().__init__(parent)
        
        self.logger = Logger()
        self.title = title
        self.file_types = file_types or []  # Extensions like ['.png', '.jpg']
        self.max_files = max_files
        self.show_browse_button = show_browse_button
        self.files = []  # List of selected file paths
        
        self.init_ui()
        self.setup_drag_drop()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Drop area
        self.drop_area = QLabel()
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_area.setMinimumHeight(160)
        self.drop_area.setWordWrap(True)
        self.update_drop_area_style()
        layout.addWidget(self.drop_area)
        
        # Browse button (only show if enabled)
        if self.show_browse_button:
            browse_layout = QHBoxLayout()
            browse_layout.addStretch()
            
            self.browse_btn = QPushButton("📁 Browse Files")
            self.browse_btn.clicked.connect(self.browse_files)
            browse_layout.addWidget(self.browse_btn)
            
            browse_layout.addStretch()
            layout.addLayout(browse_layout)
        else:
            self.browse_btn = None
        
        # Selected files list
        self.files_label = QLabel("Selected Files:")
        self.files_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.files_label.setObjectName("files_label")
        layout.addWidget(self.files_label)
        
        self.files_list = QListWidget()
        self.files_list.setMaximumHeight(150)
        # Initially hide the files list
        self.files_list.setVisible(False)
        self.files_label.setVisible(False)
        layout.addWidget(self.files_list)
        
        # Clear button
        clear_layout = QHBoxLayout()
        clear_layout.addStretch()
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self.clear_files)
        self.clear_btn.setEnabled(False)
        # Initially hide the clear button
        self.clear_btn.setVisible(False)
        clear_layout.addWidget(self.clear_btn)
        
        layout.addLayout(clear_layout)
        
        self.update_display()
    
    def setup_drag_drop(self):
        """Enable drag and drop functionality."""
        self.setAcceptDrops(True)
        self.drop_area.setAcceptDrops(True)
    
    def update_drop_area_style(self, highlight: bool = False):
        """Update the visual style of the drop area."""
        if highlight:
            style = """
                QLabel {
                    border: 3px dashed #4CAF50;
                    border-radius: 8px;
                    background-color: #E8F5E8;
                    color: #1B5E20;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 20px;
                    text-align: center;
                }
            """
            text = f"💾 Drop files here\n\n{self.title}"
        else:
            style = """
                QLabel {
                    border: 2px dashed #888;
                    border-radius: 8px;
                    background-color: #f5f5f5;
                    color: #333;
                    font-size: 13px;
                    padding: 20px;
                    text-align: center;
                }
            """
            # Create multi-line text with proper spacing
            lines = []
            lines.append(f"📁 {self.title}")
            lines.append("")
            lines.append("Drag and drop files here or click Browse")
            
            if self.file_types:
                lines.append("")
                # Split long supported formats list
                formats = ', '.join(self.file_types)
                if len(formats) > 40:  # If too long, split into multiple lines
                    words = formats.split(', ')
                    line1_words = []
                    line2_words = []
                    current_length = 0
                    for word in words:
                        if current_length + len(word) + 2 < 40:  # +2 for ", "
                            line1_words.append(word)
                            current_length += len(word) + 2
                        else:
                            line2_words.append(word)
                    
                    if line1_words:
                        lines.append(f"Supported: {', '.join(line1_words)}")
                    if line2_words:
                        lines.append(f"           {', '.join(line2_words)}")
                else:
                    lines.append(f"Supported: {formats}")
            
            text = '\n'.join(lines)
        
        self.drop_area.setStyleSheet(style)
        self.drop_area.setText(text)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.update_drop_area_style(highlight=True)
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave events."""
        self.update_drop_area_style(highlight=False)
    
    def dropEvent(self, event: QDropEvent):
        """Handle file drop events."""
        self.update_drop_area_style(highlight=False)
        
        urls = event.mimeData().urls()
        new_files = []
        
        for url in urls:
            if url.isLocalFile():
                file_path = Path(url.toLocalFile())
                if file_path.is_file():
                    new_files.append(file_path)
        
        if new_files:
            self.add_files(new_files, from_drop=True)
        
        event.acceptProposedAction()
    
    def browse_files(self):
        """Open file browser dialog."""
        # Build file filter
        if self.file_types:
            extensions = ' '.join([f'*{ext}' for ext in self.file_types])
            file_filter = f"Supported Files ({extensions});;All Files (*.*)"
        else:
            file_filter = "All Files (*.*)"
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "", file_filter
        )
        
        if file_paths:
            self.add_files([Path(fp) for fp in file_paths])
    
    def add_files(self, file_paths: List[Path], from_drop: bool = False):
        """Add files to the selection list."""
        added_count = 0
        added_files = []
        
        for file_path in file_paths:
            # Check if file already exists
            if file_path in self.files:
                continue
                
            # Check file type if specified
            if self.file_types and file_path.suffix.lower() not in self.file_types:
                QMessageBox.warning(
                    self, "Invalid File Type",
                    f"File '{file_path.name}' is not a supported type.\n"
                    f"Supported types: {', '.join(self.file_types)}"
                )
                continue
            
            # Check max files limit
            if self.max_files > 0 and len(self.files) >= self.max_files:
                QMessageBox.warning(
                    self, "File Limit Reached",
                    f"Maximum {self.max_files} files allowed."
                )
                break
            
            self.files.append(file_path)
            added_files.append(str(file_path))
            added_count += 1
        
        if added_count > 0:
            self.update_display()
            self.files_changed.emit([str(f) for f in self.files])
            
            # Emit files_dropped signal specifically for drop events
            if from_drop:
                self.files_dropped.emit(added_files)
            
            self.logger.debug(f"Added {added_count} files to drop zone")
    
    def remove_file(self, file_path: Path):
        """Remove a file from the selection."""
        if file_path in self.files:
            self.files.remove(file_path)
            self.update_display()
            self.files_changed.emit([str(f) for f in self.files])
    
    def clear_files(self):
        """Clear all selected files."""
        self.files.clear()
        self.update_display()
        self.files_changed.emit([])
    
    def get_files(self) -> List[str]:
        """Get list of selected file paths as strings."""
        return [str(f) for f in self.files]
    
    def get_file_paths(self) -> List[Path]:
        """Get list of selected file paths as Path objects."""
        return self.files.copy()
    
    def set_files(self, file_paths: List[str]):
        """Set the file list programmatically."""
        self.files = [Path(fp) for fp in file_paths if Path(fp).exists()]
        self.update_display()
        self.files_changed.emit([str(f) for f in self.files])
    
    def update_display(self):
        """Update the files list display."""
        self.files_list.clear()
        
        # Update button states and visibility
        has_files = len(self.files) > 0
        
        # Show/hide files list, label, and clear button based on whether we have files
        if has_files:
            # Show files list and populate it
            self.files_list.setVisible(True)
            self.files_label.setVisible(True)
            self.clear_btn.setVisible(True)
            self.clear_btn.setEnabled(True)
                
            for file_path in self.files:
                item = QListWidgetItem(f"📄 {file_path.name}")
                item.setToolTip(str(file_path))
                self.files_list.addItem(item)
        else:
            # Hide files list, label, and clear button when empty
            self.files_list.setVisible(False)
            self.files_label.setVisible(False)
            self.clear_btn.setVisible(False)
        
        # Update drop area text
        if has_files:
            count_text = f"{len(self.files)} file(s) selected"
            if self.max_files > 0:
                count_text += f" (max {self.max_files})"
            
            # Create properly spaced text
            lines = []
            lines.append(f"📁 {count_text}")
            lines.append("")
            lines.append("Drop more files or click Browse")
            
            # Apply similar styling but with different content
            style = """
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    background-color: #f0f8f0;
                    color: #1B5E20;
                    font-size: 13px;
                    padding: 20px;
                    text-align: center;
                }
            """
            self.drop_area.setStyleSheet(style)
            self.drop_area.setText('\n'.join(lines))
        else:
            self.update_drop_area_style()
