"""
Progress Dialog Component
Modern progress dialog with status updates and cancellation support.
"""

from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QFrame
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QMovie

from utils.logger import Logger


class ProgressDialog(QDialog):
    """Modern progress dialog with status updates and optional cancellation."""
    
    cancelled = Signal()  # Emitted when user cancels operation
    
    def __init__(self, 
                 title: str = "Operation in Progress",
                 message: str = "Please wait...",
                 can_cancel: bool = True,
                 parent=None):
        super().__init__(parent)
        
        self.logger = Logger()
        self.can_cancel = can_cancel
        self.is_cancelled = False
        
        self.init_ui(title, message)
        self.setup_connections()
        
        # Make dialog modal
        self.setModal(True)
        
        self.logger.debug(f"Progress dialog initialized: {title}")
    
    def init_ui(self, title: str, message: str):
        """Initialize the user interface."""
        self.setWindowTitle(title)
        self.setFixedSize(450, 250)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # Title label
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Status message
        self.status_label = QLabel(message)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #555; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Detailed status (initially hidden)
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(80)
        self.details_text.setReadOnly(True)
        self.details_text.setVisible(False)
        self.details_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.details_text)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        
        # Show/Hide details button
        self.details_btn = QPushButton("Show Details")
        self.details_btn.clicked.connect(self.toggle_details)
        self.details_btn.setVisible(False)  # Hidden until details are added
        button_layout.addWidget(self.details_btn)
        
        button_layout.addStretch()
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_operation)
        self.cancel_btn.setVisible(self.can_cancel)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Add some stretch to push everything to the top
        layout.addStretch()
    
    def setup_connections(self):
        """Setup signal connections."""
        pass
    
    def update_progress(self, value: int, message: str = None):
        """Update progress bar and optional status message."""
        self.progress_bar.setValue(min(max(value, 0), 100))
        
        if message:
            self.status_label.setText(message)
        
        # Force UI update
        self.repaint()
    
    def set_status(self, message: str):
        """Update the status message."""
        self.status_label.setText(message)
        self.repaint()
    
    def add_detail(self, detail: str):
        """Add a detail line to the details text area."""
        self.details_text.append(detail)
        
        # Show details button if not already visible
        if not self.details_btn.isVisible():
            self.details_btn.setVisible(True)
    
    def toggle_details(self):
        """Toggle the visibility of the details text area."""
        is_visible = self.details_text.isVisible()
        self.details_text.setVisible(not is_visible)
        
        if is_visible:
            self.details_btn.setText("Show Details")
            self.setFixedSize(450, 250)
        else:
            self.details_btn.setText("Hide Details")
            self.setFixedSize(450, 350)
    
    def set_indeterminate(self, indeterminate: bool = True):
        """Set progress bar to indeterminate mode."""
        if indeterminate:
            self.progress_bar.setRange(0, 0)  # Indeterminate mode
        else:
            self.progress_bar.setRange(0, 100)  # Normal mode
    
    def cancel_operation(self):
        """Handle cancel button click."""
        self.is_cancelled = True
        self.cancel_btn.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)
        self.set_status("Cancelling operation...")
        self.cancelled.emit()
        
        self.logger.info("User cancelled operation")
    
    def operation_completed(self, success: bool = True, message: str = None):
        """Mark operation as completed."""
        if success:
            self.progress_bar.setValue(100)
            if message:
                self.set_status(message)
            else:
                self.set_status("Operation completed successfully!")
        else:
            if message:
                self.set_status(f"Operation failed: {message}")
            else:
                self.set_status("Operation failed!")
        
        # Hide cancel button, show close
        self.cancel_btn.setText("Close")
        self.cancel_btn.clicked.disconnect()
        self.cancel_btn.clicked.connect(self.accept)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setVisible(True)
        
        # Update button style for completion
        if success:
            self.cancel_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        if not self.is_cancelled and self.can_cancel:
            # If operation is still running, ask for confirmation
            if self.progress_bar.value() < 100:
                self.cancel_operation()
        
        event.accept()
