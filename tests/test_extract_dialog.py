"""
Test for Extract Files Dialog functionality.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from ui.dialogs.extract_files_dialog import ExtractFilesDialog

@pytest.fixture(scope='session')
def qapp():
    """Create QApplication instance for tests."""
    if not QApplication.instance():
        app = QApplication([])
        yield app
        app.quit()
    else:
        yield QApplication.instance()

def test_extract_dialog_creation(qapp):
    """Test that the extract files dialog can be created."""
    dialog = ExtractFilesDialog()
    assert dialog is not None
    assert dialog.windowTitle() == "Extract Files from Image"

def test_extract_dialog_initial_state(qapp):
    """Test the initial state of the extract files dialog."""
    dialog = ExtractFilesDialog()
    
    # Check that initial state is correct
    assert dialog.stego_image_path is None
    assert dialog.output_directory is None
    assert not dialog.extract_button.isEnabled()
    assert dialog.progress_group.isHidden()

def test_extract_dialog_ready_state_validation(qapp):
    """Test the validation logic for enabling the extract button."""
    dialog = ExtractFilesDialog()
    
    # Initially not ready
    assert not dialog.extract_button.isEnabled()
    
    # Set image path but not output directory or password
    dialog.stego_image_path = "/test/path.png"
    dialog.check_ready_state()
    assert not dialog.extract_button.isEnabled()
    
    # Set output directory but no password
    dialog.output_directory = "/test/output"
    dialog.check_ready_state()
    assert not dialog.extract_button.isEnabled()
    
    # Set password but too short
    dialog.password_input.setText("12345")
    dialog.check_ready_state()
    assert not dialog.extract_button.isEnabled()
    
    # Set proper password
    dialog.password_input.setText("password123")
    dialog.check_ready_state()
    assert dialog.extract_button.isEnabled()

if __name__ == "__main__":
    pytest.main([__file__])
