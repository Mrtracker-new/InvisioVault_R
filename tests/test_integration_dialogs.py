"""
Integration test for Hide Files and Extract Files Dialogs.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from ui.dialogs.hide_files_dialog import HideFilesDialog
from ui.dialogs.extract_files_dialog import ExtractFilesDialog
from ui.main_window import MainWindow

@pytest.fixture(scope='session')
def qapp():
    """Create QApplication instance for tests."""
    if not QApplication.instance():
        app = QApplication([])
        yield app
        app.quit()
    else:
        yield QApplication.instance()

def test_both_dialogs_can_be_created(qapp):
    """Test that both dialogs can be created without errors."""
    # Test Hide Files Dialog
    hide_dialog = HideFilesDialog()
    assert hide_dialog is not None
    assert hide_dialog.windowTitle() == "Hide Files in Image"
    
    # Test Extract Files Dialog
    extract_dialog = ExtractFilesDialog()
    assert extract_dialog is not None
    assert extract_dialog.windowTitle() == "Extract Files from Image"

def test_main_window_can_create_both_dialogs(qapp):
    """Test that the main window can create both dialogs successfully."""
    main_window = MainWindow()
    assert main_window is not None
    
    # Test that the dialog creation methods exist
    assert hasattr(main_window, 'show_hide_dialog')
    assert hasattr(main_window, 'show_extract_dialog')
    assert callable(main_window.show_hide_dialog)
    assert callable(main_window.show_extract_dialog)

def test_dialog_initial_states_are_correct(qapp):
    """Test that both dialogs have correct initial states."""
    # Hide Files Dialog
    hide_dialog = HideFilesDialog()
    assert not hide_dialog.hide_button.isEnabled()  # Should be disabled initially
    assert hide_dialog.progress_group.isHidden()    # Progress should be hidden initially
    
    # Extract Files Dialog
    extract_dialog = ExtractFilesDialog()
    assert not extract_dialog.extract_button.isEnabled()  # Should be disabled initially
    assert extract_dialog.progress_group.isHidden()       # Progress should be hidden initially

if __name__ == "__main__":
    pytest.main([__file__])
