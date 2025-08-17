"""
Unicode Polyglot Integration for InVisioVault
=============================================

This module integrates the Unicode RTL polyglot method into the main InVisioVault application.
It adds a menu option to launch the Unicode RTL polyglot dialog, which provides a specialized
interface for creating executables disguised as PNG images using Unicode Right-to-Left
Override (RLO) characters.

Integration points:
- Menu option in the Tools menu
- Direct access through the Self-Executing dialog
- Standalone dialog access

Author: InVisioVault Integration Team
"""

from ui.dialogs.unicode_polyglot_dialog import UnicodePolyglotDialog

def show_unicode_polyglot_dialog(parent=None):
    """
    Show the Unicode RTL polyglot dialog.
    
    Args:
        parent: Parent widget (optional)
    
    Returns:
        Dialog execution result
    """
    dialog = UnicodePolyglotDialog(parent)
    return dialog.exec()
