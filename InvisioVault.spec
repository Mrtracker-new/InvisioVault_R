# -*- mode: python ; coding: utf-8 -*-
"""
InvisioVault PyInstaller Specification File
Advanced configuration for building Windows executable

Author: Rolan (RNR)
Purpose: Professional executable build for InvisioVault
"""

import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(os.getcwd())
assets_dir = project_root / "assets"

# Data files to include in the executable
datas = [
    # Include all assets
    (str(assets_dir / "icons"), "assets/icons"),
    (str(assets_dir / "images"), "assets/images"), 
    (str(assets_dir / "ui"), "assets/ui"),
    # Include essential docs
    (str(project_root / "README.md"), "."),
    (str(project_root / "LICENSE"), "."),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'PySide6.QtCore',
    'PySide6.QtWidgets', 
    'PySide6.QtGui',
    'numpy',
    'PIL',
    'PIL.Image',
    'cryptography.fernet',
    'cryptography.hazmat.primitives.kdf.pbkdf2',
    'cryptography.hazmat.primitives.hashes',
    'cryptography.hazmat.backends.openssl',
]

# Excluded modules to reduce file size
excludes = [
    'tkinter',
    'matplotlib',
    'scipy',
    'pandas',
    'jupyter',
    'IPython',
    'tornado',
    'zmq',
]

# Analysis of the main script
a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate entries
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create the executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='InvisioVault',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress the executable
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI application, no console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(assets_dir / "icons" / "InvisioVault.ico"),  # Application icon
    version='version_info.txt'  # Version information (will create this)
)

# Optional: Create a directory distribution instead of single file
# Uncomment the following for directory distribution (faster startup)
"""
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='InvisioVault'
)
"""
