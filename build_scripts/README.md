# InvisioVault Build Scripts

This directory contains various scripts for building the InvisioVault executable. All scripts are designed to create a professional, standalone Windows executable that requires no Python installation on target systems.

## Available Build Scripts

### 1. Advanced Python Builder ⭐ **Recommended**
**File:** `build_executable.py`

The most feature-rich and flexible build script with professional output and validation.

**Usage:**
```bash
# Standard build with cleanup
python build_scripts/build_executable.py --clean

# Debug version with console window
python build_scripts/build_executable.py --clean --debug

# Directory distribution (faster startup)
python build_scripts/build_executable.py --clean --onedir

# Verbose output for troubleshooting
python build_scripts/build_executable.py --clean --verbose

# View all options
python build_scripts/build_executable.py --help
```

**Features:**
- ✅ Automated dependency checking
- ✅ Professional build validation
- ✅ Advanced PyInstaller configuration via spec file
- ✅ Build artifact cleanup
- ✅ Detailed progress reporting
- ✅ File size optimization
- ✅ Version information embedding

### 2. Quick Batch Script
**File:** `build_exe.bat`

Simple Windows batch file for quick builds. Uses the advanced Python builder internally.

**Usage:**
- Double-click the file in Windows Explorer, or
- Run `build_scripts\build_exe.bat` from command prompt

### 3. Quick PowerShell Script
**File:** `build_exe.ps1`

PowerShell version of the quick build script. Uses the advanced Python builder internally.

**Usage:**
- Right-click → "Run with PowerShell", or
- Run `powershell -ExecutionPolicy Bypass -File build_scripts\build_exe.ps1`

## Build Output

All scripts create the executable at:
```
dist/InvisioVault.exe
```

**Executable Specifications:**
- **Type:** Single-file standalone executable
- **Size:** ~66-70 MB (optimized with UPX compression)
- **Icon:** Embedded application icon
- **Version Info:** Professional Windows version metadata
- **Dependencies:** All bundled (numpy, PySide6, cryptography, etc.)
- **Target:** Windows GUI application (no console window)

## Build Requirements

- Python 3.8+ (tested with 3.12)
- PyInstaller 6.0+ (auto-installed if missing)
- All project dependencies installed

## Troubleshooting

### Common Issues

1. **Build fails with import errors:**
   ```bash
   python build_scripts/build_executable.py --clean --verbose
   ```
   Use verbose mode to see detailed import resolution.

2. **Missing dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Icon not found warnings:**
   Ensure `assets/icons/InvisioVault.ico` exists.

4. **Large executable size:**
   The size is normal due to bundled GUI framework and scientific libraries.

### Build Files

The build process uses these configuration files:
- `InvisioVault.spec` - Advanced PyInstaller configuration
- `version_info.txt` - Windows version metadata
- `MANIFEST.in` - Package data specification

## Distribution

The generated `InvisioVault.exe` is fully portable and can be:
- ✅ Copied to any Windows computer
- ✅ Run without Python installation
- ✅ Distributed as a single file
- ✅ Used on Windows 10/11 systems

## Recommendation

For most users, use the **Advanced Python Builder**:
```bash
python build_scripts/build_executable.py --clean
```

It provides the best combination of features, error handling, and professional output.

---

## 👨‍💻 Author Information

**InvisioVault** is developed by **Rolan (RNR)** as an educational project for advanced steganography and cybersecurity learning.

- **Author**: Rolan (RNR)  
- **Contact**: [rolanlobo901@gmail.com](mailto:rolanlobo901@gmail.com)
- **Purpose**: Educational and research project
- **License**: MIT Educational License

For questions, support, or collaboration opportunities, please reach out via email or through the GitHub repository.
