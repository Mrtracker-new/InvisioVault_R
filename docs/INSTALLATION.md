# 📦 InVisioVault Installation Guide
### *Complete Setup Instructions for All Platforms*

**Version**: 1.0.1  
**Author**: Rolan (RNR) - [rolanlobo901@gmail.com](mailto:rolanlobo901@gmail.com)  
**Last Updated**: January 2025  
**Compatibility**: Python 3.8+ on Windows, macOS, and Linux

---

## 🎯 Quick Installation

### **🚀 Recommended Method (Modern)**

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R

# 2️⃣ Install in development mode
pip install -e .

# 3️⃣ Launch the application
python main.py
```

**Why this method?**
- ✅ **Automatic dependency resolution**
- ✅ **Entry point support** (`invisiovault` command)
- ✅ **Development-friendly** (changes reflect immediately)
- ✅ **Future-proof** packaging standards

---

## 📋 System Requirements

### **Minimum Requirements**
- **Operating System**: Windows 10+, macOS 12+, Ubuntu 20.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum
- **Storage**: 100MB for installation
- **Display**: 1024x768 minimum resolution

### **Recommended Specifications**
- **Python**: 3.11+ (latest stable)
- **RAM**: 8GB+ for large file operations
- **Storage**: 1GB+ for working space
- **Display**: 1920x1080+ for optimal UI experience

---

## 🛠️ Installation Methods

### **Method 1: Modern Package Install (Recommended)**

#### **Step-by-Step**
```bash
# Download the project
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R

# Modern installation
pip install -e .

# Verify installation
python -c "import invisiovault; print('✅ Installation successful!')"

# Launch application
python main.py
```

#### **Using Entry Points**
After installation, you can run from anywhere:
```bash
# GUI version
invisiovault

# CLI version (if available)
invisiovault-cli
```

### **Method 2: Direct Download + Package Install**

1. **Download**: Go to [GitHub repository](https://github.com/Mrtracker-new/InVisioVault_R)
2. **Extract**: Download ZIP and extract to desired location
3. **Install**: Open terminal in extracted folder
4. **Run**: `pip install -e .`
5. **Launch**: `python main.py`

### **Method 3: Legacy Installation (Fallback)**

If the modern method fails for any reason:

```bash
# Clone or download project
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R

# Install dependencies manually
pip install -r requirements.txt

# Launch application
python main.py
```

### **Method 4: Virtual Environment (Recommended for Development)**

```bash
# Create virtual environment
python -m venv invisiovault-env

# Activate environment
# Windows:
invisiovault-env\Scripts\activate
# macOS/Linux:
source invisiovault-env/bin/activate

# Install project
pip install -e .

# Launch
python main.py
```

---

## 🔧 Platform-Specific Instructions

### **Windows Installation**

#### **Prerequisites**
```powershell
# Check Python version
python --version

# If Python not installed, download from:
# https://python.org/downloads/
```

#### **Installation**
```powershell
# Open PowerShell or Command Prompt
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R

# Install (may require elevated permissions)
pip install -e .

# Launch
python main.py
```

#### **Common Windows Issues**
- **Permission Denied**: Run terminal as Administrator
- **Git not found**: Install Git from [git-scm.com](https://git-scm.com)
- **pip not found**: Reinstall Python with pip included

### **macOS Installation**

#### **Prerequisites**
```bash
# Check Python version
python3 --version

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python (if needed)
brew install python
```

#### **Installation**
```bash
# Clone repository
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R

# Install using pip3
pip3 install -e .

# Launch
python3 main.py
```

### **Linux Installation**

#### **Ubuntu/Debian**
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv git

# Clone and install
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R
pip3 install -e .

# Launch
python3 main.py
```

#### **CentOS/RHEL**
```bash
# Install Python and Git
sudo yum install python3 python3-pip git

# Clone and install
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R
pip3 install -e .

# Launch
python3 main.py
```

#### **Arch Linux**
```bash
# Install dependencies
sudo pacman -S python python-pip git

# Clone and install
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R
pip install -e .

# Launch
python main.py
```

---

## 🐛 Troubleshooting

### **Common Installation Issues**

#### **"pip install -e ." fails**
```bash
# Try upgrading pip first
python -m pip install --upgrade pip

# Or use --user flag
pip install -e . --user

# Or try legacy method
pip install -r requirements.txt
```

#### **Permission Denied Errors**
```bash
# Windows: Run as Administrator
# macOS/Linux: Use --user flag
pip install -e . --user

# Or use virtual environment (recommended)
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -e .
```

#### **"Command not found: git"**
- **Windows**: Install Git from [git-scm.com](https://git-scm.com)
- **macOS**: Install Xcode Command Line Tools: `xcode-select --install`
- **Linux**: Install via package manager: `sudo apt install git`

#### **Python Version Issues**
```bash
# Check Python version
python --version

# If version < 3.8, update Python
# Windows: Download from python.org
# macOS: brew install python
# Linux: Use package manager
```

### **Dependency Issues**

#### **PySide6 Installation Problems**
```bash
# Try installing PySide6 separately
pip install PySide6>=6.5.0

# If still fails, try Qt5 version (compatibility)
pip install PySide2
```

#### **NumPy/Pillow Issues**
```bash
# Install system dependencies first
# Ubuntu/Debian:
sudo apt install python3-dev libjpeg-dev zlib1g-dev

# macOS:
brew install libjpeg zlib

# Then reinstall
pip install numpy Pillow --upgrade
```

#### **Cryptography Issues**
```bash
# Install system dependencies
# Ubuntu/Debian:
sudo apt install build-essential libffi-dev libssl-dev

# macOS:
brew install libffi openssl

# CentOS/RHEL:
sudo yum install gcc openssl-devel libffi-devel
```

---

## ✅ Verification

### **Test Installation**
```bash
# Basic import test
python -c "import invisiovault; print('✅ Import successful')"

# Check entry points
invisiovault --help 2>/dev/null && echo "✅ Entry points working" || echo "ℹ️ Entry points not available (use python main.py)"

# Launch GUI
python main.py
```

### **Run Test Suite**
```bash
# Run basic tests
python -m pytest tests/ -v

# Quick functionality test
python test_main.py

# Performance benchmark
python demo_performance.py
```

---

## 🔄 Updating InVisioVault

### **Update to Latest Version**
```bash
# Navigate to project directory
cd InVisioVault_R

# Pull latest changes
git pull origin main

# Update dependencies
pip install -e . --upgrade

# Or reinstall completely
pip uninstall invisiovault
pip install -e .
```

---

## 🏗️ Development Setup

### **For Contributors/Developers**

```bash
# Clone repository
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # Windows: dev-env\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install additional development tools
pip install -e ".[test]"

# Run tests
python -m pytest tests/ -v

# Code formatting
black . --line-length 88

# Linting
flake8 . --max-line-length=88
```

---

## 📂 Directory Structure After Installation

```
📁 InVisioVault/
├── 🚀 main.py                    # Launch the application
├── 📋 requirements.txt           # Python dependencies  
├── ⚙️ setup.py                   # Package configuration
├── 📁 assets/                    # Application assets
├── 🔧 core/                      # Steganography engines
├── 📚 docs/                      # Documentation
├── ⚡ operations/                 # Core operations
├── 🎨 ui/                        # User interface
├── 🔧 utils/                     # Utility modules
└── 🧪 tests/                     # Test suite
```

---

## ❓ Need Help?

### **Getting Support**

1. **Check Documentation**: Review [User Guide](user_guide.md) first
2. **Troubleshooting**: Consult this guide's troubleshooting section
3. **GitHub Issues**: Report bugs or request features
4. **Email**: Contact [rolanlobo901@gmail.com](mailto:rolanlobo901@gmail.com)

### **Useful Resources**

- **📖 User Manual**: [`docs/user_guide.md`](user_guide.md)
- **🔧 API Reference**: [`docs/api_reference.md`](api_reference.md)
- **🛡️ Security Guide**: [`docs/security_notes.md`](security_notes.md)
- **📋 Changelog**: [`docs/changelog.md`](changelog.md)

---

## 🎉 Success!

Once installed, you should be able to:

- ✅ Launch InVisioVault with `python main.py`
- ✅ Use entry points: `invisiovault` (if supported)
- ✅ Access all features through the GUI
- ✅ Run tests and benchmarks
- ✅ Update easily with `git pull`

**Congratulations!** You're ready to start using InVisioVault for secure steganography operations.

---

**Last Updated**: January 2025  
**Version**: 1.0.1  
**Author**: Rolan (RNR)  
**License**: MIT Educational License
