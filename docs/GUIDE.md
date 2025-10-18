# 📖 InVisioVault - The Complete Guide
### *Everything you need to know about advanced steganography*

**Author**: Rolan (RNR)  
**Email**: rolanlobo901@gmail.com  
**Version**: 1.0.0  
**Last Updated**: October 2025

---

## 👋 Welcome!

Hey there! I'm Rolan, and this is InVisioVault - my advanced steganography project that I've been developing to make file hiding both powerful and accessible. This guide contains everything you need to know about installing, using, and understanding the system.

Since I'm the solo developer on this project, I've written this guide in a straightforward, personal way. Think of it as me sitting next to you and explaining how everything works.

---

## 📋 Table of Contents

1. [🚀 Getting Started](#-getting-started) - Installation and first steps
2. [💡 What is InVisioVault?](#-what-is-invisiovault) - Core concepts and features
3. [📦 Installation](#-installation) - Step-by-step setup
4. [🎯 Basic Usage](#-basic-usage) - Hide and extract files
5. [⚡ Advanced Features](#-advanced-features) - Power user capabilities
6. [🏗️ Technical Architecture](#️-technical-architecture) - How it works
7. [🛡️ Security](#️-security) - Protection and best practices
8. [🐛 Troubleshooting](#-troubleshooting) - Common issues and solutions
9. [💻 For Developers](#-for-developers) - API and development guide
10. [📞 Support](#-support) - Getting help

---

## 🚀 Getting Started

### What You'll Need

**Minimum Requirements:**
- Windows 10+, macOS 12+, or Ubuntu 20.04+
- Python 3.8 or higher
- 4GB RAM (6GB+ recommended for multimedia files)
- 200MB free storage

**For Video/Audio Steganography:**
- FFmpeg installed on your system
- Additional 6GB+ RAM recommended

### Quick Install (5 Minutes)

```bash
# 1. Clone the repository
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R

# 2. Install (modern method - recommended)
pip install -e .

# 3. Launch the app
python main.py
```

That's it! The app should open and you're ready to hide files.

---

## 💡 What is InVisioVault?

InVisioVault is a steganography tool that lets you hide files inside images, videos, and audio files. Unlike simple password protection or encryption alone, steganography makes your hidden files *invisible* - nobody can tell there's anything hidden at all.

### Why I Built This

I wanted to create a tool that combines:
- **Military-grade encryption** (AES-256)
- **Revolutionary speed** (10-100x faster than traditional methods)
- **Advanced anti-detection** (evades analysis tools)
- **Plausible deniability** (decoy files for extra security)
- **Multimedia support** (images, videos, and audio)

### Key Features

#### 🎯 Core Capabilities
- **Hide any file type** inside images (PNG, BMP, TIFF)
- **Video steganography** (MP4, AVI, MKV, MOV)
- **Audio steganography** (MP3, WAV, FLAC, AAC)
- **Password protection** with AES-256 encryption
- **Two-factor authentication** using keyfiles

#### 🛡️ Advanced Security
- **Automatic decoy mode** - Every operation includes innocent-looking decoy files
- **Multi-layer hiding** - Hide multiple datasets with different passwords
- **Anti-detection measures** - Statistical masking to evade analysis tools
- **Secure memory management** - Sensitive data automatically cleared

#### ⚡ Performance
- **Revolutionary extraction speed** - 10-100x faster than traditional methods
- **Smart capacity detection** - Automatic image analysis
- **Batch processing** - Handle multiple files efficiently
- **Memory optimized** - Efficient algorithms for large files

---

## 📦 Installation

### Method 1: Modern Install (Recommended)

This is the method I recommend because it handles all dependencies automatically:

```bash
# Clone the repository
git clone https://github.com/Mrtracker-new/InVisioVault_R.git
cd InVisioVault_R

# Install in development mode
pip install -e .

# Launch
python main.py
```

**Benefits:**
- ✅ Automatic dependency resolution
- ✅ Easy updates with `git pull`
- ✅ Entry point support (run from anywhere)
- ✅ Future-proof packaging standards

### Method 2: Direct Download

1. Download ZIP from the GitHub repository
2. Extract to your desired location
3. Open terminal in the extracted folder
4. Run: `pip install -e .`
5. Launch: `python main.py`

### Method 3: Virtual Environment (Best for Development)

If you're going to modify the code or want isolated dependencies:

```bash
# Create virtual environment
python -m venv invisiovault-env

# Activate it
# Windows:
invisiovault-env\Scripts\activate
# macOS/Linux:
source invisiovault-env/bin/activate

# Install
pip install -e .

# Launch
python main.py
```

### Installing FFmpeg (For Video/Audio)

**Windows (using Chocolatey):**
```powershell
choco install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**CentOS/RHEL:**
```bash
sudo yum install epel-release
sudo yum install ffmpeg
```

### Verify Installation

```bash
# Test the import
python -c "import invisiovault; print('✅ Installation successful!')"

# Launch the GUI
python main.py
```

---

## 🎯 Basic Usage

### Hiding Files (Simple)

Here's the basic workflow I designed for hiding files:

1. **Launch the app**: Run `python main.py`
2. **Click "Hide Files"** in the main window
3. **Choose a cover image**: Select a PNG, BMP, or TIFF image
   - *Tip: Larger images hold more data*
4. **Select files to hide**: Choose any files you want to hide
   - The app shows how much capacity you have
5. **Set a password**: Use a strong password (8+ characters)
6. **Click "Hide"**: The app creates a new image with your files hidden inside

**That's it!** Your files are now hidden with military-grade encryption and invisible to detection tools.

### Extracting Files (Simple)

To get your files back:

1. **Click "Extract Files"**
2. **Choose the image** with hidden data
3. **Enter your password** (must be exact)
4. **Choose output folder** where files will be saved
5. **Click "Extract"**: Your files are recovered!

### Important Notes

- **Passwords are case-sensitive** - "Password" ≠ "password"
- **No password recovery** - If you forget it, the files are unrecoverable
- **Don't modify images** - Editing or converting the image destroys hidden data
- **Use lossless formats** - JPEG compression will destroy your hidden files

### Image Format Guide

| Format | Recommended | Why |
|--------|-------------|-----|
| 🟢 PNG | ✅ Excellent | Lossless, great capacity |
| 🟢 BMP | ✅ Excellent | Uncompressed, maximum quality |
| 🟢 TIFF | ✅ Excellent | Professional, lossless |
| 🔴 JPEG | ❌ Avoid | Lossy compression destroys data |
| 🔴 WEBP | ❌ Avoid | Compression artifacts |

### Capacity Planning

Approximate hiding capacity by image size:

| Image Resolution | Approximate Capacity |
|------------------|---------------------|
| 800x600 (480K pixels) | ~60KB files |
| 1920x1080 (2M pixels) | ~250KB files |
| 3840x2160 (8M pixels) | ~1MB files |
| 7680x4320 (33M pixels) | ~4MB files |

---

## ⚡ Advanced Features

### 🎭 Automatic Decoy Mode

This is one of my favorite features - every time you hide files, the system automatically creates TWO layers:

1. **Decoy Layer** (outer): Contains innocent-looking files (READMEs, configs, etc.)
2. **Real Layer** (inner): Contains your actual files

**How It Works:**
- Your password accesses your real files
- The decoy layer uses a derived password
- If discovered, you can reveal the decoy files instead
- This gives you **plausible deniability**

**Security Benefits:**
- ✅ Zero extra steps required
- ✅ Works automatically on every operation
- ✅ Each layer independently encrypted
- ✅ Backward compatible with old images

### 🔐 Two-Factor Authentication

For extra security, I've implemented keyfile-based 2FA:

1. **Generate a keyfile**: Security → Generate Keyfile
   - Creates a 256KB-1MB random file
2. **Hide with 2FA**: Enable keyfile option when hiding
   - Requires both password AND keyfile
3. **Extract with 2FA**: Provide both password and keyfile

**Important:**
- Store keyfiles separately from images
- Create multiple backups
- Both password AND keyfile required to extract

### 🎬 Multimedia Steganography

I've added support for hiding files in videos and audio:

#### Video Steganography
- **Supported formats**: MP4, AVI, MKV, MOV
- **How it works**: Hides data across video frames using LSB
- **Quality**: Preserves video quality (80%+ threshold)
- **Capacity**: Much larger than images

#### Audio Steganography
- **Supported formats**: MP3, WAV, FLAC, AAC
- **Multiple techniques**:
  - LSB embedding (simple, high capacity)
  - Spread spectrum (robust, lower capacity)
  - Phase coding (high fidelity, medium capacity)

### 🗂️ Multi-Decoy Mode (Advanced)

For power users who need more control:

1. **Access Multi-Decoy Dialog**: Advanced → Multi-Decoy
2. **Create multiple datasets**: Each with its own password
3. **Set priority levels**: 1 (outer) to 5 (inner)
4. **Configure types**: Innocent, Personal, Business
5. **Hide all layers**: System hides everything at once

**Use Cases:**
- Different passwords for different people
- Graduated access levels
- Complex decoy scenarios

**⚠️ Warning:** Multi-image distribution is risky - if you lose ANY fragment, you lose ALL data permanently!

---

## 🏗️ Technical Architecture

### How It Works

I designed InVisioVault with a layered architecture:

```
┌─────────────────────────────────────┐
│        UI Layer (PySide6)           │  ← User Interface
├─────────────────────────────────────┤
│      Operations Layer               │  ← Workflow Logic
├─────────────────────────────────────┤
│      Core Engines                   │  ← Algorithms
├─────────────────────────────────────┤
│      Utilities                      │  ← Helpers & Config
└─────────────────────────────────────┘
```

### Core Components

#### 1. Steganography Engine (`core/steganography_engine.py`)
- **LSB technique**: Hides data in least significant bits
- **Revolutionary optimization**: 10-100x faster extraction
- **Smart positioning**: Password-seeded randomization
- **Anti-detection**: Statistical masking

#### 2. Encryption Engine (`core/encryption_engine.py`)
- **AES-256-CBC**: Military-grade encryption
- **PBKDF2 key derivation**: 100K-1M+ iterations
- **Secure memory**: Automatic clearing of sensitive data
- **Metadata support**: Embedded file information

#### 3. Multi-Decoy Engine (`core/multi_decoy_engine.py`)
- **Layered hiding**: Multiple independent datasets
- **Priority system**: 5 security levels
- **Universal extraction**: Works with any password
- **Automatic mode**: Transparent dual-layer protection

#### 4. Multimedia Engines
- **Video Engine**: Frame-based LSB with OpenCV/FFmpeg
- **Audio Engine**: Multiple techniques (LSB, spread spectrum, phase)
- **Analyzer**: Capacity and quality assessment

### Project Structure

```
InVisioVault/
├── main.py                    # Application entry point
├── requirements.txt           # Dependencies
├── setup.py                   # Installation config
│
├── core/                      # Core algorithms
│   ├── steganography_engine.py
│   ├── encryption_engine.py
│   ├── multi_decoy_engine.py
│   ├── video_steganography_engine.py
│   └── audio_steganography_engine.py
│
├── ui/                        # User interface
│   ├── main_window.py
│   ├── dialogs/               # Dialog windows
│   ├── components/            # Reusable widgets
│   └── themes/                # Dark theme
│
├── operations/                # High-level operations
│   ├── hide_operation.py
│   ├── extract_operation.py
│   └── analysis_operation.py
│
├── utils/                     # Utilities
│   ├── config_manager.py
│   ├── logger.py
│   └── file_utils.py
│
├── tests/                     # Test suite
└── docs/                      # Documentation (this file!)
```

### The Performance Revolution

The original steganography methods were SLOW because they had to:
1. Try thousands of possible positions
2. Check each candidate pixel
3. Reconstruct and validate repeatedly

**My optimization:**
- Direct size reading eliminates guesswork
- Pre-computed positions using password seed
- Single-pass extraction
- Result: **10-100x faster** (30+ seconds → 1-5 seconds)

---

## 🛡️ Security

### Security Architecture

I designed InVisioVault with defense-in-depth:

1. **Cryptographic Layer**: AES-256 encryption
2. **Steganographic Layer**: Randomized LSB hiding
3. **Access Control**: Password + optional keyfile
4. **Operational Security**: Secure logging and memory management
5. **Plausible Deniability**: Automatic decoy protection

### Password Security

**Strong Password Guidelines:**

✅ **Do:**
- Use 12+ characters
- Mix uppercase, lowercase, numbers, symbols
- Use unique passphrases
- Avoid personal information

❌ **Don't:**
- Use dictionary words
- Use personal info (names, dates)
- Reuse passwords
- Use short passwords (<8 characters)

**Password Strength Indicator:**
- 🔴 Very Weak: <8 characters
- 🟠 Weak: 8-10 characters
- 🟡 Moderate: 10-12 characters
- 🟢 Strong: 12+ characters
- 🔵 Very Strong: 15+ characters

### Security Levels

| Level | Iterations | Use Case | Speed |
|-------|-----------|----------|-------|
| **Standard** | 100,000 | General use | Fast |
| **High** | 500,000 | Sensitive data | Medium |
| **Maximum** | 1,000,000+ | Highly sensitive | Slower |

### Best Practices

**File Management:**
- ✅ Use unique passwords for different operations
- ✅ Store keyfiles separately from images
- ✅ Backup steganographic images securely
- ✅ Test extraction before relying on hidden data
- ✅ Use decoy data for plausible deniability

**Operational Security:**
- ✅ Use private, secure computers
- ✅ Avoid public/shared systems
- ✅ Clear temporary files after operations
- ✅ Use secure deletion for sensitive files

**Image Selection:**
- ✅ High resolution images (2MP+)
- ✅ Natural photographs with varied colors
- ✅ Lossless formats (PNG, BMP, TIFF)
- ❌ Avoid low resolution (<1MP)
- ❌ Avoid simple graphics or logos
- ❌ Never use JPEG or compressed formats

### Development Security (For Contributors)

**Never commit to Git:**
- ❌ Private keys, keyfiles, certificates
- ❌ Passwords, API keys, credentials
- ❌ Personal files and test outputs
- ❌ Config files with secrets

**Secure coding practices:**
- ✅ Use environment variables for secrets
- ✅ Validate all inputs
- ✅ Never log sensitive data
- ✅ Use secure random number generation

---

## 🐛 Troubleshooting

### Common Issues

#### "Image too small for data"

**Problem**: The image doesn't have enough capacity.

**Solutions:**
1. Use a higher resolution image
2. Compress files before hiding
3. Split data across multiple images
4. Check capacity with the analyzer

#### "Wrong password or corrupted data"

**Problem**: Extraction failed.

**Solutions:**
1. Verify password spelling and case
2. Check if keyfile is required
3. Ensure image hasn't been modified
4. Try analyzing the image first

#### "Application crashes during operation"

**Problem**: System instability.

**Solutions:**
1. Check available memory (need 4GB+ free)
2. Close other applications
3. Reduce processing chunk size in settings
4. Update dependencies: `pip install -r requirements.txt --upgrade`

#### Slow Performance

**Causes and fixes:**
1. **Large files**: Enable chunked processing
2. **Low memory**: Increase memory limit in settings
3. **CPU bottleneck**: Enable multi-threading
4. **Storage I/O**: Use SSD instead of HDD

### Multimedia Issues

#### FFmpeg Not Found

```bash
# Windows
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

#### Audio Processing Fails

```bash
# Install audio dependencies
pip install librosa --upgrade
pip install pydub --upgrade
pip install scipy --upgrade
```

### Log Files

Check logs for detailed error information:

**Location:**
- Windows: `%USERPROFILE%\.invisiovault\logs\`
- macOS/Linux: `~/.invisiovault/logs/`

**View logs:**
```bash
# Recent errors
tail -f ~/.invisiovault/logs/invisiovault.log

# Search errors
grep "ERROR" ~/.invisiovault/logs/invisiovault.log
```

---

## 💻 For Developers

### API Quick Start

```python
from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel
from pathlib import Path

# Initialize engines
stego_engine = SteganographyEngine()
encrypt_engine = EncryptionEngine(SecurityLevel.HIGH)

# Hide data
carrier = Path("carrier.png")
data = b"Secret message!"
password = "MySecurePassword123"

# Encrypt
encrypted = encrypt_engine.encrypt_with_metadata(data, password)

# Hide in image
success = stego_engine.hide_data_with_password(
    carrier_path=carrier,
    data=encrypted,
    output_path=Path("hidden.png"),
    password=password
)

# Extract data
extracted = stego_engine.extract_data_with_password(
    stego_path=Path("hidden.png"),
    password=password
)

# Decrypt
decrypted = encrypt_engine.decrypt_with_metadata(extracted, password)
print(decrypted)  # b"Secret message!"
```

### Core Classes

#### SteganographyEngine

```python
class SteganographyEngine:
    def hide_data_with_password(self, carrier_path, data: bytes, 
                                output_path, password: str) -> bool:
        """Hide data in image with password protection."""
        
    def extract_data_with_password(self, stego_path, 
                                   password: str) -> bytes:
        """Extract hidden data from image."""
```

#### EncryptionEngine

```python
class EncryptionEngine:
    def encrypt_with_metadata(self, data: bytes, 
                             password: str) -> bytes:
        """Encrypt data with AES-256."""
        
    def decrypt_with_metadata(self, encrypted: bytes, 
                             password: str) -> bytes:
        """Decrypt data."""
```

#### MultiDecoyEngine

```python
class MultiDecoyEngine:
    def hide_multiple_datasets(self, carrier_path, 
                              datasets: List[Dict]) -> bool:
        """Hide multiple datasets with different passwords."""
        
    def extract_dataset(self, stego_path, password: str) -> Dict:
        """Extract dataset matching password."""
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_steganography.py -v

# Test with coverage
python -m pytest tests/ --cov=core --cov-report=html
```

### Contributing

If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests
5. Submit a pull request

**Code standards:**
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type annotations
- Write tests for new features

---

## 📞 Support

### Getting Help

1. **Check this guide first** - Most answers are here
2. **Review logs** - Error messages often explain the issue
3. **GitHub Issues** - Report bugs or request features
4. **Email me** - rolanlobo901@gmail.com

### Useful Resources

- **GitHub Repository**: [InVisioVault_R](https://github.com/Mrtracker-new/InVisioVault_R)
- **Email Support**: rolanlobo901@gmail.com

### Educational Purpose

InVisioVault is designed for **educational purposes**. I built it to:
- Learn advanced cryptography and steganography
- Demonstrate professional software architecture
- Teach secure coding practices
- Explore performance optimization

**Always use responsibly and in compliance with local laws!**

---

## 🎯 Quick Reference

### Command Cheat Sheet

```bash
# Installation
pip install -e .

# Launch
python main.py

# Update
git pull origin main
pip install -e . --upgrade

# Run tests
python -m pytest tests/ -v

# Check logs
tail -f ~/.invisiovault/logs/invisiovault.log
```

### Keyboard Shortcuts

- `Ctrl+H` - Hide files dialog
- `Ctrl+E` - Extract files dialog
- `Ctrl+A` - Analyze image
- `Ctrl+S` - Settings
- `Ctrl+Q` - Quit

### File Format Support

**Images**: PNG ✅ | BMP ✅ | TIFF ✅ | JPEG ❌  
**Video**: MP4 ✅ | AVI ✅ | MKV ✅ | MOV ✅  
**Audio**: WAV ✅ | FLAC ✅ | MP3 ✅ | AAC ✅

---

## 🙏 Final Notes

Thanks for using InVisioVault! I've put a lot of work into making this tool both powerful and easy to use. 

If you have any questions, suggestions, or just want to say hi, feel free to reach out at rolanlobo901@gmail.com.

Remember: This is an educational project. Use it to learn about steganography, cryptography, and secure software development. Always respect privacy laws and use the tool responsibly.

Happy hiding! 🎭

— Rolan (RNR)

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**License**: MIT Educational License  
**Author**: Rolan (RNR)  
**Email**: rolanlobo901@gmail.com
