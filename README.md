# InvisioVault - Advanced Steganography Suite

**Professional-grade steganography application with AES-256 encryption, keyfile authentication, decoy mode, and multi-image distribution capabilities.**

## 🌟 Features

### Core Steganography
- **LSB (Least Significant Bit) technique** with randomized positioning
- Support for **PNG, BMP, and TIFF** lossless image formats
- Advanced **image capacity analysis** and suitability scoring
- **Compression detection** and optimization recommendations

### Security & Encryption
- **AES-256-CBC encryption** with PBKDF2-HMAC-SHA256 key derivation
- **Multiple security levels**: Standard (100k), High (500k), Maximum (1M+ iterations)
- **Keyfile-based two-factor authentication** for enhanced security
- **Secure password validation** with strength requirements

### Advanced Features
- **Decoy Mode**: Hide multiple datasets with plausible deniability
- **Two-Factor Steganography**: Distribute data across 2-8 images
- **Image Analysis Tool**: Entropy, noise level, and capacity assessment
- **Batch Processing**: Handle multiple files simultaneously

### User Interface
- **Modern PySide6 GUI** with dark/light theme support
- **Intuitive navigation** with operation-specific panels
- **Real-time progress tracking** and detailed logging
- **Responsive design** with drag & drop file handling

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

### Basic Usage

#### Hide Files in an Image
1. Launch InvisioVault
2. Select **"Basic Operations"** from the navigation panel
3. Click **"🔒 Hide Files"**
4. Select your carrier image (PNG/BMP/TIFF)
5. Choose files to hide
6. Set a strong password
7. Click **"Hide"** to create the steganographic image

#### Extract Files from an Image
1. Click **"🔓 Extract Files"**
2. Select the steganographic image
3. Enter the correct password
4. Choose output directory
5. Click **"Extract"** to recover your files

## 🔒 Security Features

### Encryption Security
- **256-bit AES encryption** in CBC mode
- **Cryptographically secure random** salt and IV generation
- **Memory protection** with automatic sensitive data clearing
- **Secure key derivation** with configurable iteration counts

### Steganography Security
- **Randomized LSB positioning** for enhanced concealment
- **Image integrity verification** with checksum validation
- **Capacity optimization** to avoid detection
- **Support for high-entropy images**

## 📊 Technical Specifications

| Feature | Specification |
|---------|---------------|
| **Encryption Algorithm** | AES-256-CBC |
| **Key Derivation** | PBKDF2-HMAC-SHA256 |
| **Iterations** | 100K - 1M+ (configurable) |
| **Supported Formats** | PNG, BMP, TIFF |
| **Maximum File Size** | Up to 50MB per operation |
| **Memory Usage** | < 500MB typical operations |
| **Platform Support** | Windows, macOS, Linux |

## 🧪 Testing

Run the test suite:
```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=ui --cov=utils
```

## 📚 Documentation

- **[User Guide](docs/user_guide.md)** - Complete user manual
- **[API Reference](docs/api_reference.md)** - Developer documentation
- **[Security Notes](docs/security_notes.md)** - Security considerations
- **[Changelog](docs/changelog.md)** - Version history

## 📄 License

This project is licensed under the **MIT License**.

## 🛡️ Security Notice

InvisioVault implements industry-standard cryptographic practices, but no software is 100% secure. For sensitive applications:
- Use strong, unique passwords
- Enable keyfile authentication
- Keep software updated
- Follow security best practices

**Important**: This software is for educational and legitimate privacy purposes only. Users are responsible for compliance with local laws and regulations.

---

**InvisioVault** - *Securing your data, one pixel at a time.*

© 2024 InvisioVault Team. All rights reserved.
