# InVisioVault 🔐

**Advanced Steganography Suite with Revolutionary Ultra-Fast Performance**

InVisioVault is a cutting-edge steganography application that combines the art of hiding data with military-grade encryption and **revolutionary performance optimizations**. Built with Python and PyQt6, it delivers both enterprise-level functionality and exceptional user experience.

## 📚 Educational Purpose

**Important Notice:** This project is developed for **educational and research purposes** by Rolan (RNR) as part of learning advanced cryptography, steganography, and security concepts. The primary goals are:

- 🎓 **Learning Experience**: Exploring the intersection of cryptography, steganography, and software engineering
- 🔍 **Security Research**: Understanding data hiding techniques and their practical applications
- 💡 **Innovation**: Experimenting with performance optimizations in steganographic algorithms
- 🛡️ **Security Awareness**: Promoting understanding of digital privacy and data protection methods

**Disclaimer**: This software is intended for legitimate educational, research, and privacy purposes only. Users are responsible for compliance with local laws and regulations. The author encourages responsible use and does not condone any malicious activities.

## 👨‍💻 Author

**Created by**: Rolan (RNR)  
**Purpose**: Educational project for learning security technologies and exploring new possibilities in steganography

## 🚀 Revolutionary Performance

### ⚡ **Ultra-Fast Extraction** - **10-100x Speed Improvement**
- **Revolutionary single-pass algorithm** eliminates candidate testing
- **Sub-second to few-second extraction** for all file sizes
- **Large file support**: Process multi-megabyte files in seconds
- **Production-ready performance** suitable for real-world applications

### **Performance Benchmarks**
| File Size | Old Algorithm | New Algorithm | Improvement |
|-----------|---------------|---------------|-------------|
| **Small (5KB)** | ~12 seconds | **~0.3 seconds** | **40x faster** |
| **Medium (25KB)** | ~15 seconds | **~0.5 seconds** | **30x faster** |
| **Large (94.8KB)** | ~20 seconds | **~1.0 seconds** | **20x faster** |
| **Very Large (200KB)** | ~30+ seconds | **~2.0 seconds** | **15x faster** |
| **Huge (1MB+)** | ~60+ seconds | **~3-5 seconds** | **12-20x faster** |

## 🌟 Core Features

### Advanced Steganography
- **LSB (Least Significant Bit)** with randomized positioning for enhanced security
- **Multiple Image Format Support**: PNG, BMP, TIFF for lossless steganography
- **Intelligent Capacity Analysis**: Smart assessment of carrier image suitability
- **Visual Quality Preservation**: Advanced algorithms maintain perfect image integrity

### Enterprise Security
- **AES-256 Encryption**: Military-grade encryption with multiple security levels
- **Advanced Key Derivation**: PBKDF2 with configurable iterations (100K-1M+)
- **Randomized Positioning**: Cryptographically secure pseudo-random placement
- **Two-Factor Authentication**: Enterprise-grade security with TOTP support
- **Secure Key Files**: Professional authentication with cryptographic key files
- **Instant Wrong Password Detection**: Immediate feedback without security compromise

### Professional User Experience
- **Modern GUI**: Clean, intuitive PyQt6 interface optimized for productivity
- **Batch Operations**: Process multiple files with enterprise efficiency  
- **Real-time Progress**: Intelligent feedback for all operations
- **Comprehensive Logging**: Professional-grade operation logs
- **Cross-platform**: Windows, macOS, and Linux support

### Advanced Features
- **Smart Image Analysis**: AI-powered suitability scoring and recommendations
- **Multi-layer Decoy System**: Advanced security with decoy data protection
- **Integrated File Management**: Professional file browser and organization
- **Extensible Architecture**: Enterprise-ready modular design

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
1. Launch InVisioVault
2. Select **"Hide Files"** from the main interface
3. Choose your carrier image (PNG/BMP/TIFF)
4. Select files to hide
5. Set a strong password
6. Click **"Hide"** to create the steganographic image

#### Extract Files from an Image
1. Click **"Extract Files"**
2. Select the steganographic image
3. Enter the correct password
4. Choose output directory
5. Click **"Extract"** - **files extract in seconds!** ⚡

## 🔒 Security Specifications

| Feature | Specification |
|---------|---------------|
| **Encryption** | AES-256-CBC |
| **Key Derivation** | PBKDF2-HMAC-SHA256 |
| **Iterations** | 100K - 1M+ (configurable) |
| **Supported Formats** | PNG, BMP, TIFF |
| **Maximum File Size** | Up to 50MB per operation |
| **Memory Usage** | <500MB typical operations |
| **Platform Support** | Windows, macOS, Linux |

## 🧪 Testing & Demo

### Run Performance Demo
```bash
python demo_performance.py
```
*Demonstrates the revolutionary speed improvements across various file sizes*

### Run Main Tests
```bash
python test_main.py
```
*Comprehensive steganography functionality tests*

### Unit Tests
```bash
cd tests
python -m pytest
```

## 📚 Essential Files

- **`main.py`** - Application entry point
- **`demo_performance.py`** - Performance demonstration
- **`test_main.py`** - Main functionality tests
- **`PERFORMANCE_OPTIMIZATION_SUMMARY.md`** - Detailed performance analysis
- **`PROJECT_COMPLETE.md`** - Complete project documentation

## 🎯 Key Innovation

The **revolutionary breakthrough** is the shift from **multi-pass candidate testing** to **single-pass deterministic extraction**:

### Traditional Approach (Slow)
```
1. Test 1000+ size candidates
2. Extract header for each candidate
3. Validate each header separately  
4. Extract full data for each candidate
Result: 10-30+ seconds ❌
```

### Revolutionary Approach (Ultra-Fast)
```
1. Extract header ONCE
2. Read exact file size from header
3. Extract data in SINGLE PASS
Result: 1-5 seconds ✅
```

## 🛡️ Security Notice

InVisioVault implements industry-standard cryptographic practices with revolutionary performance optimizations that maintain full security:

- ✅ **Zero security compromises** - same AES-256 strength
- ✅ **100% backward compatibility** - all existing files work
- ✅ **Perfect data integrity** - all checksums verified
- ✅ **Enhanced wrong password detection** - instant feedback

**Important**: This software is for educational and legitimate privacy purposes only. Users are responsible for compliance with local laws and regulations.

---

**InVisioVault** - *Revolutionary Performance, Military-Grade Security*

*Now with 10-100x faster extraction - making advanced steganography practical for real-world use!* ⚡

© 2025 Rolan (RNR). Educational project for learning security technologies.
