# 📖 InvisioVault User Manual
### *Complete Guide to Advanced Steganography*

**Version**: 1.0.0  
**Author**: Rolan (RNR)  
**Purpose**: Comprehensive user guide for InvisioVault steganography suite  

---

## 🎯 Table of Contents

1. [**Getting Started**](#-getting-started)
2. [**Basic Operations**](#-basic-operations)
3. [**Advanced Features**](#-advanced-features)
4. [**Security Features**](#-security-features)
5. [**Performance Guide**](#-performance-guide)
6. [**Troubleshooting**](#-troubleshooting)
7. [**Best Practices**](#-best-practices)

---

## 🚀 Getting Started

### **System Requirements**

| **Component** | **Requirement** | **Recommended** |
|---------------|-----------------|------------------|
| **Operating System** | Windows 10+, macOS 12+, Ubuntu 20.04+ | Latest versions |
| **Python** | 3.8+ | Python 3.11+ |
| **RAM** | 4GB minimum | 8GB+ for large files |
| **Storage** | 100MB for installation | 1GB+ for operations |
| **Display** | 1024x768 minimum | 1920x1080+ |

### **Installation Guide**

#### **Method 1: Git Clone (Recommended)**
```bash
# Clone the repository
git clone https://github.com/Mrtracker-new/InvisioVault_R.git
cd InvisioVault_R

# Install dependencies
pip install -r requirements.txt

# Launch application
python main.py
```

#### **Method 2: Direct Download**
1. Download ZIP from GitHub repository
2. Extract to desired location
3. Open terminal in extracted folder
4. Run installation commands above

#### **Method 3: Package Installation**
```bash
# Install as package
pip install -e .

# Run from anywhere
invisiovault
```

### **First Launch**

When you first launch InvisioVault:

1. **Welcome Screen**: Brief introduction to features
2. **Settings Initialization**: Default configuration setup
3. **Security Check**: Validates system compatibility
4. **UI Theme**: Choose between Dark/Light modes

---

## 🔧 Basic Operations

### **Hiding Files**

#### **Step-by-Step Guide**

1. **Launch Application**
   ```bash
   python main.py
   ```

2. **Select Operation**: Click "Hide Files" or use Ctrl+H

3. **Choose Carrier Image**:
   - Supported formats: PNG, BMP, TIFF
   - Recommended: High-resolution images
   - Avoid: JPEG (lossy compression)

4. **Select Files to Hide**:
   - Single file or multiple files
   - Any file type supported
   - Check capacity before proceeding

5. **Configure Security**:
   - **Password**: Strong password (8+ characters)
   - **Security Level**: Standard, High, or Maximum
   - **Encryption**: AES-256 (automatic)
   - **🆕 Automatic Decoy Protection**: System creates dual-layer security

6. **Advanced Options** (Optional):
   - Enable keyfile authentication
   - Configure randomization seed
   - Set compression level

7. **Execute**: Click "Hide Files" and wait for completion
   - **🛡️ Enhanced**: Every operation now includes decoy protection
   - **Dual Layers**: Innocent decoy + your real files
   - **Same Interface**: No complexity added for users

#### **Capacity Planning**

| **Image Resolution** | **Approximate Capacity** |
|---------------------|-------------------------|
| 800x600 (480K pixels) | ~60KB files |
| 1920x1080 (2M pixels) | ~250KB files |
| 3840x2160 (8M pixels) | ~1MB files |
| 7680x4320 (33M pixels) | ~4MB files |

### **Extracting Files**

#### **Step-by-Step Guide**

1. **Select Operation**: Click "Extract Files" or use Ctrl+E

2. **Choose Steganographic Image**: Select the image containing hidden data

3. **Enter Password**: Provide the correct password used during hiding

4. **Authentication** (if enabled): Select keyfile if two-factor is enabled

5. **Choose Output Directory**: Where to save extracted files

6. **Execute**: Click "Extract Files"

7. **Verification**: Check extracted files for integrity

### **Image Analysis**

#### **Analyzing Carrier Images**

1. **Select "Analyze Image"** from main menu
2. **Choose Image**: Select potential carrier image
3. **Analysis Results**:
   - **Capacity**: Maximum hideable data
   - **Entropy Score**: Image randomness (1-10)
   - **Suitability**: Overall hiding effectiveness
   - **Recommendations**: Optimization suggestions

#### **Detection Analysis**

1. **Select "Detect Hidden Data"**
2. **Choose Suspicious Image**: Image to analyze
3. **Analysis Types**:
   - **Statistical Analysis**: Histogram irregularities
   - **Entropy Analysis**: Randomness distribution
   - **LSB Analysis**: Least significant bit patterns
   - **Header Detection**: Known steganography signatures

---

## 🔥 Advanced Features

### **🆕 Transparent Decoy Mode (Revolutionary Security)**

> **🎉 NEW FEATURE**: Every basic operation now includes automatic decoy protection!

#### **Automatic Decoy Protection (Basic Mode)**

**What Happens Automatically:**
1. **Hide Files**: System creates TWO datasets in every image:
   - **🎭 Decoy Layer**: Innocent files (config files, README, etc.)
   - **🛡️ Real Layer**: Your actual files
2. **Different Passwords**: Each layer has its own password
3. **Seamless Operation**: No extra steps or complexity
4. **Universal Extraction**: Basic extract works with any password

**Security Benefits:**
- ✅ **Plausible Deniability**: Can show innocent files if discovered
- ✅ **Zero Learning Curve**: Works like regular steganography
- ✅ **Backward Compatible**: Still works with old images
- ✅ **Password Isolation**: Each dataset encrypted independently

#### **Advanced Multi-Decoy Mode**

**For Power Users Who Need More Control:**

1. **Access Multi-Decoy Dialog**: Advanced → Multi-Decoy
2. **Add Multiple Datasets**:
   - **Dataset Name**: Identifier for organization
   - **Password**: Unique password per dataset
   - **Priority Level**: Security level (1-5)
   - **Decoy Type**: Standard, Innocent, Personal, Business
3. **Configure Each Dataset**:
   - Add specific files to each dataset
   - Set individual security levels
   - Configure custom metadata
4. **Execute**: All datasets hidden with layered security

**Advanced Benefits:**
- 🎯 **Unlimited Datasets**: Hide as many as image capacity allows
- 📊 **Priority Control**: 5 security levels (outer to inner)
- 🏷️ **Type Categories**: Organize by purpose (innocent, personal, business)
- 🔍 **Granular Control**: Fine-tune each dataset individually

### **Two-Factor Authentication**

#### **Keyfile Generation**

1. **Access Keyfile Manager**: Security → Generate Keyfile
2. **Configuration**:
   - **Keyfile Size**: 256KB to 1MB
   - **Entropy Source**: System random + user input
   - **Format**: Binary keyfile
3. **Secure Storage**: Save keyfile separately from images
4. **Backup**: Create multiple copies in secure locations

#### **Using Two-Factor Authentication**

1. **Enable 2FA**: Settings → Security → Two-Factor
2. **Hide with 2FA**:
   - Enter password as usual
   - Select keyfile location
   - System combines both for encryption
3. **Extract with 2FA**:
   - Provide password AND keyfile
   - Both required for successful decryption

### **Multi-Image Distribution**

#### **Distributing Data Across Images**

1. **Enable Distribution**: Advanced → Multi-Image
2. **Configuration**:
   - **Image Count**: 2-8 images
   - **Redundancy Level**: Error correction strength
   - **Threshold**: Minimum images needed for reconstruction
3. **Select Carrier Images**: Choose multiple images
4. **Execute**: Data split across all images

#### **Reconstruction Process**

1. **Select "Reconstruct from Multiple Images"**
2. **Load Images**: Select available images from set
3. **Automatic Analysis**: System determines completeness
4. **Reconstruction**: Rebuild data from partial set
5. **Verification**: Validate reconstructed data integrity

---

## 🛡️ Security Features

### **Encryption Levels**

| **Level** | **Iterations** | **Use Case** |
|-----------|----------------|-------------|
| **Standard** | 100,000 | General use, good security |
| **High** | 500,000 | Sensitive data |
| **Maximum** | 1,000,000+ | Highly sensitive, maximum security |

### **Password Security**

#### **Strong Password Guidelines**

✅ **Recommended**:
- 12+ characters minimum
- Mix of uppercase, lowercase, numbers, symbols
- Unique phrases or passphrases
- No personal information

❌ **Avoid**:
- Dictionary words
- Personal information (names, dates)
- Common passwords
- Short passwords (<8 characters)

#### **Password Strength Indicator**

InvisioVault provides real-time password strength assessment:

- 🔴 **Very Weak**: <8 characters, common patterns
- 🟠 **Weak**: 8-10 characters, limited character types
- 🟡 **Moderate**: 10-12 characters, mixed types
- 🟢 **Strong**: 12+ characters, complex patterns
- 🔵 **Very Strong**: 15+ characters, high entropy

### **Memory Security**

- **Automatic Clearing**: Sensitive data cleared from RAM
- **Secure Input**: Password fields use secure memory
- **No Swap Files**: Prevents password exposure in swap
- **Process Isolation**: Secure process boundaries

---

## ⚡ Performance Guide

### **Optimization Settings**

#### **Memory Management**

1. **Settings → Performance → Memory**
2. **Configuration Options**:
   - **Memory Limit**: Maximum RAM usage
   - **Chunk Size**: Processing chunk size
   - **Cache Settings**: Temporary file management

#### **Processing Optimization**

1. **Multi-Threading**: Settings → Performance → Threads
2. **Options**:
   - **Auto-detect**: Use all available cores
   - **Manual**: Specify thread count
   - **Conservative**: Limit for other applications

### **Performance Monitoring**

#### **Real-Time Metrics**

- **Memory Usage**: Current RAM consumption
- **Processing Speed**: Operations per second
- **Progress Tracking**: Completion percentage
- **ETA**: Estimated time to completion

#### **Benchmarking**

```bash
# Run performance tests
python demo_performance.py

# Compare with previous versions
python test_main.py --benchmark

# Memory profiling
python -m pytest tests/ --profile-memory
```

---

## 🔧 Troubleshooting

### **Common Issues**

#### **"Image too small for data"**

**Problem**: Carrier image lacks capacity for selected files

**Solutions**:
1. Use higher resolution image
2. Compress files before hiding
3. Split data across multiple images
4. Use lossless image formats only

#### **"Wrong password or corrupted data"**

**Problem**: Extraction fails with password error

**Solutions**:
1. Verify password spelling and case
2. Check if keyfile is required
3. Ensure image hasn't been modified
4. Try different extraction methods

#### **"Application crashes during operation"**

**Problem**: System instability during processing

**Solutions**:
1. Check available memory
2. Reduce processing chunk size
3. Close other applications
4. Update dependencies: `pip install -r requirements.txt --upgrade`

### **Performance Issues**

#### **Slow Extraction Speed**

**Causes & Solutions**:
1. **Large files**: Enable chunked processing
2. **Low memory**: Increase memory limit
3. **CPU bottleneck**: Enable multi-threading
4. **Storage I/O**: Use faster storage (SSD)

#### **High Memory Usage**

**Optimization Steps**:
1. Reduce memory limit in settings
2. Process smaller chunks
3. Close unnecessary applications
4. Use 64-bit Python installation

### **Error Logs**

#### **Log Locations**

- **Windows**: `%USERPROFILE%\.invisiovault\logs\`
- **macOS**: `~/.invisiovault/logs/`
- **Linux**: `~/.invisiovault/logs/`

#### **Reading Logs**

```bash
# View recent errors
tail -f ~/.invisiovault/logs/invisiovault.log

# Search for specific errors
grep "ERROR" ~/.invisiovault/logs/invisiovault.log

# View full log
less ~/.invisiovault/logs/invisiovault.log
```

---

## 💡 Best Practices

### **Security Best Practices**

#### **File Management**

✅ **Do**:
- Use unique passwords for different operations
- Store keyfiles separately from images
- Backup steganographic images securely
- Test extraction before relying on hidden data
- Use decoy data for plausible deniability

❌ **Don't**:
- Reuse passwords across operations
- Store keyfiles with steganographic images
- Modify images after hiding data
- Share passwords through unsecured channels
- Use predictable decoy data

#### **Operational Security**

1. **Environment Security**:
   - Use private, secure computers
   - Avoid public/shared systems
   - Clear temporary files after operations
   - Use secure deletion for sensitive files

2. **Communication Security**:
   - Share passwords through secure channels
   - Use different passwords for sender/receiver
   - Verify image integrity before transmission
   - Consider additional encryption layers

### **Performance Best Practices**

#### **Image Selection**

✅ **Good Carrier Images**:
- High resolution (2MP+)
- Natural photographs
- Complex scenes with varied colors
- PNG/BMP/TIFF formats
- High entropy/randomness

❌ **Poor Carrier Images**:
- Low resolution (<1MP)
- Simple graphics or logos
- Solid colors or gradients
- JPEG or compressed formats
- Low entropy/patterns

#### **File Organization**

1. **Compression**: Compress files before hiding to maximize capacity
2. **Archiving**: Combine multiple files into archives
3. **Testing**: Always test extraction immediately after hiding
4. **Documentation**: Keep secure notes about hidden data locations

---

## 📞 Support & Resources

### **Getting Help**

1. **Documentation**: Check this user guide first
2. **FAQ**: Review common issues in troubleshooting section
3. **Logs**: Check error logs for specific issues
4. **Community**: GitHub Issues for bug reports

### **Additional Resources**

- **API Documentation**: [`docs/api_reference.md`](api_reference.md)
- **Security Guide**: [`docs/security_notes.md`](security_notes.md)
- **Performance Analysis**: [`../PERFORMANCE_OPTIMIZATION_SUMMARY.md`](../PERFORMANCE_OPTIMIZATION_SUMMARY.md)
- **Project Architecture**: [`../InvisioVault_Project_Prompt.md`](../InvisioVault_Project_Prompt.md)

### **Educational Resources**

- **Steganography Theory**: Understanding LSB techniques
- **Cryptography Basics**: AES encryption fundamentals
- **Digital Forensics**: Detection and analysis methods
- **Security Research**: Academic papers and resources

---

## 📝 Conclusion

InvisioVault provides professional-grade steganography capabilities with revolutionary performance improvements. This user guide covers all aspects from basic operations to advanced security features.

**Remember**: InvisioVault is designed for educational and legitimate privacy purposes only. Always use responsibly and in compliance with local laws and regulations.

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Author**: Rolan (RNR)  
**License**: MIT Educational License
