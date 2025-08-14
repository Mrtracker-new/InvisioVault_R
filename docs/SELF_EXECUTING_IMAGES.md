# 🚀 Self-Executing Images Feature

## Overview

The Self-Executing Images feature in InVisioVault allows you to create images that can execute embedded code when triggered. This advanced steganographic technique is designed for educational purposes and security research.

**⚠️ IMPORTANT SECURITY WARNING**
> This feature is for educational and research purposes ONLY. Always exercise extreme caution when creating or executing self-executing images. Only use content from trusted sources and in isolated environments.

## 📋 Table of Contents

- [Feature Types](#-feature-types)
- [Getting Started](#-getting-started)
- [Creating Self-Executing Images](#-creating-self-executing-images)
- [Analyzing Images](#-analyzing-images)
- [Security Considerations](#-security-considerations)
- [Technical Implementation](#-technical-implementation)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)

## 🎯 Feature Types

### 1. Polyglot Files
Create files that are simultaneously valid images AND executable programs.
- **How it works**: Combines image data with executable code
- **Usage**: File appears as normal image in viewers, but can be executed as a program
- **Best for**: Advanced steganography research, malware analysis training

### 2. Script-Executing Images
Embed executable scripts within images using steganography.
- **Supported scripts**: Python, JavaScript, PowerShell, Batch, Bash, VBScript
- **How it works**: Scripts are hidden in image pixels and extracted when analyzed
- **Security**: Can be encrypted with password protection
- **Best for**: Educational demonstrations, security training

### 3. Custom Viewer Integration
Specialized viewer for detecting and executing embedded content.
- **Safe analysis mode**: Detect content without executing
- **Interactive execution**: User-controlled execution with warnings
- **Detailed logging**: Complete execution monitoring

## 🚀 Getting Started

### Prerequisites

1. **Python Environment**: Python 3.8+ with required packages
2. **Interpreters** (for script execution):
   - Python 3.x (for .py scripts)
   - Node.js (for .js scripts)  
   - PowerShell (for .ps1 scripts)
   - Command Prompt (for .bat scripts)

### Accessing the Feature

1. **Launch InVisioVault**
2. **Navigate to "🚀 Self-Executing"** in the sidebar
3. **Click "🚀 Open Self-Executing Dialog"**

## 🔨 Creating Self-Executing Images

### Method 1: Polyglot Files

1. **Select the "Polyglot Files" tab**
2. **Choose carrier image**: Select a PNG, BMP, or TIFF image
3. **Choose executable**: Select the .exe file to embed
4. **Set output path**: Where to save the polyglot file
5. **Optional encryption**: Add password protection
6. **Click "🔨 Create Polyglot Executable"**

```
Result: A file that can be:
• Opened as an image in image viewers
• Executed as a program when run directly
```

### Method 2: Script Images

1. **Select the "Script Images" tab**
2. **Choose carrier image**: Select source image
3. **Select script type**: Choose from supported languages
4. **Write or paste script code** in the editor
5. **Configure options**:
   - Auto-execute (runs immediately when extracted)
   - Password protection
6. **Set output path**
7. **Click "🎯 Create Script-Executing Image"**

#### Quick Script Templates

**Python Hello World**:
```python
#!/usr/bin/env python3
print("Hello from self-executing image!")
import sys
print(f"Python version: {sys.version}")
input("Press Enter to continue...")
```

**JavaScript Example**:
```javascript
console.log("Hello from self-executing image!");
console.log("Node.js version:", process.version);
process.stdin.setRawMode(true);
process.stdin.resume();
process.stdin.on('data', process.exit.bind(process, 0));
console.log("Press any key to exit...");
```

## 🔍 Analyzing Images

### Using the Main Dialog

1. **Select "Analysis & Execution" tab**
2. **Choose image to analyze**
3. **Enter password** (if content is encrypted)
4. **Click "🔍 Analyze Image"** for safe analysis
5. **Review results** in the analysis panel

### Using the Standalone Viewer

1. **Launch the viewer**:
   ```bash
   python self_executing_viewer.py
   ```

2. **Or analyze directly**:
   ```bash
   python self_executing_viewer.py path/to/image.png
   ```

3. **GUI Features**:
   - Safe analysis mode
   - Interactive execution
   - Detailed results display
   - Keyboard shortcuts (Ctrl+O, Ctrl+A, F5)

### Execution Modes

- **Safe Mode**: Analyzes content without executing (recommended first step)
- **Interactive Mode**: Prompts user before execution
- **Auto Mode**: Executes immediately (use with caution!)

## 🛡️ Security Considerations

### For Creators

1. **Educational Purpose Only**
   - Only create self-executing images for legitimate educational or research purposes
   - Never embed malicious code or content intended to harm

2. **Test in Isolated Environments**
   - Use virtual machines or isolated systems
   - Never test on production systems
   - Have backups and recovery plans

3. **Responsible Disclosure**
   - If used for security research, follow responsible disclosure practices
   - Document findings appropriately

### For Analysts

1. **Trust Verification**
   - Only analyze images from known, trusted sources
   - Be extremely cautious with unknown or suspicious images
   - Verify the source and intent before analysis

2. **Execution Safety**
   - Always analyze in safe mode first
   - Use isolated environments (VMs, sandboxes)
   - Monitor system activity during execution
   - Have incident response procedures ready

3. **Detection Strategies**
   - Look for unusual image properties
   - Check file sizes vs. visual content
   - Use multiple analysis tools
   - Verify metadata and headers

### Red Flags

**Suspicious Indicators**:
- Unusually large file sizes for simple images
- Images from unknown or untrusted sources
- Requests to "run" image files
- Images with executable file extensions
- Social engineering attempts involving images

## ⚙️ Technical Implementation

### Core Components

1. **SelfExecutingEngine** (`core/self_executing_engine.py`)
   - Main logic for creating and analyzing self-executing images
   - Polyglot file generation
   - Script embedding and extraction
   - Execution management

2. **SelfExecutingDialog** (`ui/dialogs/self_executing_dialog.py`)
   - User interface for creation and analysis
   - Multi-tab design for different creation methods
   - Progress tracking and error handling

3. **Standalone Viewer** (`self_executing_viewer.py`)
   - Independent analysis tool
   - Command-line and GUI modes
   - Safe analysis capabilities

### Supported File Formats

**Images**: PNG, BMP, TIFF (lossless formats required)
**Scripts**: .py, .js, .ps1, .bat, .sh, .vbs
**Executables**: Windows PE, Linux ELF, macOS Mach-O

### Encryption Support

- **AES-256 encryption** for embedded content
- **Password-based protection**
- **PBKDF2 key derivation** for security
- **Secure memory handling**

## 📖 Examples

### Example 1: Educational Python Script

Create a demonstration script that shows system information:

```python
#!/usr/bin/env python3
"""
Educational Self-Executing Script
Demonstrates system information gathering
"""
import sys, platform, os
from datetime import datetime

print("🎓 Educational Demonstration")
print("="*40)
print(f"Execution time: {datetime.now()}")
print(f"Python: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Current user: {os.getenv('USERNAME') or os.getenv('USER')}")
print(f"Working directory: {os.getcwd()}")
print("\n✅ This script ran successfully from a steganographic image!")
input("\nPress Enter to exit...")
```

### Example 2: Security Research Tool

A script that analyzes its own execution environment:

```python
#!/usr/bin/env python3
"""
Security Research: Execution Environment Analysis
"""
import os, sys, subprocess, json

def analyze_environment():
    env_data = {
        "python_version": sys.version,
        "environment_vars": dict(os.environ),
        "working_directory": os.getcwd(),
        "command_line": sys.argv,
        "executable_path": sys.executable
    }
    
    print("🔬 Execution Environment Analysis")
    print("="*50)
    print(json.dumps(env_data, indent=2))
    
    return env_data

if __name__ == "__main__":
    analyze_environment()
    input("\nAnalysis complete. Press Enter to exit...")
```

### Example 3: Simple Batch Script

Windows batch script for demonstration:

```batch
@echo off
echo 🚀 Self-Executing Batch Script
echo ================================
echo.
echo Current Date/Time: %date% %time%
echo Computer Name: %COMPUTERNAME%
echo Username: %USERNAME%
echo Windows Version: %OS%
echo.
echo ✅ This batch script was embedded in an image!
echo.
pause
```

## 🔧 Troubleshooting

### Common Issues

#### "No executable content found"
- **Cause**: Wrong password, unsupported format, or no embedded content
- **Solution**: Verify password, check image was created with InVisioVault, try different analysis mode

#### "Script execution failed"
- **Cause**: Missing interpreter, permission issues, or script errors
- **Solution**: Install required interpreters (Python, Node.js, etc.), check permissions, debug script separately

#### "Polyglot creation failed"
- **Cause**: Unsupported executable format, file permissions, or insufficient disk space
- **Solution**: Use compatible executables, check permissions, ensure adequate storage

#### "Analysis timeout"
- **Cause**: Large files, complex content, or system performance issues
- **Solution**: Try smaller images, close other applications, use more powerful hardware

### Performance Optimization

1. **Use appropriate image sizes**
   - Larger images = more capacity but slower processing
   - Optimize for your specific use case

2. **Script optimization**
   - Keep embedded scripts concise
   - Avoid complex dependencies
   - Test scripts independently first

3. **System resources**
   - Ensure adequate RAM and CPU
   - Close unnecessary applications during processing
   - Use SSD storage for better I/O performance

### Debugging Tips

1. **Enable verbose logging**
   - Check InVisioVault logs for detailed error information
   - Use debug mode in the standalone viewer

2. **Step-by-step verification**
   - Test with simple, known-good scripts first
   - Verify each component (image, script, encryption) separately
   - Use the analysis mode before attempting execution

3. **Environment verification**
   - Ensure all required interpreters are installed and in PATH
   - Test interpreters independently: `python --version`, `node --version`
   - Check system permissions and security settings

## 📚 Additional Resources

### Educational Materials
- **Steganography fundamentals**
- **Polyglot file techniques**
- **Security analysis methodologies**
- **Ethical hacking principles**

### Security Research
- **NIST Cybersecurity Framework**
- **OWASP Security Guidelines**
- **CVE database for vulnerability research**
- **Academic papers on steganography**

### Development Resources
- **Python scripting tutorials**
- **JavaScript execution environments**
- **PowerShell automation guides**
- **Cross-platform development practices**

---

## 📝 Legal and Ethical Notice

This feature is provided for **educational and research purposes only**. Users are solely responsible for:

1. **Compliance** with all applicable laws and regulations
2. **Ethical use** of the technology
3. **Proper authorization** before testing on any systems
4. **Responsible disclosure** of security findings
5. **Safe handling** of potentially dangerous content

**Remember**: With great power comes great responsibility. Use this feature wisely and ethically.

---

*© 2025 InVisioVault - Educational Steganography Project*
*Author: Rolan (RNR)*
