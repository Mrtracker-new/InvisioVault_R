# Multi-Decoy Implementation - Educational Research

## 📚 **Educational Project by Rolan (RNR)**

**Purpose**: Research and learning project exploring advanced steganography techniques

## Overview

The multi-dataset decoy functionality in InVisioVault has been successfully implemented as part of educational research into plausible deniability techniques. This implementation demonstrates advanced steganography concepts where multiple datasets with different passwords and priority levels can be hidden in a single image.

## What Was Fixed

### Problem
The original `DecoyDialog` UI was designed to handle multiple datasets, but the underlying `DecoyEngine` only supported dual-data hiding (decoy + real). This mismatch caused the system to fail when trying to hide more than two datasets.

### Solution
Created a new `MultiDecoyEngine` that properly handles multiple datasets with layered steganography:

## New Components

### 1. MultiDecoyEngine (`core/multi_decoy_engine.py`)
- **Purpose**: Handle multiple datasets with different passwords and priorities
- **Key Features**:
  - Layered payload structure with proper metadata
  - Individual dataset encryption with separate passwords
  - ZIP archive compression for multiple files per dataset
  - Capacity calculation for multiple datasets
  - Priority-based organization (1=outer/least secure, 5=inner/most secure)

### 2. Updated DecoyWorkerThread (`ui/dialogs/decoy_dialog.py`)
- **Changes**: Simplified to use `MultiDecoyEngine` instead of complex layered logic
- **Benefits**: 
  - More reliable hiding/extraction
  - Better error handling
  - Cleaner progress reporting

## How It Works

### Hiding Multiple Datasets
1. **Dataset Preparation**: Each dataset's files are compressed into a ZIP archive
2. **Encryption**: Each archive is encrypted with its specific password using AES-256
3. **Metadata Creation**: Dataset metadata (name, priority, file count, etc.) is stored as JSON
4. **Layered Payload**: All encrypted datasets are combined into a single payload with headers
5. **Steganography**: The combined payload is hidden in the carrier image using LSB

### Extracting Datasets
1. **Payload Extraction**: The entire layered payload is extracted from the image
2. **Dataset Parsing**: Individual encrypted datasets are separated using metadata headers
3. **Decryption Attempt**: The provided password is tried against each dataset
4. **File Extraction**: Successfully decrypted dataset files are extracted to the output directory

## Features

### Plausible Deniability
- **Multiple Passwords**: Each password reveals only its corresponding dataset
- **Innocent Decoys**: Outer layers can contain innocent files to deflect suspicion
- **Priority Levels**: 5 security levels from outer (least secure) to inner (most secure)
- **Decoy Types**: Standard, Innocent, Personal, Business categorization

### Security Features
- **AES-256 Encryption**: Each dataset encrypted independently with maximum security
- **Password-Specific Seeds**: Each dataset uses password-derived randomization
- **Integrity Checking**: SHA-256 checksums verify data integrity
- **Metadata Protection**: Dataset information is encrypted within the payload

### User Interface
- **Intuitive Design**: Clear dataset configuration and overview panels
- **Priority Management**: Easy priority level selection with explanations
- **File Management**: Add/remove files per dataset with visual feedback
- **Progress Tracking**: Real-time status updates during operations

## Usage Example

### Via UI (Decoy Dialog)
1. **Select Carrier**: Choose a PNG/BMP/TIFF image
2. **Configure Datasets**: For each dataset:
   - Set name and password
   - Choose priority level (1-5)
   - Select decoy type
   - Add files
3. **Execute Operation**: Hide all datasets in the image
4. **Extract Later**: Use any dataset password to extract only that dataset

### Via Code
```python
from core.multi_decoy_engine import MultiDecoyEngine
from pathlib import Path

# Initialize engine
engine = MultiDecoyEngine(SecurityLevel.MAXIMUM)

# Define datasets
datasets = [
    {
        "name": "Innocent_Photos",
        "password": "vacation2023", 
        "priority": 1,
        "decoy_type": "innocent",
        "files": ["photo1.jpg", "photo2.jpg"]
    },
    {
        "name": "Personal_Docs",
        "password": "family_secrets",
        "priority": 5, 
        "decoy_type": "personal",
        "files": ["diary.txt", "notes.txt"]
    }
]

# Hide datasets
success = engine.hide_multiple_datasets(
    carrier_path=Path("carrier.png"),
    datasets=datasets,
    output_path=Path("hidden.png")
)

# Extract dataset
metadata = engine.extract_dataset(
    stego_path=Path("hidden.png"),
    password="vacation2023",  # Only extracts innocent photos
    output_dir=Path("extracted/")
)
```

## Technical Details

### File Format
- **Magic Header**: `INVMD` (InvisioVault Multi-Decoy)
- **Version**: `0x02 0x00`
- **Structure**: Header + Dataset Count + [Dataset Entries...]
- **Dataset Entry**: Metadata Length + Metadata JSON + Data Length + Checksum + Encrypted Data

### Capacity Management
- **Overhead Calculation**: Accounts for headers, metadata, and checksums
- **Per-Dataset Capacity**: Automatically calculated based on available space
- **Maximum Datasets**: Determined by image size and minimum data requirements

### Error Handling
- **Capacity Validation**: Prevents oversized datasets from being hidden
- **Integrity Verification**: Checksums ensure data hasn't been corrupted
- **Password Verification**: Clear error messages for wrong passwords
- **File System Errors**: Graceful handling of missing files or permissions

## Testing

The implementation includes comprehensive testing:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing  
- **Capacity Tests**: Various image sizes and dataset combinations
- **Security Tests**: Password isolation and encryption verification

Run tests with:
```bash
python test_multi_decoy.py
```

## Benefits

### For Users
- **True Multi-Dataset Support**: Hide unlimited datasets (limited by image capacity)
- **Plausible Deniability**: Each password reveals different content
- **User-Friendly Interface**: Intuitive dataset management
- **Flexible Organization**: Priority levels and decoy types

### For Security
- **Independent Encryption**: Compromising one password doesn't affect others
- **Layered Protection**: Multiple security levels within same image
- **Strong Cryptography**: AES-256 with proper key derivation
- **Data Integrity**: Built-in corruption detection

### For Development
- **Modular Design**: Clean separation between UI and core functionality
- **Extensible Architecture**: Easy to add new features or decoy types
- **Comprehensive Logging**: Detailed operation tracking for debugging
- **Error Recovery**: Graceful handling of edge cases

## Compatibility

- **Image Formats**: PNG, BMP, TIFF (lossless formats required)
- **File Types**: Any file type supported in datasets
- **Operating Systems**: Windows, macOS, Linux
- **Python Version**: 3.8+

This implementation provides an educational demonstration of robust steganographic techniques with plausible deniability, serving as a learning tool for understanding advanced data hiding methods.

---

## 📚 **Educational Purpose**

**Author**: Rolan (RNR)  
**Intent**: Educational research and learning  
**Use Case**: Cybersecurity education, steganography research, algorithm study  
**Disclaimer**: For educational and research purposes only
