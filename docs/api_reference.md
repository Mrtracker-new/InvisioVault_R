# 📚 InvisioVault API Reference
### *Complete Technical Documentation*

**Version**: 1.0.0  
**Author**: Rolan (RNR)  
**Purpose**: Comprehensive API documentation for developers and researchers  
**Last Updated**: January 2025

---

## 🎯 Table of Contents

1. [**Core Engines**](#-core-engines)
2. [**UI Components**](#-ui-components)
3. [**Utilities**](#-utilities)
4. [**Operations**](#-operations)
5. [**Configuration**](#-configuration)
6. [**Error Handling**](#-error-handling)
7. [**Examples**](#-examples)

---

## 🔧 Core Engines

### **SteganographyEngine**

#### **Class Overview**
```python
class SteganographyEngine:
    """Advanced LSB steganography with revolutionary performance."""
    
    def __init__(self):
        self.MAGIC_HEADER = b'INVS'
        self.logger = Logger()
```

#### **Key Methods**

##### **`hide_data(carrier_path, data, output_path, randomize=True, seed=None)`**
```python
def hide_data(self, carrier_path: Path, data: bytes, output_path: Path, 
              randomize: bool = True, seed: int = None) -> bool:
    """
    Hide data in carrier image using LSB steganography.
    
    Args:
        carrier_path: Path to carrier image (PNG/BMP/TIFF)
        data: Binary data to hide
        output_path: Path for output steganographic image
        randomize: Use randomized positioning (recommended)
        seed: Random seed for positioning (password-derived)
    
    Returns:
        bool: Success status
        
    Raises:
        FileNotFoundError: Carrier image not found
        ValueError: Image format not supported or insufficient capacity
        SteganographyError: Hiding operation failed
    """
```

##### **`extract_data(stego_path, randomize=True, seed=None)`**
```python
def extract_data(self, stego_path: Path, randomize: bool = True, 
                 seed: int = None) -> bytes:
    """
    Extract hidden data from steganographic image.
    
    Args:
        stego_path: Path to steganographic image
        randomize: Use randomized positioning
        seed: Random seed for positioning
    
    Returns:
        bytes: Extracted data or None if extraction failed
        
    Raises:
        FileNotFoundError: Steganographic image not found
        SteganographyError: Extraction failed or no data found
    """
```

### **EncryptionEngine**

#### **Security Levels**
```python
class SecurityLevel(Enum):
    STANDARD = "standard"    # 100,000 iterations
    HIGH = "high"           # 500,000 iterations  
    MAXIMUM = "maximum"     # 1,000,000+ iterations
```

#### **Key Methods**

##### **`encrypt_with_metadata(data, password, keyfile_path=None)`**
```python
def encrypt_with_metadata(self, data: bytes, password: str, 
                         keyfile_path: Path = None) -> bytes:
    """
    Encrypt data with metadata and optional keyfile.
    
    Args:
        data: Binary data to encrypt
        password: Encryption password
        keyfile_path: Optional keyfile for two-factor auth
    
    Returns:
        bytes: Encrypted data with metadata
    """
```

### **DecoyEngine**

#### **Class Overview**
```python
class DecoyEngine:
    """Plausible deniability with dual-dataset hiding."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.decoy_ratio = 0.3  # 30% capacity for decoy data
```

### **🆕 MultiDecoyEngine (Enhanced with Transparent Mode)**

#### **Dataset Configuration**
```python
@dataclass
class DatasetConfig:
    name: str
    password: str
    priority: int  # 1-5 (1=outer/least secure, 5=inner/most secure)
    decoy_type: str  # 'standard', 'innocent', 'personal', 'business'
    files: List[Path]
```

#### **Key Methods**

##### **`hide_multiple_datasets(carrier_path, datasets, output_path)`**
```python
def hide_multiple_datasets(self, carrier_path: Path, datasets: List[Dict], 
                          output_path: Path) -> bool:
    """
    🎉 ENHANCED: Core method for multi-decoy steganography.
    
    Now integrated with basic operations for transparent dual-layer protection.
    Creates layered steganographic images with independent encryption.
    
    Args:
        carrier_path: Path to carrier image
        datasets: List of dataset configurations:
            [
                {
                    'name': 'DatasetName',
                    'password': 'password123',
                    'priority': 1,  # 1=outer, 5=inner
                    'decoy_type': 'innocent',
                    'files': ['/path/to/file1', '/path/to/file2']
                }
            ]
        output_path: Path for output image
    
    Returns:
        bool: Success status
        
    Raises:
        SteganographyError: Insufficient capacity or operation failure
        ValueError: Invalid dataset configuration
    """
```

##### **`extract_dataset(stego_path, password, output_dir)`**
```python
def extract_dataset(self, stego_path: Path, password: str, 
                   output_dir: Path) -> Dict[str, Any]:
    """
    🎉 ENHANCED: Smart extraction that works with any password.
    
    Universal extraction method that:
    - Automatically detects image format (multi-decoy vs legacy)
    - Tries password against all available datasets
    - Returns detailed metadata about extracted files
    
    Args:
        stego_path: Path to steganographic image
        password: Password to try against datasets
        output_dir: Directory for extracted files
    
    Returns:
        dict: {
            'dataset_id': str,           # Name/ID of extracted dataset
            'extraction_path': str,      # Path where files were extracted
            'extracted_files': List[{
                'path': str,             # Full path to extracted file
                'name': str              # Filename only
            }],
            'decoy_type': str,           # Type of dataset extracted
            'priority': int,             # Security priority level
            'method': str                # 'multi_decoy' or 'legacy'
        }
        None if no dataset matches the password
        
    Raises:
        FileNotFoundError: Steganographic image not found
        SteganographyError: Image format not supported
    """
```

##### **`create_automatic_decoy_datasets(user_files, user_password)`** ⚡ *NEW*
```python
def create_automatic_decoy_datasets(self, user_files: List[str], 
                                   user_password: str) -> List[Dict]:
    """
    🎉 NEW: Create dual-layer datasets for transparent decoy protection.
    
    Called automatically by basic hide operations to create:
    1. Decoy dataset with innocent files and derived password
    2. Real dataset with user's files and actual password
    
    Args:
        user_files: List of user's files to hide
        user_password: User's chosen password
    
    Returns:
        List[Dict]: Two dataset configurations ready for hiding
        [
            {  # Decoy dataset (outer layer)
                'name': 'ProcessingData',
                'password': 'img_1234',  # Derived from user password
                'priority': 1,
                'decoy_type': 'innocent',
                'files': ['/tmp/readme.txt', '/tmp/config.ini']
            },
            {  # Real dataset (inner layer)
                'name': 'UserFiles',
                'password': user_password,
                'priority': 5,
                'decoy_type': 'personal',
                'files': user_files
            }
        ]
    """
```

---

## 🎨 UI Components

### **MainWindow**

#### **Class Overview**
```python
class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_connections()
```

### **Dialog Classes**

#### **HideFilesDialog**
```python
class HideFilesDialog(QDialog):
    """File hiding configuration dialog."""
    
    file_hiding_requested = Signal(dict)  # Configuration parameters
```

#### **ExtractFilesDialog**
```python
class ExtractFilesDialog(QDialog):
    """File extraction configuration dialog."""
    
    file_extraction_requested = Signal(dict)
```

---

## 🛠️ Utilities

### **Logger**

#### **Class Overview**
```python
class Logger:
    """Secure logging with PII redaction."""
    
    def __init__(self, name: str = "InvisioVault"):
        self.logger = logging.getLogger(name)
        self.setup_logging()
```

### **ConfigManager**

#### **Configuration Sections**
```python
class ConfigSection(Enum):
    SECURITY = "security"
    INTERFACE = "interface" 
    PERFORMANCE = "performance"
    LOGGING = "logging"
    BACKUP = "backup"
    ADVANCED = "advanced"
```

---

## 🔄 Operations

### **BaseOperation**

#### **Abstract Base Class**
```python
class BaseOperation(QObject):
    """Base class for all operations with progress tracking."""
    
    progress_updated = Signal(int, str)  # Progress percentage and message
    operation_completed = Signal(bool, str)  # Success status and message
    operation_cancelled = Signal()
```

---

## 📊 Configuration

### **Default Configuration**

```python
DEFAULT_CONFIG = {
    "security": {
        "default_security_level": "standard",
        "password_min_length": 8,
        "enable_keyfile_auth": False
    },
    "interface": {
        "theme": "dark",
        "font_size": 12,
        "enable_animations": True
    },
    "performance": {
        "max_threads": 4,
        "memory_limit_mb": 500,
        "compression_level": 6
    }
}
```

---

## 🚨 Error Handling

### **Exception Classes**
```python
class InvisioVaultError(Exception):
    """Base exception for InvisioVault."""
    pass

class SteganographyError(InvisioVaultError):
    """Steganography operation errors."""
    pass
    
class EncryptionError(InvisioVaultError):
    """Encryption/decryption errors."""
    pass
```

---

## 💡 Examples

### **Basic Steganography**

```python
from core.steganography_engine import SteganographyEngine
from core.encryption_engine import EncryptionEngine, SecurityLevel
from pathlib import Path

# Initialize engines
stego_engine = SteganographyEngine()
encryption_engine = EncryptionEngine(SecurityLevel.HIGH)

# Hide data
carrier_path = Path("carrier.png")
secret_data = b"This is my secret message!"
password = "my_secure_password_123"

# Encrypt data
encrypted_data = encryption_engine.encrypt_with_metadata(secret_data, password)

# Hide in image
success = stego_engine.hide_data(
    carrier_path=carrier_path,
    data=encrypted_data,
    output_path=Path("hidden.png"),
    randomize=True,
    seed=hash(password) % (2**32)
)

if success:
    print("Data hidden successfully!")
```

### **🆕 Transparent Decoy Mode Example**

```python
from core.multi_decoy_engine import MultiDecoyEngine
from pathlib import Path

# 🎉 NEW: Transparent decoy mode (automatic dual-layer security)
engine = MultiDecoyEngine()

# Hide files with automatic decoy protection
files_to_hide = [Path("secret_document.pdf"), Path("private_photo.jpg")]
user_password = "MySecurePassword123"

success = engine.hide_with_automatic_decoy(
    carrier_path=Path("vacation.png"),
    files=files_to_hide,
    password=user_password,  # Your files accessible with this password
    output_path=Path("hidden_image.png")
)

if success:
    print("Files hidden with automatic decoy protection!")
    
    # Extract with user password (gets real files)
    result = engine.extract_dataset(
        stego_path=Path("hidden_image.png"),
        password=user_password,
        output_dir=Path("extracted_real")
    )
    
    if result['success']:
        print(f"Extracted {len(result['files_extracted'])} real files")
        print(f"Dataset: {result['dataset_name']}")
        
    # Note: If someone else uses a different password,
    # they might get the decoy files instead!
```

### **Advanced Multi-Decoy Example**

```python
from core.multi_decoy_engine import MultiDecoyEngine
from pathlib import Path

# Initialize multi-decoy engine
engine = MultiDecoyEngine()

# Define multiple datasets with different priorities
datasets = [
    {
        "name": "Innocent Photos",
        "password": "vacation2023",
        "priority": 1,  # Outer layer (least secure)
        "decoy_type": "innocent",
        "files": ["photo1.jpg", "photo2.jpg"]
    },
    {
        "name": "Work Documents",
        "password": "work_secure_456",
        "priority": 3,  # Middle layer
        "decoy_type": "business", 
        "files": ["report.pdf", "presentation.pptx"]
    },
    {
        "name": "Personal Files",
        "password": "ultra_secret_789",
        "priority": 5,  # Inner layer (most secure)
        "decoy_type": "personal",
        "files": ["diary.txt", "private_keys.pem"]
    }
]

# Hide all datasets with layered security
success = engine.hide_multiple_datasets(
    carrier_path=Path("carrier.png"),
    datasets=datasets,
    output_path=Path("multi_hidden.png")
)

if success:
    print("Multi-layer datasets hidden successfully!")
```

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Author**: Rolan (RNR)  
**License**: MIT Educational License
