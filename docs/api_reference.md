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

### **MultiDecoyEngine**

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

### **Multi-Decoy Example**

```python
from core.multi_decoy_engine import MultiDecoyEngine
from pathlib import Path

# Initialize multi-decoy engine
engine = MultiDecoyEngine()

# Define datasets
datasets = [
    {
        "name": "Innocent Photos",
        "password": "vacation2023",
        "priority": 1,
        "decoy_type": "innocent",
        "files": ["photo1.jpg", "photo2.jpg"]
    }
]

# Hide datasets
success = engine.hide_multiple_datasets(
    carrier_path=Path("carrier.png"),
    datasets=datasets,
    output_path=Path("multi_hidden.png")
)
```

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Author**: Rolan (RNR)  
**License**: MIT Educational License
