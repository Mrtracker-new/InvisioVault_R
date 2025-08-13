# InVisioVault Technical Architecture

This document provides detailed technical guidelines and architectural patterns for developing and maintaining the InVisioVault steganography application.

## 🏗️ Architecture Overview

InVisioVault follows a modular, layered architecture designed for security, maintainability, and extensibility.

### System Layers

```
┌─────────────────────────────────────┐
│              UI Layer               │  ← Qt-based user interface
├─────────────────────────────────────┤
│           Operations Layer          │  ← File processing & workflows
├─────────────────────────────────────┤
│              Core Layer             │  ← Steganography algorithms
├─────────────────────────────────────┤
│            Utilities Layer          │  ← Config, logging, helpers
└─────────────────────────────────────┘
```

## 🎨 UI Architecture Principles

### Theme System (LOCKED IMPLEMENTATION)
```python
# ❌ NEVER DO THIS - Theme selection is prohibited
def create_theme_selector():
    theme_combo = QComboBox()
    theme_combo.addItems(['light', 'dark'])
    return theme_combo

# ✅ CORRECT - Theme is always dark
def apply_dark_theme(widget):
    # Apply consistent dark styling
    pass
```

### Settings Integration Pattern
```python
class SettingsPanel:
    def __init__(self):
        self.config = ConfigManager()
        # ❌ NEVER add: self.theme_combo = QComboBox()
        
    def load_current_settings(self):
        # Load all settings EXCEPT theme
        # Theme is automatically "dark" via ConfigManager
        pass
        
    def save_settings(self):
        # ConfigManager will automatically enforce theme="dark"
        self.config.set("interface", "theme", "dark")  # Always dark
```

## 🔧 Configuration Management

### ConfigManager Usage Patterns

```python
from utils.config_manager import ConfigManager, ConfigSection

# ✅ CORRECT Usage
config = ConfigManager()

# Getting values
theme = config.get(ConfigSection.INTERFACE, "theme")  # Always returns "dark"
window_width = config.get(ConfigSection.INTERFACE, "window_width", 1200)

# Setting values
config.set(ConfigSection.INTERFACE, "window_width", 1400)
config.set(ConfigSection.INTERFACE, "theme", "light")  # Will be forced to "dark"

# Section operations
interface_settings = config.get_section(ConfigSection.INTERFACE)
config.set_section(ConfigSection.INTERFACE, updated_settings)
```

### Theme Lock Implementation Details

The ConfigManager enforces theme locking at multiple levels:

1. **Default Configuration**: Theme defaults to "dark"
2. **Set Method**: Intercepts theme changes and forces "dark"
3. **Section Updates**: Ensures theme stays "dark" in bulk updates
4. **Validation**: Config loading/migration corrects non-dark themes
5. **Import/Export**: Imported configs are corrected automatically

## 🛡️ Security Architecture

### Cryptographic Standards

```python
# ✅ CORRECT - Use secure algorithms
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ❌ NEVER use weak algorithms
# - MD5 hashing
# - Simple XOR encryption
# - Hardcoded keys
# - Predictable random seeds
```

### Error Handling for Security

```python
# ✅ CORRECT - Secure error handling
try:
    result = perform_steganography_operation(data)
    return result
except CryptographicError as e:
    logger.error("Cryptographic operation failed")  # No details
    return None
except Exception as e:
    logger.error(f"Unexpected error in operation")  # No sensitive data
    return None

# ❌ NEVER expose sensitive information
except Exception as e:
    logger.error(f"Failed with key {secret_key}: {e}")  # WRONG!
```

## 📁 File Organization Standards

### Module Structure
```
core/
├── __init__.py
├── steganography/
│   ├── __init__.py
│   ├── lsb_encoder.py      # LSB steganography
│   ├── dct_encoder.py      # DCT-based methods
│   └── statistical_mask.py # Statistical masking
├── security/
│   ├── __init__.py
│   ├── encryption.py       # Encryption utilities
│   └── anti_detection.py   # Anti-detection measures
└── validators/
    ├── __init__.py
    └── file_validator.py   # File integrity checks
```

### Import Patterns

```python
# ✅ CORRECT - Absolute imports
from core.steganography.lsb_encoder import LSBEncoder
from utils.config_manager import ConfigManager, ConfigSection
from utils.logger import Logger

# ❌ AVOID - Relative imports in main modules
from ..utils.logger import Logger
```

## 🔍 Error Handling Architecture

### Layered Error Handling

```python
class InVisioVaultError(Exception):
    """Base exception for InVisioVault operations"""
    pass

class ConfigurationError(InVisioVaultError):
    """Configuration-related errors"""
    pass

class SteganographyError(InVisioVaultError):
    """Steganography operation errors"""
    pass

class ValidationError(InVisioVaultError):
    """Input validation errors"""
    pass
```

### Error Handling Pattern

```python
def operation_with_proper_error_handling(data):
    logger = Logger()
    
    try:
        # Validate inputs
        if not validate_input(data):
            raise ValidationError("Invalid input data")
            
        # Perform operation
        result = perform_core_operation(data)
        
        # Validate results
        if not validate_result(result):
            raise SteganographyError("Operation produced invalid result")
            
        logger.info("Operation completed successfully")
        return result
        
    except ValidationError as e:
        logger.warning(f"Validation failed: {e}")
        raise  # Re-raise for UI to handle
        
    except SteganographyError as e:
        logger.error(f"Steganography operation failed: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in operation: {type(e).__name__}")
        raise SteganographyError("Operation failed due to unexpected error")
```

## 🧪 Testing Architecture

### Test Structure
```
tests/
├── __init__.py
├── unit/
│   ├── test_config_manager.py
│   ├── test_theme_lock.py        # Critical: Theme lock tests
│   └── test_steganography.py
├── integration/
│   ├── test_ui_settings.py
│   └── test_workflow.py
└── fixtures/
    ├── sample_images/
    └── test_configs/
```

### Critical Test Cases

```python
def test_theme_lock_enforcement():
    """CRITICAL: Ensure theme cannot be changed"""
    config = ConfigManager()
    
    # Attempt various ways to change theme
    config.set("interface", "theme", "light")
    assert config.get("interface", "theme") == "dark"
    
    config.set_section("interface", {"theme": "light", "language": "en"})
    assert config.get("interface", "theme") == "dark"
    
    # Test config import doesn't bypass lock
    test_config = {"interface": {"theme": "light"}}
    config.import_config("test_light_config.json")
    assert config.get("interface", "theme") == "dark"
```

## 📊 Logging Architecture

### Logging Levels Usage

```python
# ✅ CORRECT - Appropriate logging levels
logger.debug("Entering function with parameter X")      # Development info
logger.info("Configuration loaded successfully")        # Normal operation
logger.warning("Using default value for missing config") # Recoverable issues
logger.error("Failed to save configuration")            # Error conditions
logger.critical("Critical security violation detected") # System-threatening
```

### Security-Conscious Logging

```python
# ✅ CORRECT - Safe logging
logger.info("User authentication successful")
logger.warning("Invalid configuration file format")
logger.error("Steganography operation failed")

# ❌ NEVER log sensitive data
logger.info(f"User password: {password}")        # WRONG!
logger.debug(f"Encryption key: {secret_key}")    # WRONG!
logger.info(f"Hidden message: {secret_message}") # WRONG!
```

## 🎯 Performance Considerations

### Memory Management

```python
# ✅ CORRECT - Efficient memory usage
def process_large_image(image_path):
    try:
        # Process in chunks to manage memory
        with open(image_path, 'rb') as f:
            while chunk := f.read(8192):  # Process in 8KB chunks
                process_chunk(chunk)
    finally:
        # Ensure cleanup
        gc.collect()
```

### Threading Guidelines

```python
# ✅ CORRECT - Safe threading for UI
from PyQt6.QtCore import QThread, pyqtSignal

class SteganographyWorker(QThread):
    progress_updated = pyqtSignal(int)
    operation_completed = pyqtSignal(object)
    
    def run(self):
        try:
            result = self.perform_steganography()
            self.operation_completed.emit(result)
        except Exception as e:
            logger.error(f"Worker thread error: {e}")
```

## 🚨 Critical Implementation Rules

### Absolute Prohibitions

1. **Theme System Bypass**: Never implement theme switching
2. **Config System Bypass**: Never circumvent ConfigManager
3. **Root Documentation**: Never create docs in project root
4. **Insecure Algorithms**: Never use weak cryptography
5. **Sensitive Logging**: Never log passwords/keys/secrets

### Mandatory Patterns

1. **Error Handling**: Every public method must handle exceptions
2. **Input Validation**: Validate all external inputs
3. **Configuration**: Use ConfigManager for all persistent settings
4. **Logging**: Use Logger utility for all logging
5. **Testing**: Write tests for new functionality

## 🔄 Development Workflow

### Code Review Checklist

- [ ] Theme lock not bypassed
- [ ] ConfigManager used for settings
- [ ] Proper error handling implemented
- [ ] Security best practices followed
- [ ] Documentation created in `docs/` folder
- [ ] Tests written and passing
- [ ] Logging implemented appropriately
- [ ] No sensitive data in logs/errors

### Deployment Considerations

- Ensure all dependencies are security-audited
- Validate configuration files before deployment
- Test theme lock enforcement in production builds
- Verify secure deletion functionality
- Confirm anti-detection measures are active

---

**This architecture document must be followed strictly to maintain InVisioVault's security, consistency, and quality standards.**

**Last Updated**: 2025-01-13
**Version**: 1.0
