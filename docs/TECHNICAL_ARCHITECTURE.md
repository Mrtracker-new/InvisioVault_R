# InVisioVault Technical Architecture

This document provides detailed technical guidelines and architectural patterns for developing and maintaining the InVisioVault steganography application.

## 🏗️ Architecture Overview

InVisioVault follows a modular, layered architecture designed for security, maintainability, and extensibility.

### System Layers

```
┌─────────────────────────────────────┐
│              UI Layer               │  ← Qt-based user interface + multimedia dialogs
├─────────────────────────────────────┤
│           Operations Layer          │  ← File processing & multimedia workflows
├─────────────────────────────────────┤
│              Core Layer             │  ← Steganography algorithms + multimedia engines
├─────────────────────────────────────┤
│            Utilities Layer          │  ← Config, logging, helpers, multimedia analysis
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
├── multimedia/             # 🆕 MULTIMEDIA STEGANOGRAPHY
│   ├── __init__.py
│   ├── video_steganography_engine.py  # Video processing with OpenCV/FFmpeg
│   ├── audio_steganography_engine.py  # Audio processing with multiple techniques
│   └── multimedia_analyzer.py         # Format analysis and capacity assessment
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
from core.multimedia.video_steganography_engine import VideoSteganographyEngine
from core.multimedia.audio_steganography_engine import AudioSteganographyEngine
from core.multimedia.multimedia_analyzer import MultimediaAnalyzer
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

## 🎬 Multimedia Steganography Architecture

### Core Multimedia Engines

#### **VideoSteganographyEngine**
```python
class VideoSteganographyEngine:
    """
    Advanced video steganography using frame-based LSB embedding.
    Integrates OpenCV for video processing and FFmpeg for encoding.
    """
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mkv', '.mov']
        self.frame_selection_ratio = 0.1  # Use 10% of frames
        self.quality_threshold = 0.8      # Minimum quality preservation
    
    def hide_data_in_video(self, video_path, data, output_path, password):
        # 1. Extract frames using OpenCV
        # 2. Select frames using password-seeded randomization
        # 3. Apply LSB embedding to selected frames
        # 4. Reconstruct video with FFmpeg
        pass
```

#### **AudioSteganographyEngine**
```python
class AudioSteganographyEngine:
    """
    Multi-technique audio steganography supporting:
    - LSB embedding in audio samples
    - Spread spectrum in frequency domain
    - Phase coding for high-fidelity hiding
    """
    
    def __init__(self):
        self.supported_formats = ['.wav', '.flac', '.mp3', '.aac']
        self.techniques = ['lsb', 'spread_spectrum', 'phase_coding']
    
    def select_optimal_technique(self, audio_path):
        # Analyze audio characteristics
        # Return recommended technique based on:
        # - Dynamic range, frequency content, compression level
        pass
```

#### **MultimediaAnalyzer**
```python
class MultimediaAnalyzer:
    """
    Comprehensive multimedia file analysis for:
    - Format compatibility assessment
    - Capacity calculation
    - Quality prediction
    - Technique recommendation
    """
    
    def analyze_multimedia_file(self, file_path):
        # Return detailed analysis including:
        # - Format specifications
        # - Hiding capacity estimates
        # - Quality preservation predictions
        # - Recommended techniques
        pass
```

### Multimedia UI Architecture

#### **Multimedia Dialog Structure**
```python
# Enhanced dialog system for multimedia operations
class MultimediaHideDialog(QDialog):
    """Professional multimedia file hiding interface."""
    
    def __init__(self):
        super().__init__()
        self.file_drop_zone = FileDropZone()  # Drag-and-drop with preview
        self.analysis_panel = AnalysisPanel()  # Real-time capacity analysis
        self.progress_tracker = ProgressDialog()  # Cancellable progress
    
    def setup_multimedia_preview(self):
        # Video thumbnail generation
        # Audio waveform visualization
        # Format-specific metadata display
        pass

class MultimediaExtractDialog(QDialog):
    """Multimedia extraction with format auto-detection."""
    
    def auto_detect_multimedia_format(self, file_path):
        # Detect if file contains:
        # - Video steganography
        # - Audio steganography  
        # - Legacy image steganography
        # - Multi-decoy datasets
        pass
```

### Multimedia Processing Patterns

#### **Memory Management for Large Media**
```python
# ✅ CORRECT - Streaming processing for large multimedia files
def process_large_video_file(video_path):
    """Process video in chunks to manage memory usage."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process in batches of 100 frames
        batch_size = 100
        for start_frame in range(0, frame_count, batch_size):
            frames_batch = extract_frame_batch(cap, start_frame, batch_size)
            process_frame_batch(frames_batch)
            
            # Clear processed frames from memory
            del frames_batch
            gc.collect()
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
```

#### **Quality Preservation Guidelines**
```python
# ✅ CORRECT - Quality-aware multimedia steganography
def embed_with_quality_control(media_file, data, quality_threshold=0.8):
    """Ensure multimedia quality stays above threshold."""
    
    # Pre-embedding quality assessment
    original_quality = assess_media_quality(media_file)
    
    # Embedding with quality monitoring
    embedded_media = perform_embedding(media_file, data)
    final_quality = assess_media_quality(embedded_media)
    
    # Quality validation
    quality_ratio = final_quality / original_quality
    if quality_ratio < quality_threshold:
        raise QualityDegradationError(
            f"Quality dropped to {quality_ratio:.2%}, below {quality_threshold:.2%} threshold"
        )
    
    return embedded_media
```

### Multimedia Security Architecture

#### **Format-Specific Security Measures**
```python
# Video steganography security
class VideoSecurityMeasures:
    def apply_anti_detection(self, video_frames):
        # Randomize frame selection patterns
        # Apply statistical masking across frames
        # Preserve video codec characteristics
        pass
    
    def validate_video_integrity(self, original, embedded):
        # Check codec preservation
        # Verify frame rate consistency
        # Validate metadata integrity
        pass

# Audio steganography security  
class AudioSecurityMeasures:
    def apply_frequency_masking(self, audio_data):
        # Hide data in perceptually masked frequencies
        # Preserve audio dynamics and timbre
        # Apply psychoacoustic modeling
        pass
```

### Multimedia Dependencies Architecture

#### **Dependency Integration Patterns**
```python
# ✅ CORRECT - Graceful multimedia dependency handling
try:
    import cv2
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False
    logger.warning("OpenCV not available - video steganography disabled")

try:
    import librosa
    AUDIO_ANALYSIS_SUPPORT = True
except ImportError:
    AUDIO_ANALYSIS_SUPPORT = False
    logger.warning("Librosa not available - advanced audio analysis disabled")

# Feature availability checking
def check_multimedia_capabilities():
    capabilities = {
        'video_processing': VIDEO_SUPPORT,
        'audio_analysis': AUDIO_ANALYSIS_SUPPORT,
        'ffmpeg_available': check_ffmpeg_installation(),
        'scipy_available': check_scipy_installation()
    }
    
    return capabilities
```

#### **FFmpeg Integration Architecture**
```python
# ✅ CORRECT - Secure FFmpeg integration
class FFmpegWrapper:
    """Safe wrapper for FFmpeg operations."""
    
    def __init__(self):
        self.validate_ffmpeg_installation()
        self.setup_secure_temp_directories()
    
    def convert_video_safely(self, input_path, output_path, codec_params):
        # Input validation and sanitization
        # Secure temporary file handling
        # Error handling with cleanup
        # Progress monitoring with cancellation
        pass
    
    def validate_ffmpeg_installation(self):
        # Verify FFmpeg is properly installed
        # Check for required codecs
        # Validate version compatibility
        pass
```

### Performance Optimization for Multimedia

#### **Parallel Processing Architecture**
```python
# ✅ CORRECT - Multi-threaded multimedia processing
from concurrent.futures import ThreadPoolExecutor

class MultimediaProcessor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or min(4, os.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def process_multiple_files(self, file_list, operation):
        """Process multiple multimedia files in parallel."""
        futures = []
        
        for file_path in file_list:
            future = self.executor.submit(operation, file_path)
            futures.append(future)
        
        # Collect results with progress tracking
        results = []
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            self.update_progress((i + 1) / len(futures) * 100)
        
        return results
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

**Last Updated**: 2025-08-19
**Version**: 1.1 - Multimedia Steganography Architecture
