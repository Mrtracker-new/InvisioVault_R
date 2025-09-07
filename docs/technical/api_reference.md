# 📚 InvisioVault API Reference
### *Complete Technical Documentation for Developers*

**Version**: 1.0.0
**Author**: Rolan (RNR)  
**Purpose**: Your complete API guide for building with InvisioVault  
**Last Updated**: August 2025

---

<div align="center">

### 🚀 **Build Powerful Steganography Applications**

*Everything you need to integrate InvisioVault into your projects*

</div>

## 🗺️ Quick API Navigation

### 🔧 **Core APIs**
- [🔍 Core Engines](#-core-engines) • [🎬 Multimedia Engines](#-multimedia-engines) • [🎨 UI Components](#-ui-components) • [🔧 Utilities](#-utilities)

### ⚙️ **Operations & Config**
- [⚡ Operations](#-operations) • [📝 Configuration](#-configuration) • [🚫 Error Handling](#-error-handling)

### 💡 **Examples & Integration**
- [🚀 Code Examples](#-examples) • [🎆 Quick Start](#quick-start-examples) • [⚠️ Security Warnings](#security-warnings)

---

## 📋 Table of Contents

1. [🔧 **Core Engines**](#-core-engines) - Steganography and encryption APIs
2. [🎬 **Multimedia Engines**](#-multimedia-engines) - Video and audio steganography APIs
3. [🎨 **UI Components**](#-ui-components) - Interface building blocks
4. [🔧 **Utilities**](#-utilities) - Helper functions and tools
5. [⚡ **Operations**](#-operations) - High-level operation APIs
6. [📝 **Configuration**](#-configuration) - Settings and preferences
7. [🚫 **Error Handling**](#-error-handling) - Exception management
8. [🚀 **Examples**](#-examples) - Practical code examples

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

##### **`hide_data_with_password(carrier_path, data, output_path, password, use_secure_mode=None)`**
```python
def hide_data_with_password(self, carrier_path, data: bytes, output_path, 
                           password: str, use_secure_mode: Optional[bool] = None) -> bool:
    """
    Hide data with password-based security (recommended method).
    
    Args:
        carrier_path: Path to carrier image (PNG/BMP/TIFF)
        data: Binary data to hide
        output_path: Path for output steganographic image
        password: Password for encryption and randomization
        use_secure_mode: Override default secure mode setting
    
    Returns:
        bool: Success status
        
    Raises:
        FileNotFoundError: Carrier image not found
        ValueError: Image format not supported or insufficient capacity
        SteganographyError: Hiding operation failed
    """
```

##### **`extract_data_with_password(stego_path, password, use_secure_mode=None)`**
```python
def extract_data_with_password(self, stego_path, password: str,
                              use_secure_mode: Optional[bool] = None) -> bytes:
    """
    Extract hidden data with password-based security.
    
    Args:
        stego_path: Path to steganographic image
        password: Password for decryption and positioning
        use_secure_mode: Override default secure mode setting
    
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

## 🎬 Multimedia Engines

### **VideoSteganographyEngine** ⭐ *NEW*

#### **Class Overview**
```python
class VideoSteganographyEngine:
    """Advanced video steganography using frame-based LSB embedding."""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mkv', '.mov']
        self.frame_selection_ratio = 0.1  # Use 10% of frames
        self.quality_threshold = 0.8      # Minimum quality preservation
        self.logger = Logger()
```

#### **Key Methods**

##### **`hide_data_in_video(video_path, data, output_path, password)`**
```python
def hide_data_in_video(self, video_path: Path, data: bytes, 
                      output_path: Path, password: str) -> bool:
    """
    Hide data in video using frame-based LSB steganography.
    
    Args:
        video_path: Path to input video file
        data: Binary data to hide
        output_path: Path for output video with hidden data
        password: Password for seeded randomization
    
    Returns:
        bool: Success status
        
    Raises:
        FileNotFoundError: Video file not found
        ValueError: Unsupported video format or insufficient capacity
        MultimediaError: Video processing failed
    """
```

##### **`extract_data_from_video(video_path, password)`**
```python
def extract_data_from_video(self, video_path: Path, password: str) -> bytes:
    """
    Extract hidden data from steganographic video.
    
    Args:
        video_path: Path to steganographic video
        password: Password for extraction
    
    Returns:
        bytes: Extracted data or None if extraction failed
        
    Raises:
        FileNotFoundError: Video file not found
        MultimediaError: Extraction failed or no data found
    """
```

##### **`analyze_video_capacity(video_path)`**
```python
def analyze_video_capacity(self, video_path: Path) -> Dict[str, Any]:
    """
    Analyze video file for steganography capacity.
    
    Args:
        video_path: Path to video file
    
    Returns:
        dict: {
            'total_frames': int,
            'usable_frames': int,
            'capacity_bytes': int,
            'duration_seconds': float,
            'resolution': tuple,
            'fps': float,
            'codec': str,
            'quality_score': float  # 0.0-1.0
        }
    """
```

### **AudioSteganographyEngine** ⭐ *NEW*

#### **Class Overview**
```python
class AudioSteganographyEngine:
    """Multi-technique audio steganography engine."""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.flac', '.mp3', '.aac']
        self.techniques = ['lsb', 'spread_spectrum', 'phase_coding']
        self.logger = Logger()
```

#### **Audio Techniques**
```python
class AudioTechnique(Enum):
    LSB = "lsb"                    # Least Significant Bit embedding
    SPREAD_SPECTRUM = "spread_spectrum"  # Frequency domain spreading
    PHASE_CODING = "phase_coding"       # Phase relationship manipulation
```

#### **Key Methods**

##### **`hide_data_lsb(audio_path, data, output_path, password)`**
```python
def hide_data_lsb(self, audio_path: Path, data: bytes, 
                 output_path: Path, password: str) -> bool:
    """
    Hide data in audio using LSB embedding technique.
    
    Args:
        audio_path: Path to input audio file
        data: Binary data to hide
        output_path: Path for output audio with hidden data
        password: Password for seeded randomization
    
    Returns:
        bool: Success status
        
    Raises:
        FileNotFoundError: Audio file not found
        ValueError: Unsupported audio format or insufficient capacity
        MultimediaError: Audio processing failed
    """
```

##### **`hide_data_spread_spectrum(audio_path, data, output_path, password)`**
```python
def hide_data_spread_spectrum(self, audio_path: Path, data: bytes, 
                             output_path: Path, password: str) -> bool:
    """
    Hide data using spread spectrum technique in frequency domain.
    
    Args:
        audio_path: Path to input audio file
        data: Binary data to hide
        output_path: Path for output audio with hidden data
        password: Password for frequency selection
    
    Returns:
        bool: Success status
    """
```

##### **`extract_data_from_audio(audio_path, password, technique=None)`**
```python
def extract_data_from_audio(self, audio_path: Path, password: str, 
                           technique: AudioTechnique = None) -> bytes:
    """
    Extract hidden data from steganographic audio.
    
    Args:
        audio_path: Path to steganographic audio
        password: Password for extraction
        technique: Specific technique to use (auto-detected if None)
    
    Returns:
        bytes: Extracted data or None if extraction failed
        
    Raises:
        FileNotFoundError: Audio file not found
        MultimediaError: Extraction failed or no data found
    """
```

##### **`select_optimal_technique(audio_path)`**
```python
def select_optimal_technique(self, audio_path: Path) -> AudioTechnique:
    """
    Analyze audio and recommend optimal steganography technique.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        AudioTechnique: Recommended technique based on audio characteristics
    """
```

### **MultimediaAnalyzer** ⭐ *NEW*

#### **Class Overview**
```python
class MultimediaAnalyzer:
    """Comprehensive multimedia file analysis and capacity assessment."""
    
    def __init__(self):
        self.supported_video_formats = ['.mp4', '.avi', '.mkv', '.mov']
        self.supported_audio_formats = ['.wav', '.flac', '.mp3', '.aac']
        self.logger = Logger()
```

#### **Key Methods**

##### **`analyze_multimedia_file(file_path)`**
```python
def analyze_multimedia_file(self, file_path: Path) -> Dict[str, Any]:
    """
    Comprehensive analysis of multimedia file for steganography.
    
    Args:
        file_path: Path to multimedia file
    
    Returns:
        dict: {
            'file_type': str,           # 'video' or 'audio'
            'format': str,              # File format (mp4, wav, etc.)
            'size_bytes': int,          # File size
            'capacity_bytes': int,      # Steganography capacity
            'quality_score': float,     # Quality assessment (0.0-1.0)
            'recommended_technique': str, # Optimal technique
            'processing_time_estimate': float, # Estimated processing time
            'metadata': dict           # Format-specific metadata
        }
        
    Raises:
        FileNotFoundError: File not found
        ValueError: Unsupported file format
    """
```

##### **`batch_analyze_files(file_paths)`**
```python
def batch_analyze_files(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Analyze multiple multimedia files for batch processing.
    
    Args:
        file_paths: List of paths to multimedia files
    
    Returns:
        List[Dict]: Analysis results for each file
    """
```

##### **`calculate_total_capacity(file_paths)`**
```python
def calculate_total_capacity(self, file_paths: List[Path]) -> Dict[str, Any]:
    """
    Calculate total steganography capacity across multiple files.
    
    Args:
        file_paths: List of multimedia file paths
    
    Returns:
        dict: {
            'total_capacity_bytes': int,
            'total_files': int,
            'video_files': int,
            'audio_files': int,
            'estimated_processing_time': float
        }
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

### **⚠️ Multi-Image Distribution Security Warnings**

```python
from ui.dialogs.two_factor_dialog import TwoFactorWorkerThread
from pathlib import Path

# CRITICAL WARNING: Multi-image distribution creates fragment dependencies
# If ANY fragment is lost → TOTAL DATA LOSS occurs

# Example: Fragment distribution (HIGH RISK)
worker = TwoFactorWorkerThread(
    "distribute",
    carrier_paths=[
        "/secure/location1/fragment1.png",  # ⚠️ CRITICAL: Must remain accessible
        "/secure/location2/fragment2.png",  # ⚠️ CRITICAL: Must remain accessible  
        "/secure/location3/fragment3.png"   # ⚠️ CRITICAL: Must remain accessible
    ],
    files_to_hide=["/secret/document.pdf"],
    password="FragmentPassword123",
    output_dir="/fragments/output"
)

# MANDATORY BACKUP STRATEGY:
def create_fragment_backups(fragment_paths):
    """
    ⚠️ CRITICAL: Create multiple complete backup sets of ALL fragments.
    
    Losing ANY fragment = losing ALL your data permanently!
    """
    backup_locations = [
        "/backup/local/",      # Local backup drive
        "/backup/cloud/",      # Cloud storage
        "/backup/external/"    # External USB/drive
    ]
    
    for backup_location in backup_locations:
        for fragment in fragment_paths:
            # Copy each fragment to each backup location
            backup_path = Path(backup_location) / Path(fragment).name
            shutil.copy2(fragment, backup_path)
            
            # Verify backup integrity
            if not verify_file_integrity(fragment, backup_path):
                raise Exception(f"Backup verification failed for {fragment}")
    
    print("✅ All fragments backed up to multiple secure locations")

# FRAGMENT DEPENDENCY WARNING
def validate_fragment_availability(fragment_paths):
    """
    Verify ALL fragments are accessible before attempting reconstruction.
    Missing ANY fragment will result in TOTAL DATA LOSS!
    """
    missing_fragments = []
    
    for fragment in fragment_paths:
        if not Path(fragment).exists():
            missing_fragments.append(fragment)
    
    if missing_fragments:
        raise FragmentLossError(
            f"CRITICAL: Missing fragments detected: {missing_fragments}. "
            "CANNOT RECOVER DATA! All fragments required for reconstruction."
        )
    
    return True

# Example with safety checks
try:
    # 1. Validate all fragments exist
    fragment_paths = [
        "/fragments/fragment_01.png",
        "/fragments/fragment_02.png", 
        "/fragments/fragment_03.png"
    ]
    
    validate_fragment_availability(fragment_paths)
    
    # 2. Create mandatory backups
    create_fragment_backups(fragment_paths)
    
    # 3. Proceed with reconstruction only if all fragments available
    worker = TwoFactorWorkerThread(
        "reconstruct",
        fragment_paths=fragment_paths,
        password="FragmentPassword123",
        output_dir="/reconstructed/"
    )
    
except FragmentLossError as e:
    print(f"🚨 CATASTROPHIC ERROR: {e}")
    print("🔴 DATA RECOVERY IMPOSSIBLE - PERMANENT DATA LOSS")
```

### **🎬 Multimedia Steganography Examples** ⭐ *NEW*

#### **Video Steganography Example**
```python
from core.video_steganography_engine import VideoSteganographyEngine
from pathlib import Path

# Initialize video steganography engine
video_engine = VideoSteganographyEngine()

# Analyze video capacity first
video_path = Path("vacation_video.mp4")
analysis = video_engine.analyze_video_capacity(video_path)

print(f"Video Analysis:")
print(f"- Total frames: {analysis['total_frames']}")
print(f"- Usable frames: {analysis['usable_frames']}")
print(f"- Capacity: {analysis['capacity_bytes']} bytes")
print(f"- Quality score: {analysis['quality_score']:.2f}")

# Hide data in video
secret_files = [Path("document.pdf"), Path("photo.jpg")]
password = "VideoPassword123"

# Prepare data (compress and encrypt files)
from core.file_operations import FileOperations
file_ops = FileOperations()
compressed_data = file_ops.compress_files(secret_files)

# Hide in video
success = video_engine.hide_data_in_video(
    video_path=video_path,
    data=compressed_data,
    output_path=Path("hidden_vacation.mp4"),
    password=password
)

if success:
    print("✅ Files hidden in video successfully!")
    
    # Extract from video
    extracted_data = video_engine.extract_data_from_video(
        video_path=Path("hidden_vacation.mp4"),
        password=password
    )
    
    if extracted_data:
        # Decompress and restore files
        file_ops.decompress_files(extracted_data, Path("extracted_from_video"))
        print("✅ Files extracted from video successfully!")
```

#### **Audio Steganography Example**
```python
from core.audio_steganography_engine import AudioSteganographyEngine, AudioTechnique
from pathlib import Path

# Initialize audio steganography engine
audio_engine = AudioSteganographyEngine()

# Analyze audio file and select optimal technique
audio_path = Path("music.wav")
optimal_technique = audio_engine.select_optimal_technique(audio_path)

print(f"Optimal technique for {audio_path.name}: {optimal_technique.value}")

# Hide data using recommended technique
secret_message = b"This is a secret message hidden in audio!"
password = "AudioPassword456"

if optimal_technique == AudioTechnique.LSB:
    success = audio_engine.hide_data_lsb(
        audio_path=audio_path,
        data=secret_message,
        output_path=Path("hidden_music.wav"),
        password=password
    )
elif optimal_technique == AudioTechnique.SPREAD_SPECTRUM:
    success = audio_engine.hide_data_spread_spectrum(
        audio_path=audio_path,
        data=secret_message,
        output_path=Path("hidden_music.wav"),
        password=password
    )

if success:
    print(f"✅ Data hidden using {optimal_technique.value} technique!")
    
    # Extract data (auto-detects technique)
    extracted_data = audio_engine.extract_data_from_audio(
        audio_path=Path("hidden_music.wav"),
        password=password
    )
    
    if extracted_data:
        print(f"✅ Extracted: {extracted_data.decode()}")
```

#### **Multimedia Analysis Example**
```python
from core.multimedia_analyzer import MultimediaAnalyzer
from pathlib import Path

# Initialize multimedia analyzer
analyzer = MultimediaAnalyzer()

# Analyze multiple multimedia files
files_to_analyze = [
    Path("video1.mp4"),
    Path("video2.avi"),
    Path("audio1.wav"),
    Path("audio2.flac")
]

# Batch analysis
analysis_results = analyzer.batch_analyze_files(files_to_analyze)

for result in analysis_results:
    print(f"\n{result['file_path']}:")
    print(f"  Type: {result['file_type']}")
    print(f"  Format: {result['format']}")
    print(f"  Size: {result['size_bytes']:,} bytes")
    print(f"  Capacity: {result['capacity_bytes']:,} bytes")
    print(f"  Quality: {result['quality_score']:.2f}")
    print(f"  Recommended: {result['recommended_technique']}")
    print(f"  Est. processing: {result['processing_time_estimate']:.1f}s")

# Calculate total capacity
total_capacity = analyzer.calculate_total_capacity(files_to_analyze)

print(f"\nὌa TOTAL ANALYSIS:")
print(f"Total capacity: {total_capacity['total_capacity_bytes']:,} bytes")
print(f"Total files: {total_capacity['total_files']}")
print(f"Video files: {total_capacity['video_files']}")
print(f"Audio files: {total_capacity['audio_files']}")
print(f"Est. total processing: {total_capacity['estimated_processing_time']:.1f}s")
```

#### **Combined Multimedia + Multi-Decoy Example**
```python
from core.multi_decoy_engine import MultiDecoyEngine
from core.multimedia_analyzer import MultimediaAnalyzer
from pathlib import Path

# 🎉 ADVANCED: Multimedia files with multi-decoy protection
engine = MultiDecoyEngine()
analyzer = MultimediaAnalyzer()

# Define datasets using multimedia carriers
video_carrier = Path("family_vacation.mp4")
audio_carrier = Path("favorite_song.wav")

# Analyze carriers first
video_analysis = analyzer.analyze_multimedia_file(video_carrier)
audio_analysis = analyzer.analyze_multimedia_file(audio_carrier)

print(f"Video capacity: {video_analysis['capacity_bytes']:,} bytes")
print(f"Audio capacity: {audio_analysis['capacity_bytes']:,} bytes")

# Create sophisticated multi-layer datasets
multimedia_datasets = [
    {
        "name": "Travel Photos",
        "password": "vacation2024",
        "priority": 1,  # Outer layer
        "decoy_type": "innocent",
        "files": ["beach1.jpg", "beach2.jpg"],
        "carrier": video_carrier,  # Use video as carrier
        "carrier_type": "video"
    },
    {
        "name": "Personal Documents",
        "password": "secure_docs_789",
        "priority": 5,  # Inner layer
        "decoy_type": "personal",
        "files": ["passport_scan.pdf", "bank_statement.pdf"],
        "carrier": audio_carrier,  # Use audio as carrier
        "carrier_type": "audio"
    }
]

# Hide datasets in multimedia carriers
for dataset in multimedia_datasets:
    if dataset['carrier_type'] == 'video':
        from core.video_steganography_engine import VideoSteganographyEngine
        video_engine = VideoSteganographyEngine()
        
        # Prepare and hide data in video
        success = video_engine.hide_data_in_video(
            video_path=dataset['carrier'],
            data=prepare_dataset_data(dataset['files']),
            output_path=Path(f"hidden_{dataset['name'].lower().replace(' ', '_')}.mp4"),
            password=dataset['password']
        )
        
    elif dataset['carrier_type'] == 'audio':
        from core.audio_steganography_engine import AudioSteganographyEngine
        audio_engine = AudioSteganographyEngine()
        
        # Prepare and hide data in audio
        success = audio_engine.hide_data_lsb(
            audio_path=dataset['carrier'],
            data=prepare_dataset_data(dataset['files']),
            output_path=Path(f"hidden_{dataset['name'].lower().replace(' ', '_')}.wav"),
            password=dataset['password']
        )
    
    if success:
        print(f"✅ {dataset['name']} hidden in {dataset['carrier_type']}!")

print("Ὂa Multi-layer multimedia steganography complete!")
print("ὑ0 Different passwords will reveal different datasets from different media types!")
```

---

**Last Updated**: August 2025  
**Version**: 1.0.0 - Multimedia Steganography Edition
**Author**: Rolan (RNR)  
**License**: MIT Educational License
