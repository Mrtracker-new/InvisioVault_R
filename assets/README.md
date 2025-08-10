# 🎨 InvisioVault Assets Directory
### *Application Resources and Media Files*

**Author**: Rolan (RNR)  
**Purpose**: Centralized storage for all application assets  
**Last Updated**: August 2025

---

## 📋 Quick Reference

This directory contains all visual assets, icons, and media files used throughout the InvisioVault application.

| 📁 **Directory** | 🎯 **Purpose** | 📏 **Requirements** |
|----------------|---------------|-----------------|
| `icons/` | Application icons | ICO, PNG, SVG formats |
| `images/` | Logos and branding | High-resolution images |
| `ui/` | Interface graphics | Theme-specific elements |
| `demo/` | Example files | Sample data for testing |

---

## 🎨 Icon Management

### **Application Icons**

#### **Required Formats**
- **Primary**: `InvisioVault.ico` (Windows executable icon)
- **PNG Icons**: Multiple sizes for different platforms
- **SVG Icons**: Vector format for scalability

#### **Recommended Sizes**
| Size | Usage | Platform |
|------|-------|----------|
| 16x16 | Small icons, taskbar | Windows |
| 32x32 | Standard icons | Windows/Linux |
| 48x48 | Large icons | Windows/Linux |
| 64x64 | High DPI icons | macOS |
| 128x128 | Application icon | macOS |
| 256x256 | Retina display | macOS |

### **Icon Guidelines**
- **Style**: Modern, professional appearance
- **Colors**: Consistent with application theme
- **Format**: Transparent backgrounds for PNG
- **Quality**: High resolution, crisp edges

---

## 🖼️ Image Assets

### **Branding Materials**
- **Logo**: Primary InvisioVault logo
- **Wordmark**: Text-based logo variations
- **Favicon**: Web and documentation icon
- **Splash Screen**: Application startup image

### **Documentation Images**
- **Screenshots**: Feature demonstrations
- **Diagrams**: Technical illustrations
- **Tutorials**: Step-by-step visuals
- **Comparisons**: Before/after examples

---

## 🎭 UI Graphics

### **Theme-Specific Elements**
- **Dark Theme**: Graphics optimized for dark backgrounds
- **Light Theme**: Graphics optimized for light backgrounds
- **High Contrast**: Accessibility-compliant versions

### **Interface Elements**
- **Buttons**: Custom button graphics
- **Backgrounds**: Subtle textures and patterns
- **Separators**: Visual dividers and lines
- **Progress Indicators**: Custom loading animations

---

## 📚 Demo Assets

### **Example Files**
- **Sample Images**: Carrier images for testing
- **Test Documents**: Files to hide during demos
- **Keyfiles**: Example authentication files
- **Outputs**: Expected results for validation

### **Guidelines**
- **No Personal Data**: Only generic, safe content
- **Small Sizes**: Efficient for repository storage
- **Varied Formats**: Different file types for testing
- **Clear Naming**: Descriptive filenames

---

## 📂 Directory Structure

```
📁 assets/
├── 🎨 icons/                    # Application icons
│   ├── InvisioVault.ico          # Primary Windows icon
│   ├── app-16.png                # Small icon (16x16)
│   ├── app-32.png                # Standard icon (32x32)
│   ├── app-48.png                # Large icon (48x48)
│   ├── app-64.png                # High DPI icon (64x64)
│   ├── app-128.png               # macOS icon (128x128)
│   ├── app-256.png               # Retina icon (256x256)
│   └── app-icon.svg              # Vector icon
│
├── 🖼️ images/                   # General images and branding
│   ├── logo-main.png             # Primary logo
│   ├── logo-wordmark.png         # Text logo
│   ├── favicon.png               # Web favicon
│   ├── splash-screen.png         # Startup image
│   └── screenshots/              # Application screenshots
│       ├── main-window.png       # Main interface
│       ├── hide-dialog.png       # Hide files dialog
│       └── extract-dialog.png    # Extract files dialog
│
├── 🎭 ui/                       # UI-specific graphics
│   ├── themes/                   # Theme-specific assets
│   │   ├── dark/                 # Dark theme graphics
│   │   └── light/                # Light theme graphics
│   ├── buttons/                  # Custom button graphics
│   ├── backgrounds/              # Background textures
│   └── progress/                 # Loading animations
│
└── 📚 demo/                     # Example and demo files
    ├── carrier-images/           # Sample images for hiding
    │   ├── sample-photo.png      # Test carrier image
    │   └── test-image.bmp        # Another test image
    ├── test-files/               # Files to hide in demos
    │   ├── sample-doc.txt        # Text document
    │   └── test-data.pdf         # PDF document
    └── keyfiles/                 # Example keyfiles
        └── example.key           # Sample keyfile
```

---

## 🔧 Usage Guidelines

### **Code References**
```python
# Reference assets from project root
icon_path = "assets/icons/InvisioVault.ico"
logo_path = "assets/images/logo-main.png"

# PyInstaller resource bundling
# Use relative paths for executable compatibility
resource_path = os.path.join(os.path.dirname(__file__), "assets", "icons", "app-32.png")
```

### **Best Practices**
- ✅ **Use relative paths** from project root
- ✅ **Check file existence** before loading
- ✅ **Handle missing assets** gracefully
- ✅ **Optimize file sizes** for performance
- ✅ **Use appropriate formats** (PNG for icons, SVG for vectors)

### **File Naming Convention**
- **Icons**: `app-{size}.{format}` (e.g., `app-32.png`)
- **Logos**: `logo-{variant}.{format}` (e.g., `logo-wordmark.png`)
- **UI Elements**: `{element}-{theme}.{format}` (e.g., `button-dark.png`)
- **Demo Files**: `{type}-{description}.{format}` (e.g., `sample-photo.png`)

---

## 🛠️ Maintenance

### **Regular Tasks**
- [ ] **Optimize file sizes** for faster loading
- [ ] **Update screenshots** when UI changes
- [ ] **Check broken references** in code
- [ ] **Maintain consistent style** across assets

### **Version Control**
- **Include**: Essential icons and branding
- **Exclude**: Large demo files (use .gitignore)
- **Compress**: Use appropriate compression for images
- **Document**: Update this README when adding assets

---

**Last Updated**: August 10, 2025  
**Version**: 1.0.0  
**Author**: Rolan (RNR)  
**License**: MIT Educational License
