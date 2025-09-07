# 🧪 InVisioVault Anti-Detection Testing Guide

**Version**: 1.0.0

## 🚀 Quick Start Testing

### **Step 1: Install Requirements**
```bash
# Make sure OpenCV is installed
pip install opencv-python>=4.8.0

# Verify installation
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

### **Step 2: Run Quick Test**
```bash
# From the InVisioVault directory
python quick_test.py
```

**What to look for:**
- ✅ All modules import successfully
- ✅ Hiding and extraction work
- ✅ Risk score < 0.3 (excellent) or < 0.6 (good)
- ✅ "LIKELY SAFE" status for steganalysis tools

### **Step 3: Run Comprehensive Test**
```bash
# For detailed analysis
python test_anti_detection.py
```

**What to look for:**
- ✅ High pass rate (80%+)
- ✅ Anti-Detection mode shows "LIKELY SAFE"
- ✅ Standard mode shows "LIKELY DETECTED" (proves the difference)

---

## 🔍 Testing with Real Steganalysis Tools

### **Install External Tools**

#### **Option 1: StegExpose (Java)**
```bash
# Download StegExpose.jar from:
# https://github.com/b3dk7/StegExpose

# Test usage:
java -jar StegExpose.jar your_stego_image.png
```

#### **Option 2: zsteg (Ruby)**
```bash
# Install Ruby first, then:
gem install zsteg

# Test usage:
zsteg your_stego_image.png
```

#### **Option 3: Online Tools**
- Upload images to online steganalysis services
- Compare results between standard and anti-detection images

### **Testing Workflow**

1. **Create Standard Stego Image** (should be detected):
```python
from core.steganography_engine import SteganographyEngine
engine = SteganographyEngine()
engine.hide_data(carrier, data, "standard_stego.png", randomize=False)
```

2. **Create Anti-Detection Image** (should NOT be detected):
```python
from core.enhanced_steganography_engine import EnhancedSteganographyEngine
engine = EnhancedSteganographyEngine(use_anti_detection=True)
engine.hide_data_enhanced(carrier, data, "anti_detection_stego.png", password="test123")
```

3. **Test Both Images**:
```bash
# Test standard image (should be detected)
java -jar StegExpose.jar standard_stego.png
zsteg standard_stego.png

# Test anti-detection image (should be safe)
java -jar StegExpose.jar anti_detection_stego.png  
zsteg anti_detection_stego.png
```

---

## 📊 Interpreting Results

### **Internal Test Results**

| Risk Score | Status | Meaning |
|------------|--------|---------|
| < 0.3 | 🟢 EXCELLENT | Very low detection risk |
| 0.3 - 0.6 | 🟡 GOOD | Moderate protection |
| 0.6 - 0.8 | 🟠 FAIR | Some detection risk |
| > 0.8 | 🔴 POOR | High detection risk |

### **External Tool Results**

#### **StegExpose Results**:
- ✅ **No output/low confidence**: Anti-detection working
- ❌ **High confidence detection**: Needs improvement
- ⚠️ **Medium confidence**: Moderate protection

#### **zsteg Results**:
- ✅ **No signatures found**: Anti-detection working  
- ❌ **Multiple signatures found**: Standard steganography detected
- ⚠️ **Few weak signatures**: Some protection

### **Example Good Results**:
```
StegExpose: No steganography detected
zsteg: nothing

Internal test: Risk score 0.21, LIKELY SAFE
```

### **Example Bad Results**:
```
StegExpose: Steganography detected with 95% confidence
zsteg: imagedata contains multiple LSB signatures

Internal test: Risk score 0.87, LIKELY DETECTED
```

---

## 🛠️ Troubleshooting

### **Common Issues**

#### **"Import Error" / "Module Not Found"**
```bash
# Fix: Install missing dependencies
pip install opencv-python>=4.8.0 numpy>=1.24.0 Pillow>=9.5.0

# Check imports
python -c "from core.enhanced_steganography_engine import EnhancedSteganographyEngine; print('OK')"
```

#### **"Anti-Detection Not Working" (High Risk Scores)**
```python
# Possible causes:
1. Using simple carrier images (try complex textures)
2. Hiding too much data (use <50% of capacity)  
3. Weak passwords (use strong, unique passwords)
4. Wrong settings (ensure use_anti_detection=True)
```

#### **"Test Files Not Found"**
```bash
# Make sure you're in the right directory
cd path/to/InVisioVault
python quick_test.py
```

### **Performance Issues**
- Anti-detection mode is slower (expected)
- Use smaller test files initially
- Try different carrier images

---

## 🎯 Advanced Testing

### **Batch Testing Script**
```python
# Test multiple images and data sizes
carriers = ['image1.png', 'image2.png', 'image3.png']
test_data_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB

for carrier in carriers:
    for size in test_data_sizes:
        test_data = b"x" * size
        # Test with anti-detection...
        # Record results...
```

### **Comparison Testing**
```python
# Compare different modes
engines = [
    ('Standard', SteganographyEngine()),
    ('Enhanced Fast', EnhancedSteganographyEngine(False)),
    ('Anti-Detection', EnhancedSteganographyEngine(True))
]

# Test all engines with same data
# Compare detection rates
```

### **Stress Testing**
```python
# Test with various image types
image_types = ['gradient', 'noise', 'photo', 'text', 'mixed']
# Test with different data types
data_types = ['text', 'binary', 'encrypted', 'compressed']
# Measure effectiveness
```

---

## 📋 Testing Checklist

### **Before Each Test**
- [ ] OpenCV installed and working
- [ ] Running from InVisioVault directory
- [ ] Test images available
- [ ] External tools installed (optional)

### **Quick Test Checklist**
- [ ] All imports successful
- [ ] Test data creation working
- [ ] Hiding successful
- [ ] Extraction successful  
- [ ] Data integrity verified
- [ ] Risk score calculated
- [ ] Individual tool results shown

### **Comprehensive Test Checklist**
- [ ] Basic functionality tests pass
- [ ] Steganalysis resistance tests show improvement
- [ ] Carrier analysis working
- [ ] Performance comparison completed
- [ ] Overall pass rate > 80%

### **External Tool Testing**
- [ ] Standard stego image created
- [ ] Anti-detection image created
- [ ] StegExpose test completed
- [ ] zsteg test completed
- [ ] Results compared and documented

---

## 🎉 Success Criteria

**Your anti-detection is working well if:**

✅ Internal risk scores are < 0.3  
✅ External tools find nothing or low confidence  
✅ Clear difference between standard and anti-detection images  
✅ Data integrity maintained  
✅ Performance is acceptable for your use case  

**Your anti-detection needs improvement if:**

❌ Risk scores consistently > 0.6  
❌ External tools confidently detect steganography  
❌ No difference between standard and enhanced images  
❌ Data corruption or extraction failures  

---

## 📞 Getting Help

If tests are failing:

1. **Check the logs** - Look for error messages
2. **Verify environment** - Ensure all dependencies installed
3. **Test with simple cases** - Start with small data, simple images
4. **Compare baselines** - Make sure standard steganography IS detected
5. **Check implementation** - Verify you're using the enhanced engine correctly

**Remember**: No steganography is 100% undetectable, but good anti-detection should significantly reduce detection rates compared to standard LSB steganography.

---

*Happy testing! 🕵️‍♂️*
