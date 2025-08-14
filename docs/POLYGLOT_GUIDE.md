# InVisioVault PNG/EXE Polyglot Creation Guide

**Part of InVisioVault Advanced Steganography Suite**  
*Created by Rolan (RNR) for Educational Excellence*

This guide provides complete solutions for creating files that function as both valid PNG images and Windows executables using InVisioVault's advanced polyglot features, solving the common issues you might encounter.

## Table of Contents

1. [Understanding the Problem](#understanding-the-problem)
2. [Solution Approaches](#solution-approaches)
3. [Method 1: EXE-First (Recommended)](#method-1-exe-first-recommended)
4. [Method 2: PNG-First with Embedding](#method-2-png-first-with-embedding)
5. [Method 3: PNG-First with DOS Stub Modification](#method-3-png-first-with-dos-stub-modification)
6. [Troubleshooting Common Issues](#troubleshooting-common-issues)
7. [Testing and Verification](#testing-and-verification)
8. [Advanced Techniques](#advanced-techniques)

## Understanding the Problem

### Your Issues Explained

**"This PC can't run this app" Error:**
- Windows PE loader expects specific header alignment
- DOS stub must point to correct PE header location
- File structure conflicts prevent proper execution

**"File format not supported" Error:**
- PNG parsers are strict about chunk structure
- Invalid CRC checksums cause rejection
- Missing or malformed IHDR/IEND chunks

### Root Cause Analysis

The fundamental challenge is that PNG and PE formats have incompatible requirements:

- **PNG Format**: Must start with `\x89PNG\r\n\x1a\n` signature
- **PE Format**: Must start with `MZ` signature followed by DOS stub

## Solution Approaches

### Approach Comparison

| Method | PNG Compatibility | EXE Compatibility | Implementation Difficulty | Reliability |
|--------|-------------------|-------------------|--------------------------|-------------|
| EXE-First Overlay | ⚠️ (requires PNG offset) | ✅ Excellent | Low | High |
| PNG-First Embedded | ✅ Excellent | ⚠️ (requires extraction) | Medium | High |
| PNG-First Direct | ✅ Good | ❌ Poor | High | Low |
| Hybrid Approach | ✅ Good | ✅ Good | High | Medium |

## Method 1: EXE-First (Recommended)

This is the most reliable approach for ensuring execution compatibility.

### Technical Details

1. **Structure**: `[PE File] + [Padding] + [PNG Data] + [Metadata]`
2. **Execution**: Windows loads PE directly, ignores PNG data as overlay
3. **Image Viewing**: Requires PNG location marker or extraction

### Implementation

```python
# Use advanced_polyglot.py
python advanced_polyglot.py your_program.exe [optional_image.png] output.exe
```

### Step-by-Step Process

1. **Start with valid PE file**
   ```
   PE Headers: [DOS] [PE] [Sections...]
   ```

2. **Add PNG data as overlay**
   ```
   [PE File] [Padding to 512-byte boundary] [PNG Data]
   ```

3. **Add location markers**
   ```
   [PE + PNG] [Metadata with PNG offset information]
   ```

### Code Example

```python
from advanced_polyglot import AdvancedPolyglotCreator

creator = AdvancedPolyglotCreator()
success = creator.create_hybrid_polyglot_v2(
    pe_path="your_program.exe",
    png_path="your_image.png",  # Optional
    output_path="polyglot.exe"
)
```

## Method 2: PNG-First with Embedding

This approach guarantees PNG compatibility by embedding the PE in PNG chunks.

### Technical Details

1. **Structure**: `[PNG Headers] + [PNG Chunks] + [PE Chunk] + [IEND]`
2. **PNG View**: Standard PNG, PE chunk ignored by viewers
3. **Execution**: Extract PE from chunk, then run

### Implementation

```python
# Use existing polyglot_creator.py or png_first_polyglot.py
python png_first_polyglot.py image.png program.exe output.png embedded
```

### Extraction and Execution

```python
# Extract embedded PE
python png_first_polyglot.py --extract output.png

# Or use the original extractor
python polyglot_creator.py --extract output.png
```

### Advantages

✅ Perfect PNG compatibility
✅ No image viewer issues  
✅ Secure execution (manual extraction)

### Disadvantages

❌ Requires extraction step
❌ Not direct execution
❌ Larger file size (compression helps)

## Method 3: PNG-First with DOS Stub Modification

Advanced technique that modifies the PE structure to handle PNG prefix.

### Technical Details

1. **Structure**: `[PNG Data] + [Padding] + [Modified PE]`
2. **Key Innovation**: Custom DOS stub with adjusted PE offset
3. **Challenge**: Complex PE header manipulation

### Implementation

```python
# Use png_first_polyglot.py with direct method
python png_first_polyglot.py image.png program.exe output.png direct
```

### DOS Stub Modification Process

1. **Calculate PNG size and padding**
   ```python
   png_size = len(png_data)
   padding = 512 - (png_size % 512) if png_size % 512 != 0 else 0
   pe_start_offset = png_size + padding
   ```

2. **Modify PE's e_lfanew field**
   ```python
   original_pe_offset = struct.unpack('<I', pe_data[60:64])[0]
   new_pe_offset = pe_start_offset + original_pe_offset
   modified_pe[60:64] = struct.pack('<I', new_pe_offset)
   ```

3. **Assemble polyglot**
   ```python
   result = png_data + padding + modified_pe
   ```

## Troubleshooting Common Issues

### "This PC can't run this app"

**Possible Causes:**
- Invalid PE header offset
- Corrupted DOS stub
- Misaligned sections
- Windows SmartScreen blocking

**Solutions:**

1. **Verify PE structure**
   ```bash
   python polyglot_verifier.py your_polyglot.exe
   ```

2. **Check alignment**
   - Ensure 512-byte boundaries
   - Verify e_lfanew points to correct location

3. **Test with simple executable**
   ```bash
   python advanced_polyglot.py --create-test
   ```

4. **Disable SmartScreen temporarily**
   - Run as administrator
   - Add security exception

### "File format not supported"

**Possible Causes:**
- Invalid PNG signature
- Corrupted chunk CRC
- Missing IHDR or IEND chunks

**Solutions:**

1. **Validate PNG structure**
   ```bash
   python polyglot_verifier.py your_polyglot.png
   ```

2. **Check chunk integrity**
   - Verify CRC checksums
   - Ensure proper chunk ordering

3. **Test with different image viewers**
   - Windows Photos
   - Paint
   - Browser (Chrome, Firefox)

### File Size Issues

**Large files (>10MB):**
- Use compression in embedded method
- Consider splitting large PEs
- Optimize PNG images

**Code Example:**
```python
# Compress PE data before embedding
import gzip
compressed_pe = gzip.compress(pe_data)
# Can reduce size by 50-80%
```

## Testing and Verification

### Comprehensive Testing

```bash
# Verify both formats work
python polyglot_verifier.py polyglot_file.exe

# Test PNG compatibility
copy polyglot_file.exe test_image.png
# Open test_image.png in image viewer

# Test EXE functionality
polyglot_file.exe
# Should execute normally
```

### Test Checklist

**PNG Tests:**
- [ ] Displays in Windows Photos
- [ ] Opens in web browsers
- [ ] Loads in Paint/image editors
- [ ] Shows correct dimensions
- [ ] No error messages

**EXE Tests:**
- [ ] Runs without "can't run app" error
- [ ] Executes main program logic
- [ ] Exit code is correct
- [ ] No SmartScreen warnings
- [ ] Works on different Windows versions

**Structure Tests:**
- [ ] File has both PNG and PE signatures
- [ ] Headers are properly aligned
- [ ] CRC checksums are valid
- [ ] PE sections are intact

## Advanced Techniques

### Custom PNG Chunks

Create application-specific chunks that won't interfere with viewers:

```python
def create_custom_chunk(chunk_type, data):
    """Create PNG chunk with proper CRC."""
    chunk_crc = zlib.crc32(chunk_type + data) & 0xffffffff
    return (
        struct.pack('>I', len(data)) +    # Length
        chunk_type +                       # Type (4 bytes)
        data +                            # Data
        struct.pack('>I', chunk_crc)      # CRC
    )

# Example: Store metadata
metadata_chunk = create_custom_chunk(b'mEta', b'Created by polyglot tool')
```

### PE Section Manipulation

For advanced users, you can modify PE sections to create space:

```python
def find_pe_gaps(pe_data):
    """Find unused space in PE file structure."""
    # Analyze section table
    # Find gaps between sections
    # Return available space locations
    pass
```

### Multi-Format Support

Extend to support additional formats:

```python
# Example: PNG/EXE/ZIP polyglot
# Structure: [PNG][ZIP central directory][PE][ZIP data]
```

## Working Examples

### Example 1: Simple Console App

```bash
# Create test executable
echo 'print("Hello from polyglot!")' > test.py
python -m py_compile test.py
# Use existing EXE or create one with PyInstaller

# Create polyglot
python advanced_polyglot.py test.exe simple_polyglot.exe

# Test both formats
simple_polyglot.exe  # Should print message
copy simple_polyglot.exe test.png  # Should display as image
```

### Example 2: GUI Application

```bash
# Use existing GUI app
python advanced_polyglot.py notepad.exe my_image.png gui_polyglot.exe

# Test
gui_polyglot.exe  # Should open notepad
# Rename to .png to view image
```

### Example 3: Embedded Approach

```bash
# Create PNG-first polyglot with extraction
python polyglot_creator.py image.png program.exe embedded_poly

# To run the program
python polyglot_creator.py --extract embedded_poly
```

## Security Considerations

### Antivirus Software

Many antivirus programs flag polyglot files as suspicious:

**Mitigation strategies:**
- Add security exceptions
- Submit to vendors for whitelisting
- Use code signing certificates
- Document legitimate use cases

### Best Practices

1. **Always validate inputs**
   - Check PE integrity
   - Verify PNG structure
   - Sanitize file sizes

2. **Use safe execution**
   - Test in isolated environments
   - Limit execution timeouts
   - Validate exit codes

3. **Clear documentation**
   - Document polyglot purpose
   - Include extraction instructions
   - Provide verification tools

## Conclusion

Creating PNG/EXE polyglots requires careful attention to both format specifications. The **EXE-first approach** is recommended for maximum execution compatibility, while the **PNG-first embedded approach** ensures perfect image compatibility.

Choose the method that best fits your specific requirements:

- **Need reliable execution?** → Use EXE-first method
- **Need perfect PNG compatibility?** → Use embedded method  
- **Need direct dual functionality?** → Use DOS stub modification (advanced)

All three approaches are implemented in the provided tools. Start with the verification tool to diagnose any existing polyglot issues, then use the appropriate creation method for your needs.

## Quick Start Commands

```bash
# Create EXE-first polyglot (recommended)
python advanced_polyglot.py program.exe image.png output.exe

# Create PNG-first embedded polyglot  
python png_first_polyglot.py image.png program.exe output.png embedded

# Verify any polyglot file
python polyglot_verifier.py polyglot_file

# Test with demo files
python advanced_polyglot.py --create-test
```

The tools handle all the complex header manipulation, alignment, and verification automatically, solving the issues you were experiencing with manual approaches.
