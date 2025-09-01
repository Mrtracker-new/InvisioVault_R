#!/usr/bin/env python3
"""
Audio Steganography Demo Script

This script demonstrates the new audio steganography system with various
features including different techniques, security levels, and error recovery.

Usage:
    python examples/audio_steganography_demo.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Try importing the new audio steganography system
    from core.audio.audio_steganography import (
        AudioSteganographyEngine, 
        EmbeddingConfig,
        create_audio_steganography_engine
    )
    from core.encryption_engine import SecurityLevel
    from utils.logger import Logger
    
    # Also try importing audio libraries to create test files
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("pip install librosa soundfile pydub scipy numpy")
    sys.exit(1)


def create_test_audio_file(output_path: Path, duration: float = 10.0, 
                          sample_rate: int = 44100) -> bool:
    """
    Create a test audio file for demonstration purposes.
    
    Args:
        output_path: Where to save the test audio
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Success status
    """
    try:
        print(f"🎵 Creating test audio file: {output_path.name}")
        
        # Generate simple test audio (sine waves)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create stereo audio with different frequencies for left/right
        left_channel = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4)
        right_channel = 0.3 * np.sin(2 * np.pi * 523 * t)  # 523 Hz (C5)
        
        # Combine channels
        audio_data = np.vstack([left_channel, right_channel])
        
        # Save as WAV file
        sf.write(str(output_path), audio_data.T, sample_rate)
        
        print(f"✅ Created {duration}s test audio at {sample_rate}Hz")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test audio: {e}")
        return False


def demonstrate_basic_usage():
    """Demonstrate basic hide and extract operations."""
    print("\n" + "="*60)
    print("🔹 BASIC USAGE DEMONSTRATION")
    print("="*60)
    
    # Create test audio file
    test_audio = Path("temp_carrier.wav")
    if not create_test_audio_file(test_audio, duration=5.0):
        return
    
    try:
        # Initialize engine
        print("\n🚀 Initializing Audio Steganography Engine...")
        engine = create_audio_steganography_engine(SecurityLevel.STANDARD)
        
        # Create configuration
        config = engine.create_config(
            technique='lsb',
            mode='balanced',
            password='DemoPassword123!',
            randomize_positions=True
        )
        
        print(f"⚙️  Configuration: {config.technique} technique, {config.mode} mode")
        
        # Test data to hide
        secret_message = "Hello from the new InVisioVault audio steganography system! 🎵🔐"
        print(f"📝 Secret message: {secret_message}")
        
        # Analyze capacity first
        print("\n📊 Analyzing carrier capacity...")
        capacity_info = engine.analyze_capacity(test_audio, config.technique)
        
        if 'error' in capacity_info:
            print(f"❌ Capacity analysis failed: {capacity_info['error']}")
            return
        
        print(f"📈 Available capacity: {capacity_info.get('effective_bytes', 0)} bytes")
        print(f"🎯 Suitability score: {capacity_info.get('overall_suitability', 0):.2f}/1.0")
        
        if capacity_info.get('effective_bytes', 0) < len(secret_message.encode()):
            print("❌ Insufficient capacity for demo message")
            return
        
        # Hide data
        print("\n🔒 Hiding secret message...")
        stego_file = Path("temp_stego.wav")
        
        result = engine.hide_data(
            audio_path=test_audio,
            data=secret_message,
            output_path=stego_file,
            config=config
        )
        
        if result.success:
            print(f"✅ Embedding successful!")
            print(f"   📊 Capacity utilization: {result.capacity_utilization:.1f}%")
            print(f"   ⚡ Processing time: {result.processing_time:.2f}s")
            
            if result.warnings:
                for warning in result.warnings:
                    print(f"   ⚠️  Warning: {warning}")
        else:
            print(f"❌ Embedding failed: {result.message}")
            return
        
        # Extract data
        print("\n🔓 Extracting hidden message...")
        
        extraction_result = engine.extract_data(
            audio_path=stego_file,
            config=config
        )
        
        if extraction_result.success and extraction_result.data:
            extracted_message = extraction_result.data.decode('utf-8')
            print(f"✅ Extraction successful!")
            print(f"   📝 Extracted: {extracted_message}")
            print(f"   ⚡ Processing time: {extraction_result.processing_time:.2f}s")
            
            # Verify message matches
            if extracted_message == secret_message:
                print("   🎯 Message integrity verified!")
            else:
                print("   ⚠️  Message mismatch detected")
        else:
            print(f"❌ Extraction failed: {extraction_result.message}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        # Cleanup
        for temp_file in [test_audio, Path("temp_stego.wav")]:
            if temp_file.exists():
                temp_file.unlink()


def demonstrate_multiple_techniques():
    """Demonstrate different embedding techniques."""
    print("\n" + "="*60)
    print("🔹 MULTIPLE TECHNIQUES DEMONSTRATION")
    print("="*60)
    
    # Create longer test audio for better capacity
    test_audio = Path("temp_long_carrier.wav")
    if not create_test_audio_file(test_audio, duration=15.0):
        return
    
    try:
        engine = create_audio_steganography_engine(SecurityLevel.MAXIMUM)
        
        # Test different techniques
        techniques = [
            ('lsb', 'Least Significant Bit'),
            ('spread_spectrum', 'Spread Spectrum'),
            ('phase_coding', 'Phase Coding'),
            ('echo', 'Echo Hiding')
        ]
        
        test_data = "Testing multiple steganography techniques! 🎵"
        
        print(f"📝 Test data: {test_data}")
        print(f"📏 Data size: {len(test_data)} bytes\n")
        
        for technique_code, technique_name in techniques:
            print(f"🔬 Testing {technique_name} ({technique_code})...")
            
            # Analyze capacity for this technique
            capacity_info = engine.analyze_capacity(test_audio, technique_code)
            
            if 'error' in capacity_info:
                print(f"   ❌ Analysis failed: {capacity_info['error']}")
                continue
            
            available_capacity = capacity_info.get('effective_bytes', 0)
            print(f"   📊 Capacity: {available_capacity} bytes")
            
            if available_capacity < len(test_data.encode()):
                print(f"   ❌ Insufficient capacity")
                continue
            
            # Test embedding
            config = engine.create_config(
                technique=technique_code,
                mode='secure',
                password='TechniqueTest456',
                randomize_positions=True
            )
            
            output_file = Path(f"temp_{technique_code}.wav")
            
            result = engine.hide_data(
                audio_path=test_audio,
                data=test_data,
                output_path=output_file,
                config=config
            )
            
            if result.success:
                print(f"   ✅ Embedding: {result.processing_time:.2f}s")
                
                # Test extraction
                extraction_result = engine.extract_data(
                    audio_path=output_file,
                    config=config
                )
                
                if extraction_result.success and extraction_result.data:
                    extracted = extraction_result.data.decode('utf-8')
                    if extracted == test_data:
                        print(f"   ✅ Extraction: {extraction_result.processing_time:.2f}s")
                        print(f"   🎯 Integrity: VERIFIED")
                    else:
                        print(f"   ⚠️  Integrity: FAILED")
                else:
                    print(f"   ❌ Extraction failed: {extraction_result.message}")
                
                # Cleanup
                if output_file.exists():
                    output_file.unlink()
            else:
                print(f"   ❌ Embedding failed: {result.message}")
            
            print()
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        # Cleanup
        if test_audio.exists():
            test_audio.unlink()


def demonstrate_error_recovery():
    """Demonstrate error recovery capabilities."""
    print("\n" + "="*60)
    print("🔹 ERROR RECOVERY DEMONSTRATION")
    print("="*60)
    
    test_audio = Path("temp_recovery_carrier.wav")
    if not create_test_audio_file(test_audio, duration=8.0):
        return
    
    try:
        engine = create_audio_steganography_engine(SecurityLevel.MAXIMUM)
        
        # Create configuration with maximum redundancy
        config = engine.create_config(
            technique='lsb',
            mode='maximum',  # 5x redundancy
            password='RecoveryTest789',
            error_correction=True
        )
        
        test_data = "This message tests error recovery capabilities! 🛡️"
        print(f"📝 Test message: {test_data}")
        
        # Hide with maximum redundancy
        print("\n🔒 Hiding with maximum redundancy (5x)...")
        stego_file = Path("temp_recovery_stego.wav")
        
        result = engine.hide_data(
            audio_path=test_audio,
            data=test_data,
            output_path=stego_file,
            config=config
        )
        
        if not result.success:
            print(f"❌ Embedding failed: {result.message}")
            return
        
        print(f"✅ Embedded with {result.redundancy_level}x redundancy")
        
        # Test standard extraction
        print("\n🔓 Testing standard extraction...")
        extraction_result = engine.extract_data(
            audio_path=stego_file,
            config=config,
            max_attempts=1  # Only standard attempt
        )
        
        if extraction_result.success:
            print("✅ Standard extraction successful")
        else:
            print(f"❌ Standard extraction failed: {extraction_result.message}")
        
        # Test with multiple recovery strategies
        print("\n🛠️  Testing advanced recovery (5 attempts)...")
        recovery_result = engine.extract_data(
            audio_path=stego_file,
            config=config,
            max_attempts=5  # All recovery strategies
        )
        
        if recovery_result.success and recovery_result.data:
            extracted = recovery_result.data.decode('utf-8')
            print(f"✅ Recovery successful using {recovery_result.recovery_method}")
            print(f"🎯 Message verified: {extracted == test_data}")
            print(f"📊 Attempts made: {recovery_result.attempts_made}")
            
            if recovery_result.confidence_score:
                print(f"🔍 Confidence: {recovery_result.confidence_score:.1%}")
        else:
            print(f"❌ All recovery attempts failed: {recovery_result.message}")
        
        # Test auto-technique detection
        print("\n🔍 Testing auto-technique detection...")
        auto_config = engine.create_config(
            technique='auto',  # Try all techniques
            password='RecoveryTest789'
        )
        
        auto_result = engine.extract_data(
            audio_path=stego_file,
            config=auto_config,
            max_attempts=3
        )
        
        if auto_result.success:
            print(f"✅ Auto-detection successful!")
            print(f"🔬 Detected technique: {auto_result.technique_detected}")
        else:
            print(f"❌ Auto-detection failed: {auto_result.message}")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        # Cleanup
        for temp_file in [test_audio, Path("temp_recovery_stego.wav")]:
            if temp_file.exists():
                temp_file.unlink()


def demonstrate_security_features():
    """Demonstrate security and anti-detection features."""
    print("\n" + "="*60)
    print("🔹 SECURITY FEATURES DEMONSTRATION")
    print("="*60)
    
    test_audio = Path("temp_security_carrier.wav")
    if not create_test_audio_file(test_audio, duration=12.0):
        return
    
    try:
        engine = create_audio_steganography_engine(SecurityLevel.MAXIMUM)
        
        # Test different security levels
        security_tests = [
            ('fast', 'Basic security'),
            ('balanced', 'Standard security'),
            ('secure', 'High security + anti-detection'),
            ('maximum', 'Maximum security + recovery')
        ]
        
        test_data = "Confidential data requiring maximum security! 🛡️🔐"
        
        for mode, description in security_tests:
            print(f"\n🔐 Testing {mode.upper()} mode ({description})...")
            
            config = engine.create_config(
                technique='lsb',
                mode=mode,
                password=f'SecurePass_{mode}_999',
                randomize_positions=True
            )
            
            output_file = Path(f"temp_security_{mode}.wav")
            
            # Hide data
            result = engine.hide_data(
                audio_path=test_audio,
                data=test_data,
                output_path=output_file,
                config=config
            )
            
            if result.success:
                print(f"   ✅ Embedded with {result.redundancy_level}x redundancy")
                
                if result.anti_detection_score is not None:
                    print(f"   🛡️  Detection risk: {result.anti_detection_score:.3f}/1.0")
                
                # Test extraction
                extraction_result = engine.extract_data(
                    audio_path=output_file,
                    config=config
                )
                
                if extraction_result.success and extraction_result.data:
                    extracted = extraction_result.data.decode('utf-8')
                    print(f"   ✅ Extraction successful")
                    print(f"   🎯 Data integrity: {'VERIFIED' if extracted == test_data else 'FAILED'}")
                else:
                    print(f"   ❌ Extraction failed: {extraction_result.message}")
                
                # Test wrong password
                wrong_config = engine.create_config(
                    technique='lsb',
                    mode=mode,
                    password='WrongPassword123'
                )
                
                wrong_result = engine.extract_data(
                    audio_path=output_file,
                    config=wrong_config
                )
                
                if wrong_result.success:
                    print(f"   ⚠️  WARNING: Wrong password succeeded (security issue)")
                else:
                    print(f"   ✅ Wrong password correctly rejected")
            else:
                print(f"   ❌ Embedding failed: {result.message}")
            
            # Cleanup
            if output_file.exists():
                output_file.unlink()
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        # Cleanup
        if test_audio.exists():
            test_audio.unlink()


def print_system_info():
    """Print system information and capabilities."""
    print("🎵 InVisioVault Audio Steganography System - Demo")
    print("="*60)
    
    try:
        engine = create_audio_steganography_engine()
        
        print("📋 SYSTEM INFORMATION:")
        print(f"   Version: {engine.PROTOCOL_VERSION}")
        print(f"   Security Level: {engine.security_level.value}")
        
        print("\n🔧 AVAILABLE TECHNIQUES:")
        techniques = engine.get_available_techniques()
        for code, info in techniques.items():
            print(f"   • {info['name']} ({code})")
            print(f"     {info['description']}")
        
        print("\n⚙️  AVAILABLE MODES:")
        modes = engine.get_available_modes()
        for code, info in modes.items():
            print(f"   • {info['name']} Mode")
            print(f"     Redundancy: {info['redundancy']}x, "
                  f"Error Correction: {info['error_correction']}, "
                  f"Anti-Detection: {info['anti_detection']}")
        
        print(f"\n🎯 FEATURES:")
        print("   • Multiple embedding techniques (LSB, Spread Spectrum, Phase Coding, Echo)")
        print("   • Advanced error recovery with up to 5 strategies")
        print("   • AES-256 encryption with PBKDF2 key derivation")
        print("   • Anti-detection measures and statistical masking")
        print("   • Redundant storage with error correction")
        print("   • Format validation and capacity analysis")
        print("   • Comprehensive logging and progress tracking")
        
    except Exception as e:
        print(f"❌ Failed to get system info: {e}")


def main():
    """Main demo function."""
    print_system_info()
    
    try:
        # Run demonstrations
        demonstrate_basic_usage()
        demonstrate_multiple_techniques()
        demonstrate_error_recovery()
        demonstrate_security_features()
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETE")
        print("="*60)
        print("\n✅ All demonstrations completed successfully!")
        print("🔗 For more information, see: docs/AUDIO_STEGANOGRAPHY_REWRITE.md")
        print("🧪 To run tests: pytest tests/test_audio_steganography.py -v")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
