"""
Comprehensive Test Suite for Audio Steganography System

Tests all components of the new audio steganography implementation including:
- Audio processing utilities
- Embedding techniques
- Main steganography engine
- Error handling and recovery
- Security features
"""

import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import hashlib
import json

# Test the imports
try:
    from core.audio.audio_processor import AudioProcessor, AudioInfo, AudioFormat
    from core.audio.embedding_techniques import (
        LSBEmbedding, SpreadSpectrumEmbedding, PhaseCodingEmbedding, 
        EchoHidingEmbedding, EmbeddingTechniqueFactory
    )
    from core.audio.audio_steganography import (
        AudioSteganographyEngine, EmbeddingConfig, EmbeddingResult, 
        ExtractionResult, StegoMode
    )
    from core.encryption_engine import SecurityLevel
    from utils.logger import Logger
except ImportError as e:
    pytest.skip(f"Audio steganography modules not available: {e}", allow_module_level=True)


class TestAudioProcessor:
    """Test suite for AudioProcessor class."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor instance for testing."""
        return AudioProcessor()
    
    @pytest.fixture
    def mock_audio_data(self):
        """Create mock audio data for testing."""
        # Stereo audio, 2 seconds at 44.1kHz
        channels = 2
        samples = 44100 * 2
        return np.random.uniform(-1, 1, (channels, samples)).astype(np.float32)
    
    @pytest.fixture
    def temp_wav_file(self, tmp_path):
        """Create temporary WAV file for testing."""
        wav_file = tmp_path / "test.wav"
        
        # Create a simple WAV file structure (simplified)
        sample_rate = 44100
        duration = 2.0
        samples = int(sample_rate * duration)
        audio_data = np.random.uniform(-1, 1, (2, samples)).astype(np.float32)
        
        # Mock the file creation
        wav_file.touch()
        wav_file.write_bytes(b'RIFF' + b'\x00' * 100)  # Minimal WAV header
        
        return wav_file
    
    def test_init(self, audio_processor):
        """Test AudioProcessor initialization."""
        assert audio_processor is not None
        assert audio_processor.DEFAULT_SAMPLE_RATE == 44100
        assert audio_processor.MAX_DURATION == 3600
        assert audio_processor.MIN_CAPACITY_BYTES == 1024
    
    @patch('soundfile.info')
    def test_analyze_audio_file_success(self, mock_sf_info, audio_processor, temp_wav_file):
        """Test successful audio file analysis."""
        # Mock soundfile.info
        mock_info = Mock()
        mock_info.samplerate = 44100
        mock_info.channels = 2
        mock_info.frames = 88200
        mock_info.duration = 2.0
        mock_sf_info.return_value = mock_info
        
        result = audio_processor.analyze_audio_file(temp_wav_file)
        
        assert result is not None
        assert result.format == 'wav'
        assert result.sample_rate == 44100
        assert result.channels == 2
        assert result.frames == 88200
        assert result.duration == 2.0
        assert result.is_lossless == True
    
    def test_analyze_audio_file_not_found(self, audio_processor):
        """Test analysis of non-existent file."""
        with pytest.raises(FileNotFoundError):
            audio_processor.analyze_audio_file(Path("nonexistent.wav"))
    
    def test_get_format_info(self, audio_processor):
        """Test format information extraction."""
        # Test lossless format
        wav_info = audio_processor._get_format_info('.wav')
        assert wav_info['lossless'] == True
        assert wav_info['extension'] == 'wav'
        
        # Test lossy format
        mp3_info = audio_processor._get_format_info('.mp3')
        assert mp3_info['lossless'] == False
        assert mp3_info['extension'] == 'mp3'
        
        # Test unknown format
        unknown_info = audio_processor._get_format_info('.xyz')
        assert unknown_info['lossless'] == False
    
    def test_calculate_capacity(self, audio_processor):
        """Test capacity calculation."""
        # Lossless format
        bits, bytes_cap = audio_processor._calculate_capacity(88200, 2, True)
        expected_bits = (88200 * 2) // 4  # Skip factor of 4
        assert bits == expected_bits
        assert bytes_cap == expected_bits // 8
        
        # Lossy format (much lower capacity)
        bits_lossy, bytes_lossy = audio_processor._calculate_capacity(88200, 2, False)
        assert bits_lossy < bits
        assert bytes_lossy < bytes_cap
    
    def test_validate_audio_file(self, audio_processor, temp_wav_file):
        """Test audio file validation."""
        # Valid file
        assert audio_processor._validate_audio_file(temp_wav_file) == True
        
        # Non-existent file
        assert audio_processor._validate_audio_file(Path("nonexistent.wav")) == False
        
        # Empty file
        empty_file = temp_wav_file.parent / "empty.wav"
        empty_file.touch()
        assert audio_processor._validate_audio_file(empty_file) == False
    
    def test_check_output_format(self, audio_processor, tmp_path):
        """Test output format checking."""
        # Lossless format
        wav_path = tmp_path / "output.wav"
        warnings = audio_processor._check_output_format(wav_path)
        assert len(warnings) == 0
        
        # Lossy format
        mp3_path = tmp_path / "output.mp3"
        warnings = audio_processor._check_output_format(mp3_path)
        assert len(warnings) > 0
        assert "WARNING" in warnings[0]


class TestEmbeddingTechniques:
    """Test suite for embedding techniques."""
    
    @pytest.fixture
    def logger(self):
        """Create logger for testing."""
        return Logger()
    
    @pytest.fixture
    def audio_data(self):
        """Create test audio data."""
        # Stereo, 1 second at 44.1kHz
        return np.random.uniform(-1, 1, (2, 44100)).astype(np.float32)
    
    @pytest.fixture
    def test_data(self):
        """Create test data to embed."""
        return b"Hello, World! This is test data for steganography."
    
    def test_lsb_embedding_init(self, logger):
        """Test LSB embedding initialization."""
        lsb = LSBEmbedding(logger)
        assert lsb.technique_name == "lsb"
        assert lsb.skip_factor == 4
        assert lsb.randomize == True
    
    def test_lsb_embedding_capacity(self, logger, audio_data):
        """Test LSB capacity calculation."""
        lsb = LSBEmbedding(logger)
        capacity = lsb.calculate_capacity(audio_data, 44100)
        
        channels, samples = audio_data.shape
        expected = (channels * samples) // (lsb.skip_factor * 8)
        assert capacity == expected
    
    def test_lsb_embedding_embed_extract(self, logger, audio_data, test_data):
        """Test LSB embed and extract cycle."""
        lsb = LSBEmbedding(logger, randomize=False)  # Non-random for deterministic test
        password = "test_password_123"
        
        # Check capacity
        capacity = lsb.calculate_capacity(audio_data, 44100)
        assert len(test_data) <= capacity, f"Test data too large: {len(test_data)} > {capacity}"
        
        # Embed data
        modified_audio = lsb.embed(audio_data, test_data, password, 44100)
        assert modified_audio is not None
        assert modified_audio.shape == audio_data.shape
        
        # Extract data
        extracted_data = lsb.extract(modified_audio, password, 44100, len(test_data))
        assert extracted_data is not None
        assert extracted_data == test_data
    
    def test_lsb_embedding_wrong_password(self, logger, audio_data, test_data):
        """Test LSB extraction with wrong password."""
        lsb = LSBEmbedding(logger, randomize=False)
        password = "correct_password"
        wrong_password = "wrong_password"
        
        # Embed with correct password
        modified_audio = lsb.embed(audio_data, test_data, password, 44100)
        assert modified_audio is not None
        
        # Try to extract with wrong password
        extracted_data = lsb.extract(modified_audio, wrong_password, 44100, len(test_data))
        # Should get different data (not None, but wrong)
        assert extracted_data != test_data
    
    def test_lsb_data_too_large(self, logger, audio_data):
        """Test LSB with data larger than capacity."""
        lsb = LSBEmbedding(logger)
        large_data = b"x" * 100000  # Very large data
        
        # Should fail due to capacity check
        result = lsb.embed(audio_data, large_data, "password", 44100)
        assert result is None
    
    def test_spread_spectrum_embedding_init(self, logger):
        """Test spread spectrum initialization."""
        ss = SpreadSpectrumEmbedding(logger)
        assert ss.technique_name == "spread_spectrum"
        assert ss.chip_rate == 1000
        assert ss.amplitude == 0.001
    
    def test_spread_spectrum_capacity(self, logger, audio_data):
        """Test spread spectrum capacity calculation."""
        ss = SpreadSpectrumEmbedding(logger)
        capacity = ss.calculate_capacity(audio_data, 44100)
        
        channels, samples = audio_data.shape
        expected = samples // (ss.chip_rate * 8)
        assert capacity == expected
    
    def test_phase_coding_embedding_init(self, logger):
        """Test phase coding initialization."""
        pc = PhaseCodingEmbedding(logger)
        assert pc.technique_name == "phase_coding"
        assert pc.segment_length == 2048
        assert pc.phase_shift == np.pi/4
    
    def test_echo_hiding_embedding_init(self, logger):
        """Test echo hiding initialization."""
        eh = EchoHidingEmbedding(logger)
        assert eh.technique_name == "echo_hiding"
        assert eh.delay_0 == 100
        assert eh.delay_1 == 150
        assert eh.attenuation == 0.1
    
    def test_embedding_technique_factory(self, logger):
        """Test embedding technique factory."""
        factory = EmbeddingTechniqueFactory(logger)
        
        # Test creating LSB technique
        lsb = factory.create_technique('lsb')
        assert lsb is not None
        assert isinstance(lsb, LSBEmbedding)
        
        # Test creating unknown technique
        unknown = factory.create_technique('unknown')
        assert unknown is None
        
        # Test getting available techniques
        available = factory.get_available_techniques()
        assert 'lsb' in available
        assert 'spread_spectrum' in available
    
    def test_seed_generation(self, logger):
        """Test deterministic seed generation."""
        lsb = LSBEmbedding(logger)
        
        # Same password should generate same seed
        seed1 = lsb._generate_seed_sequence("password123")
        seed2 = lsb._generate_seed_sequence("password123")
        assert seed1 == seed2
        
        # Different passwords should generate different seeds
        seed3 = lsb._generate_seed_sequence("different_password")
        assert seed1 != seed3


class TestAudioSteganographyEngine:
    """Test suite for main audio steganography engine."""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for testing."""
        return AudioSteganographyEngine(SecurityLevel.STANDARD)
    
    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create temporary audio file."""
        audio_file = tmp_path / "carrier.wav"
        audio_file.touch()
        # Add some minimal content
        audio_file.write_bytes(b'RIFF' + b'\x00' * 1000)
        return audio_file
    
    @pytest.fixture
    def basic_config(self):
        """Create basic embedding configuration."""
        return EmbeddingConfig(
            technique='lsb',
            mode='balanced',
            password='test_password_123',
            redundancy_level=2,
            error_correction=True,
            randomize_positions=True
        )
    
    def test_engine_init(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert engine.PROTOCOL_VERSION == "3.0"
        assert engine.MAGIC_HEADER == b'INVV_AUD'
        assert engine.security_level == SecurityLevel.STANDARD
    
    def test_create_config(self, engine):
        """Test configuration creation."""
        config = engine.create_config(
            technique='lsb',
            password='test123',
            mode='secure'
        )
        assert config.technique == 'lsb'
        assert config.password == 'test123'
        assert config.mode == 'secure'
    
    def test_get_available_techniques(self, engine):
        """Test getting available techniques."""
        techniques = engine.get_available_techniques()
        assert isinstance(techniques, dict)
        assert 'lsb' in techniques
        assert 'name' in techniques['lsb']
        assert 'description' in techniques['lsb']
    
    def test_get_available_modes(self, engine):
        """Test getting available modes."""
        modes = engine.get_available_modes()
        assert isinstance(modes, dict)
        assert 'fast' in modes
        assert 'balanced' in modes
        assert 'secure' in modes
        assert 'maximum' in modes
        
        # Check mode properties
        secure_mode = modes['secure']
        assert secure_mode['redundancy'] > 1
        assert secure_mode['error_correction'] == True
    
    def test_validate_inputs_success(self, engine, temp_audio_file, basic_config, tmp_path):
        """Test successful input validation."""
        output_path = tmp_path / "output.wav"
        result = engine._validate_inputs(temp_audio_file, output_path, basic_config)
        assert result.success == True
    
    def test_validate_inputs_missing_file(self, engine, basic_config, tmp_path):
        """Test validation with missing audio file."""
        missing_file = tmp_path / "missing.wav"
        output_path = tmp_path / "output.wav"
        result = engine._validate_inputs(missing_file, output_path, basic_config)
        assert result.success == False
        assert "not found" in result.message.lower()
    
    def test_validate_inputs_short_password(self, engine, temp_audio_file, tmp_path):
        """Test validation with too short password."""
        config = EmbeddingConfig(password="123")  # Too short
        output_path = tmp_path / "output.wav"
        result = engine._validate_inputs(temp_audio_file, output_path, config)
        assert result.success == False
        assert "password" in result.message.lower()
    
    def test_prepare_data_bytes(self, engine):
        """Test data preparation with bytes."""
        test_data = b"Hello, World!"
        result = engine._prepare_data(test_data)
        assert result == test_data
    
    def test_prepare_data_string(self, engine):
        """Test data preparation with string."""
        test_string = "Hello, World!"
        result = engine._prepare_data(test_string)
        assert result == test_string.encode('utf-8')
    
    def test_prepare_data_file(self, engine, tmp_path):
        """Test data preparation with file path."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = b"File content for testing"
        test_file.write_bytes(test_content)
        
        result = engine._prepare_data(test_file)
        assert result == test_content
    
    def test_prepare_data_missing_file(self, engine, tmp_path):
        """Test data preparation with missing file."""
        missing_file = tmp_path / "missing.txt"
        result = engine._prepare_data(missing_file)
        assert result is None
    
    def test_apply_mode_config(self, engine):
        """Test mode configuration application."""
        config = EmbeddingConfig(mode='secure')
        updated_config = engine._apply_mode_config(config)
        
        # Should have secure mode settings
        assert updated_config.redundancy_level > 1
        assert updated_config.error_correction == True
        assert updated_config.anti_detection == True
    
    def test_determine_extraction_techniques_auto(self, engine):
        """Test technique determination with auto mode."""
        techniques = engine._determine_extraction_techniques('auto')
        assert isinstance(techniques, list)
        assert len(techniques) > 1
        assert 'lsb' in techniques
    
    def test_determine_extraction_techniques_specific(self, engine):
        """Test technique determination with specific technique."""
        techniques = engine._determine_extraction_techniques('lsb')
        assert techniques[0] == 'lsb'
        # Should have fallback
        assert len(techniques) >= 1
    
    def test_get_extraction_strategies(self, engine):
        """Test extraction strategy determination."""
        # With max_attempts = 1, should only get standard
        strategies = engine._get_extraction_strategies(1)
        assert strategies == ['standard']
        
        # With more attempts, should get more strategies
        strategies_many = engine._get_extraction_strategies(5)
        assert len(strategies_many) > 1
        assert 'standard' in strategies_many
    
    @patch('core.audio.audio_steganography.AudioProcessor')
    @patch('core.audio.audio_steganography.EmbeddingTechniqueFactory')
    def test_hide_data_validation_failure(self, mock_factory, mock_processor, engine, tmp_path):
        """Test hide_data with validation failure."""
        audio_path = tmp_path / "missing.wav"  # Doesn't exist
        output_path = tmp_path / "output.wav"
        config = EmbeddingConfig(password="test123")
        data = b"test data"
        
        result = engine.hide_data(audio_path, data, output_path, config)
        assert result.success == False
        assert "not found" in result.message.lower()
    
    def test_generate_warnings(self, engine, tmp_path):
        """Test warning generation."""
        # Test with different formats
        wav_input = tmp_path / "input.wav"
        mp3_output = tmp_path / "output.mp3"
        config = EmbeddingConfig(redundancy_level=1, error_correction=False)
        
        warnings = engine._generate_warnings(wav_input, mp3_output, config)
        
        # Should warn about lossy output format
        assert any("lossy" in w.lower() for w in warnings)
        # Should warn about low redundancy
        assert any("redundancy" in w.lower() for w in warnings)
        # Should warn about disabled error correction
        assert any("error correction" in w.lower() for w in warnings)
    
    def test_calculate_detection_risk(self, engine):
        """Test detection risk calculation."""
        # Create original and slightly modified audio
        original = np.random.uniform(-1, 1, (2, 1000))
        modified = original + np.random.uniform(-0.01, 0.01, original.shape)  # Small changes
        
        risk = engine._calculate_detection_risk(modified, original)
        assert 0.0 <= risk <= 1.0
        
        # Test with identical audio (should be very low risk)
        risk_identical = engine._calculate_detection_risk(original, original)
        assert risk_identical < risk
    
    def test_generate_capacity_recommendations(self, engine):
        """Test capacity recommendation generation."""
        # Create mock audio info with various characteristics
        from core.audio.audio_processor import AudioInfo
        
        # Low capacity audio
        low_capacity_info = AudioInfo(
            path=Path("test.mp3"),
            format="mp3",
            sample_rate=22050,
            channels=1,
            frames=22050,  # 1 second
            duration=1.0,
            bit_depth=16,
            is_lossless=False,
            capacity_bits=100,
            capacity_bytes=12,
            file_size=1000
        )
        
        capacity_info = {'effective_bytes': 10}
        recommendations = engine._generate_capacity_recommendations(low_capacity_info, capacity_info)
        
        # Should recommend format conversion, longer audio, higher sample rate, stereo
        assert any("lossless" in r.lower() for r in recommendations)
        assert any("small" in r.lower() or "longer" in r.lower() for r in recommendations)
        assert any("sample rate" in r.lower() for r in recommendations)
        assert any("stereo" in r.lower() for r in recommendations)


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for integration tests."""
        return AudioSteganographyEngine(SecurityLevel.STANDARD)
    
    @pytest.fixture
    def mock_audio_data(self):
        """Create larger mock audio data for integration tests."""
        # 5 seconds of stereo audio
        return np.random.uniform(-1, 1, (2, 44100 * 5)).astype(np.float32)
    
    def test_stego_mode_configurations(self, engine):
        """Test different steganography modes."""
        for mode_name in ['fast', 'balanced', 'secure', 'maximum']:
            config = EmbeddingConfig(
                mode=mode_name,
                password="test123",
                technique='lsb'
            )
            
            # Apply mode configuration
            updated_config = engine._apply_mode_config(config)
            
            # Verify mode-specific settings were applied
            if mode_name == 'fast':
                assert updated_config.redundancy_level == 1
                assert updated_config.error_correction == False
                assert updated_config.anti_detection == False
            elif mode_name == 'maximum':
                assert updated_config.redundancy_level == 5
                assert updated_config.error_correction == True
                assert updated_config.anti_detection == True
    
    def test_error_handling_chain(self, engine):
        """Test error handling throughout the system."""
        # Test with various invalid inputs
        config = EmbeddingConfig(password="test123")
        
        # Invalid audio path
        result = engine.hide_data(
            Path("nonexistent.wav"), 
            b"data", 
            Path("output.wav"), 
            config
        )
        assert result.success == False
        assert "not found" in result.message.lower()
        
        # Invalid password for extraction
        extraction_result = engine.extract_data(
            Path("nonexistent.wav"),
            EmbeddingConfig(password="wrong")
        )
        assert extraction_result.success == False
    
    @patch('core.audio.audio_processor.AudioProcessor.load_audio')
    @patch('core.audio.audio_processor.AudioProcessor.analyze_audio_file')
    def test_capacity_analysis_integration(self, mock_analyze, mock_load, engine, tmp_path):
        """Test capacity analysis with mocked audio processor."""
        from core.audio.audio_processor import AudioInfo
        
        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        
        # Mock audio info
        mock_info = AudioInfo(
            path=audio_file,
            format="wav",
            sample_rate=44100,
            channels=2,
            frames=88200,
            duration=2.0,
            bit_depth=16,
            is_lossless=True,
            capacity_bits=4410,
            capacity_bytes=551,
            file_size=10000
        )
        mock_analyze.return_value = mock_info
        
        # Test capacity analysis
        capacity_result = engine.analyze_capacity(audio_file, 'lsb')
        
        assert 'file_info' in capacity_result
        assert 'quality_metrics' in capacity_result
        assert 'overall_suitability' in capacity_result
        assert 'recommendations' in capacity_result
        
        # Check file info
        file_info = capacity_result['file_info']
        assert file_info['format'] == 'wav'
        assert file_info['is_lossless'] == True
    
    def test_configuration_serialization(self, engine):
        """Test configuration to/from dictionary conversion."""
        config = EmbeddingConfig(
            technique='spread_spectrum',
            mode='secure',
            password='complex_password_123!',
            redundancy_level=3,
            randomize_positions=False
        )
        
        # Convert to dictionary
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['technique'] == 'spread_spectrum'
        assert config_dict['password'] == 'complex_password_123!'
        
        # Convert back from dictionary
        restored_config = EmbeddingConfig.from_dict(config_dict)
        assert restored_config.technique == config.technique
        assert restored_config.mode == config.mode
        assert restored_config.password == config.password
        assert restored_config.redundancy_level == config.redundancy_level


class TestErrorRecoveryScenarios:
    """Test error recovery and robustness features."""
    
    @pytest.fixture
    def engine(self):
        return AudioSteganographyEngine(SecurityLevel.STANDARD)
    
    def test_extraction_with_multiple_strategies(self, engine):
        """Test extraction using multiple recovery strategies."""
        strategies = engine._get_extraction_strategies(5)
        
        expected_strategies = ['standard', 'error_correction', 'redundant', 'partial', 'brute_force']
        assert strategies == expected_strategies
    
    def test_technique_fallback_order(self, engine):
        """Test technique fallback ordering."""
        # Auto mode should try multiple techniques
        auto_techniques = engine._determine_extraction_techniques('auto')
        assert len(auto_techniques) > 1
        assert 'lsb' in auto_techniques  # LSB should be included
        
        # Specific technique should have fallbacks
        lsb_techniques = engine._determine_extraction_techniques('spread_spectrum')
        assert lsb_techniques[0] == 'spread_spectrum'
        assert 'lsb' in lsb_techniques  # LSB as fallback
    
    def test_error_message_generation(self, engine):
        """Test comprehensive error message generation."""
        techniques_tried = ['lsb', 'spread_spectrum', 'phase_coding']
        max_attempts = 3
        
        error_msg = engine._generate_extraction_error_message(techniques_tried, max_attempts)
        
        # Should mention number of techniques and strategies
        assert str(len(techniques_tried)) in error_msg
        assert str(max_attempts) in error_msg
        
        # Should list possible causes
        assert "password" in error_msg.lower()
        assert "hidden data" in error_msg.lower()
        assert "format" in error_msg.lower()
        assert "compression" in error_msg.lower()


# Utility functions for testing
def create_test_audio_data(duration_seconds=2, sample_rate=44100, channels=2):
    """Create test audio data with specified parameters."""
    samples = int(duration_seconds * sample_rate)
    return np.random.uniform(-1, 1, (channels, samples)).astype(np.float32)


def create_test_wav_file(file_path, audio_data, sample_rate=44100):
    """Create a minimal WAV file for testing (mock implementation)."""
    # This is a simplified mock - real implementation would create proper WAV format
    file_path.touch()
    file_path.write_bytes(b'RIFF' + b'\x00' * 1000)  # Minimal header
    return file_path


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v"])
