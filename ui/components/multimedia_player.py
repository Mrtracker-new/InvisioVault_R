"""
Multimedia Player Component
Professional video/audio player widget for previewing multimedia files in InVisioVault.
"""

import sys
from pathlib import Path
from typing import Optional, List

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QSlider, QPushButton, QLabel,
        QFrame, QSizePolicy, QMessageBox, QApplication
    )
    from PySide6.QtCore import Qt, QUrl, QTimer, Signal, QSize
    from PySide6.QtGui import QIcon, QFont, QPalette
    
    # Import multimedia components
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtMultimediaWidgets import QVideoWidget
    
except ImportError as e:
    print(f"Warning: Multimedia components not available: {e}")
    print("Please ensure PySide6-Addons is installed: pip install PySide6-Addons")


class MultimediaPlayerWidget(QWidget):
    """Professional multimedia player widget with video and audio support."""
    
    # Signals
    file_loaded = Signal(str)  # Emitted when a file is loaded
    playback_started = Signal()
    playback_paused = Signal()
    playback_stopped = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Media player components
        self.media_player = None
        self.audio_output = None
        self.video_widget = None
        
        # Player state
        self.current_file = None
        self.is_video = False
        self.duration = 0
        
        # UI components
        self.position_slider = None
        self.volume_slider = None
        self.play_button = None
        self.time_label = None
        
        # Timer for position updates
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_position)
        
        self.init_ui()
        self.setup_media_player()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Video display area (will be hidden for audio files)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(400, 300)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setStyleSheet("""
            QVideoWidget {
                background-color: #000000;
                border: 2px solid #333333;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.video_widget)
        
        # Audio-only placeholder (for audio files)
        self.audio_placeholder = QLabel()
        self.audio_placeholder.setText("🎵 Audio Player\n\nSelect an audio file to play")
        self.audio_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.audio_placeholder.setMinimumSize(400, 200)
        self.audio_placeholder.setStyleSheet("""
            QLabel {
                background-color: #2c2c2c;
                color: #ffffff;
                border: 2px solid #333333;
                border-radius: 4px;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.audio_placeholder)
        
        # File info label
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(self.file_info_label)
        
        # Controls container
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_frame.setMaximumHeight(80)
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        
        # Position slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(100)
        self.position_slider.setValue(0)
        self.position_slider.sliderPressed.connect(self.on_position_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_position_slider_released)
        controls_layout.addWidget(self.position_slider)
        
        # Bottom controls
        bottom_controls = QHBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton("▶️")
        self.play_button.setFixedSize(40, 40)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        bottom_controls.addWidget(self.play_button)
        
        # Stop button
        self.stop_button = QPushButton("⏹️")
        self.stop_button.setFixedSize(40, 40)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)
        bottom_controls.addWidget(self.stop_button)
        
        # Time display
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(100)
        self.time_label.setStyleSheet("color: #333; font-weight: bold; padding: 0 10px;")
        bottom_controls.addWidget(self.time_label)
        
        bottom_controls.addStretch()
        
        # Volume control
        volume_label = QLabel("🔊")
        bottom_controls.addWidget(volume_label)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(70)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self.set_volume)
        bottom_controls.addWidget(self.volume_slider)
        
        controls_layout.addLayout(bottom_controls)
        layout.addWidget(controls_frame)
        
        # Initially show audio placeholder
        self.show_audio_mode()
        
    def setup_media_player(self):
        """Initialize the media player components."""
        try:
            # Create media player
            self.media_player = QMediaPlayer()
            
            # Create audio output
            self.audio_output = QAudioOutput()
            self.media_player.setAudioOutput(self.audio_output)
            
            # Set video output
            self.media_player.setVideoOutput(self.video_widget)
            
            # Connect signals
            self.media_player.positionChanged.connect(self.on_position_changed)
            self.media_player.durationChanged.connect(self.on_duration_changed)
            self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)
            self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
            self.media_player.errorOccurred.connect(self.on_media_error)
            
            # Set initial volume
            self.set_volume(70)
            
            # Test if multimedia backend is available
            test_media = QUrl("dummy://test")
            self.media_player.setSource(test_media)
            
        except Exception as e:
            print(f"Warning: Failed to initialize media player: {e}")
            self._setup_fallback_mode()
    
    def load_file(self, file_path: Path) -> bool:
        """
        Load a multimedia file for playback.
        
        Args:
            file_path: Path to the multimedia file
            
        Returns:
            True if file loaded successfully, False otherwise
        """
        if not self.media_player:
            # Fallback mode - just show file information
            return self._load_file_fallback(file_path)
            
        try:
            if not file_path.exists():
                self.show_error_message(f"File not found: {file_path}")
                return False
            
            # Determine file type
            video_formats = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
            audio_formats = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
            
            file_ext = file_path.suffix.lower()
            
            if file_ext in video_formats:
                self.is_video = True
                self.show_video_mode()
            elif file_ext in audio_formats:
                self.is_video = False
                self.show_audio_mode()
            else:
                self.show_error_message(f"Unsupported file format: {file_ext}")
                return False
            
            # Load the media
            media_url = QUrl.fromLocalFile(str(file_path))
            self.media_player.setSource(media_url)
            
            self.current_file = file_path
            
            # Update file info
            file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
            file_type = "Video" if self.is_video else "Audio"
            self.file_info_label.setText(f"{file_type}: {file_path.name} ({file_size:.1f} MB)")
            
            # Enable controls
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            
            # Emit signal
            self.file_loaded.emit(str(file_path))
            
            return True
            
        except Exception as e:
            self.show_error_message(f"Failed to load file: {e}")
            return False
    
    def show_video_mode(self):
        """Show video display and hide audio placeholder."""
        self.video_widget.show()
        self.audio_placeholder.hide()
        
    def show_audio_mode(self):
        """Show audio placeholder and hide video display."""
        self.video_widget.hide()
        self.audio_placeholder.show()
        self.audio_placeholder.setText(f"🎵 Audio Player\n\n{self.current_file.name if self.current_file else 'Select an audio file to play'}")
    
    def toggle_playback(self):
        """Toggle between play and pause."""
        if not self.media_player:
            return
            
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start or resume playback."""
        if self.media_player:
            self.media_player.play()
            self.position_timer.start(100)  # Update position every 100ms
            
    def pause_playback(self):
        """Pause playback."""
        if self.media_player:
            self.media_player.pause()
            self.position_timer.stop()
            
    def stop_playback(self):
        """Stop playback."""
        if self.media_player:
            self.media_player.stop()
            self.position_timer.stop()
            self.position_slider.setValue(0)
            self.update_time_display(0, self.duration)
    
    def set_volume(self, volume: int):
        """Set playback volume (0-100)."""
        if self.audio_output:
            # Convert percentage to linear volume (0.0 - 1.0)
            linear_volume = volume / 100.0
            self.audio_output.setVolume(linear_volume)
    
    def set_position(self, position: int):
        """Set playback position (0-100 percentage)."""
        if self.media_player and self.duration > 0:
            ms_position = (position / 100.0) * self.duration
            self.media_player.setPosition(int(ms_position))
    
    def on_position_changed(self, position: int):
        """Handle position change from media player."""
        if self.duration > 0:
            # Update slider if not being dragged by user
            if not self.position_slider.isSliderDown():
                percentage = (position / self.duration) * 100
                self.position_slider.setValue(int(percentage))
            
            self.update_time_display(position, self.duration)
    
    def on_duration_changed(self, duration: int):
        """Handle duration change from media player."""
        self.duration = duration
        self.update_time_display(0, duration)
    
    def on_media_status_changed(self, status):
        """Handle media status changes."""
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            # Media is ready to play
            pass
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            self.show_error_message("Invalid media file or unsupported format")
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            # Playback finished
            self.stop_playback()
    
    def on_playback_state_changed(self, state):
        """Handle playback state changes."""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("⏸️")
            self.playback_started.emit()
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.play_button.setText("▶️")
            self.playback_paused.emit()
        elif state == QMediaPlayer.PlaybackState.StoppedState:
            self.play_button.setText("▶️")
            self.playback_stopped.emit()
    
    def on_media_error(self, error, error_string):
        """Handle media player errors."""
        self.show_error_message(f"Playback error: {error_string}")
    
    def on_position_slider_pressed(self):
        """Handle position slider press (pause position updates)."""
        self.position_timer.stop()
    
    def on_position_slider_released(self):
        """Handle position slider release (seek to new position)."""
        position = self.position_slider.value()
        self.set_position(position)
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.position_timer.start(100)
    
    def update_position(self):
        """Update position display (called by timer)."""
        if self.media_player:
            position = self.media_player.position()
            self.on_position_changed(position)
    
    def update_time_display(self, position: int, duration: int):
        """Update the time display label."""
        pos_time = self.format_time(position)
        dur_time = self.format_time(duration)
        self.time_label.setText(f"{pos_time} / {dur_time}")
    
    def format_time(self, milliseconds: int) -> str:
        """Format time in MM:SS format."""
        seconds = milliseconds // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def _setup_fallback_mode(self):
        """Setup fallback mode when multimedia backend is not available."""
        self.media_player = None
        self.audio_output = None
        
        # Update UI to show that playback is not available
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.volume_slider.setEnabled(False)
        self.position_slider.setEnabled(False)
        
        # Update placeholder text
        self.audio_placeholder.setText(
            "🎵 Multimedia Preview Not Available\n\n"
            "Qt multimedia backend is not installed.\n"
            "File information will still be displayed."
        )
        self.video_widget.hide()
        self.audio_placeholder.show()
        
        print("Multimedia player running in fallback mode - preview not available")
    
    def show_error_message(self, message: str):
        """Show an error message to the user."""
        print(f"Multimedia Player: {message}")
    
    def clear(self):
        """Clear the current media and reset the player."""
        if self.media_player:
            self.media_player.stop()
            self.media_player.setSource(QUrl())
        
        self.current_file = None
        self.is_video = False
        self.duration = 0
        
        self.file_info_label.setText("No file loaded")
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.position_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
        
        # Show audio placeholder
        self.show_audio_mode()
        
    def get_supported_formats(self) -> List[str]:
        """Get list of supported multimedia formats."""
        return [
            # Video formats
            '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
            # Audio formats  
            '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'
        ]
    
    def is_supported_format(self, file_path: Path) -> bool:
        """Check if a file format is supported."""
        return file_path.suffix.lower() in self.get_supported_formats()
    
    def _load_file_fallback(self, file_path: Path) -> bool:
        """Load file in fallback mode (no playback, just info display)."""
        try:
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return False
            
            # Determine file type
            video_formats = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
            audio_formats = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
            
            file_ext = file_path.suffix.lower()
            
            if file_ext in video_formats:
                self.is_video = True
                self.show_video_mode_fallback()
            elif file_ext in audio_formats:
                self.is_video = False
                self.show_audio_mode_fallback()
            else:
                print(f"Unsupported file format: {file_ext}")
                return False
            
            self.current_file = file_path
            
            # Update file info
            file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
            file_type = "Video" if self.is_video else "Audio"
            self.file_info_label.setText(f"{file_type}: {file_path.name} ({file_size:.1f} MB)")
            
            # Emit signal
            self.file_loaded.emit(str(file_path))
            
            return True
            
        except Exception as e:
            print(f"Failed to load file: {e}")
            return False
    
    def show_video_mode_fallback(self):
        """Show video mode in fallback (no actual video playback)."""
        self.video_widget.hide()
        self.audio_placeholder.show()
        self.audio_placeholder.setText(
            f"🎬 Video File Loaded\n\n"
            f"{self.current_file.name if self.current_file else 'Video file'}\n\n"
            "Playback not available - Qt multimedia backend missing"
        )
    
    def show_audio_mode_fallback(self):
        """Show audio mode in fallback (no actual audio playback)."""
        self.video_widget.hide()
        self.audio_placeholder.show()
        self.audio_placeholder.setText(
            f"🎵 Audio File Loaded\n\n"
            f"{self.current_file.name if self.current_file else 'Audio file'}\n\n"
            "Playback not available - Qt multimedia backend missing"
        )


# Test function for standalone running
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    player = MultimediaPlayerWidget()
    player.setWindowTitle("InvisioVault - Multimedia Player Test")
    player.resize(600, 500)
    player.show()
    
    sys.exit(app.exec())
