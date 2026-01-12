"""
mmWave Vibration Visualization Application

Main application window with real-time spectrogram display
and vibration status indicator.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFrame, QLabel, QSplitter
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon

from components.spectrogram_widget import SpectrogramWidget
from components.status_indicator import StatusIndicator
from core.mock_data import MockDataGenerator
from core.model_connector import MockModelConnector


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("mmWave Vibration Visualizer")
        self.setMinimumSize(1200, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
        """)
        
        # Initialize components
        self.data_generator = MockDataGenerator(n_freq_bins=128, buffer_length=200)
        self.model_connector = MockModelConnector()
        
        # Timer for updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update)
        self.update_interval = 50  # milliseconds
        
        self._is_running = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the main UI."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Header
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Main content area with splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Spectrogram
        self.spectrogram_widget = SpectrogramWidget(n_freq_bins=128, buffer_length=200)
        content_splitter.addWidget(self.spectrogram_widget)
        
        # Right side - Status and controls
        right_panel = self._create_right_panel()
        content_splitter.addWidget(right_panel)
        
        # Set splitter proportions (70% spectrogram, 30% status)
        content_splitter.setSizes([700, 300])
        content_splitter.setHandleWidth(2)
        content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3d3d3d;
            }
        """)
        
        main_layout.addWidget(content_splitter)
    
    def _create_header(self) -> QFrame:
        """Create the header frame."""
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-radius: 10px;
            }
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Title
        title = QLabel("mmWave Vibration Visualizer")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #ffffff;
            }
        """)
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Status indicator dot
        self.header_status = QLabel("● Ready")
        self.header_status.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #6b7280;
            }
        """)
        layout.addWidget(self.header_status)
        
        return header
    
    def _create_right_panel(self) -> QFrame:
        """Create the right panel with status and controls."""
        panel = QFrame()
        panel.setMinimumWidth(300)
        panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-radius: 10px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Status indicator
        self.status_indicator = StatusIndicator()
        layout.addWidget(self.status_indicator)
        
        layout.addStretch()
        
        # Controls section
        controls_frame = QFrame()
        controls_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setSpacing(10)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶ Start")
        self.start_button.setFixedHeight(45)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #16a34a;
            }
            QPushButton:pressed {
                background-color: #15803d;
            }
        """)
        self.start_button.clicked.connect(self._toggle_running)
        button_layout.addWidget(self.start_button)
        
        self.reset_button = QPushButton("↺ Reset")
        self.reset_button.setFixedHeight(45)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)
        self.reset_button.clicked.connect(self._reset)
        button_layout.addWidget(self.reset_button)
        
        controls_layout.addLayout(button_layout)
        
        # FPS label
        self.fps_label = QLabel("Update Rate: 20 FPS")
        self.fps_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #9ca3af;
            }
        """)
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.fps_label)
        
        layout.addWidget(controls_frame)
        
        # Info section
        info_label = QLabel("Demo Mode - Mock Data")
        info_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #6b7280;
                padding: 5px;
            }
        """)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        return panel
    
    def _toggle_running(self):
        """Toggle the running state."""
        if self._is_running:
            self._stop()
        else:
            self._start()
    
    def _start(self):
        """Start the visualization."""
        self._is_running = True
        self.start_button.setText("⏸ Pause")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #f59e0b;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #d97706;
            }
            QPushButton:pressed {
                background-color: #b45309;
            }
        """)
        
        self.header_status.setText("● Running")
        self.header_status.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #22c55e;
            }
        """)
        
        self.update_timer.start(self.update_interval)
    
    def _stop(self):
        """Stop the visualization."""
        self._is_running = False
        self.start_button.setText("▶ Start")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #16a34a;
            }
            QPushButton:pressed {
                background-color: #15803d;
            }
        """)
        
        self.header_status.setText("● Paused")
        self.header_status.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #f59e0b;
            }
        """)
        
        self.update_timer.stop()
    
    def _reset(self):
        """Reset the visualization."""
        self._stop()
        self.data_generator.reset()
        self.status_indicator.set_status('unknown')
        
        # Clear spectrogram
        import numpy as np
        empty_data = np.zeros((128, 200))
        self.spectrogram_widget.update_spectrogram(empty_data)
        
        self.header_status.setText("● Ready")
        self.header_status.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #6b7280;
            }
        """)
    
    def _update(self):
        """Update the visualization with new data."""
        # Get new spectrogram data
        spectrogram = self.data_generator.update()
        
        # Get prediction from model connector
        prediction = self.model_connector.predict(spectrogram)
        
        # Update the mock data pattern based on prediction
        # (In real use, this would be reversed - prediction comes from model)
        self.data_generator.set_pattern(prediction)
        
        # Update displays
        self.spectrogram_widget.update_spectrogram(spectrogram)
        self.status_indicator.set_status(prediction)
    
    def closeEvent(self, event):
        """Handle window close."""
        self.update_timer.stop()
        event.accept()
