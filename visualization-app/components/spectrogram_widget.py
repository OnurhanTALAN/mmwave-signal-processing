"""
Spectrogram Widget

Real-time spectrogram display using pyqtgraph for high performance.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
import pyqtgraph as pg


class SpectrogramWidget(QWidget):
    """
    A widget that displays a real-time scrolling spectrogram.
    Uses pyqtgraph ImageItem for efficient updates.
    """
    
    def __init__(
        self,
        n_freq_bins: int = 128,
        buffer_length: int = 200,
        parent=None
    ):
        super().__init__(parent)
        
        self.n_freq_bins = n_freq_bins
        self.buffer_length = buffer_length
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title label
        self.title_label = QLabel("Spectrogram")
        self.title_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
                padding: 5px;
                background-color: #2d2d2d;
            }
        """)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        
        # Configure axes
        self.plot_widget.setLabel('left', 'Frequency', units='Hz')
        self.plot_widget.setLabel('bottom', 'Time', units='frames')
        
        # Create image item for spectrogram
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        
        # Create colormap (viridis-like)
        colors = [
            (0, 0, 0),
            (68, 1, 84),
            (72, 40, 120),
            (62, 74, 137),
            (49, 104, 142),
            (38, 130, 142),
            (31, 158, 137),
            (53, 183, 121),
            (109, 205, 89),
            (180, 222, 44),
            (253, 231, 37)
        ]
        positions = np.linspace(0, 1, len(colors))
        color_map = pg.ColorMap(positions, colors)
        
        # Create and add color bar
        self.color_bar = pg.ColorBarItem(
            values=(-80, 0),
            colorMap=color_map,
            label='dB'
        )
        self.color_bar.setImageItem(self.img_item)
        
        # Apply colormap to image
        self.img_item.setLookupTable(color_map.getLookupTable(nPts=256))
        
        # Set initial data range
        self.img_item.setLevels([-80, 0])
        
        layout.addWidget(self.plot_widget)
        
        # Initialize with empty data
        empty_data = np.zeros((self.n_freq_bins, self.buffer_length))
        self.update_spectrogram(empty_data)
    
    def update_spectrogram(self, data: np.ndarray):
        """
        Update the spectrogram display with new data.
        
        Parameters:
        -----------
        data : np.ndarray
            2D array with shape (n_freq_bins, n_time_frames)
        """
        # Transpose for correct orientation (frequency on y-axis)
        self.img_item.setImage(data.T, autoLevels=False)
    
    def set_title(self, title: str):
        """Set the widget title."""
        self.title_label.setText(title)
