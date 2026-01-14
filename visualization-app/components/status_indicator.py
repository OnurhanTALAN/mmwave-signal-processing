"""
Status Indicator Widget

Displays the current vibration status with visual feedback.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont


class StatusIndicator(QWidget):
    """
    A widget that displays the current vibration status.
    Shows: Constant, Increasing, or Decreasing with color coding.
    """
    
    # Status colors
    COLORS = {
        'constant': '#f59e0b',     # Amber/Yellow
        'increasing': '#ef4444',   # Red
        'decreasing': '#22c55e',   # Green
        'unknown': '#6b7280'       # Gray
    }
    
    # Status icons/arrows
    ICONS = {
        'constant': '━',      # Horizontal line
        'increasing': '▲',    # Up arrow
        'decreasing': '▼',    # Down arrow
        'unknown': '?'
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._current_status = 'unknown'
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the widget UI."""
        self.setMinimumHeight(150)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Vibration Status")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Status display container
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        
        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Icon label
        self.icon_label = QLabel(self.ICONS['unknown'])
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                font-size: 48px;
                color: {self.COLORS['unknown']};
                padding: 10px;
            }}
        """)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.icon_label)
        
        # Status text label
        self.status_label = QLabel("UNKNOWN")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 32px;
                font-weight: bold;
                color: {self.COLORS['unknown']};
                padding: 10px;
            }}
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(self.status_frame)
        
        # Connection status
        self.connection_label = QLabel("● Connected (Mock Mode)")
        self.connection_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #22c55e;
            }
        """)
        self.connection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.connection_label)
        
        # Set background
        self.setStyleSheet("""
            StatusIndicator {
                background-color: #1e1e1e;
                border-radius: 10px;
            }
        """)
    
    def set_status(self, status: str):
        """
        Set the current vibration status.
        
        Parameters:
        -----------
        status : str
            One of: 'constant', 'increasing', 'decreasing'
        """
        if status not in self.COLORS:
            status = 'unknown'
        
        if status == self._current_status:
            return
        
        self._current_status = status
        color = self.COLORS[status]
        icon = self.ICONS[status]
        
        # Update icon
        self.icon_label.setText(icon)
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                font-size: 48px;
                color: {color};
                padding: 10px;
            }}
        """)
        
        # Update status text
        self.status_label.setText(status.upper())
        self.status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 32px;
                font-weight: bold;
                color: {color};
                padding: 10px;
            }}
        """)
    
    def set_connection_status(self, connected: bool, mode: str = "Mock"):
        """Update the connection status display."""
        if connected:
            self.connection_label.setText(f"● Connected ({mode} Mode)")
            self.connection_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #22c55e;
                }
            """)
        else:
            self.connection_label.setText("○ Disconnected")
            self.connection_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #ef4444;
                }
            """)
    
    def get_status(self) -> str:
        """Get the current status."""
        return self._current_status
