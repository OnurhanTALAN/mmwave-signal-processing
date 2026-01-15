"""
PyQt6 Radar GUI Application

Main application for radar recording, spectrogram visualization,
and vibration classification using CNN-LSTM model.
"""

import sys
import os
import numpy as np
import librosa
import librosa.display

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFrame, QGroupBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPalette, QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import TREND_LABELS, PRESENCE_LABELS, HOP_LENGTH
from radar_worker import RadarWorker
from motor_controller import get_motor_controller


class SpectrogramCanvas(FigureCanvas):
    """Matplotlib canvas for spectrogram display."""
    
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e2e')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e2e')
        super().__init__(self.fig)
        
        # Configure axes colors
        self.axes.tick_params(colors='white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.title.set_color('white')
        for spine in self.axes.spines.values():
            spine.set_color('white')
        
        self._colorbar = None
        self.clear_spectrogram()
    
    def clear_spectrogram(self):
        """Clear the spectrogram display."""
        self.axes.clear()
        self.axes.set_xlabel('Zaman (s)', color='white')
        self.axes.set_ylabel('Frekans (Hz)', color='white')
        self.axes.set_title('Spectrogram', color='white', fontsize=12)
        self.axes.text(0.5, 0.5, 'Kayıt bekleniyor...',
                      transform=self.axes.transAxes,
                      ha='center', va='center',
                      fontsize=14, color='#6c7086')
        self.axes.tick_params(colors='white')
        self.draw()
    
    def update_spectrogram(self, data: dict):
        """Update the spectrogram display with new data."""
        # Clear the entire figure and recreate axes to avoid colorbar issues
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e2e')
        
        spectrogram = data['spectrogram']
        
        # Use librosa.display.specshow
        im = librosa.display.specshow(
            spectrogram,
            sr=data['sample_rate'],
            hop_length=HOP_LENGTH,
            x_axis='time',
            y_axis='log',
            ax=self.axes,
        )
        
        self.axes.set_ylim(0, 512)
        
        self.axes.set_xlabel('Zaman (s)', color='white')
        self.axes.set_ylabel('Frekans (Hz)', color='white')
        self.axes.set_title('Radar Spectrogram (0-512 Hz)', color='white', fontsize=12)
        self.axes.tick_params(colors='white')
        for spine in self.axes.spines.values():
            spine.set_color('white')
        
        # Add colorbar
        self._colorbar = self.fig.colorbar(im, ax=self.axes, format='%+2.0f dB')
        self._colorbar.ax.yaxis.set_tick_params(color='white')
        self._colorbar.ax.yaxis.label.set_color('white')
        for t in self._colorbar.ax.yaxis.get_ticklabels():
            t.set_color('white')
        
        self.fig.tight_layout()
        self.draw()


class ResultPanel(QFrame):
    """Panel for displaying model prediction results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the result panel UI."""
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border: 2px solid #313244;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Model Sonuçları")
        title.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #cdd6f4; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Presence result
        presence_group = QGroupBox("Titreşim Durumu")
        presence_group.setStyleSheet("""
            QGroupBox {
                color: #89b4fa;
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        presence_layout = QVBoxLayout(presence_group)
        
        self.presence_label = QLabel("Bekleniyor...")
        self.presence_label.setFont(QFont('Segoe UI', 16, QFont.Weight.Bold))
        self.presence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.presence_label.setStyleSheet("color: #6c7086; border: none;")
        presence_layout.addWidget(self.presence_label)
        
        self.presence_confidence = QLabel("")
        self.presence_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.presence_confidence.setStyleSheet("color: #9399b2; border: none;")
        presence_layout.addWidget(self.presence_confidence)
        
        layout.addWidget(presence_group)
        
        # Trend result
        trend_group = QGroupBox("Trend Analizi")
        trend_group.setStyleSheet("""
            QGroupBox {
                color: #f9e2af;
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        trend_layout = QVBoxLayout(trend_group)
        
        self.trend_label = QLabel("Bekleniyor...")
        self.trend_label.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        self.trend_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trend_label.setStyleSheet("color: #6c7086; border: none;")
        trend_layout.addWidget(self.trend_label)
        
        self.trend_confidence = QLabel("")
        self.trend_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trend_confidence.setStyleSheet("color: #9399b2; border: none;")
        trend_layout.addWidget(self.trend_confidence)
        
        # Trend probabilities
        self.trend_probs = QLabel("")
        self.trend_probs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trend_probs.setStyleSheet("color: #7f849c; border: none; font-size: 11px;")
        trend_layout.addWidget(self.trend_probs)
        
        layout.addWidget(trend_group)
        
        layout.addStretch()
    
    def update_results(self, prediction: dict):
        """Update the results display with new prediction."""
        # Update presence
        if prediction['has_vibration']:
            self.presence_label.setText("✓ TİTREŞİM TESPİT EDİLDİ")
            self.presence_label.setStyleSheet("color: #a6e3a1; border: none;")  # Green
        else:
            self.presence_label.setText("✗ TİTREŞİM YOK")
            self.presence_label.setStyleSheet("color: #f38ba8; border: none;")  # Red
        
        confidence = prediction['presence_confidence'] * 100
        self.presence_confidence.setText(f"Güven: %{confidence:.1f}")
        
        # Update trend
        if prediction['has_vibration']:
            trend_class = prediction['trend_class']
            trend_label = prediction['trend_label']
            
            if trend_class == 0:  # Constant
                self.trend_label.setText("➡ " + trend_label)
                self.trend_label.setStyleSheet("color: #89b4fa; border: none;")  # Blue
            elif trend_class == 1:  # Increasing
                self.trend_label.setText("↗ " + trend_label)
                self.trend_label.setStyleSheet("color: #f9e2af; border: none;")  # Yellow
            else:  # Decreasing
                self.trend_label.setText("↘ " + trend_label)
                self.trend_label.setStyleSheet("color: #fab387; border: none;")  # Orange
            
            self.trend_confidence.setText(f"Güven: %{prediction['trend_confidence'] * 100:.1f}")
            
            probs = prediction['trend_probabilities']
            self.trend_probs.setText(
                f"Sabit: %{probs['constant']*100:.1f} | "
                f"Artan: %{probs['increasing']*100:.1f} | "
                f"Azalan: %{probs['decreasing']*100:.1f}"
            )
        else:
            self.trend_label.setText("N/A")
            self.trend_label.setStyleSheet("color: #6c7086; border: none;")
            self.trend_confidence.setText("Titreşim olmadığı için trend analizi yapılmadı")
            self.trend_probs.setText("")
    
    def clear_results(self):
        """Clear the results display."""
        self.presence_label.setText("Bekleniyor...")
        self.presence_label.setStyleSheet("color: #6c7086; border: none;")
        self.presence_confidence.setText("")
        self.trend_label.setText("Bekleniyor...")
        self.trend_label.setStyleSheet("color: #6c7086; border: none;")
        self.trend_confidence.setText("")
        self.trend_probs.setText("")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.thread = None
        self._selected_motor_mode = 'none'  # Store selected mode
        self.setup_ui()
        self.setup_worker()
    
    def setup_ui(self):
        """Setup the main window UI."""
        self.setWindowTitle("mmWave Radar - Titreşim Analizi")
        self.setMinimumSize(1200, 700)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #11111b;
            }
            QWidget {
                background-color: #11111b;
                color: #cdd6f4;
            }
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header = QLabel("mmWave Radar Titreşim Analizi")
        header.setFont(QFont('Segoe UI', 20, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #89b4fa; padding: 10px;")
        main_layout.addWidget(header)
        
        # Splitter for content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Spectrogram
        left_panel = QFrame()
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border: 2px solid #313244;
                border-radius: 10px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        
        self.spectrogram_canvas = SpectrogramCanvas(self, width=8, height=5)
        left_layout.addWidget(self.spectrogram_canvas)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Results
        # Right panel - Results
        self.result_panel = ResultPanel()
        self.result_panel.setMinimumWidth(250)  # Slightly reduced minimum width
        # Removed MaximumWidth to allow resizing
        splitter.addWidget(self.result_panel)
        
        # Configure splitter behavior
        splitter.setCollapsible(1, False)  # Prevent right panel from collapsing completely
        splitter.setStretchFactor(0, 1)    # Let the spectrogram expand more
        splitter.setStretchFactor(1, 0)    # Keep result panel near its minimum/content size
        
        splitter.setSizes([800, 300])
        main_layout.addWidget(splitter, 1)
        
        # Control panel
        control_panel = QFrame()
        control_panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border: 2px solid #313244;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        control_layout = QHBoxLayout(control_panel)
        
        # Record button
        self.record_btn = QPushButton("🔴 KAYIT AL")
        self.record_btn.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        self.record_btn.setMinimumSize(200, 60)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #11111b;
                border: none;
                border-radius: 10px;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: #eba0ac;
            }
            QPushButton:pressed {
                background-color: #f38ba8;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
        """)
        self.record_btn.clicked.connect(self.start_recording)
        control_layout.addWidget(self.record_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #313244;
                border: none;
                border-radius: 5px;
                height: 30px;
                text-align: center;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 5px;
            }
        """)
        control_layout.addWidget(self.progress_bar, 1)
        
        # Status label
        self.status_label = QLabel("Hazır")
        self.status_label.setFont(QFont('Segoe UI', 11))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMinimumWidth(200)
        self.status_label.setStyleSheet("color: #a6e3a1; border: none;")
        control_layout.addWidget(self.status_label)
        
        main_layout.addWidget(control_panel)
        
        # Motor control panel
        motor_panel = QFrame()
        motor_panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border: 2px solid #313244;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        motor_layout = QHBoxLayout(motor_panel)
        
        # Motor control label
        motor_label = QLabel("Motor Kontrolü:")
        motor_label.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
        motor_label.setStyleSheet("color: #f9e2af; border: none;")
        motor_layout.addWidget(motor_label)
        
        # Motor button style template
        motor_btn_style = """
            QPushButton {{
                background-color: {bg_color};
                color: #11111b;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:checked {{
                background-color: {checked_color};
                border: 3px solid #cdd6f4;
            }}
        """
        
        # Constant vibration button
        self.btn_constant = QPushButton("➡ Sabit")
        self.btn_constant.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        self.btn_constant.setCheckable(True)
        self.btn_constant.setStyleSheet(motor_btn_style.format(
            bg_color='#89b4fa', hover_color='#b4befe', checked_color='#74c7ec'
        ))
        self.btn_constant.clicked.connect(lambda: self.select_motor_mode('constant'))
        motor_layout.addWidget(self.btn_constant)
        
        # Increasing vibration button
        self.btn_increasing = QPushButton("↗ Artan")
        self.btn_increasing.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        self.btn_increasing.setCheckable(True)
        self.btn_increasing.setStyleSheet(motor_btn_style.format(
            bg_color='#a6e3a1', hover_color='#b5e8b0', checked_color='#94e2d5'
        ))
        self.btn_increasing.clicked.connect(lambda: self.select_motor_mode('increasing'))
        motor_layout.addWidget(self.btn_increasing)
        
        # Decreasing vibration button
        self.btn_decreasing = QPushButton("↘ Azalan")
        self.btn_decreasing.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        self.btn_decreasing.setCheckable(True)
        self.btn_decreasing.setStyleSheet(motor_btn_style.format(
            bg_color='#fab387', hover_color='#f9c597', checked_color='#f9e2af'
        ))
        self.btn_decreasing.clicked.connect(lambda: self.select_motor_mode('decreasing'))
        motor_layout.addWidget(self.btn_decreasing)
        
        # No vibration button
        self.btn_none = QPushButton("⏹ Durdur")
        self.btn_none.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold))
        self.btn_none.setCheckable(True)
        self.btn_none.setChecked(True)  # Default: motor off
        self.btn_none.setStyleSheet(motor_btn_style.format(
            bg_color='#f38ba8', hover_color='#eba0ac', checked_color='#f38ba8'
        ))
        self.btn_none.clicked.connect(lambda: self.select_motor_mode('none'))
        motor_layout.addWidget(self.btn_none)
        
        # Motor connection status
        self.motor_status_label = QLabel("Motor: Bağlanıyor...")
        self.motor_status_label.setFont(QFont('Segoe UI', 10))
        self.motor_status_label.setStyleSheet("color: #f9e2af; border: none;")
        self.motor_status_label.setMinimumWidth(150)
        motor_layout.addWidget(self.motor_status_label)
        
        main_layout.addWidget(motor_panel)
        
        # Store motor buttons for easy access
        self.motor_buttons = {
            'constant': self.btn_constant,
            'increasing': self.btn_increasing,
            'decreasing': self.btn_decreasing,
            'none': self.btn_none
        }
        
        # Connect to motor controller
        self.motor_controller = get_motor_controller()
        self.connect_motor()
    
    def setup_worker(self):
        """Setup the radar worker thread."""
        self.thread = QThread()
        self.worker = RadarWorker()
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.worker.started.connect(self.on_recording_started)
        self.worker.progress.connect(self.on_progress)
        self.worker.spectrogram_ready.connect(self.on_spectrogram_ready)
        self.worker.prediction_ready.connect(self.on_prediction_ready)
        self.worker.finished.connect(self.on_recording_finished)
        self.worker.error.connect(self.on_error)
        
        # Connect thread started to worker method (for triggering)
        self.thread.start()
    
    def start_recording(self):
        """Start a radar recording."""
        self.record_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.spectrogram_canvas.clear_spectrogram()
        self.result_panel.clear_results()
        
        # Start motor with selected mode
        if hasattr(self, 'motor_controller') and self.motor_controller.is_connected():
            self.motor_controller.set_mode(self._selected_motor_mode)
            self.status_label.setText(f"Motor başlatıldı: {self._selected_motor_mode}")
        
        # Use QMetaObject.invokeMethod to call method in worker thread
        from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(self.worker, "start_recording", Qt.ConnectionType.QueuedConnection)
    
    def on_recording_started(self):
        """Handle recording started signal."""
        self.status_label.setText("Kayıt başladı...")
        self.status_label.setStyleSheet("color: #f9e2af; border: none;")
    
    def on_progress(self, message: str, percentage: int):
        """Handle progress update signal."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
    
    def on_spectrogram_ready(self, data: dict):
        """Handle spectrogram ready signal."""
        self.spectrogram_canvas.update_spectrogram(data)
    
    def on_prediction_ready(self, prediction: dict):
        """Handle prediction ready signal."""
        self.result_panel.update_results(prediction)
    
    def on_recording_finished(self):
        """Handle recording finished signal."""
        # Stop motor when recording is finished
        if hasattr(self, 'motor_controller') and self.motor_controller.is_connected():
            self.motor_controller.set_mode('none')
        
        self.record_btn.setEnabled(True)
        self.status_label.setText("Tamamlandı ✓")
        self.status_label.setStyleSheet("color: #a6e3a1; border: none;")
    
    def on_error(self, message: str):
        """Handle error signal."""
        # Stop motor on error
        if hasattr(self, 'motor_controller') and self.motor_controller.is_connected():
            self.motor_controller.set_mode('none')
            
        self.record_btn.setEnabled(True)
        self.status_label.setText(f"Hata: {message}")
        self.status_label.setStyleSheet("color: #f38ba8; border: none;")
        self.progress_bar.setValue(0)
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop motor and disconnect
        if hasattr(self, 'motor_controller') and self.motor_controller:
            self.motor_controller.disconnect()
        
        if self.thread and self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
        event.accept()
    
    def connect_motor(self):
        """Connect to the motor controller."""
        if self.motor_controller.connect():
            self.motor_status_label.setText("Motor: Bağlı ✓")
            self.motor_status_label.setStyleSheet("color: #a6e3a1; border: none;")
        else:
            self.motor_status_label.setText("Motor: Bağlantı Yok")
            self.motor_status_label.setStyleSheet("color: #f38ba8; border: none;")
    
    def select_motor_mode(self, mode: str):
        """Select the motor vibration mode (will be applied on record start)."""
        self._selected_motor_mode = mode
        
        # Update button states
        for btn_mode, btn in self.motor_buttons.items():
            btn.setChecked(btn_mode == mode)
        
        mode_labels = {
            'constant': 'Sabit Titreşim',
            'increasing': 'Artan Titreşim',
            'decreasing': 'Azalan Titreşim',
            'none': 'Motor Kapalı'
        }
        self.motor_status_label.setText(f"Seçili: {mode_labels.get(mode, mode)}")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application-wide font
    app.setFont(QFont('Segoe UI', 10))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
