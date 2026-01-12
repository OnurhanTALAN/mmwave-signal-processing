"""
mmWave Vibration Visualization App
Entry point for the application.
"""

import sys
from PySide6.QtWidgets import QApplication
from app import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("mmWave Vibration Visualizer")
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
