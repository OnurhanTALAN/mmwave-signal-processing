"""
Configuration for Radar GUI Application
"""

import os
import sys

# =============================================================================
# RADAR PARAMETERS (must match Lua script)
# =============================================================================
FRAME_NUM = 110
CHIRP_NUM = 64
NUM_SAMPLES = 256
NUM_RX = 4
CHIRP_PERIOD = 20  # ms

# Calculated values
MS_PER_RECORD = FRAME_NUM * CHIRP_PERIOD  # 2200 ms

# =============================================================================
# STFT PARAMETERS
# =============================================================================
N_FFT = 4096
HOP_LENGTH = 256
TRIM_SECONDS = 0.15  # Time to trim from start and end (seconds)
MAX_FREQ = 512       # Maximum frequency to keep (Hz)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Model9: CNN-LSTM Multi-output model
# - presence: binary (vibration vs no vibration)
# - trend: 3-class (constant, increasing, decreasing)
# Check if running as compiled executable
if getattr(sys, 'frozen', False):
    # If frozen, logic depends on where we bundle the data
    # We will bundle 'models' folder at the root level of the temp directory
    BASE_PATH = sys._MEIPASS
    MODEL_PATH = os.path.join(BASE_PATH, "models", "model6", "best_cnn_lstm.keras")
    NORM_STATS_PATH = os.path.join(BASE_PATH, "models", "model6", "best_cnn_lstm_norm_stats.npy")
else:
    # If running as script
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model6", "best_cnn_lstm.keras")
    NORM_STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model6", "best_cnn_lstm_norm_stats.npy")

# Model input shape
INPUT_SHAPE = (163, 97, 1)
TARGET_SHAPE = (163, 97)

# =============================================================================
# FILE-BASED COMMUNICATION
# =============================================================================
# Signal file for communication between Python and Lua
SIGNAL_FILE = "C:/ti/mmwave_studio_02_01_01_00/mmWaveStudio/PostProc/gui_signal.txt"

# Output directory for radar recordings
OUTPUT_DIR = "C:/ti/mmwave_studio_02_01_01_00/mmWaveStudio/PostProc/gui-output/"

# =============================================================================
# TREND LABELS
# =============================================================================
TREND_LABELS = {
    0: "Sabit (Constant)",
    1: "Artıyor (Increasing)",
    2: "Azalıyor (Decreasing)"
}

PRESENCE_LABELS = {
    0: "Titreşim Yok (No Vibration)",
    1: "Titreşim Var (Vibration Detected)"
}
