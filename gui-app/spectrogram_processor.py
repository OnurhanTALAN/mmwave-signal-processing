"""
Spectrogram Processor for Radar GUI Application

Generates spectrograms from mmWave radar ADC data (.bin files)
"""

import numpy as np
import librosa
import sys
import os

# Add parent directory to path for mmwave module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmwave.dataloader import DCA1000
from config import (
    FRAME_NUM, CHIRP_NUM, NUM_SAMPLES, NUM_RX, CHIRP_PERIOD,
    N_FFT, HOP_LENGTH, TARGET_SHAPE
)


def generate_spectrogram(file_path: str) -> tuple:
    """
    Generate a spectrogram from mmWave radar ADC data.
    
    Parameters:
    -----------
    file_path : str
        Path to the binary ADC data file
        
    Returns:
    --------
    S_db : np.ndarray
        Spectrogram in dB scale
    Fs : int
        Sample rate in Hz
    """
    # Load ADC data
    adc_data = np.fromfile(file_path, dtype=np.int16)
    
    # Pad the data if necessary
    expected_size = FRAME_NUM * NUM_RX * NUM_SAMPLES * CHIRP_NUM * 2
    adc_data_padded = np.ones(expected_size) * 1E-8
    adc_data_padded[:min(adc_data.shape[0], expected_size)] = adc_data[:min(adc_data.shape[0], expected_size)]
    adc_data = adc_data_padded.reshape(FRAME_NUM, -1)
    
    # Organize the data
    adc_data = np.apply_along_axis(
        DCA1000.organize,
        1,
        adc_data,
        num_chirps=CHIRP_NUM,
        num_rx=NUM_RX,
        num_samples=NUM_SAMPLES
    )
    
    # Average across chirps and RX antennas
    adc_data = np.mean(adc_data, axis=1, keepdims=True)
    adc_data_sq = np.squeeze(adc_data)
    adc_data_mean = np.mean(adc_data_sq, axis=1)
    
    # Flatten and convert to float32
    flattened_frames = adc_data_mean.flatten().astype(np.float32)
    
    # Calculate sample rate
    Fs = NUM_SAMPLES * (1000 // CHIRP_PERIOD)
    
    # Compute the STFT
    D = librosa.stft(flattened_frames, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    return S_db, Fs


def prepare_for_model(S_db: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Prepare spectrogram for model inference.
    
    Parameters:
    -----------
    S_db : np.ndarray
        Spectrogram in dB scale
    mean : float
        Normalization mean from training
    std : float
        Normalization std from training
        
    Returns:
    --------
    np.ndarray
        Normalized spectrogram ready for model input
    """
    from scipy.ndimage import zoom
    
    # Resize to target shape if necessary
    if S_db.shape != TARGET_SHAPE:
        zoom_factors = (TARGET_SHAPE[0] / S_db.shape[0], TARGET_SHAPE[1] / S_db.shape[1])
        S_db = zoom(S_db, zoom_factors, order=1)
    
    # Normalize using training statistics
    S_normalized = (S_db - mean) / (std + 1e-8)
    
    # Add batch and channel dimensions
    S_input = np.expand_dims(S_normalized, axis=(0, -1))
    
    return S_input


def get_spectrogram_for_display(S_db: np.ndarray, Fs: int) -> dict:
    """
    Get spectrogram data formatted for display.
    
    Parameters:
    -----------
    S_db : np.ndarray
        Spectrogram in dB scale
    Fs : int
        Sample rate in Hz
        
    Returns:
    --------
    dict : Display data including spectrogram, time axis, frequency axis
    """
    # Calculate time and frequency axes
    num_frames = S_db.shape[1]
    total_time = (FRAME_NUM * CHIRP_PERIOD) / 1000.0  # seconds
    
    time_axis = np.linspace(0, total_time, num_frames)
    freq_axis = np.linspace(0, Fs / 2, S_db.shape[0])
    
    return {
        'spectrogram': S_db,
        'time_axis': time_axis,
        'freq_axis': freq_axis,
        'sample_rate': Fs,
        'total_time': total_time
    }


if __name__ == "__main__":
    # Test with a sample file
    import matplotlib.pyplot as plt
    
    test_file = "../pipe-data/pipe-mixed-25-64-256-20/adc_data_11.bin"
    if os.path.exists(test_file):
        S_db, Fs = generate_spectrogram(test_file)
        print(f"Spectrogram shape: {S_db.shape}")
        print(f"Sample rate: {Fs} Hz")
        
        display_data = get_spectrogram_for_display(S_db, Fs)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(display_data['spectrogram'], aspect='auto', origin='lower',
                   extent=[0, display_data['total_time'], 0, Fs/2])
        plt.colorbar(label='dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Test Spectrogram')
        plt.ylim(0, 512)
        plt.show()
    else:
        print(f"Test file not found: {test_file}")
