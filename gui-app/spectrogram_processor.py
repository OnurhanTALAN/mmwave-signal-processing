"""
Spectrogram Processor for Radar GUI Application

Generates spectrograms from mmWave radar ADC data (.bin files)
"""

import numpy as np
import librosa
import sys
import os

# Add parent directory to path for mmwave module
if not getattr(sys, 'frozen', False):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmwave.dataloader import DCA1000
from config import (
    FRAME_NUM, CHIRP_NUM, NUM_SAMPLES, NUM_RX, CHIRP_PERIOD,
    N_FFT, HOP_LENGTH, TARGET_SHAPE, TRIM_SECONDS, MAX_FREQ
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
    
    # Trim time (match offline processing)
    if TRIM_SECONDS > 0:
        frame_duration = HOP_LENGTH / Fs
        trim_frames = int(TRIM_SECONDS / frame_duration)
        if trim_frames > 0:
            S_db = S_db[:, trim_frames:-trim_frames]
            
    # Trim frequency (0-512 Hz) - REMOVED from here to keep UI full spectrum
    # if MAX_FREQ is not None:
    #     freq_resolution = Fs / N_FFT
    #     max_bin = int(MAX_FREQ / freq_resolution)
    #     S_db = S_db[:max_bin, :]
    
    return S_db, Fs


def prepare_for_model(spectrogram: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Prepare spectrogram for model inference (trim frequency, resize, normalize, add dims).
    
    Parameters:
    -----------
    spectrogram : np.ndarray
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

    # 1. Frequency Trimming (0-512 Hz) for Model Only
    if MAX_FREQ is not None:
        # Re-calculate Fs to determine bins
        Fs = NUM_SAMPLES * (1000 // CHIRP_PERIOD)
        freq_resolution = Fs / N_FFT
        max_bin = int(MAX_FREQ / freq_resolution)
        spectrogram = spectrogram[:max_bin, :]

    # 2. Resize to target shape if necessary
    if spectrogram.shape != TARGET_SHAPE:
        zoom_factors = (TARGET_SHAPE[0] / spectrogram.shape[0], TARGET_SHAPE[1] / spectrogram.shape[1])
        spectrogram = zoom(spectrogram, zoom_factors, order=1)
    
    # 3. Normalize using training statistics
    S_normalized = (spectrogram - mean) / (std + 1e-8)
    
    # 4. Add batch and channel dimensions
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
    return {
        'spectrogram': S_db,
        'sample_rate': Fs
    }


if __name__ == "__main__":
    # Test with a sample file
    import matplotlib.pyplot as plt
    
    #test_file = "../pipe-data/pipe-mixed-25-64-256-20/adc_data_11.bin"
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
