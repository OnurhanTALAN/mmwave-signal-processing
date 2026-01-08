import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from mmwave.dataloader import DCA1000

def generate_spectrogram(
    file_path,
    num_frames=25,
    num_rx=4,
    num_chirps=64,
    num_samples=256,
    chirp_period=20,
    n_fft=4096,
    hop_length=256
):
    """
    Generate a spectrogram from mmWave radar ADC data.
    
    Parameters:
    -----------
    file_path : str
        Path to the binary ADC data file
    num_frames : int
        Number of frames in the radar data
    num_rx : int
        Number of receive antennas
    num_chirps : int
        Number of chirps per frame
    num_samples : int
        Number of samples per chirp
    chirp_period : int
        Chirp period in milliseconds
    n_fft : int
        FFT size for STFT
    hop_length : int
        Hop length for STFT
        
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
    adc_data_padded = np.ones(num_frames * num_rx * num_samples * num_chirps * 2) * 1E-8
    adc_data_padded[:adc_data.shape[0]] = adc_data        
    adc_data = adc_data_padded.reshape(num_frames, -1)
    
    # Organize the data
    adc_data = np.apply_along_axis(
        DCA1000.organize,
        1,
        adc_data,
        num_chirps=num_chirps,
        num_rx=num_rx,
        num_samples=num_samples
    )
    
    adc_data = np.mean(adc_data, axis=1, keepdims=True)
    adc_data_sq = np.squeeze(adc_data)
    adc_data_mean = np.mean(adc_data_sq, axis=1)
    
    # Flatten and convert to float32
    flattened_frames = adc_data_mean.flatten().astype(np.float32)
    
    # Calculate sample rate
    Fs = num_samples * (1000 // chirp_period)
    
    # Compute the STFT
    D = librosa.stft(flattened_frames, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    return S_db, Fs


def plot_spectrogram(S_db, Fs, hop_length=256, title='Spectrogram of mmWave Radar ADC Samples', figsize=(15, 6)):
    """
    Plot a spectrogram.
    
    Parameters:
    -----------
    S_db : np.ndarray
        Spectrogram in dB scale
    Fs : int
        Sample rate in Hz
    hop_length : int
        Hop length used in STFT
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    librosa.display.specshow(S_db, sr=Fs, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate spectrogram for a single file
    file_path = "pipe-data\\pipe-mixed-25-64-256-20\\adc_data_11.bin"
    S_db, Fs = generate_spectrogram(file_path)
    
    print(f"Spectrogram shape: {S_db.shape}")
    plot_spectrogram(S_db, Fs, title='Concatenated Spectrogram of mmWave Radar ADC Samples')
