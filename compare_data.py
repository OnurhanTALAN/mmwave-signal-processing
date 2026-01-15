import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import sys

# Add path to access project modules if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import processing logic to replicate exactly what the model sees
try:
    from mmwave.dataloader import DCA1000
except ImportError:
    # DCA1000 mock if not found, just to allow running if strict dependency missing
    class DCA1000:
        @staticmethod
        def organize(adc_data, num_chirps, num_rx, num_samples):
            # Fallback simple reshaping for inspection
            ret = np.zeros((num_chirps, num_rx, num_samples, 2), dtype=np.int16)
            ret[:,:,:,0] = adc_data[::2].reshape(num_chirps, num_rx, num_samples)
            ret[:,:,:,1] = adc_data[1::2].reshape(num_chirps, num_rx, num_samples)
            return ret

# Config params (Must match your current config)
FRAME_NUM = 110
CHIRP_NUM = 64
NUM_SAMPLES = 256
NUM_RX = 4
CHIRP_PERIOD = 20
N_FFT = 4096
HOP_LENGTH = 256
TRIM_SECONDS = 0.15
MAX_FREQ = 512

def process_file(file_path, label):
    print(f"--- Processing {label}: {file_path} ---")
    
    # 1. Load Raw
    adc_data = np.fromfile(file_path, dtype=np.int16)
    print(f"[{label}] Raw data size: {adc_data.size}")
    
    # 2. Pad
    expected_size = FRAME_NUM * NUM_RX * NUM_SAMPLES * CHIRP_NUM * 2
    adc_data_padded = np.ones(expected_size) * 1E-8
    adc_data_padded[:min(adc_data.shape[0], expected_size)] = adc_data[:min(adc_data.shape[0], expected_size)]
    adc_data = adc_data_padded.reshape(FRAME_NUM, -1)
    
    # 3. Organize
    # Assuming mmwave module is available, otherwise use mock
    try:
        from mmwave.dataloader import DCA1000
        adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=CHIRP_NUM, num_rx=NUM_RX, num_samples=NUM_SAMPLES)
    except:
        print("Warning: Standard DCA1000 organize failed, results may be inaccurate.")
        return None, None

    # 4. Average
    adc_data = np.mean(adc_data, axis=1, keepdims=True)
    adc_data_sq = np.squeeze(adc_data)
    adc_data_mean = np.mean(adc_data_sq, axis=1)
    
    # 5. STFT
    flattened_frames = adc_data_mean.flatten().astype(np.float32)
    Fs = NUM_SAMPLES * (1000 // CHIRP_PERIOD)
    D = librosa.stft(flattened_frames, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # 6. Convert to dB (CRITICAL STEP)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(f"[{label}] Max Value (ref=np.max): {np.max(np.abs(D)):.4f}")
    print(f"[{label}] Mean dB: {np.mean(S_db):.4f}, Min dB: {np.min(S_db):.4f}, Max dB: {np.max(S_db):.4f}")
    
    # 7. Time Trim
    if TRIM_SECONDS > 0:
        frame_duration = HOP_LENGTH / Fs
        trim_frames = int(TRIM_SECONDS / frame_duration)
        if trim_frames > 0:
            S_db = S_db[:, trim_frames:-trim_frames]
            
    # 8. Freq Trim (0-512 Hz)
    if MAX_FREQ is not None:
        freq_resolution = Fs / N_FFT
        max_bin = int(MAX_FREQ / freq_resolution)
        S_db = S_db[:max_bin, :]
        
    return S_db, Fs

def compare_files(train_file, new_file):
    # Process both
    S_train, Fs = process_file(train_file, "TRAIN (Working)")
    S_new, Fs = process_file(new_file, "NEW (Failing)")
    
    if S_train is None or S_new is None:
        print("Processing failed.")
        return

    # Visual Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Spectrograms
    vmin = min(np.min(S_train), np.min(S_new))
    vmax = max(np.max(S_train), np.max(S_new))
    
    im1 = axes[0, 0].imshow(S_train, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"Train File Spectrogram\n(Mean dB: {np.mean(S_train):.2f})")
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(S_new, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"New File Spectrogram\n(Mean dB: {np.mean(S_new):.2f})")
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot Histograms (Distribution of dB values)
    axes[1, 0].hist(S_train.flatten(), bins=50, color='blue', alpha=0.7, label='Train')
    axes[1, 0].hist(S_new.flatten(), bins=50, color='red', alpha=0.7, label='New')
    axes[1, 0].legend()
    axes[1, 0].set_title("Histogram comparison of dB values")
    
    # Plot Mean Spectrum (Frequency content)
    axes[1, 1].plot(np.mean(S_train, axis=1), label='Train', color='blue')
    axes[1, 1].plot(np.mean(S_new, axis=1), label='New', color='red')
    axes[1, 1].legend()
    axes[1, 1].set_title("Average Frequency Spectrum (Energy vs Freq Bin)")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_result.png')
    print("\nComparison saved to 'comparison_result.png'")
    # plt.show() # Uncomment if running interactively

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_data.py <train_file_path> <new_file_path>")
    else:
        compare_files(sys.argv[1], sys.argv[2])
