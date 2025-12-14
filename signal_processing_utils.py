"""
Signal processing utility functions for radar data analysis.
"""
import numpy as np


def range_Doppler(data, mode=0, padding_size=None):
    """
    Perform Range-Doppler processing on radar data.
    
    Applies Hanning window in both dimensions and computes 2D FFT
    to generate Range-Doppler Image (RDI).
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input radar data of shape [num_chirps, num_samples, num_channels]
    mode : int, optional
        Processing mode (default: 0)
        - 0: Return raw FFT output
        - 1: Return absolute value with FFT shift
        - 2: Return both raw and absolute values
    padding_size : list or tuple, optional
        Zero-padding size for FFT as [doppler_padding, range_padding]
    
    Returns:
    --------
    numpy.ndarray or list
        Depending on mode:
        - mode 0: Complex FFT output
        - mode 1: Absolute magnitude with FFT shift applied
        - mode 2: List containing [raw_fft, absolute_magnitude]
    
    Raises:
    -------
    ValueError
        If mode is not 0, 1, or 2
    """
    # Create 2D Hanning window
    window_1 = np.reshape(np.hanning(data.shape[1]), (data.shape[1], -1))
    window_2 = np.reshape(np.hanning(data.shape[0]), (-1, data.shape[0]))
    window = window_1 * window_2
    
    # Apply window to each channel
    channel = data.shape[2]
    for i in range(channel):
        data[:, :, i] = data[:, :, i] * window.T

    # Compute 2D FFT based on mode
    if mode == 0:
        return np.fft.fft2(data, s=padding_size, axes=[0, 1])
    elif mode == 1:
        rdi_abs = np.fft.fft2(data, s=padding_size, axes=[0, 1])
        rdi_abs = np.transpose(np.fft.fftshift(np.abs(rdi_abs), axes=0), [1, 0, 2])
        return rdi_abs
    elif mode == 2:
        rdi_raw = np.fft.fft2(data, s=padding_size, axes=[0, 1])
        rdi_abs = np.transpose(np.fft.fftshift(np.abs(rdi_raw), axes=0), [1, 0, 2])
        return [rdi_raw, rdi_abs]
    else:
        raise ValueError("Error mode")


def find_top_peaks(rdi_img, resolution, num_peaks=3):
    """
    Find the top N peaks in a Range-Doppler image.
    
    Parameters:
    -----------
    rdi_img : numpy.ndarray
        2D Range-Doppler image array
    resolution : float
        Range resolution in meters
    num_peaks : int, optional
        Number of top peaks to find (default: 3)
    
    Returns:
    --------
    list of dict
        List containing information about each peak with keys:
        - 'range_bin': Range bin index
        - 'doppler_bin': Doppler bin index
        - 'strength': Signal strength (magnitude)
        - 'distance_m': Distance in meters
    """
    # Flatten the image and find top N indices
    flattened = rdi_img.flatten()
    top_flat_indices = np.argsort(flattened)[-num_peaks:][::-1]
    top_rows, top_cols = np.unravel_index(top_flat_indices, rdi_img.shape)
    
    # Collect peak information
    peaks_info = []
    for i in range(num_peaks):
        r_bin = top_rows[i]
        d_bin = top_cols[i]
        strength = rdi_img[r_bin, d_bin]
        distance_m = r_bin * resolution
        
        peak_data = {
            'range_bin': r_bin,
            'doppler_bin': d_bin,
            'strength': strength,
            'distance_m': distance_m
        }
        peaks_info.append(peak_data)
        
        # Print peak information
        print(f"Signal #{i+1}: Strength={strength:.1f} | Distance: {distance_m:.3f} m (Bin: {r_bin}) | Doppler Bin: {d_bin}")
    
    return peaks_info
