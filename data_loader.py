import numpy as np


def load_radar_data(file_path, num_frames, num_chirps, num_rx, num_samples):
    """
    Load and process radar data from a binary file.
    
    Parameters:
    -----------
    file_path : str
        Path to the binary file containing radar data
    num_frames : int
        Number of frames in the data
    num_chirps : int
        Number of chirps per frame
    num_rx : int
        Number of receiver antennas
    num_samples : int
        Number of ADC samples per chirp
    
    Returns:
    --------
    numpy.ndarray or None
        Processed frame data of shape [num_chirps, num_samples, num_rx] or None if loading fails
    """
    ints_per_frame = num_chirps * num_rx * num_samples * 2
    
    try:
        # Read binary file
        full_adc_data = np.fromfile(file_path, dtype=np.int16)
        
        # Reshape to frames
        framewise_raw = full_adc_data.reshape(num_frames, ints_per_frame)
        
        # Extract first frame
        first_frame_raw = framewise_raw[0]
        
        # Process the frame data: separate I/Q components and reshape
        frameData = np.reshape(first_frame_raw, [-1, 4])
        frameData = frameData[:, 0:2:] + 1j * frameData[:, 2::]
        frameData = np.reshape(frameData, [num_chirps, -1, num_samples])
        frameData = frameData.transpose([0, 2, 1])
        
        return frameData
        
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found. Please check the path.")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load radar data: {e}")
        return None

