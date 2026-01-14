"""
Mock Data Generator

Generates synthetic spectrogram-like data for testing the visualization
without needing real radar data or a trained model.
"""

import numpy as np
from typing import Tuple


class MockDataGenerator:
    """
    Generates synthetic spectrogram data that simulates different
    vibration patterns: constant, increasing, and decreasing.
    """
    
    def __init__(
        self,
        n_freq_bins: int = 128,
        buffer_length: int = 200,
        update_rate: float = 0.05  # seconds between updates
    ):
        """
        Initialize the mock data generator.
        
        Parameters:
        -----------
        n_freq_bins : int
            Number of frequency bins in the spectrogram
        buffer_length : int
            Number of time frames to maintain in buffer
        update_rate : float
            Simulated time between frame updates
        """
        self.n_freq_bins = n_freq_bins
        self.buffer_length = buffer_length
        self.update_rate = update_rate
        
        # Initialize spectrogram buffer
        self.spectrogram_buffer = np.zeros((n_freq_bins, buffer_length))
        
        # State variables
        self.time_step = 0
        self.current_pattern = 'constant'
        self.base_intensity = 0.5
        self.vibration_frequency = 40  # Target frequency bin for vibration
        
    def set_pattern(self, pattern: str):
        """Set the current vibration pattern."""
        if pattern in ['constant', 'increasing', 'decreasing']:
            self.current_pattern = pattern
    
    def generate_frame(self) -> np.ndarray:
        """
        Generate a single new spectrogram frame.
        
        Returns:
        --------
        np.ndarray
            A column vector of frequency bin values
        """
        self.time_step += 1
        
        # Base noise floor
        frame = np.random.uniform(-80, -60, self.n_freq_bins)
        
        # Add vibration pattern
        intensity = self._calculate_intensity()
        
        # Create vibration signature (peak at vibration frequency)
        vibration_width = 10
        vibration_center = self.vibration_frequency
        
        for i in range(max(0, vibration_center - vibration_width), 
                       min(self.n_freq_bins, vibration_center + vibration_width)):
            distance = abs(i - vibration_center)
            peak = intensity * np.exp(-distance**2 / (2 * (vibration_width/3)**2))
            frame[i] += peak
        
        # Add some harmonics
        for harmonic in [2, 3]:
            harmonic_center = vibration_center * harmonic
            if harmonic_center < self.n_freq_bins:
                harmonic_intensity = intensity / (harmonic * 1.5)
                for i in range(max(0, harmonic_center - 5), 
                               min(self.n_freq_bins, harmonic_center + 5)):
                    distance = abs(i - harmonic_center)
                    frame[i] += harmonic_intensity * np.exp(-distance**2 / 8)
        
        # Add temporal variation
        frame += np.random.normal(0, 2, self.n_freq_bins)
        
        return frame
    
    def _calculate_intensity(self) -> float:
        """Calculate intensity based on current pattern."""
        base = 30  # dB above noise floor
        
        if self.current_pattern == 'constant':
            # Small random variation around base
            variation = np.sin(self.time_step * 0.1) * 3
            return base + variation
            
        elif self.current_pattern == 'increasing':
            # Ramp up over time with oscillation
            ramp = min(20, self.time_step * 0.3)
            variation = np.sin(self.time_step * 0.15) * 2
            return base + ramp + variation
            
        elif self.current_pattern == 'decreasing':
            # Ramp down over time
            ramp = max(-20, -self.time_step * 0.3)
            variation = np.sin(self.time_step * 0.1) * 2
            return max(5, base + ramp + variation)
        
        return base
    
    def update(self) -> np.ndarray:
        """
        Generate a new frame and update the buffer.
        
        Returns:
        --------
        np.ndarray
            The complete spectrogram buffer
        """
        new_frame = self.generate_frame()
        
        # Shift buffer left and add new frame on the right
        self.spectrogram_buffer = np.roll(self.spectrogram_buffer, -1, axis=1)
        self.spectrogram_buffer[:, -1] = new_frame
        
        return self.spectrogram_buffer
    
    def get_spectrogram(self) -> np.ndarray:
        """Get the current spectrogram buffer."""
        return self.spectrogram_buffer
    
    def reset(self):
        """Reset the generator state."""
        self.spectrogram_buffer = np.zeros((self.n_freq_bins, self.buffer_length))
        self.time_step = 0
