"""
Model Connector Interface

Abstract interface for connecting classification models.
Replace the MockModelConnector with your real model when ready.
"""

from abc import ABC, abstractmethod
import random
from typing import Literal


VibrationStatus = Literal['constant', 'increasing', 'decreasing']


class ModelConnector(ABC):
    """Abstract base class for model connectors."""
    
    @abstractmethod
    def predict(self, spectrogram_data) -> VibrationStatus:
        """
        Make a prediction based on spectrogram data.
        
        Parameters:
        -----------
        spectrogram_data : np.ndarray
            The spectrogram data to classify
            
        Returns:
        --------
        VibrationStatus
            One of: 'constant', 'increasing', 'decreasing'
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the model is connected and ready."""
        pass


class MockModelConnector(ModelConnector):
    """
    Mock model connector for testing without a real model.
    Generates random predictions with realistic patterns.
    """
    
    def __init__(self):
        self._connected = True
        self._current_state = 'constant'
        self._counter = 0
        self._transition_threshold = 30  # Frames before potential state change
    
    def predict(self, spectrogram_data) -> VibrationStatus:
        """Generate mock predictions with some persistence."""
        self._counter += 1
        
        # Only potentially change state after threshold frames
        if self._counter >= self._transition_threshold:
            if random.random() < 0.3:  # 30% chance to change state
                states = ['constant', 'increasing', 'decreasing']
                states.remove(self._current_state)
                self._current_state = random.choice(states)
                self._counter = 0
        
        return self._current_state
    
    def is_connected(self) -> bool:
        return self._connected


class TensorFlowModelConnector(ModelConnector):
    """
    TensorFlow/Keras model connector.
    
    TODO: Implement when the model is ready.
    
    Example usage:
        connector = TensorFlowModelConnector('models/vibration_classifier.h5')
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the TensorFlow model."""
        try:
            # Uncomment when model is ready:
            # from tensorflow.keras.models import load_model
            # self.model = load_model(self.model_path)
            raise NotImplementedError("Model not yet available")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None
    
    def predict(self, spectrogram_data) -> VibrationStatus:
        """Make prediction using the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # TODO: Implement actual prediction logic
        # prediction = self.model.predict(spectrogram_data)
        # return self._decode_prediction(prediction)
        
        raise NotImplementedError("Model prediction not yet implemented")
    
    def is_connected(self) -> bool:
        return self.model is not None
