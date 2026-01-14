"""
Model Inference Module for Radar GUI Application

Loads the CNN-LSTM model and performs inference on spectrograms.
Model outputs:
- presence: Binary (vibration vs no vibration)
- trend: 3-class (constant, increasing, decreasing)
"""

import numpy as np
import tensorflow as tf
import os

from config import MODEL_PATH, NORM_STATS_PATH, TREND_LABELS, PRESENCE_LABELS


# Define the freq_avg_pool function at module level for proper serialization
# This function is used in a Lambda layer to average over the frequency dimension
# Input shape: (batch, 97, 21, 64) -> Output: (batch, 97, 64)
def freq_avg_pool(t):
    """Average pooling over frequency dimension (axis=2)."""
    return tf.reduce_mean(t, axis=2)


# Custom layer class as alternative approach for Lambda deserialization
@tf.keras.utils.register_keras_serializable()
class FreqAvgPoolLayer(tf.keras.layers.Layer):
    """Custom layer that performs average pooling over frequency dimension (axis=2)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=2)
    
    def compute_output_shape(self, input_shape):
        # Input: (batch, time, freq, channels)
        # Output: (batch, time, channels)
        return (input_shape[0], input_shape[1], input_shape[3])
    
    def get_config(self):
        return super().get_config()


class ModelInference:
    """
    Handles loading and inference for the CNN-LSTM model.
    """
    
    def __init__(self):
        self.model = None
        self.mean = None
        self.std = None
        self._loaded = False
    
    def load(self) -> bool:
        """
        Load the model and normalization statistics.
        
        Returns:
        --------
        bool : True if loading successful, False otherwise
        """
        try:
            # Check if files exist
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            if not os.path.exists(NORM_STATS_PATH):
                raise FileNotFoundError(f"Normalization stats not found: {NORM_STATS_PATH}")
            
            print(f"Loading model from: {MODEL_PATH}")
            
            # Try to load with the custom FreqAvgPoolLayer substitution
            # First, we need to load the model architecture and weights separately
            # then reconstruct with proper output_shape
            
            # Attempt 1: Load with Lambda wrapper that has compute_output_shape
            class FreqAvgPoolLambda(tf.keras.layers.Layer):
                """Wrapper for freq_avg_pool that properly computes output shape."""
                def __init__(self, **kwargs):
                    # Filter out keys that come from Lambda layer but aren't valid for Layer
                    # The original Lambda config includes 'function', 'arguments', 'dtype' (with nested config)
                    invalid_keys = ['function', 'arguments', 'dtype', 'trainable', 'output_shape', 'output_shape_type', 'mask']
                    for key in invalid_keys:
                        kwargs.pop(key, None)
                    
                    # Extract name or use default
                    name = kwargs.pop('name', 'freq_avg_pool')
                    super().__init__(name=name, **kwargs)
                
                def call(self, inputs, mask=None):
                    # Accept mask argument to be compatible with Lambda layer's inbound_nodes
                    return tf.reduce_mean(inputs, axis=2)
                
                def compute_output_shape(self, input_shape):
                    # Input: (batch, time, freq, channels)
                    # Output: (batch, time, channels) - removes freq dimension
                    return tf.TensorShape([input_shape[0], input_shape[1], input_shape[3]])
                
                def get_config(self):
                    # Return only the name for clean serialization
                    return {'name': self.name}
                
                @classmethod
                def from_config(cls, config):
                    return cls(**config)
            
            # Register the custom layer
            custom_objects = {
                'FreqAvgPoolLambda': FreqAvgPoolLambda,
            }
            
            # Load model by reconstructing architecture
            import zipfile
            import json
            
            # Read the config from the .keras file
            with zipfile.ZipFile(MODEL_PATH, 'r') as z:
                config_json = z.read('config.json').decode('utf-8')
                model_config = json.loads(config_json)
            
            # Find and replace the Lambda layer with our custom layer
            def replace_lambda_in_config(config):
                """Recursively find and replace freq_avg_pool Lambda layers."""
                if isinstance(config, dict):
                    # Check if this is the Lambda layer we're looking for
                    if config.get('class_name') == 'Lambda':
                        layer_config = config.get('config', {})
                        if layer_config.get('name') == 'freq_avg_pool':
                            # Get and clean inbound_nodes - remove mask from kwargs
                            inbound_nodes = config.get('inbound_nodes', [])
                            cleaned_nodes = []
                            for node in inbound_nodes:
                                cleaned_node = dict(node)
                                if 'kwargs' in cleaned_node:
                                    # Remove mask from kwargs if present
                                    cleaned_kwargs = {k: v for k, v in cleaned_node['kwargs'].items() if k != 'mask'}
                                    cleaned_node['kwargs'] = cleaned_kwargs
                                cleaned_nodes.append(cleaned_node)
                            
                            # Replace with our custom layer, preserving essential keys
                            return {
                                'class_name': 'FreqAvgPoolLambda',
                                'config': {'name': 'freq_avg_pool'},
                                'module': None,
                                'registered_name': 'FreqAvgPoolLambda',
                                # Preserve these keys from the original Lambda layer
                                'name': config.get('name', 'freq_avg_pool'),
                                'inbound_nodes': cleaned_nodes,
                                'build_config': config.get('build_config', {})
                            }
                    # Recursively process all values
                    return {k: replace_lambda_in_config(v) for k, v in config.items()}
                elif isinstance(config, list):
                    return [replace_lambda_in_config(item) for item in config]
                return config
            
            modified_config = replace_lambda_in_config(model_config)
            
            # Remove compile_config since we only need inference (not training)
            # This avoids needing to register custom metrics like F1Score
            if 'compile_config' in modified_config:
                del modified_config['compile_config']
            
            # Rebuild the model from modified config
            self.model = tf.keras.models.model_from_json(
                json.dumps(modified_config),
                custom_objects=custom_objects
            )
            
            # Load weights from the .keras file
            with zipfile.ZipFile(MODEL_PATH, 'r') as z:
                # Extract weights to a temp location
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Extract model.weights.h5
                    z.extract('model.weights.h5', tmpdir)
                    weights_path = os.path.join(tmpdir, 'model.weights.h5')
                    self.model.load_weights(weights_path)
            
            # Load normalization statistics
            print(f"Loading normalization stats from: {NORM_STATS_PATH}")
            norm_stats = np.load(NORM_STATS_PATH, allow_pickle=True).item()
            self.mean = norm_stats['mean']
            self.std = norm_stats['std']
            
            print(f"Model loaded successfully!")
            print(f"  - Normalization mean: {self.mean:.4f}")
            print(f"  - Normalization std: {self.std:.4f}")
            
            self._loaded = True
            return True
            
        except Exception as e:
            import traceback
            print(f"Error loading model: {e}")
            print("Full traceback:")
            traceback.print_exc()
            self._loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def predict(self, spectrogram: np.ndarray) -> dict:
        """
        Run inference on a spectrogram.
        
        Parameters:
        -----------
        spectrogram : np.ndarray
            Prepared spectrogram input (batch, height, width, channels)
            
        Returns:
        --------
        dict : Prediction results including probabilities and labels
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Run prediction
        predictions = self.model.predict(spectrogram, verbose=0)
        
        # Extract outputs
        presence_prob = float(predictions['presence'][0][0])
        trend_probs = predictions['trend'][0]
        
        # Determine classes
        presence_class = 1 if presence_prob >= 0.5 else 0
        trend_class = int(np.argmax(trend_probs))
        
        result = {
            # Presence output
            'presence_probability': presence_prob,
            'presence_class': presence_class,
            'presence_label': PRESENCE_LABELS[presence_class],
            'has_vibration': presence_class == 1,
            
            # Trend output (only meaningful if vibration is present)
            'trend_probabilities': {
                'constant': float(trend_probs[0]),
                'increasing': float(trend_probs[1]),
                'decreasing': float(trend_probs[2])
            },
            'trend_class': trend_class,
            'trend_label': TREND_LABELS[trend_class],
            
            # Confidence
            'presence_confidence': presence_prob if presence_class == 1 else (1 - presence_prob),
            'trend_confidence': float(np.max(trend_probs))
        }
        
        return result
    
    def get_normalization_stats(self) -> tuple:
        """Get normalization mean and std."""
        return self.mean, self.std


# Global instance for easy access
_model_instance = None


def get_model() -> ModelInference:
    """Get the global model instance, loading if necessary."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelInference()
    return _model_instance


if __name__ == "__main__":
    # Test model loading
    model = ModelInference()
    if model.load():
        print("\nModel loaded successfully!")
        
        # Create a dummy input for testing
        dummy_input = np.random.randn(1, 163, 97, 1).astype(np.float32)
        result = model.predict(dummy_input)
        
        print("\nTest prediction results:")
        print(f"  Presence: {result['presence_label']}")
        print(f"    Probability: {result['presence_probability']:.4f}")
        print(f"    Confidence: {result['presence_confidence']:.4f}")
        print(f"  Trend: {result['trend_label']}")
        print(f"    Probabilities: {result['trend_probabilities']}")
        print(f"    Confidence: {result['trend_confidence']:.4f}")
    else:
        print("Failed to load model!")
