import sys
import os
import numpy as np
import tensorflow as tf

import librosa

# Add current directory to sys.path to ensure modules can be found if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mmwave.dataloader import DCA1000
except ImportError:
    # If running from root, try adding root to path or gui-app logic
    pass

# Hardcoded parameters to match project config exactly
FRAME_NUM = 110
CHIRP_NUM = 64
NUM_SAMPLES = 256
NUM_RX = 4
CHIRP_PERIOD = 20
N_FFT = 4096
HOP_LENGTH = 256
TARGET_SHAPE = (163, 97)
TRIM_SECONDS = 0.15
MAX_FREQ = 512

def freq_avg_pool(t):
    return tf.reduce_mean(t, axis=2)

@tf.keras.utils.register_keras_serializable()
class FreqAvgPoolLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=2)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[3])

def process_bin_file(bin_path, mean, std):
    """
    Process a raw .bin file into a normalized spectrogram ready for the model.
    Steps match batch_process_spectrograms.py logic exactly.
    """
    print(f"Processing raw binary file: {bin_path}")
    
    # 1. Load ADC data
    adc_data = np.fromfile(bin_path, dtype=np.int16)
    
    # 2. Pad
    expected_size = FRAME_NUM * NUM_RX * NUM_SAMPLES * CHIRP_NUM * 2
    adc_data_padded = np.ones(expected_size) * 1E-8
    adc_data_padded[:min(adc_data.shape[0], expected_size)] = adc_data[:min(adc_data.shape[0], expected_size)]
    adc_data = adc_data_padded.reshape(FRAME_NUM, -1)
    
    # 3. Organize (DCA1000)
    adc_data = np.apply_along_axis(
        DCA1000.organize,
        1,
        adc_data,
        num_chirps=CHIRP_NUM,
        num_rx=NUM_RX,
        num_samples=NUM_SAMPLES
    )
    
    # 4. Average
    adc_data = np.mean(adc_data, axis=1, keepdims=True)
    adc_data_sq = np.squeeze(adc_data)
    adc_data_mean = np.mean(adc_data_sq, axis=1)
    
    # 5. Flatten
    flattened_frames = adc_data_mean.flatten().astype(np.float32)
    
    # 6. STFT
    Fs = NUM_SAMPLES * (1000 // CHIRP_PERIOD)
    D = librosa.stft(flattened_frames, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # 7. Trim Time (Array-level)
    print(f"Spectrogram shape before time trim: {S_db.shape}")
    if TRIM_SECONDS > 0:
        frame_duration = HOP_LENGTH / Fs
        trim_frames = int(TRIM_SECONDS / frame_duration)
        if trim_frames > 0:
            S_db = S_db[:, trim_frames:-trim_frames]
    print(f"Spectrogram shape after time trim: {S_db.shape}")

    # 8. Trim Frequency (0-512 Hz)
    if MAX_FREQ is not None:
        freq_resolution = Fs / N_FFT
        max_bin = int(MAX_FREQ / freq_resolution)
        S_db = S_db[:max_bin, :]
        print(f"Spectrogram shape after freq trim (0-{MAX_FREQ}Hz): {S_db.shape}")
            
    # 9. Resize (if needed) and Normalize
    from scipy.ndimage import zoom
    if S_db.shape != TARGET_SHAPE:
        print(f"Resizing spectrogram from {S_db.shape} to {TARGET_SHAPE}")
        zoom_factors = (TARGET_SHAPE[0] / S_db.shape[0], TARGET_SHAPE[1] / S_db.shape[1])
        S_db = zoom(S_db, zoom_factors, order=1)
        
    # Normalize
    S_normalized = (S_db - mean) / (std + 1e-8)
    
    # Add batch and channel dims
    # Result: (1, 163, 97, 1)
    S_input = np.expand_dims(S_normalized, axis=(0, -1))
    
    return S_input

def main():
    # 1. Provide Arguments
    if len(sys.argv) < 2:
        print("Usage: python predict_npy.py <path_to_file>")
        print("Supported formats: .npy (pre-processed), .bin (raw radar data)")
        return

    file_path = sys.argv[1]
    
    # Path to model8
    model_path = os.path.join("models", "model9", "best_cnn_lstm.keras")
    stats_path = os.path.join("models", "model9", "best_cnn_lstm_norm_stats.npy")

    # 2. Check File
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at: {file_path}")
        return

    try:
        # Check extension
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.bin':
            # Need normalization stats for .bin processing
            if not os.path.exists(stats_path):
                print(f"Error: Stats file not found at {stats_path}. Cannot process .bin file.")
                return
                
            stats = np.load(stats_path, allow_pickle=True).item()
            mean, std = stats['mean'], stats['std']
            print(f"Loaded normalization stats: mean={mean:.4f}, std={std:.4f}")
            
            # Process bin file
            model_input = process_bin_file(file_path, mean, std)
            
        elif ext.lower() == '.npy':
            data = np.load(file_path)
            print(f"Loaded .npy file. Original shape: {data.shape}")
            model_input = data
            
            # Reshape logic matches previous version
            if model_input.ndim == 2:
                model_input = np.expand_dims(model_input, axis=0)
            if model_input.ndim == 3:
                model_input = np.expand_dims(model_input, axis=-1)
        else:
            print(f"Unsupported file extension: {ext}")
            return
            
        print(f"Input shape for model: {model_input.shape}")
        print(f"Input stats - Min: {np.min(model_input):.4f}, Max: {np.max(model_input):.4f}, Mean: {np.mean(model_input):.4f}")

    except Exception as e:
        print(f"Error loading or processing file: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Load the Model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at: {model_path}")
        return

    print(f"Loading model from: {model_path} ...")
    # Enable unsafe deserialization if possible
    try:
        import keras
        if hasattr(keras.config, 'enable_unsafe_deserialization'):
            keras.config.enable_unsafe_deserialization()
    except:
        pass

    try:
        class FreqAvgPoolLambda(tf.keras.layers.Layer):
            """Wrapper for freq_avg_pool that properly computes output shape."""
            def __init__(self, **kwargs):
                invalid_keys = ['function', 'arguments', 'dtype', 'trainable', 'output_shape', 'output_shape_type', 'mask']
                for key in invalid_keys:
                    kwargs.pop(key, None)
                name = kwargs.pop('name', 'freq_avg_pool')
                super().__init__(name=name, **kwargs)
            
            def call(self, inputs, mask=None):
                return tf.reduce_mean(inputs, axis=2)
            
            def compute_output_shape(self, input_shape):
                return tf.TensorShape([input_shape[0], input_shape[1], input_shape[3]])
            
            def get_config(self):
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
        with zipfile.ZipFile(model_path, 'r') as z:
            config_json = z.read('config.json').decode('utf-8')
            model_config = json.loads(config_json)
        
        # Find and replace the Lambda layer with our custom layer
        def replace_lambda_in_config(config):
            """Recursively find and replace freq_avg_pool Lambda layers."""
            if isinstance(config, dict):
                if config.get('class_name') == 'Lambda':
                    layer_config = config.get('config', {})
                    if layer_config.get('name') == 'freq_avg_pool':
                        inbound_nodes = config.get('inbound_nodes', [])
                        cleaned_nodes = []
                        for node in inbound_nodes:
                            cleaned_node = dict(node)
                            if 'kwargs' in cleaned_node:
                                cleaned_kwargs = {k: v for k, v in cleaned_node['kwargs'].items() if k != 'mask'}
                                cleaned_node['kwargs'] = cleaned_kwargs
                            cleaned_nodes.append(cleaned_node)
                        
                        return {
                            'class_name': 'FreqAvgPoolLambda',
                            'config': {'name': 'freq_avg_pool'},
                            'module': None,
                            'registered_name': 'FreqAvgPoolLambda',
                            'name': config.get('name', 'freq_avg_pool'),
                            'inbound_nodes': cleaned_nodes,
                            'build_config': config.get('build_config', {})
                        }
                return {k: replace_lambda_in_config(v) for k, v in config.items()}
            elif isinstance(config, list):
                return [replace_lambda_in_config(item) for item in config]
            return config
        
        modified_config = replace_lambda_in_config(model_config)
        
        if 'compile_config' in modified_config:
            del modified_config['compile_config']
        
        # Rebuild the model from modified config
        model = tf.keras.models.model_from_json(
            json.dumps(modified_config),
            custom_objects=custom_objects
        )
        
        # Load weights
        with zipfile.ZipFile(model_path, 'r') as z:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                z.extract('model.weights.h5', tmpdir)
                weights_path = os.path.join(tmpdir, 'model.weights.h5')
                model.load_weights(weights_path)
                    
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: If this is a Lambda layer error, the model might require the complex loading logic from gui-app/model_inference.py")
        return

    # 5. Predict
    print("\nRunning prediction...")
    try:
        predictions = model.predict(model_input, verbose=0)
        
        # 6. Display Results
        # Assuming the model outputs a dictionary or list for 'presence' and 'trend'
        # Adjust logic below based on your specific model output structure
        
        print("\n--- Prediction Results ---")
        
        if isinstance(predictions, dict):
            presence = float(predictions['presence'][0][0])
            trend = predictions['trend'][0]
            
            print(f"Presence Probability: {presence:.6f}")
            print(f"Vibration Detected:   {'YES' if presence > 0.5 else 'NO'}")
            print("-" * 30)
            print("Trend Probabilities:")
            print(f"  Constant:   {float(trend[0]):.6f}")
            print(f"  Increasing: {float(trend[1]):.6f}")
            print(f"  Decreasing: {float(trend[2]):.6f}")
            
            classes = ['Constant', 'Increasing', 'Decreasing']
            print(f"Dominant Trend: {classes[np.argmax(trend)]}")
            
        elif isinstance(predictions, list):
            # Fallback if output is a list [presence, trend] or [trend, presence]
            print("Output is a list:", predictions)
        else:
            print("Output tensor:", predictions)
            
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
