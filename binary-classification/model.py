"""
Binary Classification Model for Radar Spectrogram Classification

Simple CNN model for classifying spectrograms as:
- Closed (0): No vibration
- Full-speed (1): Vibration present

This is a standard binary classification without MIL.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def create_cnn_model(input_shape=(1024, 64, 1), dropout_rate=0.5):
    """
    Create a CNN model for binary spectrogram classification.
    
    Architecture:
    - 4 Conv2D blocks with BatchNorm and MaxPooling
    - Global Average Pooling
    - Dense layers with Dropout
    - Sigmoid output for binary classification
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input spectrogram (height, width, channels)
    dropout_rate : float
        Dropout rate for regularization
        
    Returns:
    --------
    model : tf.keras.Model
        Compiled CNN model
    """
    
    inputs = layers.Input(shape=input_shape, name='spectrogram_input')
    
    x = layers.Conv2D(32, (7, 3), padding='same', activation='relu', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    x = layers.Conv2D(64, (5, 3), padding='same', activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D((1, 2), name='pool3')(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.MaxPooling2D((1, 2), name='pool4')(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    x = layers.Dense(64, activation='relu', name='fc2')(x)
    x = layers.Dropout(dropout_rate, name='dropout2')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='binary_cnn_classifier')
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_cnn_model(input_shape=(1024, 64, 1), dropout_rate=0.5)
    model.summary()
