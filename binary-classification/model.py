"""
Binary Classification Model for Radar Spectrogram Classification

Simple CNN model for classifying spectrograms as:
- Closed (0): No vibration
- Full-speed (1): Vibration present

This is a standard binary classification without MIL.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def create_cnn_model(input_shape=(163, 97, 1), dropout_rate=0.5, l2_reg=1e-4):
    """
    Create a CNN model for binary spectrogram classification.
    
    Architecture:
    - 4 Conv2D blocks with BatchNorm, Dropout, and MaxPooling
    - Global Average Pooling
    - Dense layers with Dropout
    - Sigmoid output for binary classification
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input spectrogram (height, width, channels)
    dropout_rate : float
        Dropout rate for regularization
    l2_reg : float
        L2 regularization strength
        
    Returns:
    --------
    model : tf.keras.Model
        CNN model
    """
    
    inputs = layers.Input(shape=input_shape, name='spectrogram_input')
    
    # Block 1: Large frequency kernels for broad pattern capture
    x = layers.Conv2D(32, (11, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg), name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.5, name='spatial_drop1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # Block 2
    x = layers.Conv2D(64, (7, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg), name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.5, name='spatial_drop2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # Block 3
    x = layers.Conv2D(128, (5, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg), name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.5, name='spatial_drop3')(x)
    x = layers.MaxPooling2D((1, 2), name='pool3')(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg), name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.5, name='spatial_drop4')(x)
    x = layers.MaxPooling2D((1, 2), name='pool4')(x)
    
    # Global Average Pooling to reduce spatial dimensions
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Smaller dense layers with dropout
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg), name='fc1')(x)
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg), name='fc2')(x)
    x = layers.Dropout(dropout_rate, name='dropout2')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='binary_cnn_classifier')
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_cnn_model(input_shape=(163, 97, 1), dropout_rate=0.5)
    model.summary()
