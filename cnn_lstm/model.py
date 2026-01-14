"""
CNN-LSTM Multi-Output Model for Radar Spectrogram Classification

Two output heads:
- Presence: Binary classification (vibration vs no vibration)
- Trend: 3-class classification (constant, increasing, decreasing)
  Only applies when presence=1
"""

import tensorflow as tf
from tensorflow.keras import regularizers


def create_cnn_lstm_model(input_shape=(163, 97, 1), dropout_rate=0.4, l2_reg=1e-4):
    """
    Create a CNN-LSTM model with two output heads.
    
    Architecture:
    - CNN blocks with frequency-only downsampling (preserves temporal resolution)
    - Feature dimension reduction before LSTM
    - Bidirectional LSTM for temporal modeling
    - Two output heads: presence (binary) and trend (3-class)
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input spectrogram (freq, time, channels)
    dropout_rate : float
        Dropout rate for regularization
    l2_reg : float
        L2 regularization strength
    
    Returns:
    --------
    model : tf.keras.Model
        Multi-output CNN-LSTM model
    """
    inputs = tf.keras.Input(shape=input_shape, name="spectrogram")

    # ------------------------------------------------------------------
    # CNN feature extractor (frequency downsampling ONLY)
    # Preserves all 97 time steps for LSTM temporal modeling
    # ------------------------------------------------------------------

    # Conv Block 1 – wide frequency view
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(9, 3),
        strides=(2, 1),  # Downsample freq only
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.SpatialDropout2D(0.2, name="sd1")(x)

    # Conv Block 2 – mid-level patterns
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(7, 3),
        strides=(2, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.SpatialDropout2D(0.25, name="sd2")(x)

    # Conv Block 3 – higher abstraction
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 3),
        strides=(2, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv3",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.SpatialDropout2D(0.3, name="sd3")(x)

    # ------------------------------------------------------------------
    # Reduce feature dimension before LSTM
    # Current shape: (batch, 21, 97, 128) → reduce channels
    # ------------------------------------------------------------------
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="channel_reduce",
    )(x)
    
    # Global pooling over frequency dimension to reduce LSTM input
    # Shape: (batch, 21, 97, 64) → (batch, 97, 64) via averaging over freq
    x = tf.keras.layers.Permute((2, 1, 3), name="permute_time_first")(x)
    # Now: (batch, 97, 21, 64)
    
    # Average over frequency dimension
    x = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=2),
        name="freq_avg_pool"
    )(x)
    # Now: (batch, 97, 64) - much smaller input for LSTM!

    # ------------------------------------------------------------------
    # Temporal modeling with Bidirectional LSTM
    # ------------------------------------------------------------------
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2),
        name="bilstm1",
    )(x)
    
    # Second LSTM layer for deeper temporal understanding
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.2),
        name="bilstm2",
    )(x)

    x = tf.keras.layers.Dropout(dropout_rate, name="lstm_dropout")(x)

    # Shared representation
    shared = tf.keras.layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="shared_dense"
    )(x)
    shared = tf.keras.layers.Dropout(0.3, name="shared_dropout")(shared)

    # ------------------------------------------------------------------
    # Output heads
    # ------------------------------------------------------------------
    
    # Presence head (binary: vibration vs no vibration)
    presence_output = tf.keras.layers.Dense(
        1, activation="sigmoid", name="presence"
    )(shared)

    # Trend head (3-class: constant, increasing, decreasing)
    # Note: Only trained on samples where presence=1
    trend_output = tf.keras.layers.Dense(
        3, activation="softmax", name="trend"
    )(shared)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "presence": presence_output,
            "trend": trend_output,
        },
        name="cnn_lstm_vibration_v3",
    )

    return model


if __name__ == "__main__":
    # Test model creation
    model = create_cnn_lstm_model(input_shape=(163, 97, 1), dropout_rate=0.4)
    model.summary()
