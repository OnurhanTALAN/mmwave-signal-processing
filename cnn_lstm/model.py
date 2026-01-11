import tensorflow as tf


def create_cnn_lstm_model(input_shape=(163, 97, 1), dropout_rate=0.4):
    inputs = tf.keras.Input(shape=input_shape, name="spectrogram")

    # ------------------------------------------------------------------
    # CNN feature extractor (frequency downsampling ONLY)
    # ------------------------------------------------------------------

    # Conv Block 1 – wide frequency view
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(9, 3),
        strides=(2, 1),   
        padding="same",
        activation="relu",
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
        name="conv3",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.SpatialDropout2D(0.3, name="sd3")(x)

    # ------------------------------------------------------------------
    # Prepare sequence for LSTM
    # Shape: (batch, freq, time, channels)
    # → (batch, time, freq * channels)
    # ------------------------------------------------------------------
    x = tf.keras.layers.Permute((2, 1, 3), name="permute_time_first")(x)

    time_steps = x.shape[1]
    freq_dim = x.shape[2]
    channels = x.shape[3]

    x = tf.keras.layers.Reshape(
        (time_steps, freq_dim * channels),
        name="reshape_seq",
    )(x)

    # ------------------------------------------------------------------
    # Temporal modeling
    # ------------------------------------------------------------------
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=False),
        name="bilstm",
    )(x)

    x = tf.keras.layers.Dropout(dropout_rate, name="lstm_dropout")(x)

    # Shared representation
    x = tf.keras.layers.Dense(128, activation="relu", name="shared_dense")(x)
    x = tf.keras.layers.Dropout(0.3, name="shared_dropout")(x)

    # ------------------------------------------------------------------
    # Output heads
    # ------------------------------------------------------------------
    presence_output = tf.keras.layers.Dense(
        1, activation="sigmoid", name="presence"
    )(x)

    trend_output = tf.keras.layers.Dense(
        3, activation="softmax", name="trend"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "presence": presence_output,
            "trend": trend_output,
        },
        name="cnn_lstm_vibration_v2",
    )

    return model
