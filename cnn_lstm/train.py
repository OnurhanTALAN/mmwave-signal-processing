# CNN‑LSTM Multi‑Output Training Script for Radar Spectrograms
"""
Training script for vibration detection and trend classification.
It loads spectrograms from four folders, builds a CNN‑LSTM model, and trains
with two heads:
  * presence – binary (0/1)
  * trend    – 3‑class (constant, increasing, decreasing)
"""

import os
import glob
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# Model import (CNN‑LSTM)
# ---------------------------------------------------------------------------
import os
import sys
# Add repository root to sys.path for module imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_root)
from cnn_lstm.model import create_cnn_lstm_model

# ---------------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------------
class TimeLoggingCallback(tf.keras.callbacks.Callback):
    """Log epoch duration and total training time."""
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.train_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f} seconds")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start_time
        avg_epoch_time = np.mean(self.epoch_times)
        print("=" * 60)
        print("TRAINING TIME SUMMARY")
        print("=" * 60)
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Average epoch time: {avg_epoch_time:.2f}s")
        print(f"Fastest epoch: {min(self.epoch_times):.2f}s")
        print(f"Slowest epoch: {max(self.epoch_times):.2f}s")
        print("=" * 60)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_spectrograms(data_path, target_shape=(163, 97)):
    """Load spectrograms from the four category folders.

    Returns:
        X          – np.ndarray of shape (N, H, W, 1)
        y_presence – np.ndarray of 0/1 (no‑vibration vs vibration)
        y_trend    – np.ndarray of 0/1/2 (constant, inc, dec). For the
                      no‑vibration class we assign 0 (constant) – it will be
                      ignored during loss weighting.
    """
    X, y_presence, y_trend = [], [], []
    # Mapping folder → (presence, trend)
    mapping = {
        "no-vibration": (0, 0),          # no vibration, trend placeholder
        "full-vibration": (1, 0),        # constant amplitude
        "increasing-vibration": (1, 1), # increasing amplitude
        "decreasing-vibration": (1, 2), # decreasing amplitude
    }
    for folder, (pres, tr) in mapping.items():
        pattern = os.path.join(data_path, folder, "*.npy")
        files = glob.glob(pattern)
        print(f"Loading {len(files)} files from '{folder}' ...")
        for f in files:
            spec = np.load(f)
            if spec.shape != target_shape:
                zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
                spec = zoom(spec, zoom_factors, order=1)
            X.append(spec)
            y_presence.append(pres)
            y_trend.append(tr)
    X = np.array(X, dtype=np.float32)
    y_presence = np.array(y_presence, dtype=np.int32)
    y_trend = np.array(y_trend, dtype=np.int32)
    # Add channel dimension
    X = np.expand_dims(X, axis=-1)
    print("\nLoaded spectrograms:")
    print(f"  Total samples : {len(X)}")
    print(f"  Shape per sample: {X.shape[1:]} (H, W, C)")
    print(f"  Presence distribution: 0={np.sum(y_presence==0)}, 1={np.sum(y_presence==1)}")
    print(f"  Trend distribution : 0={np.sum(y_trend==0)}, 1={np.sum(y_trend==1)}, 2={np.sum(y_trend==2)}")
    return X, y_presence, y_trend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "../spectrograms-output/underground-pipe-vibration-2s"
TARGET_SHAPE = (163, 97)  # frequency bins × time steps
INPUT_SHAPE = (163, 97, 1)
DROPOUT_RATE = 0.4

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MODEL_SAVE_PATH = "models/best_cnn_lstm.keras"
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("CNN‑LSTM MULTI‑OUTPUT TRAINING – CONFIGURATION")
print("=" * 60)
print(f"Data path       : {DATA_PATH}")
print(f"Input shape     : {INPUT_SHAPE}")
print(f"Dropout rate    : {DROPOUT_RATE}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Epochs          : {EPOCHS}")
print(f"Learning rate   : {LEARNING_RATE}")
print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading spectrograms...")
load_start = time.time()
X, y_presence, y_trend = load_spectrograms(DATA_PATH, target_shape=TARGET_SHAPE)
print(f"Data loading time: {time.time() - load_start:.2f}s")

# Split data (70/15/15) with stratification on presence
X_train, X_tmp, y_pres_train, y_pres_tmp, y_trend_train, y_trend_tmp = train_test_split(
    X, y_presence, y_trend, test_size=0.30, random_state=42, stratify=y_presence
)
X_val, X_test, y_pres_val, y_pres_test, y_trend_val, y_trend_test = train_test_split(
    X_tmp, y_pres_tmp, y_trend_tmp, test_size=0.50, random_state=42, stratify=y_pres_tmp
)
print(f"Train samples : {len(X_train)}")
print(f"Val   samples : {len(X_val)}")
print(f"Test  samples : {len(X_test)}")

# TensorFlow datasets – multi‑output
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, {"presence": y_pres_train, "trend": y_trend_train})).shuffle(len(X_train)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, {"presence": y_pres_val, "trend": y_trend_val})).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, {"presence": y_pres_test, "trend": y_trend_test})).batch(BATCH_SIZE)

print("\nDatasets prepared.\n")

# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------
print("Creating CNN‑LSTM model...")
model = create_cnn_lstm_model(input_shape=INPUT_SHAPE, dropout_rate=DROPOUT_RATE)
model.summary()

# ---------------------------------------------------------------------------
# Compile – multi‑output
# ---------------------------------------------------------------------------
optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
losses = {
    "presence": tf.keras.losses.BinaryCrossentropy(),
    "trend": tf.keras.losses.SparseCategoricalCrossentropy(),
}
loss_weights = {"presence": 1.0, "trend": 0.5}
metrics = {
    "presence": [tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    "trend": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
}
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
print("Model compiled.\n")

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
time_callback = TimeLoggingCallback()
callbacks = [
    time_callback,
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print("Starting training...\n")
train_start = time.time()
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=callbacks, verbose=1)
print(f"Training completed in {time.time() - train_start:.2f}s\n")

# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------
print("Evaluating on test set...")
eval_start = time.time()
test_results = model.evaluate(test_dataset, verbose=1)
print(f"Evaluation time: {time.time() - eval_start:.2f}s\n")
print("Test results (loss + metrics):")
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

# ---------------------------------------------------------------------------
# Presence‑only confusion matrix (binary classification)
# ---------------------------------------------------------------------------
print("Generating confusion matrix for vibration presence...")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Predict presence probabilities on the raw test arrays (not the dataset) for simplicity
presence_proba, _ = model.predict(X_test, verbose=0)
y_pred_presence = (presence_proba > 0.5).astype(int).flatten()
cm = confusion_matrix(y_pres_test, y_pred_presence)
plt.figure(figsize=(6, 5))
sns = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix – Vibration Presence')
plt.tight_layout()
plt.savefig('confusion_matrix_presence.png', dpi=150)
print("Confusion matrix saved to: confusion_matrix_presence.png")

print("\nClassification report (presence):")
print(classification_report(y_pres_test, y_pred_presence, target_names=['No', 'Yes']))

print("\nAll done.")
