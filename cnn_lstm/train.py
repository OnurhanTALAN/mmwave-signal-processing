# CNN-LSTM Multi-Output Training Script for Radar Spectrograms
"""
Training script for vibration detection and trend classification.
It loads spectrograms from four folders, builds a CNN-LSTM model, and trains
with two heads:
  * presence binary (0/1)
  * trend    3-class (constant, increasing, decreasing) - masked for no-vibration samples
"""

import os
import glob
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# Model import (CNN-LSTM)
# ---------------------------------------------------------------------------
import sys
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


class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate warmup scheduler."""
    def __init__(self, warmup_epochs, target_lr):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            self.model.optimizer.learning_rate.assign(lr)
            print(f"\nWarmup LR: {lr:.6f}")

# ---------------------------------------------------------------------------
# Custom F1 metrics
# ---------------------------------------------------------------------------
class F1Score(tf.keras.metrics.Metric):
    """Custom F1 Score metric for binary classification."""
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
def augment_fn(inputs, targets):
    """
    Augmentation function for tf.data.Dataset.
    Applies SpecAugment-style augmentations with 50% probability.
    """
    image = inputs
    
    # Apply augmentation with 50% probability
    if tf.random.uniform([]) > 0.5:
        # Random horizontal flip (time reversal)
        image = tf.image.random_flip_left_right(image)
    
    # Time masking: zero out random time slices
    if tf.random.uniform([]) > 0.5:
        width = tf.shape(image)[1]
        mask_width = tf.random.uniform([], 1, 10, dtype=tf.int32)
        mask_start = tf.random.uniform([], 0, width - mask_width, dtype=tf.int32)
        
        mask = tf.concat([
            tf.ones([tf.shape(image)[0], mask_start, tf.shape(image)[2]]),
            tf.zeros([tf.shape(image)[0], mask_width, tf.shape(image)[2]]),
            tf.ones([tf.shape(image)[0], width - mask_start - mask_width, tf.shape(image)[2]])
        ], axis=1)
        image = image * mask
    
    # Frequency masking: zero out random frequency bands
    if tf.random.uniform([]) > 0.5:
        height = tf.shape(image)[0]
        mask_height = tf.random.uniform([], 1, 15, dtype=tf.int32)
        mask_start = tf.random.uniform([], 0, height - mask_height, dtype=tf.int32)
        
        mask = tf.concat([
            tf.ones([mask_start, tf.shape(image)[1], tf.shape(image)[2]]),
            tf.zeros([mask_height, tf.shape(image)[1], tf.shape(image)[2]]),
            tf.ones([height - mask_start - mask_height, tf.shape(image)[1], tf.shape(image)[2]])
        ], axis=0)
        image = image * mask
    
    return image, targets

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_spectrograms_by_class(data_path, target_shape=(163, 97)):
    """
    Load spectrograms from the four category folders, sorted for time-based split.

    Returns:
        Dictionary with arrays for each class
    """
    data = {}
    # Mapping folder → (presence, trend)
    # trend=0 for no-vibration is a placeholder, will be masked during training
    mapping = {
        "no-vibration":          (0, 0),    # no vibration
        "full-vibration":        (1, 0),    # constant amplitude
        "increasing-vibration":  (1, 1),    # increasing amplitude
        "decreasing-vibration":  (1, 2),    # decreasing amplitude
    }
    
    for folder, (pres, tr) in mapping.items():
        pattern = os.path.join(data_path, folder, "*.npy")
        files = sorted(glob.glob(pattern))  # Sort for time-based split
        print(f"Loading {len(files)} files from '{folder}' (sorted)...")
        
        spectrograms = []
        for f in files:
            spec = np.load(f)
            if spec.shape != target_shape:
                zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
                spec = zoom(spec, zoom_factors, order=1)
            spectrograms.append(spec)
        
        if len(spectrograms) > 0:
            X = np.array(spectrograms, dtype=np.float32)
            X = np.expand_dims(X, axis=-1)  # Add channel dimension
            data[folder] = {
                'X': X,
                'presence': pres,
                'trend': tr,
            }
            print(f"  → Loaded {len(X)} samples, shape {X.shape}")
    
    return data


def time_based_split_multiclass(data, train_ratio=0.7, val_ratio=0.15):
    """
    Split data chronologically for each class.
    
    Returns arrays and sample weights for trend masking.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    
    X_train, X_val, X_test = [], [], []
    y_pres_train, y_pres_val, y_pres_test = [], [], []
    y_trend_train, y_trend_val, y_trend_test = [], [], []
    trend_weight_train, trend_weight_val, trend_weight_test = [], [], []
    
    for folder, class_data in data.items():
        X = class_data['X']
        pres = class_data['presence']
        trend = class_data['trend']
        
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split
        X_train.append(X[:train_end])
        X_val.append(X[train_end:val_end])
        X_test.append(X[val_end:])
        
        # Labels
        y_pres_train.extend([pres] * train_end)
        y_pres_val.extend([pres] * (val_end - train_end))
        y_pres_test.extend([pres] * (n - val_end))
        
        y_trend_train.extend([trend] * train_end)
        y_trend_val.extend([trend] * (val_end - train_end))
        y_trend_test.extend([trend] * (n - val_end))
        
        # Trend weights: 0 for no-vibration (presence=0), 1 otherwise
        weight = 1.0 if pres == 1 else 0.0
        trend_weight_train.extend([weight] * train_end)
        trend_weight_val.extend([weight] * (val_end - train_end))
        trend_weight_test.extend([weight] * (n - val_end))
    
    # Concatenate
    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    X_test = np.concatenate(X_test)
    
    y_pres_train = np.array(y_pres_train, dtype=np.int32)
    y_pres_val = np.array(y_pres_val, dtype=np.int32)
    y_pres_test = np.array(y_pres_test, dtype=np.int32)
    
    y_trend_train = np.array(y_trend_train, dtype=np.int32)
    y_trend_val = np.array(y_trend_val, dtype=np.int32)
    y_trend_test = np.array(y_trend_test, dtype=np.int32)
    
    trend_weight_train = np.array(trend_weight_train, dtype=np.float32)
    trend_weight_val = np.array(trend_weight_val, dtype=np.float32)
    trend_weight_test = np.array(trend_weight_test, dtype=np.float32)
    
    print(f"\nTime-based split (chronological):")
    print(f"  Train: first {train_ratio*100:.0f}% of each class")
    print(f"  Val:   next {val_ratio*100:.0f}% of each class")
    print(f"  Test:  last {test_ratio*100:.0f}% of each class")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"\nTrend weight distribution (1=vibration, 0=masked):")
    print(f"  Train: {np.sum(trend_weight_train > 0):.0f} samples with trend labels")
    print(f"  Val:   {np.sum(trend_weight_val > 0):.0f} samples with trend labels")
    print(f"  Test:  {np.sum(trend_weight_test > 0):.0f} samples with trend labels")
    
    return (X_train, X_val, X_test, 
            y_pres_train, y_pres_val, y_pres_test,
            y_trend_train, y_trend_val, y_trend_test,
            trend_weight_train, trend_weight_val, trend_weight_test)


def normalize_spectrograms(X_train, X_val, X_test):
    """Normalize spectrograms using training set statistics."""
    mean = np.mean(X_train)
    std = np.std(X_train)
    
    print(f"\nNormalization statistics from training set:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std:  {std:.4f}")
    
    X_train_norm = (X_train - mean) / (std + 1e-8)
    X_val_norm = (X_val - mean) / (std + 1e-8)
    X_test_norm = (X_test - mean) / (std + 1e-8)
    
    return X_train_norm, X_val_norm, X_test_norm, mean, std

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "../spectrograms-output/underground-pipe-vibration-2s"
TARGET_SHAPE = (163, 97)  # frequency bins × time steps
INPUT_SHAPE = (163, 97, 1)
DROPOUT_RATE = 0.4
L2_REG = 1e-4

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1

# Split configuration
USE_TIME_BASED_SPLIT = True

MODEL_SAVE_PATH = "models/best_cnn_lstm.keras"
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("CNN-LSTM MULTI-OUTPUT TRAINING – CONFIGURATION")
print("=" * 60)
print(f"Data path       : {DATA_PATH}")
print(f"Input shape     : {INPUT_SHAPE}")
print(f"Dropout rate    : {DROPOUT_RATE}")
print(f"L2 regularization: {L2_REG}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Epochs          : {EPOCHS}")
print(f"Learning rate   : {LEARNING_RATE}")
print(f"Warmup epochs   : {WARMUP_EPOCHS}")
print(f"Label smoothing : {LABEL_SMOOTHING}")
print(f"Split method    : {'TIME-BASED' if USE_TIME_BASED_SPLIT else 'RANDOM'}")
print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Load and split data
# ---------------------------------------------------------------------------
print("Loading spectrograms...")
load_start = time.time()
data = load_spectrograms_by_class(DATA_PATH, target_shape=TARGET_SHAPE)
print(f"Data loading time: {time.time() - load_start:.2f}s")

if len(data) == 0:
    print("\nNo data found! Please check DATA_PATH and folder structure.")
    print("Expected folders: no-vibration, full-vibration, increasing-vibration, decreasing-vibration")
    exit(1)

# Time-based split with trend weights
(X_train, X_val, X_test, 
 y_pres_train, y_pres_val, y_pres_test,
 y_trend_train, y_trend_val, y_trend_test,
 trend_weight_train, trend_weight_val, trend_weight_test) = time_based_split_multiclass(data)

# Normalize
print("\nNormalizing spectrograms...")
X_train, X_val, X_test, train_mean, train_std = normalize_spectrograms(X_train, X_val, X_test)

# Save normalization stats
np.save("models/normalization_stats.npy", {"mean": train_mean, "std": train_std})
print("Normalization statistics saved to models/normalization_stats.npy")

# ---------------------------------------------------------------------------
# Create TensorFlow datasets with sample weights for trend masking
# ---------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

# Training dataset with augmentation
train_dataset = tf.data.Dataset.from_tensor_slices((
    X_train,
    {"presence": y_pres_train, "trend": y_trend_train},
    {"presence": np.ones_like(y_pres_train, dtype=np.float32), "trend": trend_weight_train}
))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
train_dataset = train_dataset.map(
    lambda x, y, w: (augment_fn(x, y)[0], y, w),
    num_parallel_calls=AUTOTUNE
)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(AUTOTUNE)

# Validation dataset (no augmentation)
val_dataset = tf.data.Dataset.from_tensor_slices((
    X_val,
    {"presence": y_pres_val, "trend": y_trend_val},
    {"presence": np.ones_like(y_pres_val, dtype=np.float32), "trend": trend_weight_val}
))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((
    X_test,
    {"presence": y_pres_test, "trend": y_trend_test},
    {"presence": np.ones_like(y_pres_test, dtype=np.float32), "trend": trend_weight_test}
))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("\nDatasets prepared with trend masking for no-vibration samples.\n")

# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------
print("Creating CNN-LSTM model...")
model = create_cnn_lstm_model(input_shape=INPUT_SHAPE, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG)
model.summary()

# ---------------------------------------------------------------------------
# Compile with multi-output losses and sample weights
# ---------------------------------------------------------------------------
print("\nCompiling model...")
optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

losses = {
    "presence": tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    "trend": tf.keras.losses.SparseCategoricalCrossentropy(),
}
loss_weights = {"presence": 1.0, "trend": 1.0}

metrics = {
    "presence": [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        F1Score(name="f1_score"),
    ],
    "trend": [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    ],
}

model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
print("Model compiled.\n")

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
time_callback = TimeLoggingCallback()
warmup_callback = WarmUpLearningRateScheduler(WARMUP_EPOCHS, LEARNING_RATE)

callbacks = [
    time_callback,
    warmup_callback,
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60 + "\n")

train_start = time.time()
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)
print(f"\nTraining completed in {time.time() - train_start:.2f}s\n")

# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------
print("Evaluating on test set...")
eval_start = time.time()
test_results = model.evaluate(test_dataset, verbose=1)
print(f"Evaluation time: {time.time() - eval_start:.2f}s\n")

print("=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")
print("=" * 60)
print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------
print("\nGenerating confusion matrices...")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Predict on test set
predictions = model.predict(X_test, verbose=0)
presence_proba = predictions["presence"]
trend_proba = predictions["trend"]

y_pred_presence = (presence_proba > 0.5).astype(int).flatten()
y_pred_trend = np.argmax(trend_proba, axis=1)

# --- Presence confusion matrix ---
cm_presence = confusion_matrix(y_pres_test, y_pred_presence)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_presence, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Vibration', 'Vibration'],
            yticklabels=['No Vibration', 'Vibration'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Vibration Presence')
plt.tight_layout()
plt.savefig('confusion_matrix_presence.png', dpi=150)
print("Confusion matrix (presence) saved to: confusion_matrix_presence.png")

print("\nClassification report (presence):")
print(classification_report(y_pres_test, y_pred_presence, 
                          target_names=['No Vibration', 'Vibration']))

# --- Trend confusion matrix (only for vibration samples) ---
# Filter to only samples where presence=1
vibration_mask = y_pres_test == 1
y_trend_test_filtered = y_trend_test[vibration_mask]
y_pred_trend_filtered = y_pred_trend[vibration_mask]

if len(y_trend_test_filtered) > 0:
    cm_trend = confusion_matrix(y_trend_test_filtered, y_pred_trend_filtered)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_trend, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Constant', 'Increasing', 'Decreasing'],
                yticklabels=['Constant', 'Increasing', 'Decreasing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Vibration Trend\n(Only vibration samples)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_trend.png', dpi=150)
    print("\nConfusion matrix (trend) saved to: confusion_matrix_trend.png")
    
    print("\nClassification report (trend - vibration samples only):")
    print(classification_report(y_trend_test_filtered, y_pred_trend_filtered,
                              target_names=['Constant', 'Increasing', 'Decreasing']))

# ---------------------------------------------------------------------------
# Training history plots
# ---------------------------------------------------------------------------
print("\nGenerating training history plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Presence metrics
axes[0, 0].plot(history.history['presence_loss'], label='Train')
axes[0, 0].plot(history.history['val_presence_loss'], label='Val')
axes[0, 0].set_title('Presence Loss')
axes[0, 0].legend()

axes[0, 1].plot(history.history['presence_accuracy'], label='Train')
axes[0, 1].plot(history.history['val_presence_accuracy'], label='Val')
axes[0, 1].set_title('Presence Accuracy')
axes[0, 1].legend()

axes[0, 2].plot(history.history['presence_auc'], label='Train')
axes[0, 2].plot(history.history['val_presence_auc'], label='Val')
axes[0, 2].set_title('Presence AUC')
axes[0, 2].legend()

# Trend metrics
axes[1, 0].plot(history.history['trend_loss'], label='Train')
axes[1, 0].plot(history.history['val_trend_loss'], label='Val')
axes[1, 0].set_title('Trend Loss')
axes[1, 0].legend()

axes[1, 1].plot(history.history['trend_accuracy'], label='Train')
axes[1, 1].plot(history.history['val_trend_accuracy'], label='Val')
axes[1, 1].set_title('Trend Accuracy')
axes[1, 1].legend()

# Overall loss
axes[1, 2].plot(history.history['loss'], label='Train')
axes[1, 2].plot(history.history['val_loss'], label='Val')
axes[1, 2].set_title('Total Loss')
axes[1, 2].legend()

for ax in axes.flat:
    ax.set_xlabel('Epoch')

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("Training history saved to: training_history.png")

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
