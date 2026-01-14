"""
Binary Classification Training Script for Radar Spectrogram Classification

Trains a CNN model to classify spectrograms as:
- Closed (0): No vibration
- Full-speed (1): Vibration present

Uses the same metrics, optimizer, and loss functions as the MIL model.
"""

import os
import glob
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

from model import create_cnn_model

# ============================================================================
# CUSTOM CALLBACKS
# ============================================================================

class TimeLoggingCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to log time for each epoch.
    """
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
        print(f"\n" + "="*60)
        print(f"TRAINING TIME SUMMARY")
        print("="*60)
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Average epoch time: {avg_epoch_time:.2f}s")
        print(f"Fastest epoch: {min(self.epoch_times):.2f}s")
        print(f"Slowest epoch: {max(self.epoch_times):.2f}s")
        print("="*60)


class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Learning rate warmup scheduler.
    Gradually increases learning rate from 0 to target LR over warmup_epochs.
    """
    def __init__(self, warmup_epochs, target_lr):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            # Keras 3 compatible way to set learning rate
            self.model.optimizer.learning_rate.assign(lr)
            print(f"\nWarmup LR: {lr:.6f}")


class F1Score(tf.keras.metrics.Metric):
    """
    Custom F1 Score metric for binary classification.
    """
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


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def create_augmentation_layer():
    """
    Create a SpecAugment-style data augmentation layer.
    Applies time and frequency masking to spectrograms.
    """
    return tf.keras.Sequential([
        # Random horizontal flip (time reversal)
        layers.RandomFlip("horizontal"),
        # Small random zoom for scale invariance
        layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
    ], name='augmentation')


# Import layers for augmentation
from tensorflow.keras import layers


def spec_augment(spectrogram, freq_mask_param=10, time_mask_param=10, num_masks=2):
    """
    Apply SpecAugment-style augmentation to spectrograms.
    
    Parameters:
    -----------
    spectrogram : tf.Tensor
        Input spectrogram (batch, height, width, channels)
    freq_mask_param : int
        Maximum frequency mask width
    time_mask_param : int
        Maximum time mask width
    num_masks : int
        Number of masks to apply
        
    Returns:
    --------
    tf.Tensor : Augmented spectrogram
    """
    spec = spectrogram
    height = tf.shape(spec)[1]
    width = tf.shape(spec)[2]
    
    # Frequency masking
    for _ in range(num_masks):
        f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, height - f, dtype=tf.int32)
        indices = tf.reshape(tf.range(f0, f0 + f), (1, -1, 1, 1))
        mask = tf.scatter_nd(
            tf.transpose(tf.concat([
                tf.zeros((1, f, 1, 1), dtype=tf.int32),
                tf.cast(indices, tf.int32),
                tf.zeros((1, f, 1, 1), dtype=tf.int32),
                tf.zeros((1, f, 1, 1), dtype=tf.int32)
            ], axis=0), [1, 2, 3, 0]),
            tf.ones([f]),
            tf.shape(spec)
        )
        spec = spec * (1 - mask)
    
    return spec


def augment_fn(image, label):
    """
    Augmentation function for tf.data.Dataset.
    Applies random augmentations with 50% probability.
    """
    # Apply augmentation with 50% probability
    if tf.random.uniform([]) > 0.5:
        # Random horizontal flip
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
    
    return image, label


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_spectrograms_by_class(data_path, target_shape=(163, 97)):
    """
    Load spectrograms separately by class, preserving file order for time-based split.
    
    Parameters:
    -----------
    data_path : str
        Path to the spectrograms-output directory
    target_shape : tuple
        Target shape for resizing spectrograms (height, width)
        
    Returns:
    --------
    X_closed, X_fullspeed : np.ndarray
        Spectrograms for each class, sorted by filename (temporal order)
    """
    
    def load_class_spectrograms(class_path, class_name):
        files = sorted(glob.glob(class_path))  # Sort by filename to preserve temporal order
        print(f"Loading {len(files)} {class_name} spectrograms (sorted by filename)...")
        
        spectrograms = []
        for file in files:
            spec = np.load(file)
            if spec.shape != target_shape:
                zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
                spec = zoom(spec, zoom_factors, order=1)
            spectrograms.append(spec)
        
        return np.array(spectrograms)
    
    # Load each class separately, sorted by filename
    X_closed = load_class_spectrograms(
        os.path.join(data_path, 'closed', '*.npy'), 'closed'
    )
    X_fullspeed = load_class_spectrograms(
        os.path.join(data_path, 'full-speed', '*.npy'), 'full-speed'
    )
    
    # Add channel dimension
    X_closed = np.expand_dims(X_closed, axis=-1)
    X_fullspeed = np.expand_dims(X_fullspeed, axis=-1)
    
    print(f"\nLoaded spectrograms:")
    print(f"  Closed (no vibration): {len(X_closed)} samples, shape {X_closed.shape}")
    print(f"  Full-speed (vibration): {len(X_fullspeed)} samples, shape {X_fullspeed.shape}")
    
    return X_closed, X_fullspeed


def time_based_split(X_closed, X_fullspeed, train_ratio=0.7, val_ratio=0.15):
    """
    Split data chronologically (by recording order) instead of randomly.
    
    This ensures test set contains recordings from a DIFFERENT time period
    than training, validating true generalization vs session-specific learning.
    
    Parameters:
    -----------
    X_closed, X_fullspeed : np.ndarray
        Spectrograms for each class, already sorted by time
    train_ratio : float
        Proportion of data for training (default: 0.7)
    val_ratio : float
        Proportion of data for validation (default: 0.15)
        
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    
    def split_class(X, label):
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = np.full(len(X_train), label)
        y_val = np.full(len(X_val), label)
        y_test = np.full(len(X_test), label)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # Split each class chronologically
    X_train_0, X_val_0, X_test_0, y_train_0, y_val_0, y_test_0 = split_class(X_closed, 0)
    X_train_1, X_val_1, X_test_1, y_train_1, y_val_1, y_test_1 = split_class(X_fullspeed, 1)
    
    # Combine classes
    X_train = np.concatenate([X_train_0, X_train_1])
    X_val = np.concatenate([X_val_0, X_val_1])
    X_test = np.concatenate([X_test_0, X_test_1])
    y_train = np.concatenate([y_train_0, y_train_1])
    y_val = np.concatenate([y_val_0, y_val_1])
    y_test = np.concatenate([y_test_0, y_test_1])
    
    print(f"\nTime-based split (chronological):")
    print(f"  Train: first {train_ratio*100:.0f}% of each class")
    print(f"  Val:   next {val_ratio*100:.0f}% of each class")
    print(f"  Test:  last {test_ratio*100:.0f}% of each class")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} ({np.sum(y_train==0)} closed, {np.sum(y_train==1)} full-speed)")
    print(f"  Val:   {len(X_val)} ({np.sum(y_val==0)} closed, {np.sum(y_val==1)} full-speed)")
    print(f"  Test:  {len(X_test)} ({np.sum(y_test==0)} closed, {np.sum(y_test==1)} full-speed)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def random_split(X_closed, X_fullspeed, test_size=0.3, val_size=0.5, random_state=42):
    """
    Traditional random stratified split.
    """
    # Combine classes
    X = np.concatenate([X_closed, X_fullspeed])
    y = np.concatenate([np.zeros(len(X_closed)), np.ones(len(X_fullspeed))])
    
    # Random stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    print(f"\nRandom stratified split:")
    print(f"  Train: {len(X_train)} ({np.sum(y_train==0):.0f} closed, {np.sum(y_train==1):.0f} full-speed)")
    print(f"  Val:   {len(X_val)} ({np.sum(y_val==0):.0f} closed, {np.sum(y_val==1):.0f} full-speed)")
    print(f"  Test:  {len(X_test)} ({np.sum(y_test==0):.0f} closed, {np.sum(y_test==1):.0f} full-speed)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_spectrograms(X_train, X_val, X_test):
    """
    Normalize spectrograms using training set statistics.
    
    Parameters:
    -----------
    X_train, X_val, X_test : np.ndarray
        Spectrogram arrays
        
    Returns:
    --------
    Normalized arrays and statistics (mean, std)
    """
    # Compute mean and std from training set only
    mean = np.mean(X_train)
    std = np.std(X_train)
    
    print(f"\nNormalization statistics from training set:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std:  {std:.4f}")
    
    # Normalize all sets using training statistics
    X_train_norm = (X_train - mean) / (std + 1e-8)
    X_val_norm = (X_val - mean) / (std + 1e-8)
    X_test_norm = (X_test - mean) / (std + 1e-8)
    
    return X_train_norm, X_val_norm, X_test_norm, mean, std


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data configuration
DATA_PATH = "../spectrograms-output"  # Relative path from binary-classification folder
TARGET_SHAPE = (163, 97)  # Preserve frequency resolution

# Model configuration
INPUT_SHAPE = (163, 97, 1)
DROPOUT_RATE = 0.4  # Slightly reduced since we added SpatialDropout2D
L2_REG = 1e-4

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1

# Split configuration
# Set to True for time-based split (validates cross-session generalization)
# Set to False for random stratified split (traditional approach)
USE_TIME_BASED_SPLIT = True

# Paths
MODEL_SAVE_PATH = "models/best_model_classification.keras"
os.makedirs("models", exist_ok=True)

print("="*60)
print("BINARY CLASSIFICATION - CONFIGURATION")
print("="*60)
print(f"Data path: {DATA_PATH}")
print(f"Input shape: {INPUT_SHAPE}")
print(f"Dropout rate: {DROPOUT_RATE}")
print(f"L2 regularization: {L2_REG}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Warmup epochs: {WARMUP_EPOCHS}")
print(f"Label smoothing: {LABEL_SMOOTHING}")
print(f"Split method: {'TIME-BASED (chronological)' if USE_TIME_BASED_SPLIT else 'RANDOM (stratified)'}")
print("="*60 + "\n")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("Loading spectrograms...")
data_load_start = time.time()
X_closed, X_fullspeed = load_spectrograms_by_class(DATA_PATH, target_shape=TARGET_SHAPE)
data_load_time = time.time() - data_load_start
print(f"Data loading completed in {data_load_time:.2f}s")

print("\nSplitting data...")
# Split: 70% train, 15% validation, 15% test
if USE_TIME_BASED_SPLIT:
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
        X_closed, X_fullspeed, train_ratio=0.7, val_ratio=0.15
    )
else:
    X_train, X_val, X_test, y_train, y_val, y_test = random_split(
        X_closed, X_fullspeed, test_size=0.3, val_size=0.5, random_state=42
    )

# Normalize data
print("\nNormalizing spectrograms...")
X_train, X_val, X_test, train_mean, train_std = normalize_spectrograms(X_train, X_val, X_test)

# Save normalization stats for inference
np.save("models/normalization_stats.npy", {"mean": train_mean, "std": train_std})
print("Normalization statistics saved to models/normalization_stats.npy")

# Create TensorFlow datasets with prefetching
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
train_dataset = train_dataset.map(augment_fn, num_parallel_calls=AUTOTUNE)  # Data augmentation
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(AUTOTUNE)  # Prefetch for GPU efficiency

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE")
print("="*60 + "\n")

# ============================================================================
# MODEL CREATION
# ============================================================================

print("Creating model...")
model = create_cnn_model(
    input_shape=INPUT_SHAPE, 
    dropout_rate=DROPOUT_RATE,
    l2_reg=L2_REG
)

# Print model summary
model.summary()

# ============================================================================
# MODEL COMPILATION
# ============================================================================

print("\nCompiling model...")
# Optimizer with weight decay (same as MIL model)
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Loss function with label smoothing
loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)

# Metrics including F1 score
metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    F1Score(name='f1_score')
]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)
print("Model compiled successfully!")

# ============================================================================
# CALLBACKS
# ============================================================================
time_callback = TimeLoggingCallback()
warmup_callback = WarmUpLearningRateScheduler(WARMUP_EPOCHS, LEARNING_RATE)

callbacks = [
    time_callback,
    warmup_callback,
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

training_start_time = time.time()
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)
training_total_time = time.time() - training_start_time

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Total training time: {training_total_time:.2f}s ({training_total_time/60:.2f} minutes)")
print("="*60 + "\n")

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================

print("Evaluating on test set...")
evaluation_start_time = time.time()
test_results = model.evaluate(test_dataset, verbose=1)
evaluation_time = time.time() - evaluation_start_time

print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)
for metric_name, value in zip(model.metrics_names, test_results):
    print(f"{metric_name}: {value:.4f}")
print("="*60)
print(f"Evaluation time: {evaluation_time:.2f}s")
print("="*60)

print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

print("\nGenerating confusion matrix...")
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Get predictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Vibration (0)', 'Vibration (1)'],
            yticklabels=['No Vibration (0)', 'Vibration (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Binary Classification')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Confusion matrix saved to: confusion_matrix.png")

# Print classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['No Vibration', 'Vibration']))
print("="*60)

# Print confusion matrix values
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Values:")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")

# ============================================================================
# TRAINING HISTORY PLOTS
# ============================================================================

print("\nGenerating training history plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Train')
axes[0, 0].plot(history.history['val_loss'], label='Validation')
axes[0, 0].set_title('Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()

# Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train')
axes[0, 1].plot(history.history['val_accuracy'], label='Validation')
axes[0, 1].set_title('Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()

# AUC
axes[0, 2].plot(history.history['auc'], label='Train')
axes[0, 2].plot(history.history['val_auc'], label='Validation')
axes[0, 2].set_title('AUC')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].legend()

# Precision
axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Validation')
axes[1, 0].set_title('Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()

# Recall
axes[1, 1].plot(history.history['recall'], label='Train')
axes[1, 1].plot(history.history['val_recall'], label='Validation')
axes[1, 1].set_title('Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()

# F1 Score
axes[1, 2].plot(history.history['f1_score'], label='Train')
axes[1, 2].plot(history.history['val_f1_score'], label='Validation')
axes[1, 2].set_title('F1 Score')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("Training history saved to: training_history.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_execution_time = time.time() - data_load_start

print("\n" + "="*60)
print("COMPLETE EXECUTION SUMMARY")
print("="*60)
print(f"Data loading time: {data_load_time:.2f}s")
print(f"Model training time: {training_total_time:.2f}s ({training_total_time/60:.2f} min)")
print(f"Test evaluation time: {evaluation_time:.2f}s")
print(f"Total execution time: {total_execution_time:.2f}s ({total_execution_time/60:.2f} min)")
print("="*60 + "\n")
