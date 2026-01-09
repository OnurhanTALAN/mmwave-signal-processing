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


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_spectrograms(data_path, target_shape=(1024, 64)):
    """
    Load all spectrograms from closed and full-speed directories.
    
    Parameters:
    -----------
    data_path : str
        Path to the spectrograms-output directory
    target_shape : tuple
        Target shape for resizing spectrograms (height, width)
        
    Returns:
    --------
    X : np.ndarray
        Array of spectrograms with shape (num_samples, height, width, 1)
    y : np.ndarray
        Array of labels (0 for closed, 1 for full-speed)
    """
    X = []
    y = []
    
    # Load closed spectrograms (label 0 - no vibration)
    closed_path = os.path.join(data_path, 'closed', '*.npy')
    closed_files = glob.glob(closed_path)
    print(f"Loading {len(closed_files)} closed spectrograms...")
    
    for file in closed_files:
        spec = np.load(file)
        if spec.shape != target_shape:
            zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
            spec = zoom(spec, zoom_factors, order=1)
        X.append(spec)
        y.append(0) # closed
    
    # Load full-speed spectrograms (label 1 - vibration present)
    fullspeed_path = os.path.join(data_path, 'full-speed', '*.npy')
    fullspeed_files = glob.glob(fullspeed_path)
    print(f"Loading {len(fullspeed_files)} full-speed spectrograms...")
    
    for file in fullspeed_files:
        spec = np.load(file)
        if spec.shape != target_shape:
            zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
            spec = zoom(spec, zoom_factors, order=1)
        X.append(spec)
        y.append(1) # full-speed
    
    X = np.array(X)
    y = np.array(y)
    
    # Add channel dimension
    X = np.expand_dims(X, axis=-1)
    
    print(f"\nLoaded {len(X)} spectrograms")
    print(f"Shape: {X.shape}")
    print(f"Closed samples (no vibration): {np.sum(y == 0)}")
    print(f"Full-speed samples (vibration): {np.sum(y == 1)}")
    
    return X, y


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data configuration
DATA_PATH = "../spectrograms-output"  # Relative path from binary-classification folder
TARGET_SHAPE = (1024, 64)  # Preserve frequency resolution

# Model configuration
INPUT_SHAPE = (1024, 64, 1)
DROPOUT_RATE = 0.3

# Training configuration
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Paths
MODEL_SAVE_PATH = "models/best_model.keras"
os.makedirs("models", exist_ok=True)

print("="*60)
print("BINARY CLASSIFICATION - CONFIGURATION")
print("="*60)
print(f"Data path: {DATA_PATH}")
print(f"Input shape: {INPUT_SHAPE}")
print(f"Dropout rate: {DROPOUT_RATE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*60 + "\n")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("Loading spectrograms...")
data_load_start = time.time()
X, y = load_spectrograms(DATA_PATH, target_shape=TARGET_SHAPE)
data_load_time = time.time() - data_load_start
print(f"Data loading completed in {data_load_time:.2f}s")

print("\nSplitting data...")
# Split: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} samples")
print(f"Val: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE")
print("="*60 + "\n")


# ============================================================================
# MODEL CREATION
# ============================================================================

print("Creating model...")
model = create_cnn_model(
    input_shape=INPUT_SHAPE, 
    dropout_rate=DROPOUT_RATE
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
# Loss function
loss = tf.keras.losses.BinaryCrossentropy()
# Metrics
metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
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
callbacks = [
    time_callback,
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
            xticklabels=['Closed (0)', 'Full-speed (1)'],
            yticklabels=['Closed (0)', 'Full-speed (1)'])
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
print(classification_report(y_test, y_pred, target_names=['Closed', 'Full-speed']))
print("="*60)

# Print confusion matrix values
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Values:")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")


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
