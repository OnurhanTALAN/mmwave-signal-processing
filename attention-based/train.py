import os
import glob
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

from model import create_model

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
        print(f"\n⏱️  Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
    
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
# DATA LOADING FUNCTIONS (Frequency Band MIL)
# ============================================================================

def load_spectrograms_as_bags(data_path, num_instances=10, target_time_width=64):
    """
    Load spectrograms and convert each one to a bag of frequency band instances.
    
    Each spectrogram is split into frequency bands:
    - 1 spectrogram = 1 bag
    - Frequency bands = instances within the bag
    
    This is natural MIL:
    - Full-speed bag: Some frequency bands show vibration (positive instances)
    - Closed bag: All frequency bands are quiet (all negative instances)
    
    Parameters:
    -----------
    data_path : str
        Path to the spectrograms-output directory
    num_instances : int
        Number of frequency bands per spectrogram (instances per bag)
    target_time_width : int
        Target width (time dimension) for each instance
        
    Returns:
    --------
    X_bags : np.ndarray
        Array of bags with shape (num_bags, num_instances, band_height, time_width, 1)
    y_bags : np.ndarray
        Array of bag labels (0 for closed, 1 for full-speed)
    """
    X_bags = []
    y_bags = []
    
    # Load closed spectrograms (label 0)
    closed_path = os.path.join(data_path, 'closed', '*.npy')
    closed_files = glob.glob(closed_path)
    print(f"Loading {len(closed_files)} closed spectrograms...")
    
    for file in closed_files:
        spec = np.load(file)  # Shape: (freq_bins, time_frames) e.g., (2049, 26)
        
        # Split into frequency bands (instances)
        instances = split_into_frequency_bands(spec, num_instances, target_time_width)
        
        X_bags.append(instances)
        y_bags.append(0)  # Closed = negative bag
    
    # Load full-speed spectrograms (label 1)
    fullspeed_path = os.path.join(data_path, 'full-speed', '*.npy')
    fullspeed_files = glob.glob(fullspeed_path)
    print(f"Loading {len(fullspeed_files)} full-speed spectrograms...")
    
    for file in fullspeed_files:
        spec = np.load(file)
        
        # Split into frequency bands (instances)
        instances = split_into_frequency_bands(spec, num_instances, target_time_width)
        
        X_bags.append(instances)
        y_bags.append(1)  # Full-speed = positive bag
    
    X_bags = np.array(X_bags)
    y_bags = np.array(y_bags)
    
    print(f"\nLoaded {len(X_bags)} bags (1 spectrogram = 1 bag)")
    print(f"Bag shape: {X_bags.shape}")
    print(f"  - {X_bags.shape[0]} bags")
    print(f"  - {X_bags.shape[1]} instances (frequency bands) per bag")
    print(f"  - Each instance: {X_bags.shape[2]}x{X_bags.shape[3]} (freq x time)")
    print(f"Closed bags: {np.sum(y_bags == 0)}")
    print(f"Full-speed bags: {np.sum(y_bags == 1)}")
    
    return X_bags, y_bags


def split_into_frequency_bands(spectrogram, num_bands, target_time_width):
    """
    Split a spectrogram into frequency bands.
    
    Parameters:
    -----------
    spectrogram : np.ndarray
        Input spectrogram of shape (freq_bins, time_frames)
    num_bands : int
        Number of frequency bands to create
    target_time_width : int
        Target width for time dimension (will resize if needed)
        
    Returns:
    --------
    bands : np.ndarray
        Array of shape (num_bands, band_height, target_time_width, 1)
    """
    freq_bins, time_frames = spectrogram.shape
    
    # Calculate band height
    band_height = freq_bins // num_bands
    
    # Resize time dimension if needed
    if time_frames != target_time_width:
        zoom_factor = target_time_width / time_frames
        spectrogram = zoom(spectrogram, (1, zoom_factor), order=1)
    
    bands = []
    for i in range(num_bands):
        start_freq = i * band_height
        end_freq = start_freq + band_height
        
        # Extract band
        band = spectrogram[start_freq:end_freq, :]
        
        # Add channel dimension
        band = np.expand_dims(band, axis=-1)
        
        bands.append(band)
    
    return np.array(bands)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data configuration
DATA_PATH = "spectrograms-output"
NUM_INSTANCES = 10  # Number of frequency bands per spectrogram
TARGET_TIME_WIDTH = 64  # Time dimension for each band (original: 26)

# Calculate band height from original spectrogram size
# Original: (2049, 26) -> Split into 10 bands -> Each band: (204, 64)
ORIGINAL_FREQ_BINS = 2049
BAND_HEIGHT = ORIGINAL_FREQ_BINS // NUM_INSTANCES  # 204

# Model configuration
INPUT_SHAPE = (BAND_HEIGHT, TARGET_TIME_WIDTH, 1)  # (204, 64, 1) per instance
DROPOUT_RATE = 0.5
ATTENTION_DIM = 128

# Training configuration
BATCH_SIZE = 32  # Increased since we have more bags now (4800)
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Paths
MODEL_SAVE_PATH = "models/best_model.keras"
os.makedirs("models", exist_ok=True)

print("="*60)
print("CONFIGURATION (Frequency Band MIL)")
print("="*60)
print(f"Data path: {DATA_PATH}")
print(f"Original spectrogram: (2049, 26)")
print(f"Instances per bag: {NUM_INSTANCES} frequency bands")
print(f"Instance shape: ({BAND_HEIGHT}, {TARGET_TIME_WIDTH}, 1)")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*60 + "\n")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("Loading spectrograms as bags...")
data_load_start = time.time()
X_bags, y_bags = load_spectrograms_as_bags(
    DATA_PATH, 
    num_instances=NUM_INSTANCES, 
    target_time_width=TARGET_TIME_WIDTH
)
data_load_time = time.time() - data_load_start
print(f"Data loading completed in {data_load_time:.2f}s ({data_load_time/60:.2f} minutes)")

print("\nSplitting data...")
# Split bags into train/val/test
X_train_bags, X_temp, y_train, y_temp = train_test_split(
    X_bags, y_bags, test_size=0.3, random_state=42, stratify=y_bags
)
X_val_bags, X_test_bags, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train_bags)} bags")
print(f"Val: {len(X_val_bags)} bags")
print(f"Test: {len(X_test_bags)} bags")

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_bags, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train_bags)).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_bags, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_bags, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE")
print("="*60 + "\n")

# ============================================================================
# MODEL CREATION
# ============================================================================

print("Creating model...")
model = create_model(
    input_shape=INPUT_SHAPE, 
    num_instances=NUM_INSTANCES,
    dropout_rate=DROPOUT_RATE,
    attention_dim=ATTENTION_DIM
)

# Print model summary
model.summary()

# ============================================================================
# MODEL COMPILATION
# ============================================================================

print("\nCompiling model...")
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc")
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
# FINAL SUMMARY
# ============================================================================

total_execution_time = time.time() - data_load_start

print("\n" + "="*60)
print("COMPLETE EXECUTION SUMMARY")
print("="*60)
print(f"Data loading time: {data_load_time:.2f}s ({data_load_time/60:.2f} min)")
print(f"Model training time: {training_total_time:.2f}s ({training_total_time/60:.2f} min)")
print(f"Test evaluation time: {evaluation_time:.2f}s")
print(f"Total execution time: {total_execution_time:.2f}s ({total_execution_time/60:.2f} min)")
print("="*60 + "\n")

