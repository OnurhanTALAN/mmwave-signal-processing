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
# DATA LOADING FUNCTIONS
# ============================================================================

def load_spectrograms(data_path, target_shape=(128, 128)):
    """
    Load all spectrograms from closed and full-speed directories.
    
    Parameters:
    -----------
    data_path : str
        Path to the spectrograms-output directory
    target_shape : tuple
        Target shape for resizing spectrograms
        
    Returns:
    --------
    X : np.ndarray
        Array of spectrograms with shape (num_samples, height, width, 1)
    y : np.ndarray
        Array of labels (0 for closed, 1 for full-speed)
    """
    X = []
    y = []
    
    # Load closed spectrograms (label 0)
    closed_path = os.path.join(data_path, 'closed', '*.npy')
    closed_files = glob.glob(closed_path)
    print(f"Loading {len(closed_files)} closed spectrograms.")
    
    for file in closed_files:
        spec = np.load(file)
        # Resize if needed
        if spec.shape != target_shape:
            zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
            spec = zoom(spec, zoom_factors, order=1)
        X.append(spec)
        y.append(0)
    
    # Load full-speed spectrograms (label 1)
    fullspeed_path = os.path.join(data_path, 'full-speed', '*.npy')
    fullspeed_files = glob.glob(fullspeed_path)
    print(f"Loading {len(fullspeed_files)} full-speed spectrograms.")
    
    for file in fullspeed_files:
        spec = np.load(file)
        # Resize if needed
        if spec.shape != target_shape:
            zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
            spec = zoom(spec, zoom_factors, order=1)
        X.append(spec)
        y.append(1)
    
    X = np.array(X)
    y = np.array(y)
    
    # Add channel dimension
    X = np.expand_dims(X, axis=-1)
    
    print(f"\n✅ Loaded {len(X)} spectrograms")
    print(f"   Shape: {X.shape}")
    print(f"   Closed samples: {np.sum(y == 0)}")
    print(f"   Full-speed samples: {np.sum(y == 1)}")
    
    return X, y


def create_bags(X, y, num_instances=10, min_positive=2, max_positive=5, shuffle=True):
    """
    Create bags for Multiple Instance Learning with proper MIL assumption.
    
    MIL Assumption:
    - Negative bags: ALL instances are negative (closed)
    - Positive bags: AT LEAST 1 positive instance + rest are negative (mixed)
    
    Parameters:
    -----------
    X : np.ndarray
        Array of spectrograms
    y : np.ndarray
        Array of labels (0=closed, 1=full-speed)
    num_instances : int
        Number of instances per bag
    min_positive : int
        Minimum number of positive instances in positive bags
    max_positive : int
        Maximum number of positive instances in positive bags
    shuffle : bool
        Whether to shuffle data within each class before creating bags
        
    Returns:
    --------
    X_bags : np.ndarray
        Array of bags with shape (num_bags, num_instances, height, width, channels)
    y_bags : np.ndarray
        Array of bag labels
    """
    # Separate negative and positive instances
    negative_indices = np.where(y == 0)[0]
    positive_indices = np.where(y == 1)[0]
    
    X_negative = X[negative_indices].copy()
    X_positive = X[positive_indices].copy()
    
    if shuffle:
        np.random.shuffle(X_negative)
        np.random.shuffle(X_positive)
    
    X_bags = []
    y_bags = []
    positive_counts = []
    
    # Calculate how many positive bags we can create
    # Each positive bag needs ~(num_instances - avg_positive) negatives
    avg_positive = (min_positive + max_positive) / 2
    avg_negative_per_pos_bag = num_instances - avg_positive
    
    # Max positive bags based on positive instances
    max_pos_bags_from_positives = int(len(X_positive) / avg_positive)
    
    # Reserve negatives for positive bags first
    negatives_for_pos_bags = int(max_pos_bags_from_positives * avg_negative_per_pos_bag)
    negatives_for_neg_bags = len(X_negative) - negatives_for_pos_bags
    
    # Ensure we have at least some negative bags
    if negatives_for_neg_bags < num_instances:
        # Reduce positive bags to allow some negative bags
        negatives_for_neg_bags = (len(X_negative) // num_instances // 2) * num_instances
        negatives_for_pos_bags = len(X_negative) - negatives_for_neg_bags
    
    # Create NEGATIVE bags (pure closed instances)
    num_negative_bags = negatives_for_neg_bags // num_instances
    neg_idx = 0
    
    for i in range(num_negative_bags):
        bag = X_negative[neg_idx:neg_idx + num_instances]
        neg_idx += num_instances
        X_bags.append(bag)
        y_bags.append(0)
        positive_counts.append(0)
    
    # Remaining negatives for positive bags
    remaining_negatives = X_negative[neg_idx:]
    
    # Create POSITIVE bags (mixed: closed + full-speed)
    pos_idx = 0
    neg_for_pos_idx = 0
    
    while pos_idx < len(X_positive) and neg_for_pos_idx + (num_instances - max_positive) <= len(remaining_negatives):
        # Random number of positive instances (1-4)
        num_pos = np.random.randint(min_positive, max_positive + 1)
        num_neg = num_instances - num_pos
        
        # Check if we have enough instances
        if pos_idx + num_pos > len(X_positive):
            break
        if neg_for_pos_idx + num_neg > len(remaining_negatives):
            break
        
        # Get instances
        pos_instances = X_positive[pos_idx:pos_idx + num_pos]
        neg_instances = remaining_negatives[neg_for_pos_idx:neg_for_pos_idx + num_neg]
        
        pos_idx += num_pos
        neg_for_pos_idx += num_neg
        
        # Combine and shuffle within bag
        bag = np.concatenate([pos_instances, neg_instances])
        if shuffle:
            bag = bag[np.random.permutation(num_instances)]
        
        X_bags.append(bag)
        y_bags.append(1)
        positive_counts.append(num_pos)
    
    X_bags = np.array(X_bags)
    y_bags = np.array(y_bags)
    positive_counts = np.array(positive_counts)
    
    # Final shuffle of bags
    if shuffle:
        shuffle_idx = np.random.permutation(len(X_bags))
        X_bags = X_bags[shuffle_idx]
        y_bags = y_bags[shuffle_idx]
        positive_counts = positive_counts[shuffle_idx]
    
    # Print statistics
    print(f"\nCreated {len(X_bags)} bags (MIL with mixed positive bags)")
    print(f"Bag shape: {X_bags.shape}")
    print(f"Negative bags (pure closed): {np.sum(y_bags == 0)}")
    print(f"Positive bags (mixed): {np.sum(y_bags == 1)}")
    
    # Positive bag statistics
    pos_bag_counts = positive_counts[y_bags == 1]
    if len(pos_bag_counts) > 0:
        print(f"Full-speed instances per positive bag: min={pos_bag_counts.min()}, max={pos_bag_counts.max()}, avg={pos_bag_counts.mean():.1f}")
        for c in range(min_positive, max_positive + 1):
            count = np.sum(pos_bag_counts == c)
            if count > 0:
                print(f"  Bags with {c} full-speed: {count}")
    
    return X_bags, y_bags


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data configuration
DATA_PATH = "spectrograms-output"
TARGET_SHAPE = (1024, 64)  # Preserve frequency resolution (original: 2049 x 26)
NUM_INSTANCES = 10  # 10 instances per bag for better attention learning

# Model configuration
INPUT_SHAPE = (1024, 64, 1)  # Match target shape
DROPOUT_RATE = 0.5  # Increased from 0.5 to combat overfitting
ATTENTION_DIM = 128

# Training configuration
BATCH_SIZE = 16  # Increased from 8 for better generalization
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Paths
MODEL_SAVE_PATH = "models/best_model.keras"
os.makedirs("models", exist_ok=True)

print("="*60)
print("CONFIGURATION")
print("="*60)
print(f"Data path: {DATA_PATH}")
print(f"Original spectrogram shape: (2049, 26)")
print(f"Target shape (resized): {TARGET_SHAPE}")
print(f"Num instances per bag: {NUM_INSTANCES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Note: Preserving frequency resolution (2049→{TARGET_SHAPE[0]})")
print("="*60 + "\n")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("Loading spectrograms...")
data_load_start = time.time()
X, y = load_spectrograms(DATA_PATH, target_shape=TARGET_SHAPE)
data_load_time = time.time() - data_load_start
print(f"Data loading completed in {data_load_time:.2f}s ({data_load_time/60:.2f} minutes)")

print("\nSplitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Val: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

print("\nCreating bags for MIL...")
X_train_bags, y_train_bags = create_bags(X_train, y_train, num_instances=NUM_INSTANCES)
X_val_bags, y_val_bags = create_bags(X_val, y_val, num_instances=NUM_INSTANCES)
X_test_bags, y_test_bags = create_bags(X_test, y_test, num_instances=NUM_INSTANCES)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_bags, y_train_bags))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train_bags)).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_bags, y_val_bags))
val_dataset = val_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_bags, y_test_bags))
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
print(f"⏱️  Total training time: {training_total_time:.2f}s ({training_total_time/60:.2f} minutes)")
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

