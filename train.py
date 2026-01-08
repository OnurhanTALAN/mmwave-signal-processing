import tensorflow as tf
from model import create_model

# Create the Attention-based MIL model
model = create_model(
    input_shape=(128, 128, 1), 
    num_instances=10,
    dropout_rate=0.5,
    attention_dim=128
)

# Print model summary
model.summary()

# define optimizer, loss function and metrics
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=1e-4
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

# define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_loss",
        save_best_only=True,
    )
]

# Train the model
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=callbacks
)
