
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# -----------------------------
# 1. Configuration
# -----------------------------
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 75
SEED = 42

MODEL_PATH = "facial_expression_model.keras"
BEST_MODEL_PATH = "best_model.keras"

tf.keras.utils.set_random_seed(SEED)


# -----------------------------
# 2. Check dataset
# -----------------------------
if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        "Dataset not found. Expected data/train and data/test."
    )

class_names = sorted(
    folder.name for folder in TRAIN_DIR.iterdir() if folder.is_dir()
)

if len(class_names) != 7:
    raise ValueError(
        f"Expected 7 expression folders, found {len(class_names)}: "
        f"{class_names}. Check your dataset structure."
    )

test_class_names = sorted(
    folder.name for folder in TEST_DIR.iterdir() if folder.is_dir()
)

if class_names != test_class_names:
    raise ValueError(
        "The train and test folders must contain the same class folders."
    )

print("Expression classes:", class_names)


# -----------------------------
# 3. Load training images
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)


# -----------------------------
# 4. Improve input pipeline
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# -----------------------------
# 5. Handle class imbalance (Smoothed)
# -----------------------------
import math

train_labels = np.concatenate([
    labels.numpy() for _, labels in train_ds
])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=train_labels
)

# Apply square root to dampen extreme weight values
class_weights = {
    index: math.sqrt(float(weight)) 
    for index, weight in enumerate(weights)
}

print("Smoothed Class weights:", class_weights)

# -----------------------------
# 6. Build the CNN
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    
    # Data Augmentation (Highly Recommended)
    tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]),
    
    tf.keras.layers.Rescaling(1.0 / 255),

    # Convolution block 1
    tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # Convolution block 2 (Restored)
    tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # Convolution block 3 (Upgraded to double-conv)
    tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding="same"), # Added
    tf.keras.layers.LeakyReLU(alpha=0.1),                # Added
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # Convolution block 4 (Upgraded to double-conv)
    tf.keras.layers.Conv2D(256, (3, 3), padding="same"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), padding="same"), # Added
    tf.keras.layers.LeakyReLU(alpha=0.1),                # Added
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),  

    # --- UPGRADED CLASSIFICATION HEAD ---
    tf.keras.layers.Flatten(), # Restored spatial awareness
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.3), # Relaxed dropout
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

# Lowered learning rate to prevent dead neurons
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# -----------------------------
# 7. Training callbacks
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15, # <--- Change this from 5 to 15
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3, # Increased slightly so it doesn't decay too fast
        min_lr=1e-6
    )
]


# -----------------------------
# 8. Train the model
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# -----------------------------
# 9. Save model and labels
# -----------------------------
# Load the best checkpoint saved during training.
model = tf.keras.models.load_model(BEST_MODEL_PATH)
model.save(MODEL_PATH)

with open("class_names.json", "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

print(f"Model saved to {MODEL_PATH}")


# -----------------------------
# 10. Evaluate on test data
# -----------------------------
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# -----------------------------
# 11. Classification report
# -----------------------------
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

y_prob = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
))


# -----------------------------
# 12. Confusion matrix
# -----------------------------
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(class_names))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(xticks_rotation=45, values_format="d")
plt.title("FER-2013 Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()


# -----------------------------
# 13. Training curves
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
