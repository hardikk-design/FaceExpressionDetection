
import json
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("best_model.keras")

# Load the exact class ordering used during training
with open("class_names.json", "r", encoding="utf-8") as file:
    class_names = json.load(file)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    label_mode="int",
    class_names=class_names,
    shuffle=False
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
