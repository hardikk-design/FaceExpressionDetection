
import json
import sys

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "best_model.keras"
LABELS_PATH = "class_names.json"


def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


def prepare_face(face):
    # Convert to grayscale and resize to the training dimensions.
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))

    # Shape: (1, 48, 48, 1)
    face = face.astype("float32")
    face = np.expand_dims(face, axis=(0, -1))

    # Normalization is already inside the saved model.
    return face


def predict_expression(image_path):
    model, class_names = load_model_and_labels()

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print("No face detected. Try a clear, front-facing image.")
        return

    # Predict the largest detected face.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    
    face = image[y:y + h, x:x + w]
    processed_face = prepare_face(face)

    probabilities = model.predict(
        processed_face, verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))
    label = class_names[predicted_index]
    confidence = probabilities[predicted_index] * 100

    print(f"Predicted expression: {label}")
    print(f"Confidence: {confidence:.2f}%")

    # --- NEW: Draw bounding box and label on the image ---
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    text = f"{label} ({confidence:.1f}%)"
    cv2.putText(
        image, text, (x, y - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

    # Show the image until the user presses any key
    cv2.imshow("Facial Expression Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python predict.py "my_face.jpg"')
        sys.exit(1)

    predict_expression(sys.argv[1])
