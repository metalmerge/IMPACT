import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def preprocess_images(folder_path, target_size=(224, 224)):
    """
    Detects and resizes faces in the images in the given folder.

    Parameters:
        folder_path (str): Path to the folder containing images.
        target_size (tuple): Desired output size for the face images.

    Returns:
        np.ndarray, np.ndarray: Arrays of processed images and their labels.
    """
    images = []
    labels = []

    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file_name)

        if os.path.isfile(img_path):  # Ensure it is a file
            img = cv2.imread(img_path)
            face = detect_face(img)
            if face is not None:
                face_resized = cv2.resize(face, target_size)
                images.append(face_resized)
                labels.append(
                    1 if "target" in folder_path else 0
                )  # Update the condition based on folder name

    return np.array(images), np.array(labels)


def detect_face(image):
    """
    Detects a single face in an image using OpenCV's Haar Cascades.

    Parameters:
        image (np.ndarray): The image to process.

    Returns:
        np.ndarray or None: The detected face or None if no face was found.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return image[y : y + h, x : x + w]
    return None


def create_model(input_shape=(224, 224, 3)):
    """
    Creates a ResNet-based model for face classification.

    Parameters:
        input_shape (tuple): The shape of the input images.

    Returns:
        Model: The compiled Keras model.
    """
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def find_target_in_image(image, model, step=32, window_size=(224, 224)):
    """
    Detects target faces in a multi-person image using a sliding window approach.

    Parameters:
        image (np.ndarray): The multi-person image to scan.
        model (Model): The trained Keras model for face classification.
        step (int): Step size for sliding window.
        window_size (tuple): Size of the window for cropping faces.

    Returns:
        list: List of coordinates where the target face was detected.
    """
    height, width = image.shape[:2]
    detected_faces = []

    for y in range(0, height - window_size[1], step):
        for x in range(0, width - window_size[0], step):
            window = image[y : y + window_size[1], x : x + window_size[0]]
            face = detect_face(window)
            if face is not None:
                face_resized = cv2.resize(face, window_size)
                face_input = np.expand_dims(face_resized, axis=0)

                prediction = model.predict(face_input)
                if prediction[0][1] > 0.9:
                    detected_faces.append(
                        (x, y, x + window_size[0], y + window_size[1])
                    )

    return detected_faces


def main():
    # TODO THIS CODE IS UNFISHED BUT DOES RUN
    # Check if the required number of command-line arguments is provided
    if len(sys.argv) < 2:
        print("Usage: python specific_face_finding_model_training.py <mode>")
        sys.exit(1)

    mode = sys.argv[1]

    # Paths to datasets
    target_folder = "dataset/Arnn"  # Folder containing target images
    false_folder = "dataset/FACES_ A database of facial expressions in young, middle-aged, and older women and men (publicly available datasets)"  # Folder containing false images
    multi_person_folder = (
        "dataset/PC_Photos_v3"  # Folder containing multi-person images
    )

    if mode == "1":
        # Load and preprocess images
        target_images, target_labels = preprocess_images(target_folder)
        false_images, false_labels = preprocess_images(false_folder)

        # Prepare data for training
        X = np.concatenate((target_images, false_images), axis=0)
        y = np.concatenate(
            (np.ones(len(target_images)), np.zeros(len(false_images))), axis=0
        )

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train the model
        model = create_model()
        model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16
        )

        # Save model
        model.save("target_face_detector_model.h5")
        print("Model saved as 'target_face_detector_model.h5'.")

    elif mode == "2":
        # Load the model
        model = tf.keras.models.load_model("target_face_detector_model.h5")

        # Run detection on all multi-person images
        results = []
        for img_name in os.listdir(multi_person_folder):
            img_path = os.path.join(multi_person_folder, img_name)
            img = cv2.imread(img_path)
            faces = find_target_in_image(img, model)
            results.append({"image": img_name, "faces": faces})

        # Print results
        for result in results:
            print(f"Image {result['image']} - Target Faces Found: {result['faces']}")

    else:
        print("Invalid mode. Use '1' for training or '2' for detection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
