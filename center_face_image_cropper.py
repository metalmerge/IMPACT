import pandas as pd
from collections import defaultdict
import cv2
from PIL import Image
import numpy as np
import os
from PIL.ExifTags import TAGS
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tempfile


def crop_center_vertical(image_path, output_dir):
    """
    Crops the image into three vertical sections and keeps only the center section.

    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the cropped center section.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Calculate the width of each section
    section_width = width // 3

    # Define the coordinates for the center section
    start_x = section_width
    end_x = 2 * section_width

    # Crop the center section
    center_section = image[:, start_x:end_x]

    # Convert to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB))

    # Save the cropped image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_image_path = temp_file.name
        pil_image.save(temp_image_path)

    # Perform face detection on the temporary file
    detect_and_crop_faces(temp_image_path, output_dir)

    # Remove the temporary file
    os.remove(temp_image_path)

    print(f"Cropped center section saved to {output_dir}")


def detect_and_crop_faces(image_path, output_dir):
    """
    Detects faces in an image, crops the largest face, and saves it with 300 DPI.

    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the cropped face image.
    """
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        print("No faces found in the image.")
        return

    # Find the largest face based on area (width * height)
    largest_face = None
    largest_area = 0
    for x, y, w, h in faces:
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_face = (x, y, w, h)

    if largest_face is None:
        print("No valid face found.")
        return

    # Extract the largest face coordinates
    x, y, w, h = largest_face

    # Crop the image to the largest face
    cropped_image = image[y : y + h, x : x + w]

    # Convert the cropped image to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the output path for the largest face image
    output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}_face.png")

    # Save the cropped image with 300 DPI
    pil_image.save(output_path, dpi=(300, 300))

    print(f"Largest cropped face image saved to {output_path} with 300 DPI.")


def select_images():
    # Open file dialog to select multiple images
    file_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")],
    )
    if file_paths:
        for file_path in file_paths:
            process_image(file_path)


def process_image(input_image_path):
    # Image crop and find face
    output_image_folder = (
        "cropped_images_of_faces"  # Path to save the cropped face image
    )

    # Create output folder if it doesn't exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    # Perform cropping and face detection
    crop_center_vertical(input_image_path, output_image_folder)

    # messagebox.showinfo(
    #     "Success", f"Image processing completed for {input_image_path}!"
    # )


def main():
    # Create the main window
    root = tk.Tk()
    root.title("Image Cropper and Face Detector")

    # Create a button to select images
    select_button = tk.Button(root, text="Select Images", command=select_images)
    select_button.pack(pady=20)

    # Run the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    main()
