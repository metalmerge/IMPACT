import pandas as pd
from collections import defaultdict
import cv2
from PIL import Image
import numpy as np
import os
from PIL.ExifTags import TAGS
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tempfile
from pathlib import Path


def crop_center_vertical(image_path, output_dir):
    """
    Crops the image into three vertical sections and keeps only the center section.

    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the cropped center section.
    """
    try:
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
    except Exception as e:
        messagebox.showerror(
            "Error", f"An error occurred while processing the image: {e}"
        )


def detect_and_crop_faces(image_path, output_dir):
    """
    Detects faces in an image, crops the largest face, and saves it with 300 DPI.

    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the cropped face image.
    """
    try:
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
        output_path = os.path.join(
            output_dir, f"{os.path.basename(image_path)}_face.png"
        )

        # Save the cropped image with 300 DPI
        pil_image.save(output_path, dpi=(300, 300))

        print(f"Largest cropped face image saved to {output_path} with 300 DPI.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while detecting faces: {e}")


def select_images():
    # Open file dialog to select multiple images
    file_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")],
    )
    if file_paths:
        process_images(file_paths)


def process_images(file_paths):
    # Disable the select button and show the progress bar
    select_button.config(state=tk.DISABLED)
    progress_bar.grid(row=3, column=0, columnspan=2, pady=10)
    progress_bar.start()

    try:
        for file_path in file_paths:
            process_image(file_path)
        messagebox.showinfo("Success", "Image processing completed!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        # Enable the select button and hide the progress bar
        select_button.config(state=tk.NORMAL)
        progress_bar.stop()
        progress_bar.grid_remove()


def process_image(input_image_path):
    # Get the user's Downloads folder
    downloads_folder = str(Path.home() / "Downloads")

    # Image crop and find face
    output_image_folder = downloads_folder  # Path to save the cropped face image

    # Create output folder if it doesn't exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    # Perform cropping and face detection
    crop_center_vertical(input_image_path, output_image_folder)


def main():
    global select_button, progress_bar

    # Create the main window
    root = tk.Tk()
    root.title("Image Cropper and Face Detector")

    # Create a label with instructions
    instructions = tk.Label(root, text="Select images to crop and detect faces.")
    instructions.grid(row=0, column=0, columnspan=2, pady=10)

    # Create a button to select images
    select_button = tk.Button(root, text="Select Images", command=select_images)
    select_button.grid(row=1, column=0, columnspan=2, pady=10)

    # Create a progress bar
    progress_bar = ttk.Progressbar(root, mode="indeterminate")
    progress_bar.grid(row=3, column=0, columnspan=2, pady=10)
    progress_bar.grid_remove()

    # Run the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    main()
