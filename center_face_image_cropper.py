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
from main import process_image, crop_center_vertical, crop_image_above_certificate
import sys


def select_images():
    # Open file dialog to select multiple images
    file_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")],
    )
    if file_paths:
        process_images(file_paths)


def resource_path(relative_path):
    """Get the absolute path to a resource, works for both development and PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores the path in `sys._MEIPASS`
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def detect_and_crop_face_above_certificate(
    image_path, output_dir, certificate_template_path
):
    """
    Detects the certificate in the image, finds the face above it, crops the face, and saves it with 300 DPI.

    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the cropped face image.
        certificate_template_path (str): Path to the certificate template image.
    """
    try:
        # Load the certificate template and input image
        certificate_template = cv2.imread(
            certificate_template_path, cv2.IMREAD_GRAYSCALE
        )
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform template matching to find the certificate in the image
        result = cv2.matchTemplate(
            gray_image, certificate_template, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        print(f"Max value: {max_val} {image_path}")

        # Set a threshold to detect the template
        threshold = 0.7
        if max_val >= threshold:
            template_w, template_h = certificate_template.shape[::-1]
            top_left = max_loc
            bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
            certificate_area = (top_left, bottom_right)
            print(f"Certificate found at {certificate_area}")
            # save image of certificate for testing .crop((x1, y1, x2, y2))
            # certificate_image = image[
            #     certificate_area[0][1] : certificate_area[1][1],
            #     certificate_area[0][0] : certificate_area[1][0],
            # ]
            # cv2.imwrite("certificate_image.jpg", certificate_image)
            image = crop_image_above_certificate(image, certificate_area)
        else:
            print(f"Certificate not found in the image. {image_path}")
            image = crop_center_vertical(image_path, output_dir)
            # return  # TODO, crop center vertical and try

        # Detect faces in the entire image
        # Load Haar Cascade classifier from the resource directory
        haarcascades_path = resource_path(
            "resources/haarcascades/haarcascade_frontalface_default.xml"
        )
        face_cascade = cv2.CascadeClassifier(haarcascades_path)

        if face_cascade.empty():
            raise FileNotFoundError(
                "Haar Cascade XML file not found or failed to load."
            )

        faces = face_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            print("No faces found in the image.")
            return

        # Find the face above the certificate
        largest_face = None
        largest_area = 0
        for x, y, w, h in faces:
            # if y < face_area[1]:  # Face must be above the certificate
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, w, h)

        if largest_face is None:
            print("No valid face found above the certificate.")
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
            output_dir,
            f"{os.path.basename(image_path)}.png",
        )

        # Save the cropped image with 300 DPI
        pil_image.save(output_path, dpi=(300, 300))

        print(f"Largest cropped face image saved to {output_path} with 300 DPI.")
    except Exception as e:
        print(f"An error occurred: {e}")


def process_image(input_image_path):
    # Get the user's Downloads folder
    downloads_folder = str(Path.home() / "Downloads")

    # Image crop and find face
    output_image_folder = downloads_folder  # Path to save the cropped face image

    # Create output folder if it doesn't exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    # Pass the correct path to detect_and_crop_face_above_certificate
    certificate_template_path = resource_path("certificate_template.jpg")
    detect_and_crop_face_above_certificate(
        input_image_path, output_image_folder, certificate_template_path
    )


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
