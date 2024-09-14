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
from main import process_image


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
