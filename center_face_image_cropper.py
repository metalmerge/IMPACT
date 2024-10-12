import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from main import (
    process_image,
    extract_title_from_windows_properties,
    extract_names_from_title,
)


def select_images():
    # Open file dialog to select multiple images
    file_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")],
    )
    if file_paths:
        process_images(file_paths)


def extract_names_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image '{image_path}' does not exist.")
        return None

    # Check if the file is a valid image file
    if image_path.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
        # Extract title metadata
        title = extract_title_from_windows_properties(image_path)

        # Extract names from title metadata
        names = extract_names_from_title(title)

        # Return the image file name and extracted names
        return (os.path.basename(image_path), names)
    else:
        print(f"File '{image_path}' is not a valid image.")
        return None


def process_images(file_paths):
    # Disable the select button and show the progress bar
    select_button.config(state=tk.DISABLED)
    progress_bar.grid(row=3, column=0, columnspan=2, pady=10)
    progress_bar.start()

    # # Print or save the metadata list
    for file_path in file_paths:
        # meta_data = extract_names_from_image(file_path)
        # if meta_data[1]:
        #     print(f"Meta data in: {file_path}:{meta_data}")
        #     process_image(file_path, meta_data[1])
        # else:
        # print(f"No metadata in: {file_path}")
        file_name = os.path.basename(file_path)
        print(f"Processing image: {file_name}")
        process_image(file_path, file_name)
    try:
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
