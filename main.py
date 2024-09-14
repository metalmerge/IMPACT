import pandas as pd
from collections import defaultdict
import cv2
from PIL import Image
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tempfile
from pathlib import Path
from tkinter import messagebox
from PIL.ExifTags import TAGS


def read_and_validate_csv(csv_file_path):
    """
    This function reads a CSV file and extracts the 'First Name', 'Last Name', and 'LookupID' columns.
    It also validates the data to ensure there are no missing values in these columns.

    Parameters:
        csv_file_path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing valid entries with First Name, Last Name, and LookupID.
    """
    try:
        # Read the CSV file while skipping lines with errors
        data = pd.read_csv(csv_file_path, on_bad_lines="skip")

        # Required columns
        required_columns = ["First Name", "Last Name", "LookupID", "Title"]

        # Validate required columns exist
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"CSV file must contain '{col}' column.")

        # Extract relevant columns and clean data
        relevant_data = data[required_columns]

        # Convert columns to string type
        relevant_data = relevant_data.astype(str)

        valid_data = relevant_data.dropna(subset=required_columns)
        valid_data = valid_data[
            (valid_data["First Name"].str.strip() != "")
            & (valid_data["Last Name"].str.strip() != "")
            & (valid_data["LookupID"].str.strip() != "")
            & (valid_data["Title"].str.strip() != "")
        ]

        if valid_data.empty:
            raise ValueError("No valid data found after filtering.")

        print(f"Successfully extracted {len(valid_data)} valid entries.")
        return valid_data

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: Issue parsing the CSV file '{csv_file_path}'.")
    except ValueError as ve:
        print(f"Data validation error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None


def detect_and_crop_faces(image_path, output_dir):
    """
    Detects faces in an image, crops the images around each face, and saves them with 300 DPI.

    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the cropped face images.
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
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(300, 300)
    )

    if len(faces) == 0:
        print("No faces found in the image.")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all detected faces and save each cropped face image
    for i, (x, y, w, h) in enumerate(faces):
        # Crop the image to the face
        cropped_image = image[y : y + h, x : x + w]

        # Convert the cropped image to PIL format
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        # Save the cropped image with 300 DPI
        output_path = os.path.join(output_dir, f"face_{i+1}.png")
        pil_image.save(output_path, dpi=(300, 300))

        print(f"Cropped face image saved to {output_path} with 300 DPI.")


def extract_metadata_from_image(image_path):
    """
    Extracts metadata from an image, specifically the caption_abstract if available.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        tuple: (image_name, caption_abstract) where image_name is the name of the image file
               and caption_abstract is the metadata value.
    """
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Extract EXIF data
            exif_data = img._getexif()

            if exif_data is not None:
                # Find the tag for 'caption_abstract' if it exists
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag.lower() == "caption_abstract":
                        return (os.path.basename(image_path), value)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    # Return None if 'caption_abstract' is not found
    return (os.path.basename(image_path), None)


def process_images_from_folder(folder_path):
    """
    Processes all images in a folder, extracting the caption_abstract metadata.

    Parameters:
        folder_path (str): Path to the folder containing image files.

    Returns:
        list of tuples: Each tuple contains (image_name, caption_abstract).
    """
    metadata_list = []

    # Iterate through all files in the directory
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is an image (you can add more image formats as needed)
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_metadata = extract_metadata_from_image(file_path)
            if image_metadata:
                metadata_list.append(image_metadata)

    return metadata_list


def load_data(csv_path):
    return pd.read_csv(csv_path)


# Determine gender based on title
def determine_gender(title):
    if title in ["Mr."]:
        return "Male"
    elif title in ["Mrs.", "Miss.", "Ms."]:
        return "Female"
    else:
        return "Unknown"


def classify_members(data):
    # Data structures to store classified groups
    groups = {
        "man_multiple": [],
        "married_man_twice": [],
        "man_once": [],
        "woman_multiple": [],
        "married_woman_twice": [],
        "woman_once": [],
        "doctors_multiple": [],
        "doctors_twice": [],
        "doctors_once": [],
    }

    # Dictionaries to count occurrences of last names
    last_name_count = defaultdict(int)
    member_info = {}

    for _, row in data.iterrows():
        first_name = row["First Name"]
        last_name = row["Last Name"]
        lookup_id = row["LookupID"]
        title = row["Title"]

        # Count last names for marital status determination
        last_name_count[last_name] += 1

        # Determine gender
        gender = determine_gender(title)

        # Store member information
        member_info[lookup_id] = {
            "first_name": first_name,
            "last_name": last_name,
            "title": title,
            "gender": gender,
        }

        # Add to doctors list with count of last names
        if title == "Dr.":
            if last_name_count[last_name] == 2:
                groups["doctors_twice"].append(
                    (
                        first_name,
                        last_name,
                        lookup_id,
                        title,
                        last_name_count[last_name],
                    )
                )
            elif last_name_count[last_name] == 1:
                groups["doctors_once"].append(
                    (
                        first_name,
                        last_name,
                        lookup_id,
                        title,
                        last_name_count[last_name],
                    )
                )
            elif last_name_count[last_name] > 2:
                groups["doctors_multiple"].append(
                    (
                        first_name,
                        last_name,
                        lookup_id,
                        title,
                        last_name_count[last_name],
                    )
                )

    # Classify based on last name occurrences and gender
    for lookup_id, info in member_info.items():
        first_name = info["first_name"]
        last_name = info["last_name"]
        title = info["title"]
        gender = info["gender"]

        if last_name_count[last_name] == 2:
            if gender == "Male":
                groups["married_man_twice"].append(
                    (
                        first_name,
                        last_name,
                        lookup_id,
                        title,
                        last_name_count[last_name],
                    )
                )
            elif gender == "Female":
                groups["married_woman_twice"].append(
                    (
                        first_name,
                        last_name,
                        lookup_id,
                        title,
                        last_name_count[last_name],
                    )
                )
        elif last_name_count[last_name] == 1:
            if gender == "Male":
                groups["man_once"].append(
                    (
                        first_name,
                        last_name,
                        lookup_id,
                        title,
                        last_name_count[last_name],
                    )
                )
            elif gender == "Female":
                groups["woman_once"].append(
                    (
                        first_name,
                        last_name,
                        lookup_id,
                        title,
                        last_name_count[last_name],
                    )
                )
        else:
            if gender == "Male":
                groups["man_multiple"].append(
                    (
                        first_name,
                        last_name,
                        lookup_id,
                        title,
                        last_name_count[last_name],
                    )
                )
            elif gender == "Female":
                groups["woman_multiple"].append(
                    (
                        first_name,
                        last_name,
                        lookup_id,
                        title,
                        last_name_count[last_name],
                    )
                )
            else:
                print(f"Unknown: {first_name} {last_name} {gender}")

    return groups


def select_images(images_dir):
    # Directory containing the images

    # List all image files in the directory
    file_paths = [
        os.path.join(images_dir, file_name)
        for file_name in os.listdir(images_dir)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]

    if file_paths:
        process_images(file_paths)
    else:
        messagebox.showinfo(
            "No Images Found", "No images found in the 'images/' directory."
        )


def process_images(file_paths):
    for file_path in file_paths:
        process_image(file_path)


def process_image(input_image_path):
    # Get the user's Downloads folder
    downloads_folder = Path.home() / "Downloads"

    # Create a new folder called "cropped faces" in the Downloads folder
    output_image_folder = downloads_folder / "cropped_faces"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    # Perform cropping and face detection
    # crop_center_vertical(input_image_path, output_image_folder)
    detect_and_crop_face_above_certificate(
        input_image_path, output_image_folder, "certificate_template.jpg"
    )


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

        # Further crop the center section to keep only the top half
        # top_half_center_section = center_section[: height // 2, :]

        # Convert to PIL format
        # pil_image = Image.fromarray(cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB))
        return center_section
        # Save the cropped image to a temporary file
        # with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        #     temp_image_path = temp_file.name
        #     pil_image.save(temp_image_path)

        # # Perform face detection on the temporary file
        # # detect_and_crop_face_above_certificate(
        # #     temp_image_path, output_dir, "certificate_template.jpg"
        # # )

        # # Remove the temporary file
        # os.remove(temp_image_path)

        print(f"Cropped center section saved to {output_dir}")
    except Exception as e:
        messagebox.showerror(
            "Error", f"An error occurred while processing the image: {e}"
        )


def crop_image_above_certificate(image, certificate_area):
    """
    Crops the image to keep everything above the Y-coordinate where the certificate was found
    and horizontally centered around the certificate's center X-coordinate.

    Parameters:
        image (numpy.ndarray): The input image.
        certificate_area (tuple): Coordinates of the detected certificate, given as ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)).

    Returns:
        numpy.ndarray: The cropped image.
    """
    try:
        # Unpack certificate_area
        (top_left_x, top_left_y), (bottom_right_x, _) = certificate_area

        # Image dimensions
        _, width = image.shape[:2]

        # Calculate the center X-coordinate of the certificate
        center_x = (top_left_x + bottom_right_x) // 2

        # Define the width of the cropped area (you can adjust the width as needed)
        crop_width = (bottom_right_x - top_left_x) * 2
        left_x = max(center_x - crop_width // 2, 0)
        right_x = min(center_x + crop_width // 2, width)

        # Define the cropping coordinates
        crop_top = 0
        crop_bottom = top_left_y

        # Crop the image
        cropped_image = image[crop_top:crop_bottom, left_x:right_x]

        # Convert the numpy.ndarray to a PIL Image object
        # pil_image = Image.fromarray(cropped_image)

        # Save the PIL Image object
        # pil_image.save("cropped_image.jpg")

        return cropped_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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
        threshold = 0.8
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
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
            output_dir, f"{os.path.basename(image_path)}_face.png"
        )

        # Save the cropped image with 300 DPI
        pil_image.save(output_path, dpi=(300, 300))

        print(f"Largest cropped face image saved to {output_path} with 300 DPI.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # extract info from csv
    # csv_file_path = r"CCA PC members for photos.csv"
    # valid_data = read_and_validate_csv(csv_file_path)
    # # if valid_data is not None:
    # #     print(valid_data)
    # groups = classify_members(valid_data)  # TODO check if works
    # for group_name, members in groups.items():
    #     print(f"\n{group_name.replace('_', ' ').title()}:")
    #     for i, member in enumerate(members):
    #         if i >= 5:
    #             break
    #         print(f"Name: {member[0]} {member[1]}, {member[3]} ({member[4]})")

    # Image crop and find face
    images_dir = "images/"
    select_images(images_dir)

    # Extract metadata from images in a folder
    # folder_path = "path_to_your_image_folder"  # Path to the folder with images
    # metadata_list = process_images_from_folder(folder_path)
    # # Print or save the metadata list
    # for image_name, caption_abstract in metadata_list:
    #     print(f"Image: {image_name}, Caption Abstract: {caption_abstract}")


if __name__ == "__main__":
    main()
