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
import win32com.client
import re
from pathlib import Path
import sys
import csv
from tqdm import tqdm
import shutil
import gender_guesser.detector as gender

EXTENSIONS = [
    ".JPG",
    ".jpg",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".bmp",
    ".BMP",
    ".tiff",
    ".TIFF",
    ".webp",
    ".WEBP",
]


def read_and_validate_csv(csv_file_path):
    """
    This function reads a CSV file and extracts the 'First Name', 'Last Name', 'LookupID', and 'Title' columns.
    It also validates the data to ensure there are no missing values in these columns and removes duplicate entries.

    Parameters:
        csv_file_path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing valid entries with First Name, Last Name, LookupID, and Title.
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

        # Remove rows with missing values in required columns
        valid_data = relevant_data.dropna(subset=required_columns)
        valid_data = valid_data[
            (valid_data["First Name"].str.strip() != "")
            & (valid_data["Last Name"].str.strip() != "")
            & (valid_data["LookupID"].str.strip() != "")
            & (valid_data["Title"].str.strip() != "")
        ]

        # Remove duplicate rows based on 'First Name' and 'Last Name'
        valid_data = valid_data.drop_duplicates(subset=["First Name", "Last Name"])

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
    # Load Haar Cascade classifier from the resource directory
    haarcascades_path = resource_path(
        "resources/haarcascades/haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(haarcascades_path)

    if face_cascade.empty():
        raise FileNotFoundError("Haar Cascade XML file not found or failed to load.")

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
        output_path = os.path.join(output_dir, f"face_{i + 1}.png")
        pil_image.save(output_path, dpi=(300, 300))

        print(f"Cropped face image saved to {output_path} with 300 DPI.")


def extract_title_from_windows_properties(image_path):
    """
    Extracts and prints the 'Title' from Windows file properties (Details > Description).

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted title if found, otherwise None.
    """
    try:
        # Ensure the file exists
        if not os.path.exists(image_path):
            print(f"File does not exist: {image_path}")
            return None

        # Get the absolute path to the folder
        folder_path = os.path.abspath(os.path.dirname(image_path))
        shell = win32com.client.Dispatch("Shell.Application")
        folder = shell.Namespace(folder_path)

        # Ensure that the folder object is valid
        if folder is None:
            print(f"Could not open folder: {folder_path}")
            return None

        file = folder.ParseName(os.path.basename(image_path))

        # Ensure the file object is valid
        if file is None:
            print(f"Could not open file: {image_path}")
            return None

        # Extract the "Title" (in Windows, Title is property 21)
        title = folder.GetDetailsOf(file, 21)

        if title:
            return title
        else:
            print(f"No Title metadata found for {os.path.basename(image_path)}.")
            return None

    except Exception as e:
        print(f"Could not extract metadata for {image_path}: {e}")
        return None


def clean_title(title):
    """
    Removes unwanted words and phrases that are not names from the title.

    Parameters:
        title (str): The title text containing names and other information.

    Returns:
        str: Cleaned title text with only potential names left.
    """
    # Common words/phrases to remove
    remove_phrases = [
        "PC Certificate presented at",
        "San Antonio",
        "TX",
        "May",
        "February",
        "Parents Weekend",
        "Argyle",
        "Hillsdale College",
        "Dinner",
        "Broadlawn",
        "Reception",
        "during",
        "Presentation",
        "Not pictured",
        "Certifice Presenti",
        "Hillsdale College",
        "Arnn September",
        "Club Freshman",
        "Parent Farewell",
        "Presidents Club",
        "Freshman Farewell",
        "Dinner August",
        "Hilt St",
        "Club Certificate",
        "Diamond Member",
        "Photographer",
        "Certificate",
        "Lunche Vero",
        "Metz Photographer",
        "Steering Committee",
        "Florida January",
        "Irving Cventi",
        "Member",
        "Presented",
        "Society",
        "Kirryher Photographer",
        "Simi Valley",
        "Founders Circle",
        "China",
        "Hotel",
        "September",
        "Grand Plaza Hotel",
        "Searle Center",
        "Socialism November",
        "Fort Des",
        "Palm Beach",
        "Moines Iowa",
        "Irving Convention",
        "Center Irving",
        "Center",
        "March",
        "Hartford Connecticut",
        r"\b\d{4}\b",  # Remove years
        r"\b\d{1,2}[a-z]{2},?\s+\d{4}\b",  # Remove dates like "May 22, 2019"
    ]

    if "Hillsdale College hosts the National Leadership Seminar" in title:
        return ""

    # Remove each unwanted phrase
    for phrase in remove_phrases:
        title = re.sub(phrase, "", title, flags=re.IGNORECASE)

    # Remove extra whitespace
    return " ".join(title.split())


def clean_title_around_larry_arnn(title):
    """
    Cleans the title by extracting up to 4 words before and 4 words after 'Larry Arnn',
    while excluding 'Larry Arnn' itself.

    Parameters:
        title (str): The title text containing names and other information.

    Returns:
        str: The cleaned title with up to 4 words before and 4 words after 'Larry Arnn'.
    """
    if title is None:
        return None

    # Split the title into words
    words = title.replace(",", "").replace(";", "").split()

    # Find the index of "Larry Arnn" in the title
    try:
        larry_index = words.index(
            "Larry"
        )  # Find 'Larry' and assume 'Arnn' follows immediately
    except ValueError:
        # If "Larry" is not found, return an empty string
        return ""

    # Ensure "Arnn" is the next word to confirm we have "Larry Arnn"
    if larry_index + 1 < len(words) and words[larry_index + 1] == "Arnn":
        # Extract up to 4 words before "Larry Arnn"
        start_index = max(0, larry_index - 4)

        # Extract up to 4 words after "Arnn"
        end_index = min(
            len(words), larry_index + 2 + 4
        )  # +2 to skip both "Larry" and "Arnn"

        # Get the words before and after "Larry Arnn"
        cleaned_words = (
            words[start_index:larry_index] + words[larry_index + 2 : end_index]
        )
        cleaned_title = " ".join(cleaned_words)
    else:
        # If "Arnn" is not immediately after "Larry," return an empty string
        cleaned_title = ""

    return cleaned_title


def split_and_return_names(names):
    """
    Splits names containing '&' into separate names and returns a list of individual names.

    Parameters:
        names (list): A list of names that may contain '&'.

    Returns:
        list: A list of individual names.
    """
    individual_names = []
    for name in names:
        if "&" in name:
            # Split names by '&', strip extra spaces, and add each part separately
            parts = [part.strip() for part in name.split("&")]
            for part in parts:
                if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", part):
                    individual_names.append(part)
        else:
            individual_names.append(name)

    return individual_names


def extract_names_from_title(title):
    """
    Extracts one or two names from the title while excluding 'Larry Arnn'.

    Parameters:
        title (str): The title text containing names and other information.

    Returns:
        list: A list of extracted names (if any).
    """
    title = clean_title_around_larry_arnn(title)
    # Clean the title by removing irrelevant phrases
    cleaned_title = clean_title(title)

    # Define a regex pattern to match names like "First & Last" or "First and Last"
    name_pattern = r"([A-Z][a-z]+(?:\s*[&and]\s*[A-Z][a-z]+)*\s+[A-Z][a-z]+)"

    # Extract names using the pattern
    names = re.findall(name_pattern, cleaned_title)

    individual_names = split_and_return_names(names)

    # Return the extracted names, ensuring there's a maximum of two names
    return individual_names[:2]


def extract_names_from_filename(file_name):
    """
    Extracts names from the file name.

    Parameters:
        file_name (str): The file name containing names.

    Returns:
        list: A list of extracted names.
    """
    banned_words = [
        "edit",
        "PC Photo",
        "Copy",
        "Traditional",
        "Family",
        "family",
        "present",
        "Present",
        "With",
        "with",
        "Kids",
        "kids",
        "daughter",
        "Daughter",
    ]
    # Remove the file extension

    base_name = os.path.splitext(file_name)[0]
    for phrase in banned_words:
        base_name = re.sub(phrase, "", base_name, flags=re.IGNORECASE)

    # Replace underscores and hyphens with spaces
    base_name = base_name.replace("_", " ").replace("-", " ")

    # Split names by "and" or "&"
    names = re.split(r" and | & ", base_name)

    # Trim extra spaces
    names = [name.strip() for name in names]

    # Find the last name by checking if the last name appears in the final part
    last_name = names[-1].split()[-1] if len(names) > 1 else None

    # Append the last name to first names if needed
    full_names = []
    for name in names:
        name_parts = name.split()
        if (
            len(name_parts) == 1 and last_name
        ):  # If it's only a first name, add the last name
            full_names.append(f"{name_parts[0]} {last_name}")
        else:
            full_names.append(name)

    return full_names


def process_images_in_folder(folder_path):
    """
    Goes through all images in a folder and extracts the 'Title' metadata from Windows properties.

    Returns:
        list: A list of tuples containing the file name and extracted names.
    """
    global EXTENSIONS
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return []

    names_in_image = []

    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        # Only process if it is a valid image file
        if any(char in file_name for char in "0123456789"):
            continue
        elif file_name.lower().endswith(tuple(EXTENSIONS)):
            # title = extract_title_from_windows_properties(file_path)
            # names = extract_names_from_title(title)  # TODO

            # title = file_name
            # Extract names from the file name
            names = extract_names_from_filename(file_name)
            names_in_image.append((file_name, names))
        else:
            print(f"Skipping non-image file: {file_name}")

    return names_in_image


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
    """
    Classifies members into various groups based on their titles, last names, and gender.

    Parameters:
        data (pd.DataFrame): The DataFrame containing member information.

    Returns:
        dict: A dictionary with groups as keys and lists of members as values.
    """
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


def select_images(images_dir, names):
    global EXTENSIONS
    # List all image files in the directory
    file_paths = [
        os.path.join(images_dir, file_name)
        for file_name in os.listdir(images_dir)
        if file_name.lower().endswith(tuple(EXTENSIONS))
    ]

    if file_paths:
        process_images(file_paths, names)
    else:
        messagebox.showinfo(
            "No Images Found", "No images found in the 'images/' directory."
        )


def process_images(file_paths, names):
    for file_path in file_paths:
        process_image(file_path, names)


def resource_path(relative_path):
    """Get the absolute path to a resource, works for both development and PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores the path in `sys._MEIPASS`
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def process_image(input_image_path, names):
    # Get the user's Downloads folder
    downloads_folder = Path.home() / "Downloads"

    # Create a new folder called "cropped faces" in the Downloads folder
    output_image_folder = downloads_folder / "cropped_faces"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    # Perform cropping and face detection
    # crop_center_vertical(input_image_path, output_image_folder)

    # Use the resource_path function to get the path for the template image
    certificate_template_path = resource_path(
        "image_templates/certificate_template.jpg"
    )

    # Pass the correct path to detect_and_crop_face_above_certificate

    d = gender.Detector()
    for name in names:
        names_string = str(name).replace(" ", "_")
        output_path = os.path.join(output_image_folder, f"{names_string}.png")
        if os.path.exists(output_path):
            print(f"Image already exists in {output_image_folder}")
            continue
        married = False
        if len(names) > 1:
            married = True

        first_name = name.split()[0]  # Assuming the first word is the first name
        guessed_gender = d.get_gender(first_name)
        print(f"Guessed: {first_name} to be {guessed_gender}")
        detect_and_crop_face_above_certificate(
            input_image_path,
            output_image_folder,
            certificate_template_path,
            names_string,
            guessed_gender,
            married,
        )


def crop_image_based_on_conditions(
    image_path, output_dir, guessed_gender="unknown", married=False
):
    """
    Crops the image based on the guessed gender and marital status:
    - If male and married, divide into 5 vertical slices, crop out the center (3rd) slice, and crop out the bottom fourth of the image.
    - If male and unmarried, crop out the bottom fourth of the image (horizontally).
    - Otherwise, divide into 3 vertical slices and keep only the center section.

    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Path to save the cropped image.
        guessed_gender (str): The guessed gender of the person ("male" or "female").
        married (bool): Marital status of the person (True for married, False for unmarried).
    """
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        height, width = image.shape[:2]  # Get image dimensions

        # Scenario 1: Male and Married
        if guessed_gender == "male" and married:
            # Divide into 5 vertical slices
            section_width = width // 5

            # Keep sides by cropping out the center 3rd slice
            left_section = image[:, : section_width * 2]  # Left 2/5ths
            right_section = image[:, section_width * 3 :]  # Right 2/5ths

            # Concatenate left and right sections horizontally
            image_without_center = cv2.hconcat([left_section, right_section])

            # Now crop out the bottom fourth of the image (horizontally)
            crop_bottom_y = int(height * 2 / 3)  # Keep upper 2/3rds of the image
            cropped_image = image_without_center[:crop_bottom_y, :]

        # Scenario 2: Male and Unmarried
        # elif guessed_gender == "male" and not married:
        #     # Crop out the bottom fourth of the image (horizontally)
        #     crop_bottom_y = int(height * 3 / 4)  # Keep upper 3/4ths of the image
        #     cropped_image = image[:crop_bottom_y, :]

        elif guessed_gender == "female" and married:
            # # Divide into 3 vertical slices
            section_width = width // 3

            # Keep only the center section
            start_x = section_width
            end_x = 2 * section_width
            cropped_image = image[:, start_x:end_x]
        # Scenario 3: Default (Other Cases)
        else:
            # Crop out the bottom fourth of the image (horizontally)
            crop_bottom_y = int(height * 2 / 3)  # Keep upper 2/3rds of the image
            cropped_image = image[:crop_bottom_y, :]

        # Create output directory if it does not exist
        # os.makedirs(output_dir, exist_ok=True)
        # Create output file name and save cropped image
        # output_file_name = os.path.join(
        #     output_dir,
        #     f"cropped__{guessed_gender}_{married}_{os.path.basename(image_path)}",
        # )
        # cv2.imwrite(output_file_name, cropped_image)
        # print(f"Cropped image saved to {output_file_name}")

        return cropped_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def crop_image_above_certificate(image, certificate_area, guessed_gender, married):
    """
    Crops the image based on the guessed gender and marital status:
    - If male and married, crop out the center (keep sides).
    - If male and unmarried, crop everything below the certificate.
    - Otherwise, crop the image above the certificate, centered around the certificate's X-coordinate.

    Parameters:
        image (numpy.ndarray): The input image.
        certificate_area (tuple): Coordinates of the detected certificate, given as ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)).
        guessed_gender (str): The guessed gender of the person ("male" or "female").
        married (bool): Marital status of the person (True for married, False for unmarried).

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

        if guessed_gender == "male" and married:
            # Crop out the center and keep sides
            crop_top = 0
            crop_bottom = top_left_y
            left_x = 0  # Keep the left side
            crop_width = bottom_right_x - top_left_x
            right_x = center_x - crop_width // 2  # Crop center-left
            cropped_image_left = image[crop_top:crop_bottom, left_x:right_x]

            left_x = center_x + crop_width // 2  # Crop center-right
            right_x = width  # Keep the right side
            cropped_image_right = image[crop_top:crop_bottom, left_x:right_x]

            # Combine the left and right cropped sides
            cropped_image = cv2.hconcat([cropped_image_left, cropped_image_right])

        elif guessed_gender == "male" and not married:
            # Crop everything below the certificate
            crop_top = 0
            crop_bottom = top_left_y
            cropped_image = image[crop_top:crop_bottom, :]

        else:
            # Default behavior: crop centered around certificate
            left_x = max(center_x - crop_width // 2, 0)
            right_x = min(center_x + crop_width // 2, width)
            crop_top = 0
            crop_bottom = top_left_y

            # Crop the image
            cropped_image = image[crop_top:crop_bottom, left_x:right_x]

        return cropped_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def test_crop_image_above_certificate():
    # Example images and certificate areas for testing
    test_images = [
        (
            "male_married.jpg",
            "male",
            True,
        ),  # Male, married (crop center out, keep sides)
        (
            "male_unmarried.jpg",
            "male",
            False,
        ),  # Male, unmarried (crop below certificate)
        ("female.jpg", "female", False),  # Female (default cropping behavior)
    ]

    # Fake certificate area for testing purposes (you can adjust based on the image resolution)
    # Load the certificate template and input image
    certificate_template_path = resource_path(
        "image_templates/certificate_template.jpg"
    )
    certificate_template = cv2.imread(certificate_template_path, cv2.IMREAD_GRAYSCALE)

    for image_name, guessed_gender, married in test_images:

        # Load the image (ensure these images exist in the same directory)
        image = cv2.imread(image_name)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image is None:
            print(f"Image {image_name} not found.")
            continue
        # Perform template matching to find the certificate in the image
        result = cv2.matchTemplate(
            gray_image, certificate_template, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        # Set a threshold to detect the template
        threshold = 0.7
        if max_val >= threshold:
            template_w, template_h = certificate_template.shape[::-1]
            top_left = max_loc
            bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
            certificate_area = (top_left, bottom_right)

        output_dir = "output_tests"
        os.makedirs(output_dir, exist_ok=True)
        # Call the function
        cropped_image = crop_image_above_certificate(
            image, certificate_area, guessed_gender, married
        )

        if cropped_image is not None:
            # Save the resulting cropped image
            output_image_path = os.path.join(output_dir, f"cropped_{image_name}")
            cv2.imwrite(output_image_path, cropped_image)
            print(f"Saved cropped image for {image_name} at {output_image_path}")
        else:
            print(f"Failed to crop {image_name}")


def detect_and_crop_face_above_certificate(
    image_path,
    output_dir,
    certificate_template_path,
    names_string,
    guessed_gender="unknown",
    married=False,
):
    """
    Detects the certificate in the image, finds Larry Arnn's face, replaces it with black pixels,
    finds the face above it, crops the face (and upper body), and saves it with 300 DPI.
    """
    output_path = os.path.join(output_dir, f"{names_string}.png")
    try:
        image_path = str(image_path)

        image = cv2.imread(image_path)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load Larry Arnn's template for face detection
        larry_template = cv2.imread(
            "image_templates/Larry_Arnn_template.png", cv2.IMREAD_GRAYSCALE
        )
        # Find Larry Arnn's face using template matching
        larry_result = cv2.matchTemplate(
            gray_image, larry_template, cv2.TM_CCOEFF_NORMED
        )
        _, larry_max_val, _, larry_max_loc = cv2.minMaxLoc(larry_result)

        # Set a threshold for Larry Arnn's face detection
        print(
            f"Max value for Larry Arnn's face: {larry_max_val}"
        )  # TODO ineffective in finding Larry Arnn's face
        larry_threshold = 0.6
        if larry_max_val >= larry_threshold:
            larry_template_w, larry_template_h = larry_template.shape[::-1]
            larry_top_left = larry_max_loc
            larry_bottom_right = (
                larry_top_left[0] + larry_template_w,
                larry_top_left[1] + larry_template_h,
            )

            image = cv2.rectangle(
                image, larry_top_left, larry_bottom_right, (0, 0, 0), -1
            )
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Save image to see if Larry Arnn's face was detected
            # cv2.imwrite("test.png", image)
            print("Larry Arnn's face has been replaced with black pixels.")

        # Load the certificate template and input image
        certificate_template = cv2.imread(
            certificate_template_path, cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            error_message = f"Error: Unable to load image at {image_path}"
            print(error_message)
            with open("errors.txt", "a") as error_file:
                error_file.write(f"{error_message}\n")
            return
        # Perform template matching to find the certificate in the image
        result = cv2.matchTemplate(
            gray_image, certificate_template, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        # Set a threshold to detect the template
        threshold = 0.7
        if max_val >= threshold:
            template_w, template_h = certificate_template.shape[::-1]
            top_left = max_loc
            bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
            certificate_area = (top_left, bottom_right)
            image = crop_image_above_certificate(
                image, certificate_area, guessed_gender, married
            )
        else:
            print(f"Certificate not found in {image_path}")
            image = crop_image_based_on_conditions(
                image_path, output_dir, guessed_gender, married
            )
        # Load Haar Cascade classifier for face detection
        haarcascades_path = resource_path(
            "resources/haarcascades/haarcascade_frontalface_default.xml"
        )
        face_cascade = cv2.CascadeClassifier(haarcascades_path)
        if face_cascade.empty():
            raise FileNotFoundError(
                "Haar Cascade XML file not found or failed to load."
            )

        # Detect faces
        faces = face_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            print("No faces found in the image.")
            return

        # Find the largest detected face (not including Larry Arnn)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3], default=None)
        if largest_face is None:
            print("No valid face found above the certificate.")
            return

        # Expand the bounding box to include the upper body
        x, y, w, h = largest_face
        expansion_factor = 2  # How much to expand the height
        expanded_y = max(0, y - h // 2)  # Expand upwards by half the face height
        expanded_h = min(
            image.shape[0] - expanded_y, int(h * expansion_factor)
        )  # Expand height by 2x the face height
        expanded_x = max(0, x - w // 4)  # Expand width slightly to capture shoulders
        expanded_w = min(
            image.shape[1] - expanded_x, int(w * 1.5)
        )  # Expand width by 1.5x the face width

        # Crop the expanded region (upper body)
        cropped_image = image[
            expanded_y : expanded_y + expanded_h, expanded_x : expanded_x + expanded_w
        ]

        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        os.makedirs(output_dir, exist_ok=True)
        pil_image.save(output_path, dpi=(300, 300))

        print(f"Image saved to {output_path}\n")
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")


def match_and_save_images(csv_data, images_folder, output_folder):
    """
    Matches each data entry in the CSV file with an image from the folder of images using their first and last name,
    and then saves the image in the format "first_name_last_name_lookup_id".

    Parameters:
        csv_data (pd.DataFrame): The validated CSV data.
        images_folder (str): The folder containing the images.
        output_folder (str): The folder to save the matched images.
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    matches = 0

    # Initialize the gender detector
    d = gender.Detector()

    # Iterate over each row in the CSV data with a progress bar
    for _, row in tqdm(
        csv_data.iterrows(), total=csv_data.shape[0], desc="Processing images"
    ):
        first_name = row["First Name"].strip()
        last_name = row["Last Name"].strip()
        lookup_id = row["LookupID"].strip()

        # Construct the image file name
        image_file_name = f"{first_name}_{last_name}"
        guessed_gender = d.get_gender(first_name)
        print(f"\nGuessed: {first_name} to be {guessed_gender}")
        image_file_path = None

        # Search for the image file in the images folder
        for file in os.listdir(images_folder):
            if image_file_name in file or (
                guessed_gender == "female" and last_name in file
            ):
                potential_path = Path(images_folder) / file
                if potential_path.exists():
                    image_file_path = potential_path
                    out_name = file
                    break

        if image_file_path:
            # Load the image
            image = Image.open(image_file_path)

            # Construct the output file name
            output_file_name = f"{out_name.replace('.png', '')}_{lookup_id}.png"
            output_file_path = Path(output_folder) / output_file_name

            # Save the image in the desired format
            image.save(output_file_path, dpi=(300, 300))

            print(f"Image saved to {output_file_path}")
            matches += 1
        else:
            print(f"No matching image found for {first_name} {last_name}")

    print(f"Total matches: {matches}")


def extract_names(filename):
    names = set()  # Use a set to avoid duplicates

    with open(filename, mode="r", newline="", encoding="ISO-8859-1") as file:
        reader = csv.reader(file)
        for row in reader:
            combined_row = " ".join(row)
            found_names = re.findall(
                r"([A-Z][a-z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-z]+)?)\s-\s(\d+)", combined_row
            )
            for name, _ in found_names:
                name_parts = name.split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = name_parts[-1]
                    names.add(f"{first_name} {last_name}")  # Add to set

    return names


def main():
    # TODO: based on how many last names, find the corressponding image and crop the face, make a flow chart
    """
    Main function to process PC Members photos.
    Depending on the command line arguments, it either extracts and filters member data, processes images,
    or matches images to corresponding members.
    """
    # Define the CSV file containing member information
    filename = "PC Members Photos 20240916.csv"

    # Extract names from the CSV and store them
    extracted_names = extract_names(filename)

    # Read and validate CSV file containing PC member details
    csv_file_path = r"CCA PC members for photos.csv"
    valid_data = read_and_validate_csv(csv_file_path)

    if valid_data is not None:
        # Filter out entries that match the extracted names
        filtered_data = valid_data[
            ~valid_data.apply(
                lambda row: f"{row['First Name']} {row['Last Name']}"
                in extracted_names,
                axis=1,
            )
        ]
        print(f"Filtered out {len(valid_data) - len(filtered_data)} entries.")

        # Classify members into groups based on certain criteria
        groups = classify_members(filtered_data)

        # Display group classification details
        for group_name, members in groups.items():
            print(f"Number of {group_name.replace('_', ' ').title()}: {len(members)}")
            print(f"\n{group_name.replace('_', ' ').title()}:")

            # Print the first three members in each group
            for i, member in enumerate(
                members[:3]
            ):  # Only displaying the first three members
                print(f"Name: {member[0]} {member[1]}, {member[3]} ({member[4]})")

    # Path for storing the cropped face images
    output_folder = Path.home() / "Downloads" / "cropped_faces"

    # Process based on command-line arguments
    if sys.argv[1] == "2":
        # Extract metadata from images in the folder
        folder_path = "PC_Photos_v3/"  # Folder containing member images
        metadata_list = process_images_in_folder(folder_path)

        # Print or process each image metadata
        for image_name, names in tqdm(metadata_list, desc="Processing images"):
            if names:
                print(f"\nNames in: {image_name}: {names}")
                process_image(Path(folder_path) / image_name, names)

    elif sys.argv[1] == "3":
        # Example: Image crop and find face examples
        images_dir = "images/"
        select_images(images_dir, [("John", "Doe"), ("Jane", "Doe")])
        # test_crop_image_above_certificate()
        # return

    elif sys.argv[1] == "4":
        # Match and save images to members
        match_and_save_images(valid_data, output_folder, str(output_folder) + "_id")


if __name__ == "__main__":
    main()
