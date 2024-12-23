# IMPACT Project

## Overview

The IMPACT project is designed to process images and CSV data to extract, validate, and classify member information. It includes functionalities for detecting and cropping faces in images, extracting metadata, and matching images to corresponding members based on their names.

## Features

- **CSV Data Validation**: Reads and validates CSV files to ensure required columns are present and data is clean.
- **Face Detection and Cropping**: Detects faces in images, crops them, and saves the cropped images with 300 DPI.
- **Metadata Extraction**: Extracts 'Title' metadata from Windows file properties.
- **Name Extraction**: Extracts names from image titles and filenames.
- **Member Classification**: Classifies members into various groups based on their titles, last names, and gender.
- **Image Matching**: Matches images to members based on their names and saves the images with appropriate filenames.

## Requirements

- Python 3.x
- Required Python packages:
  - pandas
  - numpy
  - opencv-python
  - Pillow
  - tqdm
  - gender-guesser
  - pywin32 (for Windows metadata extraction)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/impact.git
   cd impact
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Arguments

The script can be run with different command-line arguments to perform various tasks:

1. **Extract and Filter Member Data**:
   ```sh
   python main.py 1
   ```

2. **Extract Metadata from Images**:
   ```sh
   python main.py 2
   ```

3. **Image Crop and Face Detection**:
   ```sh
   python main.py 3
   ```

4. **Match and Save Images to Members**:
   ```sh
   python main.py 4
   ```

### Example

To process images in the `PC_Photos_v3/` folder and extract metadata:
```sh
python main.py 2
```

## Functions

### `read_and_validate_csv(csv_file_path)`
Reads and validates a CSV file to ensure required columns are present and data is clean.

### `detect_and_crop_faces(image_path, output_dir)`
Detects faces in an image, crops them, and saves the cropped images with 300 DPI.

### `extract_title_from_windows_properties(image_path)`
Extracts the 'Title' metadata from Windows file properties.

### `clean_title(title)`
Removes unwanted words and phrases from the title.

### `clean_title_around_larry_arnn(title)`
Cleans the title by extracting up to 4 words before and after 'Larry Arnn'.

### `split_and_return_names(names)`
Splits names containing '&' into separate names and returns a list of individual names.

### `extract_names_from_title(title)`
Extracts names from the title while excluding 'Larry Arnn'.

### `extract_names_from_filename(file_name)`
Extracts names from the file name.

### `process_images_in_folder(folder_path)`
Processes images in a folder and extracts the 'Title' metadata.

### `load_data(csv_path)`
Loads data from a CSV file.

### `determine_gender(title)`
Determines gender based on the title.

### `classify_members(data)`
Classifies members into various groups based on their titles, last names, and gender.

### `select_images(images_dir, names)`
Selects images from a directory based on the provided names.

### `process_images(file_paths, names)`
Processes a list of image file paths.

### `resource_path(relative_path)`
Gets the absolute path to a resource.

### `process_image(input_image_path, names)`
Processes an image and performs cropping and face detection.

### `crop_image_based_on_conditions(image_path, output_dir, guessed_gender, married)`
Crops the image based on the guessed gender and marital status.

### `crop_image_above_certificate(image, certificate_area, guessed_gender, married)`
Crops the image above the certificate based on the guessed gender and marital status.

### `test_crop_image_above_certificate()`
Tests the `crop_image_above_certificate` function with example images.

### `detect_and_crop_face_above_certificate(image_path, output_dir, certificate_template_path, names_string, guessed_gender, married)`
Detects the certificate in the image, finds Larry Arnn's face, replaces it with black pixels, finds the face above it, crops the face, and saves it with 300 DPI.

### `match_and_save_images(csv_data, images_folder, output_folder)`
Matches each data entry in the CSV file with an image from the folder of images and saves the image in the format "first_name_last_name_lookup_id".

### `extract_names(filename)`
Extracts names from a CSV file.

### `main()`
Main function to process PC Members photos based on command-line arguments.

## License

This project is licensed under the MIT License.
