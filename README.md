# IMPACT: Image Matching and Processing for Cropped Transformation

## Project Overview
The **IMPACT** project (Image Matching and Processing for Cropped Transformation) is designed to automate the process of matching individuals' data from a CSV file with image metadata, cropping the matched images to show only the person’s face, and renaming the cropped image files using a unique lookup ID provided in the CSV. This tool will streamline image processing by making the workflow efficient and consistent, allowing users to work with clean, properly formatted images associated with the correct individuals.

## Key Features
- **CSV Data Processing**: Import and read data from a CSV file, which includes individuals’ names and a unique lookup ID.
- **Image Metadata Matching**: Search through a folder of images and match each person by name using the metadata embedded in the images.
- **Face Detection and Cropping**: Automatically detect the face of the matched person and crop the image to display only the face.
- **Image Renaming and Export**: Save the cropped image with a new filename based on the unique lookup ID from the CSV file.

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Workflow](#project-workflow)
4. [Tasks](#tasks)
5. [Contributing](#contributing)
6. [License](#license)

---

## Installation

1. **Clone the Repository**  
   Run the following command to clone the project:
   ```bash
   git clone https://github.com/yourusername/IMPACT.git
   cd IMPACT
   ```

2. **Set Up the Environment**  
   Create a virtual environment and install the required dependencies:
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. **Dependencies**  
   The project uses the following Python libraries:
   - `pandas` (for handling CSV files)
   - `opencv-python` (for image processing)
   - `Pillow` (for image manipulation)
   - `face_recognition` (for face detection and cropping)

4. **Input Requirements**  
   - A CSV file with individual data, including names and lookup IDs.
   - A folder containing images with metadata fields including individual names.

---

## Usage

To use the **IMPACT** tool, follow these steps:

1. Ensure the CSV file and the folder of images are properly prepared.
2. Run the script with the following command:
   ```bash
   python impact.py --csv_path "path/to/csvfile.csv" --image_folder "path/to/image_folder"
   ```

3. The script will output the cropped and renamed images in the `output` directory.

---

## Project Workflow

### 1. CSV Data Import
- Load the CSV file that contains each individual’s details, including:
  - **Name** (to be matched with image metadata)
  - **Unique Lookup ID** (used to rename the final processed image)
  
### 2. Image Metadata Matching
- Read through the folder of images and extract metadata for each file.
- Use the names in the metadata to match with the names in the CSV file.
- If a match is found, proceed to the cropping step. If not, log the unmatched images for review.

### 3. Face Detection and Cropping
- For each matched image, detect the person’s face using a face recognition library.
- Automatically crop the image to only include the person's face while maintaining a reasonable margin.
  
### 4. Rename and Export
- Rename the cropped image using the unique Lookup ID from the CSV file.
- Save the new image in the `output` folder, maintaining the original image format (e.g., `.jpg`, `.png`).

---

## Tasks

Below is a comprehensive list of tasks for the **IMPACT** project:

1. **Set Up Environment**  
   - Install necessary libraries and set up the development environment.
   - Ensure that dependencies are properly configured to handle CSV and image processing.

2. **CSV File Handling**  
   - Read and parse the CSV file to extract relevant data (names and lookup IDs).
   - Ensure data validation and handle cases of missing or malformed data.

3. **Image Metadata Extraction**  
   - Search through the image folder to extract metadata (name field from EXIF or similar).
   - Match image metadata with the names from the CSV file.
   - Handle any cases where no match is found by logging unmatched files.

4. **Face Detection**  
   - Implement face detection using the `face_recognition` library.
   - Handle multiple faces in one image by selecting the largest or first detected face.

5. **Image Cropping**  
   - Crop the image to display only the face, with some padding for clarity.
   - Test cropping on various image sizes and formats to ensure accuracy.

6. **File Renaming and Export**  
   - Rename each cropped image using the lookup ID from the CSV file.
   - Save the newly cropped and renamed images in the `output` folder.
   - Ensure image format and quality are preserved.

7. **Error Handling and Logging**  
   - Implement logging for unmatched names and any errors encountered during processing.
   - Provide a summary of errors or skipped items at the end of the run.

8. **Testing**  
   - Test the entire process with sample data to ensure accuracy.
   - Validate face cropping and file renaming functions work as expected.
   
---

## Contributing

We welcome contributions to **IMPACT**! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request and describe your changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact

For questions or issues, feel free to reach out via the GitHub repository's issue tracker.

