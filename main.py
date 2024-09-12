import pandas as pd


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
        required_columns = [
            "First Name",
            "Last Name",
            "LookupID",
        ]

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


def main():
    # Example usage:
    csv_file_path = r"CCA PC members for photos.csv"
    valid_data = read_and_validate_csv(csv_file_path)

    # If valid data is found, display it
    if valid_data is not None:
        print(valid_data)


if __name__ == "__main__":
    main()
