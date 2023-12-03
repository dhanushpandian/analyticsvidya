import os
import shutil
import pandas as pd

# Load CSV file
csv_file_path = 'testrun.csv'
df = pd.read_csv(csv_file_path)

# Set the source and destination directories
source_directory = '/home/dash/dash/analyticsvidya/test/test/images'
destination_directory = 'train/seg'

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    # Get the image name from the CSV file
    # Assuming the column name is 'Image Name'
    image_name = row['Image Name'].strip()

    # Construct the source path
    source_path = os.path.join(source_directory, image_name)

    # Check if the source file exists
    if not os.path.exists(source_path):
        print(f"Source file '{source_path}' not found. Skipping...")
        continue

    # Construct the destination path
    # Assuming the column name is 'Predicted Class'
    predicted_class = row['Predicted Class'].strip().lower()
    destination_path = os.path.join(
        destination_directory, predicted_class, image_name)

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Copy the file to the destination
    shutil.copyfile(source_path, destination_path)
    print(f"Successfully copied '{source_path}' to '{destination_path}'")
