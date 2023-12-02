import pandas as pd
import os
import shutil

# Load Excel file with image names and labels
excel_file_path = 'train.csv'
df = pd.read_csv(excel_file_path)

# Directory containing the images
images_directory = 'images'

# Output directory where segregated folders will be created
output_directory = 'seg'

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    image_name = row['filename']  
    label = row['label']  

    # Create a directory for the label if it doesn't exist
    label_directory = os.path.join(output_directory, str(label))
    os.makedirs(label_directory, exist_ok=True)

    # Construct paths
    image_source_path = os.path.join(images_directory, image_name)
    image_destination_path = os.path.join(label_directory, image_name)

    # Move the image to the corresponding label directory
    shutil.move(image_source_path, image_destination_path)

print("Images segregated based on labels.")
