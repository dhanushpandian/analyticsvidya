from text_generation import Client
import base64
import csv
import os
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

API_TOKEN = os.getenv("API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceM4/idefics-80b-instruct"
DECODING_STRATEGY = "Greedy"

generation_args = {
    "max_new_tokens": 256,
    "repetition_penalty": 1.0,
    "stop_sequences": ["<end_of_utterance>", "\nUser:"],
}

if DECODING_STRATEGY == "Greedy":
    generation_args["do_sample"] = False
elif DECODING_STRATEGY == "Top P Sampling":
    generation_args["temperature"] = 1.
    generation_args["do_sample"] = True
    generation_args["top_p"] = 0.95


def map_class_to_value(predicted_class):
    switch = {
        "crack": 1,
        "scratch": 2,
        "flat tyre": 3,
        "dent": 4,
        "glass shatter": 5,
        "lamp broken": 6,
    }
    # Strip leading and trailing spaces before mapping
    return switch.get(predicted_class.strip().lower(), 0)


def generate_text_for_image(image_path):
    with open(image_path, "rb") as image_file:
        # Read the image file and encode it as base64
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # Include the base64-encoded image in the query
    QUERY = f"User: reply in one word, What is the most suitable word to describe image from the options: crack, scratch, flat tyre, dent, glass shatter, lamp broken?![](data:image/jpeg;base64,{image_base64})<end_of_utterance>\nAssistant:"

    client = Client(
        base_url=API_URL,
        headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
    )

    # Retry mechanism with a maximum of 3 attempts
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = client.generate(prompt=QUERY, **generation_args)
            # Extract the generated text from the response
            generated_tokens = response.details.generated_tokens
            if isinstance(generated_tokens, int):
                return ""  # If generated_tokens is an integer, return an empty string
            generated_text = generated_tokens[0]['text'].strip(
            ) if generated_tokens else ""
            return generated_text
        except requests.exceptions.ReadTimeout as e:
            print(f"Attempt {attempt + 1}/{max_attempts} failed. Retrying...")
            if attempt == max_attempts - 1:
                raise e  # If all attempts fail, raise the exception

    return ""  # Return an empty string if all attempts fail


# Example: Path to the directory containing image files
image_directory_path = "test/test/images"
output_csv_path = "res.csv"

# Get a list of image paths in the directory
image_paths = [os.path.join(image_directory_path, filename) for filename in os.listdir(
    image_directory_path) if filename.endswith(".jpg")]

# Write CSV header
with open(output_csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["image_id", "label"])

# Loop through each image and generate text
for image_path in image_paths:
    # Extract the image name from the path
    image_name = os.path.basename(image_path)

    try:
        # Generate text for the local image
        output_text = generate_text_for_image(image_path)

        # Map the predicted class to a numerical value
        predicted_value = map_class_to_value(output_text)

        # Print a warning if the predicted value is 0
        if predicted_value == 0:
            print(
                f"Warning: Predicted value is 0 for image '{image_name}'. Please investigate.")

        # Print the actual response details for debugging
        print(
            f"Output for image '{image_name}':\nPredicted Value: {predicted_value}")
        print("Response Details:", response.details, "\n")

        # Write the results to CSV
        with open(output_csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([image_name, predicted_value])

    except Exception as e:
        print(f"Error processing image '{image_name}': {e}")
