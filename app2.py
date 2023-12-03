from text_generation import Client
from PIL import Image
from io import BytesIO
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


def generate_text_for_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    # Process the image as needed
    # ...

    # Generate text
    QUERY = f"User: What is in this image?![]({image})<end_of_utterance>\nAssistant:"
    client = Client(
        base_url=API_URL,
        headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
    )
    generated_text = client.generate(prompt=QUERY, **generation_args)
    return generated_text


# Example: Path to the file containing images
file_path = "path/to/your/image/file.jpg"

# Read image bytes from the file
with open(file_path, "rb") as file:
    image_bytes = file.read()

# Process each image and generate text
output_text = generate_text_for_image(image_bytes)
print(f"Output for image '{file_path}':\n{output_text}\n")
