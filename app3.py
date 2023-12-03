from text_generation import Client
import base64
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


def generate_text_for_image(image_path):
    with open(image_path, "rb") as image_file:
        # Read the image file and encode it as base64
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # Include the base64-encoded image in the query
    QUERY = f"User: reply in one word ,What is the most suitable word to describe image from the options : crack, scratch, flat tyre ,dent, glass shatter, lamp broken?![](data:image/jpeg;base64,{image_base64})<end_of_utterance>\nAssistant:"

    client = Client(
        base_url=API_URL,
        headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
    )
    generated_text = client.generate(prompt=QUERY, **generation_args)
    return generated_text


# Example: Path to the local image file
image_path = "test/test/images/11943.jpg"

# Generate text for the local image
output_text = generate_text_for_image(image_path)
print(f"Output for image '{image_path}':\n{output_text}\n")
