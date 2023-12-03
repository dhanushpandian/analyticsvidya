from text_generation import Client
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

API_TOKEN = os.getenv("API_TOKEN")


API_URL = "https://api-inference.huggingface.co/models/HuggingFaceM4/idefics-80b-instruct"
DECODING_STRATEGY = "Greedy"
QUERY = "User: What is in this image tell me from the options that follow only crack, scratch, tire flat,dent, glass shatter, lamp broken?![](test/test/images/7201.jpg)<end_of_utterance>\nAssistant:"

client = Client(
    base_url=API_URL,
    headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
)
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

generated_text = client.generate(prompt=QUERY, **generation_args)
print(generated_text)
