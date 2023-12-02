from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd

model = load_model("converted_keras/keras_model.h5", compile=False)

class_names = open("converted_keras/labels.txt", "r").readlines()

image_dir = "test/test/images"

predictions = []

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]

        predictions.append({
            'image_id': os.path.splitext(filename)[0],
            'label': class_name[2:].strip()
        })

df = pd.DataFrame(predictions)

output_csv_path = "result.csv"
df.to_csv(output_csv_path, index=False)

print("Predictions saved to:", output_csv_path)
