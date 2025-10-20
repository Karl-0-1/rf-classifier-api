import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
from pydantic import BaseModel
import io
from typing import Optional

# --- 1. Define Application and Response Models ---

app = FastAPI(
    title="RF Signal Classifier API",
    description="A simple API that uses a CNN model to classify RF signal images."
)

class ClassificationResponse(BaseModel):
    filename: str
    content_type: str
    predicted_class: str
    confidence: float
    error: Optional[str] = None

# --- 2. Load Model and Class Names on Startup ---

MODEL_PATH = 'rf_model.h5'
CLASSES_PATH = 'class_names.txt'
IMG_HEIGHT = 128
IMG_WIDTH = 128

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

print("Loading class names...")
with open(CLASSES_PATH, 'r') as f:
    CLASS_NAMES = [line.strip() for line in f]
print(f"Classes loaded: {CLASS_NAMES}")


# --- 3. Preprocessing Helper Function ---

def preprocess_image(file_bytes: bytes):
    """
    Loads image from bytes, resizes, and prepares for the model.
    """
    # Load the image from bytes
    img = image.load_img(io.BytesIO(file_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
    # Convert to array
    img_array = image.img_to_array(img)
    # Rescale (normalize)
    img_array = img_array / 255.0
    # Create a batch (add dimension)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# --- 4. Define API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the RF Signal Classifier API. Go to /docs to test."}


@app.post("/classify/", response_model=ClassificationResponse)
async def classify_signal(file: UploadFile = File(...)):
    """
    Upload an image of an RF signal to classify it.
    """
    try:
        # Read the file content
        contents = await file.read()

        # Preprocess the image
        image_batch = preprocess_image(contents)

        # Make prediction
        predictions = model.predict(image_batch, verbose=0)

        # Get the top prediction
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(predictions[0]))

        return ClassificationResponse(
            filename=file.filename,
            content_type=file.content_type,
            predicted_class=predicted_class_name,
            confidence=confidence
        )

    except Exception as e:
        return ClassificationResponse(
            filename=file.filename,
            content_type=file.content_type,
            predicted_class="",
            confidence=0.0,
            error=f"Error processing file: {e}"
        )

if __name__ == "__main__":
    print("Starting API server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)