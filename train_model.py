import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
)
from tensorflow.keras.utils import image_dataset_from_directory
import os
import sys

# --- 1. Setup & Configuration ---

# !! IMPORTANT: Change this path to where you unzipped the dataset !!
# Use forward slashes or double backslashes for Windows paths
# e.g., 'C:/Users/YourUser/Downloads/RF_Signal_Data'
# or 'C:\\Users\\YourUser\\Downloads\\RF_Signal_Data'
DATA_DIR = r"C:\Users\Keshab\Downloads\datasets\RF_Signal_Data"

# Model parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
EPOCHS = 20 # 20 is a good starting point

# Check if data directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory not found at '{DATA_DIR}'")
    print("Please download the Kaggle dataset and update the DATA_DIR variable.")
    sys.exit()

# --- 2. Load and Preprocess Data ---
print("Loading and preprocessing data...")

try:
    train_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
except Exception as e:
    print(f"\nError loading data: {e}")
    print("Please ensure the DATA_DIR path is correct and points to the folder")
    print("containing the subfolders for each class (e.g., 'Bluetooth', 'WiFi').")
    sys.exit()

# Get class names
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"--- Found {NUM_CLASSES} classes: {class_names} ---")

# --- 3. Build the Simple CNN Model ---
print("Building the CNN model...")
model = Sequential([
    Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# --- 4. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- 5. Train the Model ---
print("\n--- Starting Model Training ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("--- Training Finished ---")

# --- 6. Save the Model and Class Names (The Goal!) ---

# Save the model
MODEL_SAVE_PATH = 'rf_model.h5'
model.save(MODEL_SAVE_PATH)
print(f"\n--- Model saved to {MODEL_SAVE_PATH} ---")

# Save the class names
CLASSES_SAVE_PATH = 'class_names.txt'
with open(CLASSES_SAVE_PATH, 'w') as f:
    for item in class_names:
        f.write(f"{item}\n")
print(f"--- Class names saved to {CLASSES_SAVE_PATH} ---")

print("\nPart 2 is complete. You have your model!")