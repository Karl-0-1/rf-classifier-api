# End-to-End Deep Learning API for RF Signal Classification

   

This project is a complete, end-to-end MLOps workflow demonstrating how to train a robust deep learning model and serve it as a high-performance, containerized web API.

The project trains a Convolutional Neural Network (CNN) to classify 21 different types of radio frequency (RF) signals (like Bluetooth, WiFi, LoRa, etc.) from their spectrogram images. The trained model is then exposed via a FastAPI endpoint and packaged with Docker for easy deployment.

---

## 🚀 Core Features

* **Robust CNN Training:** The `train_model.py` script is built for performance and accuracy.
    * **GPU-Aware:** Automatically detects and uses an available NVIDIA GPU for training.
    * **Data Augmentation:** Uses `RandomFlip`, `RandomRotation`, and `RandomZoom` to prevent overfitting and create a more generalized model.
    * **Class Imbalance Fix:** Automatically calculates and applies **class weights** to force the model to pay more attention to rare signal types (fixing the "always predicts one class" problem).
    * **Smart Callbacks:** Uses `ModelCheckpoint` (saves only the *best* model), `EarlyStopping` (prevents wasted time), and `ReduceLROnPlateau` (helps the model find a better solution).
* **High-Speed API:**
    * Built with **FastAPI** for asynchronous, high-performance predictions.
    * Provides a simple `/classify/` endpoint that accepts an image upload.
    * Includes a `/docs` page with interactive (Swagger UI) documentation.
* **Containerized Deployment:**
    * A multi-stage `Dockerfile` creates an efficient, production-ready image.
    * `requirements.txt` and a `.dockerignore` file ensure a clean and reproducible build.
* **Large File Support:**
    * Uses **Git LFS** (Large File Storage) to correctly handle the large `rf_model.h5` model file, which is too big for standard GitHub tracking.

---

## 🛠️ Tech Stack

* **Python 3.11**
* **Model:** TensorFlow 2.x / Keras
* **API:** FastAPI
* **Server:** Uvicorn
* **Containerization:** Docker
* **Utilities:** Scikit-learn (for class weights), Numpy
* **Version Control:** Git & Git LFS

---

## 📁 Repository Structure

```
rf-classifier-api/
├── .dockerignore       # Tells Docker which files to ignore for a clean build
├── .gitignore          # Tells Git which files to ignore (like venv)
├── .gitattributes      # Configures Git LFS to track .h5 files
├── Dockerfile          # Instructions for building the Docker container
├── README.md           # You are here!
├── class_names.txt     # A list of all signal classes the model can predict
├── main.py             # The FastAPI server code
├── requirements.txt    # All Python dependencies
├── rf_model.h5         # The (LFS-tracked) trained model
└── train_model.py      # The script to train the model from scratch
```

---

## 🏁 How to Run This Project

This guide covers all steps, from training your own model to running the pre-built API.

### Part 1: How to Train the Model From Scratch (Optional)

You only need to do this if you want to re-train the model. This repository already includes a pre-trained `rf_model.h5`.

**Step 1.1: Get the Data**
1.  Go to the Kaggle dataset: [RF Signal Classification (21 Classes)](https://www.kaggle.com/datasets/halcy0nic/radio-frequecy-rf-signal-image-classification)
2.  Download and unzip the data.
3.  You will get a folder named `RF_Signal_Data` (or similar). Note the full path to this folder (e.g., `C:\Users\YourUser\Downloads\RF_Signal_Data`).

**Step 1.2: Set Up the Environment**
```bash
# Clone the repository (if you haven't already)
git clone [https://github.com/YourUsername/rf-classifier-api.git](https://github.com/YourUsername/rf-classifier-api.git)
cd rf-classifier-api

# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Install all required libraries
pip install -r requirements.txt
```

**Step 1.3: Run the Training Script**
1.  Open `train_model.py` in your code editor.
2.  Change the `DATA_DIR` variable (line 20) to point to the `RF_Signal_Data` folder you downloaded:
    ```python
    DATA_DIR = 'C:/path/to/your/RF_Signal_Data'
    ```
3.  Run the script from your activated terminal. This will automatically use your GPU if available.
    ```bash
    python train_model.py
    ```
4.  This will run for several minutes/hours and will create a new `rf_model.h5` and `class_names.txt` file, overwriting the old ones.

---

### Part 2: How to Run the Pre-Built API

This is the main way to use the project.

**Step 2.1: Prerequisites**
* [Git](https://git-scm.com/downloads)
* [Git LFS](https://git-lfs.github.com/) (Large File Storage)
* [Python 3.10+](https://www.python.org/downloads/)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (must be running in the background)

**Step 2.2: Clone the Repository (with LFS)**
Because the `rf_model.h5` file is stored with Git LFS, you must have it installed.

```bash
# 1. Install LFS (only need to do this once per computer)
git lfs install

# 2. Clone the repository
# !!! REPLACE THIS URL WITH YOUR OWN REPO URL !!!
git clone [https://github.com/Karl-0-1/rf-classifier-api.git](https://github.com/Karl-0-1/rf-classifier-api.git)
cd rf-classifier-api

# 3. Pull the large model file
# This downloads the rf_model.h5 file from LFS storage
git lfs pull
```

---

### Part 3: Choose Your Method (Docker or Local)

#### Method A: Run with Docker (Recommended)

This is the simplest, cleanest, and most reliable way to run the API.

1.  **Build the Image:** From the root of the project folder, run:
    ```bash
    docker build -t rf-classifier-api .
    ```

2.  **Run the Container:**
    ```bash
    docker run -d -p 8000:8000 --name rf_api rf-classifier-api
    ```
    * `-d`: Runs the container in detached (background) mode.
    * `-p 8000:8000`: Maps your computer's port 8000 to the container's port 8000.
    * `--name rf_api`: Gives your running container an easy-to-remember name.

The API is now running! **Skip to Part 4.**

#### Method B: Run Locally (For Development)

Use this method if you want to run the server without Docker.

1.  **Activate Environment:** Make sure you are in the project folder and your virtual environment is active.
    ```cmd
    .\venv\Scripts\activate
    ```
2.  **Install Dependencies:** (If you haven't already from Part 1)
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the API Server:**
    ```bash
    python main.py
    ```
The server will start, and you'll see a log message from Uvicorn.

---

### Part 4: How to Use the API

Once the server is running (from either Docker or locally), you can test it.

1.  Open your web browser and go to the interactive docs:
    **`http://127.0.0.1:8000/docs`**

2.  You will see the FastAPI documentation page.

3.  Click on the `POST /classify/` endpoint to expand it.

4.  Click the **"Try it out"** button.

5.  Click the **"Choose File"** button and upload an RF signal image (you can use one from the Kaggle dataset).

6.  Click the **"Execute"** button.

### Example Response

You will get a JSON response back with the model's prediction:

```json
{
  "filename": "bluetooth_image_01.png",
  "content_type": "image/png",
  "predicted_class": "Bluetooth",
  "confidence": 0.9984,
  "error": null
}
```

---

### Part 5: How to Push to GitHub (with Git LFS)

If you re-trained your model or made changes, here is how you push your code (and the large model file) to GitHub.

```bash
# 1. Install Git LFS (if you haven't)
git lfs install

# 2. Tell LFS to track all .h5 files (only need to do once)
git lfs track "*.h5"

# 3. Add the LFS tracking file
git add .gitattributes

# 4. Add all your other project files
# (Git will ignore "venv" because of your .gitignore)
git add .

# 5. Make your "commit" (a snapshot of your code)
git commit -m "Add new trained model"

# 6. Push your code to GitHub
# This will upload the small files first, then the LFS file
git push
```
