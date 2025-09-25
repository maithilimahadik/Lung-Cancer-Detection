# Lung-Cancer-Detection

This project implements a deep learning solution for classifying lung images (X-ray/CT) into three categories: **Normal**, **Malignant** (Cancerous), or **Benign** (Non-Cancerous). The system utilizes **Transfer Learning** with a **pre-trained ResNet50 model** to achieve image classification and is deployed via an interactive web application.

## Core Features

  * **Deep Learning Model:** Built on a **pre-trained ResNet50** model using the Keras API for efficient feature extraction.
  * **Transfer Learning:** Weights of the ResNet50 base were **frozen** to leverage existing knowledge, and a custom classification head was added and trained.
  * **Data Preprocessing:** Lung cancer images were preprocessed and normalized using **ImageDataGenerator**.
  * **Web Interface:** A **Flask-based web application** provides a user-friendly interface for doctors/researchers to upload images and view real-time predictions and confidence scores.
  * **Comprehensive Evaluation:** Model performance was evaluated using **Loss, Accuracy, Classification Report, and a Confusion Matrix**.

## Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | **TensorFlow, Keras** | Core machine learning framework and high-level API. |
| **Model** | **ResNet50** | Used as the base for transfer learning. |
| **Data/Math** | **NumPy, Matplotlib, Seaborn** | Handling array operations, plotting, and visualization of metrics. |
| **Web Framework** | **Flask** | Backend for the web application and REST API endpoint `/predict`. |
| **Image Handling** | **PIL (Pillow)** | Used in the Flask app for reading and resizing uploaded images. |

## Setup and Run Instructions

### Prerequisites

You will need Python (3.x) and the following libraries:

```bash
pip install tensorflow==2.* flask numpy pillow scikit-learn matplotlib seaborn
```

### 1\. Model Training (Optional, if `lung_model.h5` is missing)

To train the model, you typically run the Jupyter Notebook:

1.  **Download Dataset:** Acquire the "The IQ-OTHNCCD lung cancer dataset" and organize the training, validation, and test images into corresponding directories (as specified in the notebook).
2.  **Run Notebook:** Execute all cells in `Python_Project.ipynb`.
3.  **Output:** The fully trained model will be saved as `lung_model.h5`.

### 2\. Running the Flask Web Application

Ensure the trained model file (`lung_model.h5`) is in the same directory as `app.py`.

1.  **Run the Server:**
    ```bash
    python app.py
    ```
2.  **Access Application:** The application will start on your local machine. Open your web browser and navigate to the address provided in the terminal (usually `http://127.0.0.1:5000/`).
3.  **Predict:** Use the interface to upload a lung image. The `/predict` endpoint will process the image, make a prediction, and return the result and confidence.
