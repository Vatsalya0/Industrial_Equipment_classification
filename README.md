# Industrial_Equipment_classification
This project uses a Convolutional Neural Network (CNN) to classify industrial equipment into two categories: **defective** and **non-defective**. 
The model achieves high accuracy (98%).  
This classification automates quality control processes, reduces manual intervention, and enhances productivity.

## Problem Statement
In industrial manufacturing, identifying defective equipment is a critical yet challenging task that often relies on manual inspections, which are time-consuming, error-prone, and expensive. The aim of this project is to develop an automated solution using a Convolutional Neural Network (CNN) to classify industrial equipment into two categories: defective and non-defective. This will enhance efficiency, reduce operational costs, and ensure higher accuracy in quality control processes.

---

## Requirements

### Hardware
- System with at least **8GB RAM**.
- GPU-enabled system (optional, for faster training).

### Software
- **Python** (version 3.8 or higher).
- Libraries: 
  - TensorFlow (2.x or higher)
  - NumPy
  - Matplotlib
  - OpenCV

---

## Dataset
- **Training Data**: 1187 images categorized into defective and non-defective classes.
- **Test Data**: 254 images, distributed into the same categories.
- **Image Details**: 
  - Format: RGB
  - Resolution: 64x64 pixels

---

## Workflow
### 1. Data Loading and Preprocessing
- Load images from directories using TensorFlow's `image_dataset_from_directory`.
- Resize images to **64x64 pixels** and batch them into size 32.

### 2. CNN Model Architecture
- **Input Layer**: Accepts images of shape 64x64x3 (RGB format).
- **Convolutional Layers**:
  - 2 Conv2D layers with 64 filters each, ReLU activation, followed by MaxPooling.
- **Dropout Layer**: Prevents overfitting by deactivating 50% of neurons.
- **Flatten Layer**: Converts 2D data into 1D.
- **Dense Layers**:
  - One hidden layer with 128 neurons and ReLU activation.
  - Output layer with 2 neurons and softmax activation (for 2 classes).

### 3. Model Compilation
- **Optimizer**: RMSprop
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

### 4. Model Training
- Train the model for **30 epochs** with the training dataset.
- Achieved **97.97% training accuracy**.

### 5. Model Evaluation
- Evaluated on the test dataset:
  - **Test Accuracy**: 98.19%
  - **Test Loss**: 0.0452

### 6. Model Deployment
- Save the trained model for future use.
- Use the model for predictions to classify images.

---

## Flowchart
```plaintext
[Start] 
   ↓
[Load Dataset] 
   ↓
[Preprocess Images (Resize, Normalize)] 
   ↓
[Build CNN Model (Layers Configuration)] 
   ↓
[Train Model (Training Dataset)] 
   ↓
[Evaluate Model (Test Dataset)] 
   ↓
[Save Model for Future Use] 
   ↓
[Use Model for Predictions] 
   ↓
[End]

