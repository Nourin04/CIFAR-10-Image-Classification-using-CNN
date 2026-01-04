# CIFAR-10-Image-Classification-using-CNN



## Overview

This project implements a simple image classification system using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. OpenCV is used for basic image preprocessing and enhancement, while PyTorch is used for model development, training, and evaluation. A Gradio-based web interface is provided for interactive testing of the trained model.

---

## Dataset

**CIFAR-10**

* 60,000 color images of size 32×32
* 10 classes:
  * airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    
* Split:
  * 50,000 training images
  * 10,000 testing images

The dataset is automatically downloaded using `torchvision.datasets.CIFAR10`.

---

## Project Workflow

### 1. Image Processing with OpenCV

* Converted images from tensors to NumPy arrays for OpenCV processing
* Applied the following preprocessing and enhancement techniques:

  * RGB conversion and resizing
  * Brightness and contrast adjustment
  * Gaussian blur filtering
* Displayed original and enhanced images side-by-side for comparison

> OpenCV is used for visualization and understanding image enhancement techniques. The CNN is trained on normalized original images.

---

### 2. Model Development

* Implemented a simple CNN architecture using PyTorch:

  * Two convolutional layers with ReLU activation
  * Max pooling layers for downsampling
  * Fully connected layers for classification
* Used:

  * Loss Function: CrossEntropyLoss
  * Optimizer: Adam
* Trained the model for 20 epochs using GPU acceleration in Google Colab

---

### 3. Model Evaluation

* Evaluated the trained model on the test dataset
* Metrics used:

  * Overall classification accuracy
  * Confusion matrix
* Visualized:

  * Sample test images with predicted and actual labels
  * Training loss vs epochs graph
  * Training accuracy vs epochs graph
<img width="853" height="766" alt="result" src="https://github.com/user-attachments/assets/b1a181ee-553c-4d14-b4b6-0c6e138c7c67" />


---

### 4. Model Saving

* Saved the trained model using PyTorch’s `state_dict` method:

  ```
  cifar10_cnn.pth
  ```
* The saved model can be loaded later for inference without retraining.

---

### 5. Gradio Web Interface

* Built a simple Gradio interface to test the trained CNN
* Features:

  * Upload an image (JPG/PNG)
  * Automatic resizing and normalization
  * Displays the predicted CIFAR-10 class
* The interface uses the saved model and runs inference only

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy
* Matplotlib
* Gradio

---

## Repository Structure

```
├── cifar10_cnn.pth          # Saved trained model
├── cnn_training.ipynb       # Training, evaluation, and visualization
├── README.md
```


---

## Results
<img width="1293" height="456" alt="image" src="https://github.com/user-attachments/assets/57ba5ae2-4b3a-4fc9-a604-adbe2d712bc9" />
<img width="1310" height="469" alt="image" src="https://github.com/user-attachments/assets/4ded69bd-249f-4b21-b416-427349f2be9f" />
* The CNN achieves reasonable accuracy on the CIFAR-10 test set for a simple architecture.
* The loss decreases consistently across epochs, indicating effective learning.
* Visual inspection of predictions shows correct classification for many test samples, with some misclassifications as expected.

---

## Conclusion

This project demonstrates an end-to-end image classification pipeline using OpenCV for image preprocessing and PyTorch for deep learning. The addition of a Gradio interface enables interactive testing and makes the model easy to deploy and reuse.


