# CIFAR-10 Image Classification

This project demonstrates a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset.

## Project Structure

- `notebooks/CIFAR10_image_classification.ipynb`: The main Jupyter notebook containing the code for data loading, preprocessing, model definition, training, and evaluation.
- `requirements.txt`: List of Python dependencies required to run the project.

## Getting Started

1.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

3.  Open `notebooks/CIFAR10_image_classification.ipynb` and run the cells.

## Model Architecture

The model consists of:
- 3 Convolutional layers with ReLU activation
- 2 Max Pooling layers
- 2 Dense layers (one hidden, one output)

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.
