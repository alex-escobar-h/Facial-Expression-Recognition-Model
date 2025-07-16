# Facial Expression Recognition Model
This repository contains a  facial expression recognition model using a Convolutional Neural Network (CNN) trained on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) from Kaggle. The dataset contains 48x48 grayscale images of human faces, each labeled with one of seven emotions: 
1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## Dataset Overview
FER-2013 provides 28,709 training examples along with 3,589 examples in the public test set. We decided a CNN would be the best model architecture for this project due to its strong ability to extract spatial features from visual data and learn patterns that are useful for image classification.

## Model Architecture

The model is built using PyTorch's torchvision. The architecture consists of the following layers:

- **Input Layer:** Accepts 48x48 grayscale images.
- **Convolutional Layers:** Multiple convolutional layers with ReLU activation to extract spatial features.
- **Pooling Layers:** MaxPooling layers to reduce dimensionality and retain important features.
- **Dropout Layers:** Applied after pooling to prevent overfitting.
- **Flatten Layer:** Converts the 2D feature maps into a 1D feature vector.
- **Dense Layers:** Fully connected layers for classification.
- **Output Layer:** Softmax activation to predict one of the seven emotion classes.

## Training

The model was trained using the Adam/SGD optimizers and categorical cross-entropy loss. Data augmentation techniques such as rotation, zoom, and horizontal flipping were applied to improve generalization. Early stopping and model checkpointing can be used to prevent overfitting and save the best model.

## Requirements
- Python 3.6+
- Torch
- Matplotlib

## Usage

This model can be run in a Jupyter Notebook on a local machine or in a cloud environment. We recommend using a GPU for faster training times. 

Google Colab is a good option for running this notebook if you do not have a local GPU setup. If you want to run the notebook on Google Colab, you can add your own personal Kaggle API key to access the FER-2013 dataset directly from Kaggle.

## References

- [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
