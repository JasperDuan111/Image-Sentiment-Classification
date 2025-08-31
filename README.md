# Image-Sentiment-Classification

A lightweight deep learning project for facial emotion recognition, where custom CNNs achieve nearly the same accuracy as InceptionNet(v2) with less than 8% of its parameters.

## Features

- Custom networks with only 7.8% of the parameters of InceptionNet(v2) achieve up to 68.57% validation accuracy, comparable to the full InceptionNet.
- Implemented using PyTorch framework
- 48x48 pixel grayscale image processing
- 7-class emotion classification (angry, disgust, fear, happy, neutral, sad, surprise)
- CUDA acceleration support for training and inference
- Automatic model saving and loading functionality

## Project Structure

```
Image Sentiment Classification/
├── data.py                              # Data loading and preprocessing module
├── model.py                             # Model definition file
├── Image Sentiment Classification.ipynb # Main training and testing code
├── requirements.txt                     # Project dependencies
├── data/                               # Dataset directory
│   ├── train/                          # Training data
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/                           # Test data
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
├── model/                              # Saved trained models
│   ├── InceptionNet_acc0.6936.pth
│   ├── InceptionNet_simplified_acc0.6959.pth
│   ├── Mynet_v1_acc0.6755.pth
│   └── Mynet_v2_acc0.6857.pth
└── test_images/                        # Test images
    ├── angry.png
    ├── happy.png
    ├── neutral.png
    ├── sad.png
    └── surprise.png
```

## Data Source

You can download the data from [FER 2013](https://www.kaggle.com/datasets/msambare/fer2013)

## Examples of ouputs

![Surprising cat](test_images/surprise_result.png)

## Model Architectures

The project implements multiple deep learning models:

1. **InceptionNet**: Convolutional neural network based on Inception architecture, using multi-scale convolution kernels for feature extraction
2. **InceptionNet_simplified**: Simplified version of Inception network that reduces parameters while maintaining performance(even better)
3. **Mynet_v1**: Custom convolutional neural network architecture 
4. **Mynet_v2**: Improved version of the custom network with a little bit better performance

## Performance Results

- **InceptionNet**: 69.36% validation accuracy
- **InceptionNet_simplified**: 69.59% validation accuracy
- **Mynet_v1**: 67.55% validation accuracy
- **Mynet_v2**: 68.57% validation accuracy

## Data Preprocessing

- Image resizing to 48x48 pixels
- Convert to grayscale images(original data are grayscale images)
- Data normalization (mean=0.5, std=0.5)
- Data augmentation support

## Training Parameters

- Batch size: 32
- Learning rate: 3e-4 (using AdamW optimizer)
- Learning rate scheduler: CosineAnnealingLR
- Loss function: Cross-entropy loss
- Training epochs: 30
- Precision: bfloat16 (mixed precision training)

## Notes

- Several acceleration methods are used
- Input images must be facial images with clear facial expressions
- Models are trained at 48x48 resolution, input images will be automatically resized
