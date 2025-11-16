# Convolutional Neural Networks and Data Augmentation on CIFAR-10
**Author:** Ons Kharrat
## Overview
This project implements and compares Convolutional Neural Networks (CNNs) for image classification on the CIFAR-10 dataset, with a focus on evaluating the impact of data augmentation techniques on model performance. The project trains two identical CNN architectures—one without data augmentation (baseline) and one with augmentation—to demonstrate how data augmentation affects generalization and model accuracy.
## Features
- **Custom CNN Architecture**: A compact, efficient CNN designed for CIFAR-10 classification

   - Three convolutional blocks with Batch Normalization
   - Global Average Pooling to reduce parameters
   - Dropout regularization (30%)
   - ~95K trainable parameters

- **Data Augmentation Pipeline**:

   - Random horizontal flips
   - Random rotations (±14.4°)
   - Random zoom (±10%)

- **Training Features**:

   - Reproducible training with fixed random seeds
   - Learning rate scheduling (ReduceLROnPlateau)
   - Early stopping to prevent overfitting
   - Validation split for model evaluation

- **Evaluation & Visualization**:

   - Training/validation curves
   - Confusion matrix analysis
   - Side-by-side comparison of baseline vs. augmented models
## Dataset
**CIFAR-10**: A dataset of 60,000 32x32 color images in 10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
**Data Split**:
- Training: 45,000 images
- Validation: 5,000 images
- Test: 10,000 images
## Requirements
- Python 3.x
- TensorFlow 2.x (tested with 2.18.0)
- NumPy
- Matplotlib
- scikit-learn
## Installation
1. Clone this repository:
git clone [<repository-url>](https://github.com/OnsKharrat06/CNN-and-Data-Augemntation-on-CIFAR-10.git)
cd CNN-and-Data-Augemntation-on-CIFAR-10
2. Install required packages:
pip install tensorflow numpy matplotlib scikit-learn
## Usage
Open and run the Jupyter notebook:
jupyter notebook cnn-and-data-augmentation-on-cifar-10.ipynb
The notebook is organized into cells that:
1. Set up the environment and ensure reproducibility
2. Load and preprocess the CIFAR-10 dataset
3. Build data pipelines
4. Define and compile the CNN architecture
5. Train the baseline model (no augmentation)
6. Visualize baseline training curves
7. Evaluate baseline model on test set
8. Implement and visualize data augmentation
9. Train the augmented model
10. Compare results and generate confusion matrix
## Model Architecture
Input (32×32×3)
  ↓
ConvBlock(32) → 16×16×32
  ↓
ConvBlock(64) → 8×8×64
  ↓
ConvBlock(128) → 4×4×128
  ↓
GlobalAveragePooling2D → 128
  ↓
Dropout(0.30)
  ↓
Dense(10, softmax) → 10 classes
**ConvBlock Structure**:
- Conv2D (3×3, same padding)
- BatchNormalization
- ReLU activation
- MaxPooling2D (2×2)
## Results
### Test Set Performance
| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| Baseline (no augmentation) | 64.66% | 0.9992 |
| With Data Augmentation | 59.01% | 1.1533 |
### Key Observations
- The baseline model achieved higher test accuracy (64.66%) compared to the augmented model (59.01%)
- This result suggests that for this specific architecture and training configuration, the augmentation strategy may have been too aggressive or the model may need more training epochs to fully benefit from augmentation
- The augmented model shows different learning dynamics, which can be observed in the training curves
## Project Structure
CNN-and-Data-Augemntation-on-CIFAR-10/
├── cnn-and-data-augmentation-on-cifar-10.ipynb  # Main notebook
└── README.md                                      # This file
## Training Configuration
- **Batch Size**: 128
- **Initial Learning Rate**: 0.001 (Adam optimizer)
- **Epochs**: 8 (with early stopping)
- **Dropout Rate**: 0.30
- **Seed**: 42 (for reproducibility)
## Callbacks
- **ReduceLROnPlateau**: Reduces learning rate by factor of 0.5 when validation loss plateaus (patience=1)
- **EarlyStopping**: Stops training if validation loss doesn't improve for 2 epochs, restores best weights
## Notes
- The project uses GPU acceleration when available (CUDA-compatible GPUs)
- All random operations are seeded for reproducibility
- Images are normalized to [0, 1] range
- Data augmentation is applied on-the-fly during training
## Future Improvements
- Experiment with different augmentation strategies and intensities
- Try more sophisticated architectures (ResNet, EfficientNet)
- Implement learning rate finder
- Add more comprehensive evaluation metrics
- Experiment with longer training schedules
## License
This project is for educational purposes.
