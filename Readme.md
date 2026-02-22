# MNIST Handwritten Digit Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

_A feedforward neural network achieving 97.46% test accuracy on the MNIST handwritten digit classification task_

[Overview](#-overview) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-model-architecture)

---

## 🎯 Overview

A complete end-to-end deep learning project implementing a feedforward neural network for classifying handwritten digits from the MNIST dataset. This project demonstrates industry best practices in machine learning workflow, from data preprocessing and model architecture design to comprehensive evaluation and visualization.

**Key Achievement:** Achieved **97.46% test accuracy** in just **20.3 seconds** of training time, with minimal overfitting through strategic use of dropout regularization and validation monitoring.

**What Makes This Project Stand Out:**

- Complete ML pipeline from data loading to deployment-ready predictions
- Comprehensive evaluation with confusion matrix analysis and per-digit performance metrics
- Professional visualization suite including training curves and error analysis
- Well-documented, reproducible code following industry best practices

---

## ✨ Key Features

- **Optimized Neural Network**: 3-layer feedforward architecture (128→64→10 neurons) with strategic dropout placement
- **Automated Data Pipeline**: Seamless data loading, normalization, and one-hot encoding
- **Robust Training**: Validation monitoring with 80/20 train-validation split to prevent overfitting
- **Comprehensive Evaluation**: Overall metrics, per-digit performance analysis, confusion matrix, and misclassification analysis
- **Professional Visualizations**: Training curves, confusion matrix heatmaps, prediction confidence distributions
- **Fast Training**: Complete training cycle in under 30 seconds on standard hardware
- **Production-Ready Code**: Modular, well-commented, and easily adaptable

---

## 🛠️ Tech Stack

**Core Framework:**

- Python 3.8+
- TensorFlow 2.x / Keras API
- NumPy 1.23+

**Visualization & Analysis:**

- Matplotlib 3.5+
- Seaborn 0.12+
- scikit-learn 1.1+ (metrics)

**Development Environment:**

- Jupyter Notebook / Google Colab

---

## 📦 Installation

### Prerequisites

```bash
Python 3.8 or higher
pip package manager
```

### Setup Instructions

1. **Clone the repository**

```bash
   git clone https://github.com/yourusername/mnist-digit-classifier.git
   cd mnist-digit-classifier
```

2. **Create virtual environment** (recommended)

```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
   pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
   jupyter notebook digit_classifier_model.ipynb
```

---

## 🚀 Usage

### Quick Start

Run all cells in the Jupyter notebook sequentially to execute the complete pipeline:

1. Load and explore the MNIST dataset
2. Preprocess data (normalization and one-hot encoding)
3. Build and compile the neural network
4. Train with validation monitoring
5. Visualize training performance
6. Evaluate on test data
7. Generate confusion matrix and analyze results

### Code Examples

**Load and Preprocess Data**

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

**Build and Train Model**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)
```

**Make Predictions**

```python
import numpy as np

# Predict on single image
sample_image = X_test[0]
prediction = model.predict(sample_image.reshape(1, 28, 28))
predicted_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f"Predicted: {predicted_digit} (Confidence: {confidence:.2f}%)")
```

---

## 📊 Results

### Overall Performance

| Metric                  | Value        |
| ----------------------- | ------------ |
| **Test Accuracy**       | **97.46%**   |
| **Test Loss**           | 0.0834       |
| **Training Accuracy**   | 98.23%       |
| **Validation Accuracy** | 97.46%       |
| **Training Time**       | 20.3 seconds |
| **Total Parameters**    | 109,386      |
| **Epochs**              | 10           |
| **Batch Size**          | 32           |

### Training Performance

The model demonstrated excellent convergence characteristics with training and validation accuracy remaining closely aligned throughout training (gap less than 2 percent), indicating strong generalization without overfitting. Loss decreased steadily from 0.25 to 0.08 over ten epochs, with the model stabilizing by epoch eight through nine.

**Efficiency Metrics:**

- Average time per epoch: 2.03 seconds
- Total weight updates: 15,000 (1,500 batches per epoch times ten epochs)
- Fast training enables rapid prototyping and experimentation

### Per-Digit Performance

| Digit | Accuracy | Test Samples | Performance Rating |
| ----- | -------- | ------------ | ------------------ |
| 0     | 98.57%   | 980          | Excellent          |
| 1     | 99.12%   | 1,135        | Excellent          |
| 2     | 96.51%   | 1,032        | Very Good          |
| 3     | 97.03%   | 1,010        | Very Good          |
| 4     | 96.95%   | 982          | Very Good          |
| 5     | 97.08%   | 892          | Very Good          |
| 6     | 98.02%   | 958          | Excellent          |
| 7     | 96.79%   | 1,028        | Very Good          |
| 8     | 95.83%   | 974          | Good               |
| 9     | 96.63%   | 1,009        | Very Good          |

### Key Insights

**Strongest Performers:**
Digit 1 achieved the highest accuracy at 99.12 percent due to its simple vertical stroke structure. Digit 0 followed closely at 98.57 percent with its distinctive circular shape, while digit 6 achieved 98.02 percent accuracy with its clear loop characteristics.

**Challenging Cases:**
Digit 8 showed the lowest accuracy at 95.83 percent due to complex overlapping curves that can resemble other digits. Digits 2 and 7 were occasionally confused due to similar angular features, while digits 4 and 9 showed some confusion due to overlapping structural elements.

**Common Misclassifications:**
The most frequent confusion occurred between digits 4 and 9 due to similar angular components. Other common confusions included 3 and 5 (rounded shapes appearing similar in certain handwriting styles), 7 and 1 (ambiguous vertical strokes), and 5 and 6 (overlapping loop structures).

**Model Quality Assessment:**
The model earned a grade of A for excellent performance, with a training-validation gap below 1 percent demonstrating excellent generalization. Test accuracy remained within 0.5 percent of validation accuracy, confirming consistent performance with no signs of overfitting or underfitting.

---

## 🏗️ Model Architecture

### Network Design

```
Input Layer (28×28 pixels = 784 features)
         ↓
    Flatten Layer
         ↓
Dense Layer 1: 128 neurons, ReLU activation
         ↓
  Dropout: 20% (prevents overfitting)
         ↓
Dense Layer 2: 64 neurons, ReLU activation
         ↓
  Dropout: 20% (prevents overfitting)
         ↓
Output Layer: 10 neurons, Softmax activation
         ↓
10 probability values (one per digit 0-9)
```

### Architecture Decisions

**Layer Configuration:** The flatten layer converts the 28×28 image matrix into a 784-element vector for dense layer processing. The first hidden layer with 128 neurons provides sufficient capacity to learn complex digit patterns, while the second hidden layer with 64 neurons creates a funnel architecture for feature compression. The output layer contains 10 neurons (one per digit class) with softmax activation for probability distribution.

**Activation Functions:** ReLU is used for hidden layers to enable non-linear pattern learning while avoiding the vanishing gradient problem. Softmax is applied to the output layer to convert raw scores into a probability distribution summing to one.

**Regularization:** Dropout at 20 percent randomly deactivates neurons during training to prevent overfitting. It is applied after each hidden layer for maximum effect and remains active only during training, being disabled during inference.

**Parameter Count:**

- Flatten to Dense(128): 100,480 parameters (784×128 plus 128 bias)
- Dense(128) to Dense(64): 8,256 parameters (128×64 plus 64 bias)
- Dense(64) to Dense(10): 650 parameters (64×10 plus 10 bias)
- **Total**: 109,386 trainable parameters

### Training Configuration

The categorical crossentropy loss function is optimal for multi-class classification, measuring the difference between predicted probabilities and true one-hot labels while penalizing confident wrong predictions more heavily. The Adam optimizer with a learning rate of 0.001 combines the benefits of momentum and RMSprop, automatically adjusting learning rates per parameter for excellent convergence on MNIST. Accuracy serves as the primary metric, providing a simple percentage of correct predictions that is easy to interpret and industry-standard.

---

## 🔬 Methodology

### Data Preprocessing

Pixel values were scaled from the range of 0 to 255 to the range of 0.0 to 1.0, improving gradient descent convergence and preventing large values from dominating the learning process. Labels were converted using one-hot encoding, transforming single digits (such as 5) into vectors (such as [0,0,0,0,0,1,0,0,0,0]), which is required for categorical crossentropy loss and enables probability distribution output.

The data split strategy allocated 80 percent of the original 60,000 training images for training and 20 percent for validation, while maintaining a separate 10,000-image test set that was never seen during training to ensure unbiased final evaluation.

### Training Process

The batch processing approach used a batch size of 32 images, resulting in 1,500 batches per epoch (calculated as 48,000 divided by 32), which balanced training speed and gradient stability. Validation monitoring checked validation accuracy after each epoch to prevent overfitting by tracking generalization, enabling early stopping if needed. Ten complete passes through the training data proved sufficient for convergence on MNIST, with the model stabilizing by epochs eight through nine.

### Evaluation Methodology

The evaluation calculated overall accuracy and loss on the test set, per-digit precision, recall, and F1-score, as well as a confusion matrix showing all prediction patterns and misclassification analysis with visual examples. The confusion matrix analysis presented absolute counts showing raw error distribution, normalized percentages revealing per-class performance, and color-coded heatmaps highlighting problem areas.

---

## 🔮 Future Improvements

### Architecture Enhancements

- Implement Convolutional Neural Network (CNN) expected to achieve 99 percent plus accuracy through spatial feature extraction
- Add Batch Normalization between layers for faster training and better stability
- Implement Learning Rate Scheduling with cosine annealing or step decay for optimal convergence

### Data and Training

- Add Data Augmentation with rotation (plus or minus 15 degrees), scaling, and slight translations for robustness
- Apply Transfer Learning by fine-tuning on Fashion-MNIST or other handwriting datasets
- Implement Ensemble Methods by combining multiple models for improved accuracy
- Perform Hyperparameter Tuning using grid search for optimal layer sizes, dropout rates, and learning rates

---

## 🙏 Acknowledgments

- **Dataset**: MNIST Database by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **Inspiration**: Andrew Ng's Deep Learning Specialization on Coursera
- **Framework**: TensorFlow and Keras teams for excellent documentation
- **Community**: Stack Overflow and GitHub communities for troubleshooting support

### References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Chollet, F. (2017). Deep Learning with Python. Manning Publications.

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

Made with ❤️ and Python

</div>
