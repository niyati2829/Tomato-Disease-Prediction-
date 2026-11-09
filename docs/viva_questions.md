# Viva Voce & Oral Defense Preparation Guide

This guide contains fundamental deep learning questions and concise, clear explanations tailored for your hackathon evaluation and viva.

---

## 1. Tensors & Computer Vision Basics

### Q1: What is a Tensor, and how does an image translate into a Tensor?
**Answer:**  
A tensor is a multi-dimensional array of numerical values. In computer vision, an RGB image is translated into a 3D tensor of shape `(Channels, Height, Width)` or `(C, H, W)` in PyTorch:
- **Channels (C = 3)**: Red, Green, and Blue intensity values (typically normalized between 0.0 and 1.0).
- **Height (H) & Width (W)**: Spatial resolution of the image (e.g., 224 x 224 pixels).
- When batched for training, a 4th dimension is added: `(Batch_Size, Channels, Height, Width)` e.g. `(32, 3, 224, 224)`.

### Q2: Why do we normalize image pixel values using mean and standard deviation?
**Answer:**  
Raw pixel values range from `0` to `255`. Normalizing them (e.g., subtracting mean and dividing by std):
1. Centers data around zero, preventing vanishing or exploding gradients.
2. Ensures all features (color channels) contribute uniformly to gradient updates.
3. Accelerates convergence during gradient descent.

---

## 2. Convolutional Neural Networks (CNNs)

### Q3: Why do we use CNNs instead of standard Multi-Layer Perceptrons (MLPs / Fully Connected layers) for images?
**Answer:**  
1. **Parameter Explosion**: Flattening a `224x224x3` image produces `150,528` input features. Connecting this to even a 512-neuron hidden layer requires over 77 million weights, leading to massive memory usage and severe overfitting.
2. **Spatial Hierarchy**: MLPs ignore the 2D spatial arrangement of neighboring pixels. CNNs use localized kernels (e.g., `3x3`) to detect patterns like edges, textures, and leaf spots regardless of where they appear in the image (**translation invariance**).
3. **Weight Sharing**: A single small convolutional kernel is slid across the entire image, drastically reducing the number of trainable parameters.

### Q4: What does a Convolutional Layer do?
**Answer:**  
A convolutional layer applies a set of learnable filters (kernels) that slide across the input tensor. At each position, it performs an element-wise multiplication and sum, extracting spatial features (e.g., early layers detect leaf veins and edges; deeper layers detect complex disease lesion patterns).

### Q5: What is the purpose of Pooling (e.g., MaxPool2d)?
**Answer:**  
1. **Dimensionality Reduction**: Reduces the spatial dimensions (Height and Width) of feature maps (e.g., a `2x2` max pool with stride 2 halves the height and width), reducing computational cost.
2. **Translation Robustness**: Retains the most prominent feature in each local region, making the network slightly invariant to small shifts or rotations.

### Q6: What does an Activation Function (e.g., ReLU) do?
**Answer:**  
Without activation functions, a neural network is just a series of linear matrix multiplications, meaning it could only learn linear relationships. Activation functions like **ReLU** ($\text{ReLU}(x) = \max(0, x)$) introduce non-linearity, enabling the network to learn complex, non-linear boundaries between diseases.

---

## 3. Dataset Splits & Leakage Prevention

### Q7: Why do we split data into Train, Validation, and Test sets?
**Answer:**  
- **Training Set (e.g., 70%)**: Used by the model to learn weights via backpropagation.
- **Validation Set (e.g., 15%)**: Evaluated during training to tune hyperparameters (learning rate, architecture) and detect overfitting early (early stopping). The model does *not* backpropagate on this set.
- **Test Set (e.g., 15%)**: Held out completely until all training is finished to provide an unbiased measure of real-world generalization.

### Q8: What is Stratified Splitting, and why is it important here?
**Answer:**  
Stratified splitting ensures each split (train, val, test) maintains the exact same proportion of each disease class as the original dataset. If a rare disease makes up 5% of the total dataset, stratification guarantees it makes up 5% in train, val, and test, preventing sampling bias.

---

## 4. Loss Functions & Optimizers

### Q9: Why do we use Cross-Entropy Loss for multi-class classification?
**Answer:**  
Cross-Entropy Loss measures the distance between the model's predicted probability distribution (produced via Softmax) and the true one-hot encoded ground truth label:
$$\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$$
It heavily penalizes predictions that are confident but wrong, pushing the model to assign high probability to the correct disease class.

### Q10: How does the Adam Optimizer work compared to standard SGD?
**Answer:**  
Standard Stochastic Gradient Descent (SGD) updates weights using a single global learning rate. **Adam (Adaptive Moment Estimation)** maintains individual adaptive learning rates for each parameter by tracking:
1. First moment (exponential moving average of past gradients — momentum).
2. Second moment (moving average of squared gradients — scaling).
This leads to faster and more stable convergence, especially on sparse or noisy gradients.

---

## 5. Overfitting & Regularization

### Q11: What is Overfitting, and how can you tell if your model is overfitting?
**Answer:**  
Overfitting occurs when a model memorizes the training data (including background noise and quirks) rather than learning generalizable leaf disease patterns.  
**Symptom**: Training loss continues to decrease and training accuracy approaches 100%, while validation loss starts rising and validation accuracy plateaus or drops.

### Q12: What techniques are used to combat overfitting?
**Answer:**  
1. **Data Augmentation**: Artificially expanding dataset diversity by flipping, rotating, and color-jittering images during training.
2. **Dropout**: Randomly deactivating a percentage of neurons (e.g., 30%) during each forward pass in training, preventing neurons from co-adapting.
3. **Weight Decay (L2 Regularization)**: Penalizes large weights in the loss function, encouraging smaller, smoother weights.
4. **Early Stopping**: Halting training when validation loss stops improving for a set number of consecutive epochs.

---

## 6. Evaluation Metrics

### Q13: Why is Accuracy alone not enough for medical/pathology diagnosis?
**Answer:**  
If a dataset is imbalanced (e.g., 90% healthy leaves and 10% diseased leaves), a dumb model predicting "healthy" for every image achieves 90% accuracy while failing 100% of diseased crops.  
We must inspect:
- **Precision**: Out of all leaves predicted as *Late Blight*, how many actually have it? (Avoids false alarms).
- **Recall (Sensitivity)**: Out of all leaves that actually have *Late Blight*, how many did we catch? (Crucial to prevent disease outbreak).
- **F1-Score**: Harmonic mean of Precision and Recall.

### Q14: What is a Confusion Matrix?
**Answer:**  
An $N \times N$ grid (where $N = 10$ tomato classes) comparing ground truth labels (rows) against model predictions (columns). Diagonal cells represent correct classifications; off-diagonal cells highlight exact pairwise confusion (e.g., whether the model confuses Early Blight with Target Spot).

---

## 7. Transfer Learning

### Q15: What is Transfer Learning, and why is it valuable for this project?
**Answer:**  
Transfer Learning takes a deep neural network (e.g., ResNet or MobileNet) trained on a massive dataset (ImageNet with 1.2+ million images) and re-purposes its learned feature representations for a new domain (tomato diseases).
- **Feature Extraction**: Early layers already know how to detect lines, shapes, and textures. We freeze those layers and only train a new classification head on our tomato dataset.
- **Benefits**: Reaches significantly higher accuracy with fewer epochs, requires less training data, and converges much faster than training from scratch.
