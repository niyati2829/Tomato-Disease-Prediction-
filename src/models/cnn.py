"""Custom Baseline Convolutional Neural Network (CNN) Module.

Future Responsibility:
1. Define a beginner-friendly, modular PyTorch CNN architecture (`BaselineCNN` inheriting from `nn.Module`).
2. Architecture design:
   - Input: (B, 3, 224, 224)
   - Conv Block 1: Conv2d(3 -> 32, kernel_size=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2)
   - Conv Block 2: Conv2d(32 -> 64, kernel_size=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2)
   - Conv Block 3: Conv2d(64 -> 128, kernel_size=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2)
   - Conv Block 4: Conv2d(128 -> 256, kernel_size=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2)
   - Global Average Pooling or Flatten + Dense layer(s)
   - Dropout for regularization
   - Output Linear Layer: (num_features -> num_classes=10)
3. Provide helper function to count trainable parameters.

TODO:
- [ ] Implement `BaselineCNN(nn.Module)`
- [ ] Implement `count_parameters(model: nn.Module) -> int`
"""

# NOTE: Full CNN model architecture will be implemented during Phase 1 modeling.
