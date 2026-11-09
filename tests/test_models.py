"""Unit Tests for Model Architectures.

Future Responsibility:
1. Verify that `BaselineCNN` forward pass with dummy input `(B, 3, 224, 224)` outputs logits of shape `(B, 10)`.
2. Verify that Transfer Learning model (e.g. MobileNetV2) outputs logits of shape `(B, 10)`.
3. Verify that model weights update after a single backprop step (gradient sanity check).
4. Verify that parameter freezing correctly sets `requires_grad = False` for backbone layers.

TODO:
- [ ] Implement `test_baseline_cnn_forward_shape()`
- [ ] Implement `test_transfer_learning_forward_shape()`
- [ ] Implement `test_model_gradient_flow()`
"""
