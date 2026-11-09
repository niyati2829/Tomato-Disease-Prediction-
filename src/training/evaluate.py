"""Holdout Test Set Evaluation.

Future Responsibility:
1. Load a saved model checkpoint (`artifacts/checkpoints/*.pt`).
2. Run inference on the isolated test set (`data/splits/test.csv`) under `torch.no_grad()`.
3. Gather ground truth labels and model predictions (logits / softmax probabilities).
4. Interface with `src/evaluation/metrics.py` and `src/evaluation/plots.py` to generate:
   - Overall accuracy, macro precision, recall, and F1-score.
   - Per-class classification reports.
   - Confusion matrix visualization saved to `artifacts/figures/`.

TODO:
- [ ] Implement `load_model_for_evaluation(checkpoint_path: Path, model_arch: str, device) -> nn.Module`
- [ ] Implement `run_test_inference(model, test_loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- [ ] Implement `evaluate_pipeline(config_path: str)`
"""

# NOTE: Evaluation script will be implemented once models are trained.
