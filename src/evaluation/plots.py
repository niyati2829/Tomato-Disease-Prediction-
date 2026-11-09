"""Visualization & Performance Plotting Module.

Future Responsibility:
1. Plot and save normalized and raw Confusion Matrices using Seaborn / Matplotlib.
2. Plot and save training history curves (Training vs. Validation Loss, Training vs. Validation Accuracy per epoch).
3. Generate class distribution bar charts.
4. Export high-resolution PNG figures directly into `artifacts/figures/`.

TODO:
- [ ] Implement `plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], output_path: Path)`
- [ ] Implement `plot_learning_curves(history: dict, output_path: Path)`
- [ ] Implement `plot_class_distribution(counts: dict, output_path: Path)`
"""

# NOTE: Plotting functions will be implemented when training runs produce history data.
