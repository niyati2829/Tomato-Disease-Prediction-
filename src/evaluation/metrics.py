"""Classification Metrics Computation Module.

Future Responsibility:
1. Provide standard evaluation metrics using `scikit-learn.metrics`:
   - Accuracy
   - Macro and Weighted Precision
   - Macro and Weighted Recall
   - Macro and Weighted F1-score
   - Per-class classification report
2. Format metrics cleanly for terminal printing and logging into `reports/experiment_log.csv`.

TODO:
- [ ] Implement `compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict`
- [ ] Implement `format_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> str`
- [ ] Implement `log_experiment_result(csv_path: Path, experiment_data: dict)`
"""

# NOTE: Metric calculation functions will be implemented during evaluation phase.
