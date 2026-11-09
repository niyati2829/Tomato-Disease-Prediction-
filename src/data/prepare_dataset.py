"""Dataset Splitting & Preparation Module.

Future Responsibility:
1. Parse all image paths and associate them with integer class indices.
2. Perform deterministic, stratified Train/Validation/Test splitting (e.g., 70% / 15% / 15%)
   using `sklearn.model_selection.train_test_split` with a fixed random seed.
3. Save split indices as CSV files (`data/splits/train.csv`, `val.csv`, `test.csv`)
   to avoid copying or duplicating large image files on disk.
4. Verify split proportions and guarantee zero data leakage between sets.

TODO:
- [ ] Implement `create_stratified_splits(image_dir: Path, output_dir: Path, ratios: tuple, seed: int)`
- [ ] Implement `save_split_csv(file_paths: list, labels: list, target_csv: Path)`
- [ ] Implement `validate_no_leakage(train_csv: Path, val_csv: Path, test_csv: Path) -> bool`
"""

# NOTE: Splitting logic will be implemented once raw images are extracted.
