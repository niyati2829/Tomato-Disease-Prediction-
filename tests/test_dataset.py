"""Unit Tests for Dataset and DataLoaders.

Future Responsibility:
1. Test that `PlantVillageTomatoDataset` correctly loads images and matches labels.
2. Verify that image transforms output tensors of expected shape `(3, 224, 224)` and dtype `torch.float32`.
3. Verify that `DataLoader` yields batches of the requested batch size.
4. Ensure no index out-of-range errors or label mismatches exist in the split CSVs.

TODO:
- [ ] Implement `test_dataset_initialization()`
- [ ] Implement `test_image_transforms_shape()`
- [ ] Implement `test_dataloader_batch_dimensions()`
"""
