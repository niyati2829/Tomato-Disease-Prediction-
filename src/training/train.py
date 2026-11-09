"""Model Training & Validation Loop.

Future Responsibility:
1. Orchestrate the training pipeline using configurations defined in `configs/config.yaml`.
2. Implement clean epoch loops:
   - `train_one_epoch(...)`: Forward pass, CrossEntropyLoss computation, optimizer.zero_grad(), loss.backward(), optimizer.step().
   - `validate_one_epoch(...)`: torch.no_grad() validation loss and accuracy tracking.
3. Checkpoint management: Save best model weights (`best_model.pt`) when validation loss improves.
4. Support early stopping to prevent overfitting.
5. Log learning rate, epoch loss, and validation metrics for export to `reports/experiment_log.csv`.

TODO:
- [ ] Implement `train_one_epoch(model, dataloader, criterion, optimizer, device) -> tuple[float, float]`
- [ ] Implement `validate(model, dataloader, criterion, device) -> tuple[float, float]`
- [ ] Implement `train_model(config_path: str)`
- [ ] Implement `save_checkpoint(model, optimizer, epoch, val_loss, filepath: Path)`
"""

# NOTE: Training loops will be developed iteratively with step-by-step beginner explanations.
