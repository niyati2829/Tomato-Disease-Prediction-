"""Standardized Logging Configuration Utility.

Future Responsibility:
1. Configure Python's standard `logging` module with a clean console formatter.
2. Direct debug and training progress logs to both stdout and an optional log file.
3. Prevent duplicate handlers across interactive notebook and script executions.

TODO:
- [ ] Implement `setup_logger(name: str = "crop_disease", log_file: Path | None = None) -> logging.Logger`
"""
