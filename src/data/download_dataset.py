"""Dataset Download & Extraction Module.

Future Responsibility:
1. Provide functions to download or clone the PlantVillage dataset from the official GitHub source:
   https://github.com/spMohanty/PlantVillage-Dataset
2. Extract the downloaded archive safely into `data/raw/`.
3. Filter or isolate the 10 Tomato classes (`Tomato___*`) into a dedicated directory.
4. Verify checksums/file counts to ensure no images were corrupted during transit.

TODO:
- [ ] Implement `download_plantvillage(dest_dir: Path)`
- [ ] Implement `extract_tomato_subset(raw_dir: Path, target_dir: Path)`
- [ ] Implement `verify_download(target_dir: Path) -> bool`
"""

# NOTE: Actual download logic will be implemented in subsequent steps after user approval.
