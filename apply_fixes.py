import os

# Fix 1: dataset_builder.py - handle empty CSV
db_path = "src/preprocessing/dataset_builder.py"
if os.path.exists(db_path):
    with open(db_path, "r") as f:
        content = f.read()
    old = """    splits_file = Path(config["data"]["splits_file"])
    if not splits_file.exists():
        df = build_dataset_metadata(config)
    else:
        df = pd.read_csv(splits_file)"""
    new = """    splits_file = Path(config["data"]["splits_file"])
    if not splits_file.exists() or splits_file.stat().st_size == 0:
        if splits_file.exists() and splits_file.stat().st_size == 0:
            logger.warning("Splits file exists but is empty. Regenerating from raw data.")
        df = build_dataset_metadata(config)
    else:
        df = pd.read_csv(splits_file)"""
    if old in content:
        content = content.replace(old, new)
        with open(db_path, "w") as f:
            f.write(content)
        print("Fixed: dataset_builder.py")
    else:
        print("Skipped: dataset_builder.py (pattern not found)")
else:
    print("Skipped: dataset_builder.py (file not found)")

# Fix 2: augmentation.py - add Resize before RandomCrop
aug_path = "src/preprocessing/augmentation.py"
if os.path.exists(aug_path):
    with open(aug_path, "r") as f:
        content = f.read()
    old = "TRAIN_TRANSFORM = A.Compose([\n    A.RandomCrop(224, 224),"
    new = "TRAIN_TRANSFORM = A.Compose([\n    A.Resize(256, 256),\n    A.RandomCrop(224, 224),"
    if old in content:
        content = content.replace(old, new)
        with open(aug_path, "w") as f:
            f.write(content)
        print("Fixed: augmentation.py")
    else:
        print("Skipped: augmentation.py (pattern not found)")
else:
    print("Skipped: augmentation.py (file not found)")
