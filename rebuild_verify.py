import sys; sys.path.insert(0, '.')
import yaml
from pathlib import Path
from src.preprocessing.dataset_builder import build_dataset_metadata

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

df = build_dataset_metadata(config)

print()
print('=== FINAL COUNTS ===')
print(f'Total: {len(df)}')
train = (df["split"]=="train").sum()
val = (df["split"]=="val").sum()
test = (df["split"]=="test").sum()
print(f'By split:  train={train}  val={val}  test={test}')
auth = (df["label"]==0).sum()
forged = (df["label"]==1).sum()
print(f'By label:  authentic={auth}  forged={forged}')
print()
print('By forgery_type:')
print(df['forgery_type'].value_counts().to_string())
print()
print('By split x forgery_type:')
print(df.groupby(['split','forgery_type']).size().to_string())
print()
print('By dataset_source:')
print(df['dataset_source'].value_counts().to_string())
