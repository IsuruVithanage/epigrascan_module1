from pathlib import Path
import json

kaggle_path = Path("data/kaggle_brahmi")
classes = sorted([d.name for d in kaggle_path.iterdir() if d.is_dir()])

label_mapping = {str(i): name for i, name in enumerate(classes)}

with open("label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=2, ensure_ascii=False)

print(f"✅ Created label_mapping.json with {len(classes)} classes")