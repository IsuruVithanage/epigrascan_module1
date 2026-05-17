from pathlib import Path
import json

output_dir = Path("outputs")
json_files = list(output_dir.glob("*.json"))

print(f"Found {len(json_files)} JSON files ready for CAME\n")

for f in json_files[:5]:  # Show first 5 files
    with open(f, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"📄 {f.name}")
    print(f"   Noisy text     : {data['noisy_transliteration']}")
    print(f"   Number of [MASK]: {len(data['soft_probs'])}")
    print(f"   Confidences    : {data['confidences']}")
    print("-" * 60)