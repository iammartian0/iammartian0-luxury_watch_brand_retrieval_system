from pathlib import Path
import csv

root = Path("data/images")
metadata_path = Path("data/image_metadata.csv")

all_metadata = []

for brand_dir in root.iterdir():
    if brand_dir.is_dir():
        brand = brand_dir.name
        image_files = [f for f in brand_dir.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        
        for img_file in image_files:
            all_metadata.append({
                "brand": brand,
                "image_path": str(img_file),
                "search_term": f"{brand} watches"
            })

with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['brand', 'image_path', 'search_term'])
    writer.writeheader()
    writer.writerows(all_metadata)

print(f"Updated metadata: {len(all_metadata)} entries")