import csv
from pathlib import Path

data_dir = Path("data")
filtered_dir = data_dir / "filtered"
original_metadata_path = data_dir / "image_metadata.csv"
cleaned_metadata_path = data_dir / "cleaned_metadata.csv"

# Load original metadata to get search terms
with open(original_metadata_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    original_metadata = list(reader)

# Create lookup for search terms
metadata_lookup = {}
for row in original_metadata:
    img_path = Path(row['image_path']).name
    metadata_lookup[img_path] = row['search_term']

# Collect all filtered images
cleaned_metadata = []
for brand_dir in filtered_dir.iterdir():
    if brand_dir.is_dir():
        brand = brand_dir.name
        for img_file in brand_dir.glob("*.*"):
            if img_file.is_file():
                img_name = img_file.name
                search_term = metadata_lookup.get(img_name, "unknown")
                cleaned_metadata.append({
                    'brand': brand,
                    'image_path': f"data/images/{brand}/{img_name}",
                    'search_term': search_term,
                    'filtered_image_path': str(img_file)
                })

# Save cleaned metadata
with open(cleaned_metadata_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['brand', 'image_path', 'search_term', 'filtered_image_path'])
    writer.writeheader()
    writer.writerows(cleaned_metadata)

print(f"Created cleaned_metadata.csv with {len(cleaned_metadata)} images")