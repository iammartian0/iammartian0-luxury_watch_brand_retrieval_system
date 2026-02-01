from pathlib import Path
import csv
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BRANDS = ["Rolex", "Omega", "Breitling", "Cartier", "Seiko", 
          "Longines", "TAG Heuer", "Hublot", "Audemars Piguet", "Patek Philippe"]

def validate_text_dataset():
    csv_path = Path("data/cleaned_watches.csv")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    logger.info(f"Text dataset: {len(rows)} rows")
    
    found_brands = set(row.get('brand', '') for row in rows if row.get('brand'))
    missing_brands = set(BRANDS) - found_brands
    
    if missing_brands:
        logger.warning(f"Missing brands in text data: {missing_brands}")
    else:
        logger.info("✓ All 10 brands present in text dataset")

def validate_image_dataset():
    images_dir = Path("data/images")
    metadata_path = Path("data/image_metadata.csv")
    
    if not metadata_path.exists():
        logger.error("❌ image_metadata.csv not found")
        return
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        metadata = list(reader)
    
    logger.info(f"Image metadata: {len(metadata)} entries")
    
    brand_counts = {}
    for item in metadata:
        brand = item['brand']
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    logger.info("Brand image distribution:")
    for brand in sorted(BRANDS):
        count = brand_counts.get(brand, 0)
        logger.info(f"  {brand:20} {count}")
    
    valid_images = 0
    corrupted_images = 0
    
    for item in metadata:
        img_path = Path(item['image_path'])
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_images += 1
            except:
                corrupted_images += 1
                logger.warning(f"Corrupted image: {img_path}")
        else:
            logger.warning(f"Missing image: {img_path}")
    
    logger.info(f"Valid images: {valid_images}")
    logger.info(f"Corrupted images: {corrupted_images}")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Dataset Validation Report")
    logger.info("=" * 50)
    
    validate_text_dataset()
    logger.info("")
    validate_image_dataset()
    
    logger.info("")
    logger.info("Validation complete!")