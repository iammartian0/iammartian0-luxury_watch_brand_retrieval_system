import csv
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_metadata_alignment():
    logger.info("=" * 60)
    logger.info("Fixing Metadata Alignment")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    filtered_dir = data_dir / "filtered"
    removed_dir = data_dir / "removed"
    original_metadata_path = data_dir / "image_metadata.csv"
    
    cleaned_metadata_path = data_dir / "cleaned_metadata.csv"
    rejected_metadata_path = data_dir / "rejected_metadata.csv"
    
    logger.info(f"Scanning filtered images in: {filtered_dir}")
    logger.info(f"Scanning removed images in: {removed_dir}")
    logger.info(f"Original metadata: {original_metadata_path}")
    
    filtered_images = list(filtered_dir.rglob("*.jpg")) + list(filtered_dir.rglob("*.png")) + list(filtered_dir.rglob("*.webp"))
    removed_images = list(removed_dir.rglob("*.jpg")) + list(removed_dir.rglob("*.png")) + list(removed_dir.rglob("*.webp"))
    
    logger.info(f"Found {len(filtered_images)} images in filtered/")
    logger.info(f"Found {len(removed_images)} images in removed/")
    
    original_metadata = {}
    with open(original_metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = Path(row['image_path']).name
            original_metadata[img_name] = row
    
    logger.info(f"Loaded {len(original_metadata)} entries from original metadata")
    
    cleaned_rows = []
    missing_in_original = []
    
    for img_path in filtered_images:
        img_name = img_path.name
        
        if img_name in original_metadata:
            original_row = original_metadata[img_name]
            cleaned_rows.append({
                'brand': original_row['brand'],
                'image_path': original_row['image_path'],
                'search_term': original_row['search_term'],
                'filtered_image_path': str(img_path.relative_to(data_dir))
            })
        else:
            missing_in_original.append(img_name)
            logger.warning(f"Filtered image not found in original metadata: {img_name}")
    
    logger.info(f"Built {len(cleaned_rows)} cleaned metadata rows")
    if missing_in_original:
        logger.warning(f"{len(missing_in_original)} filtered images missing from original metadata")
    
    removed_metadata = {}
    try:
        with open(rejected_metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = Path(row['image_path']).name
                removed_metadata[img_name] = row
        logger.info(f"Loaded {len(removed_metadata)} existing rejected metadata entries")
    except FileNotFoundError:
        logger.warning("rejected_metadata.csv not found, will create new")
        removed_metadata = {}
    
    newly_rejected = []
    for img_path in removed_images:
        img_name = img_path.name
        
        if img_name not in removed_metadata:
            if img_name in original_metadata:
                new_entry = {
                    **original_metadata[img_name],
                    'reject_reason': 'manually removed - false positive/quality issue'
                }
                removed_metadata[img_name] = new_entry
                newly_rejected.append(new_entry)
                logger.info(f"Added newly rejected: {img_name} ({original_metadata[img_name]['brand']})")
            else:
                logger.warning(f"Removed image not found in original metadata: {img_name}")
    
    logger.info(f"Added {len(newly_rejected)} new entries to rejected metadata")
    
    backup_cleaned = data_dir / "cleaned_metadata_backup.csv"
    backup_rejected = data_dir / "rejected_metadata_backup.csv"
    
    if cleaned_metadata_path.exists():
        if backup_cleaned.exists():
            backup_cleaned.unlink()
        cleaned_metadata_path.rename(backup_cleaned)
        logger.info(f"Backed up cleaned_metadata.csv to {backup_cleaned}")
    
    if rejected_metadata_path.exists():
        if backup_rejected.exists():
            backup_rejected.unlink()
        rejected_metadata_path.rename(backup_rejected)
        logger.info(f"Backed up rejected_metadata.csv to {backup_rejected}")
    
    with open(cleaned_metadata_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['brand', 'image_path', 'search_term', 'filtered_image_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    logger.info(f"Saved cleaned metadata: {cleaned_metadata_path} ({len(cleaned_rows)} entries)")
    
    with open(rejected_metadata_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['brand', 'image_path', 'search_term', 'reject_reason']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        all_rejected_rows = []
        for img_name, row in removed_metadata.items():
            cleaned_row = {k: v for k, v in row.items() if k in fieldnames}
            all_rejected_rows.append(cleaned_row)
        
        writer.writerows(all_rejected_rows)
    logger.info(f"Saved rejected metadata: {rejected_metadata_path} ({len(all_rejected_rows)} entries)")
    
    logger.info("=" * 60)
    logger.info("METADATA ALIGNMENT COMPLETE")
    logger.info("=" * 60)
    print(f"\nFiltered Images:")
    print(f"  Metadata: {len(cleaned_rows)} entries")
    print(f"  Files: {len(filtered_images)} images")
    print(f"  Status: {'MATCH' if len(cleaned_rows) == len(filtered_images) else 'MISMATCH'}")
    
    print(f"\nRemoved Images:")
    print(f"  Metadata: {len(all_rejected_rows)} entries")
    print(f"  Files: {len(removed_images)} images")
    print(f"  Newly added: {len(newly_rejected)} entries")
    print(f"  Status: {'MATCH' if len(all_rejected_rows) == len(removed_images) else 'MISMATCH'}")
    
    total_original = len(original_metadata)
    total_accounted = len(cleaned_rows) + len(all_rejected_rows)
    print(f"\nTotal Accounted: {total_accounted}/{total_original} original images")
    print(f"  ({total_accounted/total_original*100:.1f}%)")
    
    if missing_in_original:
        print(f"\nWarning: {len(missing_in_original)} filtered images not in original metadata")
        for img in missing_in_original[:10]:
            print(f"    - {img}")
        if len(missing_in_original) > 10:
            print(f"    ... and {len(missing_in_original) - 10} more")

if __name__ == "__main__":
    fix_metadata_alignment()