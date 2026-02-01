import torch
from PIL import Image
import transformers
from pathlib import Path
import csv
import logging
import imagehash
import numpy as np
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

def load_clip_model():
    logger.info("Loading CLIP model...")
    model_id = "openai/clip-vit-base-patch32"
    model = transformers.CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    processor = transformers.CLIPProcessor.from_pretrained(model_id)
    logger.info("CLIP model loaded successfully")
    return model, processor

def classify_image(image, model, processor, class_prompts):
    inputs = processor(text=class_prompts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1)
    return probs[0].cpu().numpy()

def filter_single_watch(image_path, model, processor):
    try:
        image = Image.open(image_path).convert("RGB")
        
        single_watch_prompts = [
            "single watch product photograph",
            "one luxury watch isolated",
            "watch closeup product shot",
            "luxury timepiece alone"
        ]
        
        multiple_watches_prompts = [
            "multiple watches collection",
            "two watches side by side",
            "several watches display",
            "watch collection photography"
        ]
        
        person_prompts = [
            "person wearing luxury watch on wrist",
            "woman wearing wristwatch",
            "man with watch",
            "model wearing luxury watch"
        ]
        
        all_prompts = single_watch_prompts + multiple_watches_prompts + person_prompts
        probs = classify_image(image, model, processor, all_prompts)
        
        single_watch_avg = np.mean(probs[:len(single_watch_prompts)])
        multiple_watches_avg = np.mean(probs[len(single_watch_prompts):len(single_watch_prompts)+len(multiple_watches_prompts)])
        person_avg = np.mean(probs[len(single_watch_prompts)+len(multiple_watches_prompts):])
        
        logger.debug(f"Image: {image_path.name}")
        logger.debug(f"  Single watch: {single_watch_avg:.3f}")
        logger.debug(f"  Multiple watches: {multiple_watches_avg:.3f}")
        logger.debug(f"  With person: {person_avg:.3f}")
        
        is_single_watch = single_watch_avg > multiple_watches_avg
        no_person = person_avg < 0.40
        
        keep_reasons = []
        reject_reason = None
        
        if not is_single_watch:
            reject_reason = f"multiple watches (single={single_watch_avg:.3f}, multiple={multiple_watches_avg:.3f})"
        elif not no_person:
            reject_reason = f"person/model detected (person={person_avg:.3f})"
        else:
            keep_reasons.append("single watch")
            keep_reasons.append("no person")
        
        return is_single_watch and no_person, reject_reason, keep_reasons
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return False, f"processing error", []

def compute_phash(image_path):
    try:
        image = Image.open(image_path)
        return imagehash.phash(image)
    except Exception as e:
        logger.warning(f"Could not compute pHash for {image_path}: {e}")
        return None

def filter_duplicates(image_paths):
    logger.info("Filtering duplicates using pHash...")
    
    hash_to_image = {}
    duplicates = []
    
    for img_path in tqdm(image_paths, desc="Computing pHash"):
        h = compute_phash(img_path)
        if h is None:
            continue
        
        if h in hash_to_image:
            duplicates.append((img_path, hash_to_image[h], "pHash duplicate"))
        else:
            hash_to_image[h] = img_path
    
    logger.info(f"Found {len(duplicates)} pHash duplicates")
    return duplicates

def filter_near_duplicates(image_paths, embeddings, threshold=0.88):
    logger.info(f"Filtering near-duplicates with CLIP similarity > {threshold}...")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    duplicates = []
    
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            if image_paths[i].parent != image_paths[j].parent:
                continue
            
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            
            if similarity > threshold:
                duplicates.append((image_paths[i], image_paths[j], f"CLIP similarity {similarity:.3f}"))
    
    logger.info(f"Found {len(duplicates)} near-duplicate pairs")
    return duplicates

def get_image_embeddings(images, model, processor):
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy()

def clean_images():
    logger.info("=" * 60)
    logger.info("Starting Image Cleaning Pipeline (Improved Thresholds)")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    images_dir = data_dir / "images"
    metadata_path = data_dir / "image_metadata.csv"
    rejected_dir = data_dir / "removed"
    filtered_dir = data_dir / "filtered"
    cleaned_metadata_path = data_dir / "cleaned_metadata.csv"
    
    rejected_dir.mkdir(exist_ok=True)
    filtered_dir.mkdir(exist_ok=True)
    
    model, processor = load_clip_model()
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_metadata = list(reader)
    
    logger.info(f"Total images in metadata: {len(all_metadata)}")
    
    image_paths = [Path(row['image_path']) for row in all_metadata if Path(row['image_path']).exists()]
    logger.info(f"Valid image paths: {len(image_paths)}")
    
    kept_metadata = []
    rejected_metadata = []
    
    logger.info("=" * 60)
    logger.info("Step 1: Filtering for single watches (tuned thresholds)")
    logger.info("  - Single watch: avg(single) > avg(multiple)")
    logger.info("  - Person threshold: < 0.40")
    logger.info("=" * 60)
    
    for img_path, meta_row in tqdm(zip(image_paths, all_metadata), total=len(image_paths), desc="Filtering images"):
        should_keep, reject_reason, reasons = filter_single_watch(img_path, model, processor)
        
        if should_keep:
            kept_metadata.append(meta_row)
        else:
            rejected_metadata.append({
                **meta_row,
                'reject_reason': reject_reason
            })
            
            brand_dir = rejected_dir / meta_row['brand']
            brand_dir.mkdir(exist_ok=True)
            shutil.copy2(img_path, brand_dir / img_path.name)
    
    logger.info(f"\nRemoved multiple watches/people:")
    for reason in ["multiple watches", "person/model", "person/model detected"]:
        count = sum(1 for m in rejected_metadata if reason in m.get('reject_reason', ''))
        if count > 0:
            logger.info(f"  - {reason}: {count} images")
    
    low_conf_count = sum(1 for m in rejected_metadata if 'low confidence' in m.get('reject_reason', ''))
    if low_conf_count > 0:
        logger.info(f"  - low confidence single-watch: {low_conf_count} images")
    
    error_count = sum(1 for m in rejected_metadata if 'processing error' in m.get('reject_reason', ''))
    if error_count > 0:
        logger.info(f"  - processing error: {error_count} images")
    
    logger.info(f"Kept after content filtering: {len(kept_metadata)}")
    
    logger.info("=" * 60)
    logger.info("Copying filtered images to filtered directory...")
    logger.info("=" * 60)
    
    for meta_row in kept_metadata:
        img_path = Path(meta_row['image_path'])
        if img_path.exists():
            brand_dir = filtered_dir / meta_row['brand']
            brand_dir.mkdir(exist_ok=True)
            shutil.copy2(img_path, brand_dir / img_path.name)
            meta_row['filtered_image_path'] = str(brand_dir / img_path.name)
    
    logger.info(f"Copied {len(kept_metadata)} filtered images")
    
    logger.info("=" * 60)
    logger.info("Step 2: Removing exact duplicates (pHash)")
    logger.info("=" * 60)
    
    kept_paths = [Path(m['image_path']) for m in kept_metadata]
    
    phash_duplicates = filter_duplicates(kept_paths)
    
    duplicate_files = set()
    
    for dup, original, reason in phash_duplicates:
        if dup not in duplicate_files:
            for meta in list(kept_metadata):
                if meta['image_path'] == str(dup):
                    rejected_metadata.append({
                        **meta,
                        'reject_reason': reason
                    })
                    kept_metadata.remove(meta)
                    break
            duplicate_files.add(dup)
    
    logger.info(f"Kept after pHash deduplication: {len(kept_metadata)}")
    
    logger.info("=" * 60)
    logger.info("Step 3: Generating CLIP embeddings for near-duplicate detection")
    logger.info("=" * 60)
    
    final_paths = [Path(m['image_path']) for m in kept_metadata]
    
    embeddings_dict = {}
    
    for brand in set(m['brand'] for m in kept_metadata):
        brand_paths = [p for p in final_paths if p.parent.name == brand]
        if not brand_paths:
            continue
        
        logger.info(f"  Processing {brand}: {len(brand_paths)} images")
        
        brand_images = []
        valid_paths = []
        for img_path in brand_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                brand_images.append(img)
                valid_paths.append(img_path)
            except Exception as e:
                logger.warning(f"Could not load {img_path}: {e}")
                continue
        
        if brand_images:
            embeddings = get_image_embeddings(brand_images, model, processor)
            embeddings_dict[brand] = (valid_paths, embeddings)
    
    logger.info(f"Kept after CLIP embedding generation: {len(kept_metadata)}")
    
    logger.info("=" * 60)
    logger.info("Step 4: Removing near-duplicates")
    logger.info("=" * 60)
    
    for brand, (paths, embeddings) in embeddings_dict.items():
        near_dups = filter_near_duplicates(paths, embeddings)
        
        for dup, original, reason in near_dups:
            for meta in list(kept_metadata):
                if meta['image_path'] == str(dup):
                    if meta not in [m for m in rejected_metadata]:
                        rejected_metadata.append({
                            **meta,
                            'reject_reason': reason
                        })
                    if meta in kept_metadata:
                        kept_metadata.remove(meta)
                    break
    
    logger.info(f"Kept after near-duplicate removal: {len(kept_metadata)}")
    
    logger.info("=" * 60)
    logger.info("Step 5: Removing duplicate filtered images and updating metadata")
    logger.info("=" * 60)
    
    final_kept_metadata = []
    for meta_row in kept_metadata:
        img_path = Path(meta_row['image_path'])
        original_path = meta_row['filtered_image_path'] if 'filtered_image_path' in meta_row else None
        
        if original_path and Path(original_path).exists():
            final_kept_metadata.append(meta_row)
        else:
            logger.warning(f"Filtered image not found, skipping: {meta_row.get('image_path', 'unknown')}")
    
    kept_metadata = final_kept_metadata
    
    for meta_row in kept_metadata:
        img_path = Path(meta_row['image_path'])
        if img_path.exists():
            brand_dir = filtered_dir / meta_row['brand']
            brand_dir.mkdir(exist_ok=True)
            shutil.copy2(img_path, brand_dir / img_path.name)
            meta_row['filtered_image_path'] = str(brand_dir / img_path.name)
    
    logger.info(f"Final count after all filtering: {len(kept_metadata)}")
    
    logger.info("=" * 60)
    logger.info("Saving results...")
    logger.info("=" * 60)
    
    if kept_metadata:
        with open(cleaned_metadata_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['brand', 'image_path', 'search_term', 'filtered_image_path'])
            writer.writeheader()
            writer.writerows(kept_metadata)
    else:
        logger.warning("No images kept to save metadata!")
    
    cleaned_rejected_metadata = [{k: v for k, v in m.items() if k != 'filtered_image_path'} for m in rejected_metadata]
    
    with open(data_dir / 'rejected_metadata.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['brand', 'image_path', 'search_term', 'reject_reason'])
        writer.writeheader()
        writer.writerows(cleaned_rejected_metadata)
    
    logger.info("=" * 60)
    logger.info("CLEANING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Before: {len(all_metadata)} images")
    logger.info(f"After:  {len(kept_metadata)} images")
    logger.info(f"Removed: {len(rejected_metadata)} images")
    logger.info(f"Retention rate: {len(kept_metadata)/len(all_metadata)*100:.1f}%")
    logger.info(f"Cleaned metadata: {cleaned_metadata_path}")
    logger.info(f"Rejected metadata: {data_dir / 'rejected_metadata.csv'}")

if __name__ == "__main__":
    clean_images()