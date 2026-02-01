import torch
from PIL import Image
import transformers
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

def load_clip_model():
    logger.info("=" * 60)
    logger.info("Loading CLIP Model: openai/clip-vit-base-patch32")
    logger.info("=" * 60)
    
    model_id = "openai/clip-vit-base-patch32"
    model = transformers.CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    processor = transformers.CLIPProcessor.from_pretrained(model_id)
    
    logger.info("CLIP model loaded successfully")
    logger.info(f"Embedding dimension: 512")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, processor

def get_image_embeddings(image_paths, model, processor, batch_size=32):
    logger.info("=" * 60)
    logger.info(f"Generating embeddings for {len(image_paths)} images")
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 60)
    
    embeddings = []
    valid_paths = []
    valid_indices = []
    
    for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        
        batch_images = []
        batch_valid_indices = []
        batch_valid_paths = []
        
        for i, img_path in enumerate(batch_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                batch_valid_indices.append(batch_start + i)
                batch_valid_paths.append(img_path)
            except Exception as e:
                logger.warning(f"Failed to load {img_path.name}: {e}")
                continue
        
        if not batch_images:
            continue
        
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        batch_embeddings = image_features.cpu().numpy()
        
        for i in range(len(batch_embeddings)):
            embeddings.append(batch_embeddings[i])
            valid_paths.append(batch_valid_paths[i])
            valid_indices.append(batch_valid_indices[i])
    
    logger.info(f"Successfully processed {len(embeddings)}/{len(image_paths)} images")
    
    return np.array(embeddings), valid_paths, valid_indices

def get_text_embeddings(texts, model, processor, batch_size=100):
    logger.info("=" * 60)
    logger.info(f"Generating embeddings for {len(texts)} text records")
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 60)
    
    embeddings = []
    valid_indices = []
    
    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Processing text"):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        
        # Filter out non-string or empty texts
        valid_batch_texts = []
        valid_batch_indices = []
        for i, t in enumerate(batch_texts):
            if isinstance(t, str) and len(t.strip()) > 0:
                valid_batch_texts.append(t)
                valid_batch_indices.append(batch_start + i)
        
        if not valid_batch_texts:
            continue
        
        try:
            inputs = processor(text=valid_batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        except Exception as e:
            logger.warning(f"Failed to process text batch at {batch_start}-{batch_end}: {e}")
            continue
            
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        batch_embeddings = text_features.cpu().numpy()
        
        for i in range(len(batch_embeddings)):
            embeddings.append(batch_embeddings[i])
            valid_indices.append(valid_batch_indices[i])
    
    logger.info(f"Successfully processed {len(embeddings)}/{len(texts)} text records")
    
    return np.array(embeddings), valid_indices

def generate_image_embeddings():
    logger.info("=" * 60)
    logger.info("PHASE 1: IMAGE EMBEDDINGS")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    filtered_dir = data_dir / "filtered"
    metadata_path = data_dir / "cleaned_metadata.csv"
    
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        metadata = list(reader)
    
    logger.info(f"Loaded {len(metadata)} metadata entries")
    
    # Get all image paths
    image_paths = []
    for entry in metadata:
        # Try both relative and absolute paths
        relative_path = Path(entry['filtered_image_path'])
        if relative_path.exists():
            image_paths.append(relative_path)
        else:
            # Try with data directory prefix
            absolute_path = data_dir / entry['filtered_image_path']
            if absolute_path.exists():
                image_paths.append(absolute_path)
    
    logger.info(f"Found {len(image_paths)} valid image paths")
    
    # Generate embeddings
    model, processor = load_clip_model()
    embeddings, valid_paths, valid_indices = get_image_embeddings(image_paths, model, processor, batch_size=32)
    
    # Save embeddings
    embeddings_path = data_dir / "image_embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved image embeddings to {embeddings_path}")
    
    # Save metadata mapping
    mapping = {
        'total_images': len(image_paths),
        'processed_images': len(embeddings),
        'embedding_dim': 512,
        'valid_paths': [str(p) for p in valid_paths],
        'valid_indices': valid_indices
    }
    
    mapping_path = data_dir / "image_metadata_mapping.json"
    import json
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved metadata mapping to {mapping_path}")
    
    model.to('cpu')
    del model
    torch.cuda.empty_cache()
    
    return embeddings, valid_paths

def generate_text_embeddings():
    logger.info("=" * 60)
    logger.info("PHASE 2: TEXT EMBEDDINGS")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    text_data_path = data_dir / "cleaned_watches.csv"
    
    # Load text data
    logger.info(f"Loading text data from {text_data_path}")
    df = pd.read_csv(text_data_path, low_memory=False)
    
    logger.info(f"Loaded {len(df)} text records")
    
    # Extract text for embeddings (use combined_text column if available, otherwise construct)
    if 'combined_text' in df.columns:
        texts = df['combined_text'].fillna('').tolist()
    else:
        texts = [
            f"{row['brand']} {row.get('model', '')} {row['name']}" 
            for _, row in df.iterrows()
        ]
    
    logger.info(f"Processing {len(texts)} text records")
    
    # Generate embeddings
    model, processor = load_clip_model()
    embeddings, valid_indices = get_text_embeddings(texts, model, processor, batch_size=100)
    
    # Save embeddings
    embeddings_path = data_dir / "text_embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved text embeddings to {embeddings_path}")
    
    # Save metadata mapping
    mapping = {
        'total_texts': len(texts),
        'processed_texts': len(embeddings),
        'embedding_dim': 512,
        'valid_indices': valid_indices
    }
    
    mapping_path = data_dir / "text_metadata_mapping.json"
    import json
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved metadata mapping to {mapping_path}")
    
    model.to('cpu')
    del model
    torch.cuda.empty_cache()
    
    return embeddings, valid_indices

def main():
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING GENERATION STARTED")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    
    # Phase 1: Image embeddings
    image_embeddings, image_paths = generate_image_embeddings()
    
    logger.info(f"\nImage embeddings shape: {image_embeddings.shape}")
    logger.info(f"Image embeddings dtype: {image_embeddings.dtype}")
    logger.info(f"Image embeddings memory: {image_embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    # Phase 2: Text embeddings
    text_embeddings, text_indices = generate_text_embeddings()
    
    logger.info(f"\nText embeddings shape: {text_embeddings.shape}")
    logger.info(f"Text embeddings dtype: {text_embeddings.dtype}")
    logger.info(f"Text embeddings memory: {text_embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Image embeddings: {len(image_paths)} images")
    logger.info(f"Text embeddings: {len(text_indices)} text records")
    logger.info(f"Total storage used: {(image_embeddings.nbytes + text_embeddings.nbytes) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()