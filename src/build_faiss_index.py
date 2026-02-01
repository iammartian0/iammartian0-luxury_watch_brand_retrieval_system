import numpy as np
import faiss
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_image_index():
    logger.info("=" * 60)
    logger.info("PHASE 1: BUILDING IMAGE SEARCH INDEX")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    embeddings_path = data_dir / "image_embeddings.npy"
    metadata_path = data_dir / "image_metadata_mapping.json"
    index_path = data_dir / "image_index.faiss"
    cleaned_metadata_path = data_dir / "cleaned_metadata.csv"
    
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Embeddings dtype: {embeddings.dtype}")
    
    # Load metadata mapping
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Metadata loaded:")
    logger.info(f"  Total images: {metadata['total_images']}")
    logger.info(f"  Processed images: {metadata['processed_images']}")
    
    # Load cleaned metadata for brand/image mapping
    import csv
    cleaned_metadata = []
    with open(cleaned_metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_metadata.append(row)
    logger.info(f"Loaded {len(cleaned_metadata)} cleaned metadata entries")
    
    # Create FAISS index
    logger.info("Creating FAISS IndexFlatIP (cosine similarity)")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    logger.info(f"Index dimension: {dimension}")
    logger.info(f"Index type: Exact search (IndexFlatIP)")
    
    # Add vectors to index
    logger.info("Adding vectors to index...")
    index.add(embeddings.astype('float32'))
    logger.info(f"Total vectors in index: {index.ntotal}")
    
    # Save index
    logger.info(f"Saving index to {index_path}")
    faiss.write_index(index, str(index_path))
    logger.info("Index saved successfully")
    
    # Create index metadata
    index_metadata = {
        'index_type': 'IndexFlatIP',
        'dimension': dimension,
        'total_vectors': int(index.ntotal),
        'metric': 'Inner Product (cosine similarity with normalized vectors)',
        'embedding_file': str(embeddings_path.relative_to(data_dir)),
        'metadata_file': str(metadata_path.relative_to(data_dir))
    }
    
    index_metadata_path = data_dir / "image_index_metadata.json"
    with open(index_metadata_path, 'w') as f:
        json.dump(index_metadata, f, indent=2)
    logger.info(f"Index metadata saved to {index_metadata_path}")
    
    logger.info("=" * 60)
    logger.info("IMAGE INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    
    return index, metadata

def build_text_index():
    logger.info("=" * 60)
    logger.info("PHASE 2: BUILDING TEXT SEARCH INDEX")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    embeddings_path = data_dir / "text_embeddings.npy"
    metadata_path = data_dir / "text_metadata_mapping.json"
    index_path = data_dir / "text_index.faiss"
    
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Embeddings dtype: {embeddings.dtype}")
    
    # Load metadata mapping
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Metadata loaded:")
    logger.info(f"  Total texts: {metadata['total_texts']}")
    logger.info(f"  Processed texts: {metadata['processed_texts']}")
    
    # Create FAISS index
    logger.info("Creating FAISS IndexFlatIP (cosine similarity)")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    logger.info(f"Index dimension: {dimension}")
    logger.info(f"Index type: Exact search (IndexFlatIP)")
    
    # Add vectors to index
    logger.info("Adding vectors to index...")
    index.add(embeddings.astype('float32'))
    logger.info(f"Total vectors in index: {index.ntotal}")
    
    # Save index
    logger.info(f"Saving index to {index_path}")
    faiss.write_index(index, str(index_path))
    logger.info("Index saved successfully")
    
    # Create index metadata
    index_metadata = {
        'index_type': 'IndexFlatIP',
        'dimension': dimension,
        'total_vectors': int(index.ntotal),
        'metric': 'Inner Product (cosine similarity with normalized vectors)',
        'embedding_file': str(embeddings_path.relative_to(data_dir)),
        'metadata_file': str(metadata_path.relative_to(data_dir))
    }
    
    index_metadata_path = data_dir / "text_index_metadata.json"
    with open(index_metadata_path, 'w') as f:
        json.dump(index_metadata, f, indent=2)
    logger.info(f"Index metadata saved to {index_metadata_path}")
    
    logger.info("=" * 60)
    logger.info("TEXT INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    
    return index, metadata

def validate_indexes(image_index, text_index):
    logger.info("=" * 60)
    logger.info("VALIDATING INDEXES")
    logger.info("=" * 60)
    
    # Test image index
    logger.info("\nImage Index Validation:")
    logger.info(f"  Dimension: {image_index.d}")
    logger.info(f"  Total vectors: {image_index.ntotal}")
    logger.info(f"  Index type: {type(image_index).__name__}")
    
    # Perform search (test)
    if image_index.ntotal > 0:
        dummy_query = np.random.rand(1, image_index.d).astype('float32')
        distances, indices = image_index.search(dummy_query, k=5)
        logger.info(f"  Test search: Found {len(indices[0])} results")
        logger.info(f"  Distance range: {distances[0].min():.4f} to {distances[0].max():.4f}")
    
    # Test text index
    logger.info("\nText Index Validation:")
    logger.info(f"  Dimension: {text_index.d}")
    logger.info(f"  Total vectors: {text_index.ntotal}")
    logger.info(f"  Index type: {type(text_index).__name__}")
    
    # Perform search (test)
    if text_index.ntotal > 0:
        dummy_query = np.random.rand(1, text_index.d).astype('float32')
        distances, indices = text_index.search(dummy_query, k=5)
        logger.info(f"  Test search: Found {len(indices[0])} results")
        logger.info(f"  Distance range: {distances[0].min():.4f} to {distances[0].max():.4f}")

def main():
    logger.info("\n" + "=" * 60)
    logger.info("FAISS INDEX BUILDING STARTED")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    
    # Phase 1: Build image index
    image_index, image_metadata = build_image_index()
    
    # Phase 2: Build text index
    text_index, text_metadata = build_text_index()
    
    # Phase 3: Validate indexes
    validate_indexes(image_index, text_index)
    
    logger.info("\n" + "=" * 60)
    logger.info("FAISS INDEX BUILDING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nIndex Files Created:")
    logger.info(f"  Image Index: {data_dir / 'image_index.faiss'}")
    logger.info(f"  Text Index: {data_dir / 'text_index.faiss'}")
    logger.info(f"  Image Metadata: {data_dir / 'image_index_metadata.json'}")
    logger.info(f"  Text Metadata: {data_dir / 'text_index_metadata.json'}")
    logger.info(f"\nTotal Searchable Vectors:")
    logger.info(f"  Images: {image_index.ntotal:,}")
    logger.info(f"  Texts: {text_index.ntotal:,}")
    logger.info(f"  Combined: {image_index.ntotal + text_index.ntotal:,}")

if __name__ == "__main__":
    main()