import torch
from PIL import Image
import transformers
import faiss
import numpy as np
import pandas as pd
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

class WatchRetrievalSystem:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.model = None
        self.processor = None
        self.image_index = None
        self.text_index = None
        self.image_metadata = None
        self.text_metadata = None
        self.clean_metadata = []
        
        self._load_indexes()
        self._load_metadata()
    
    def _load_model(self):
        """Lazy load CLIP model"""
        if self.model is None or self.processor is None:
            logger.info("Loading CLIP model...")
            model_id = "openai/clip-vit-base-patch32"
            self.model = transformers.CLIPModel.from_pretrained(
                model_id, 
                torch_dtype=torch.float16
            ).to(device)
            self.processor = transformers.CLIPProcessor.from_pretrained(model_id)
            logger.info("CLIP model loaded successfully")
    
    def _load_indexes(self):
        """Load FAISS indexes"""
        logger.info("Loading FAISS indexes...")

        image_index_path = self.data_dir / "image_index.faiss"
        text_index_path = self.data_dir / "text_index.faiss"

        if not image_index_path.exists():
            logger.error(f"Image index not found at {image_index_path}")
            raise FileNotFoundError(f"Image index file not found. Please ensure data files are downloaded.")

        if not text_index_path.exists():
            logger.error(f"Text index not found at {text_index_path}")
            raise FileNotFoundError(f"Text index file not found. Please ensure data files are downloaded.")

        self.image_index = faiss.read_index(str(image_index_path))
        self.text_index = faiss.read_index(str(text_index_path))

        logger.info(f"Image index: {self.image_index.ntotal} vectors")
        logger.info(f"Text index: {self.text_index.ntotal} vectors")
    
    def _load_metadata(self):
        """Load metadata mappings"""
        logger.info("Loading metadata...")
        
        # Load image metadata mapping
        with open(self.data_dir / "image_metadata_mapping.json", 'r') as f:
            self.image_metadata = json.load(f)
        
        # Load text metadata mapping
        with open(self.data_dir / "text_metadata_mapping.json", 'r') as f:
            self.text_metadata = json.load(f)
        
        # Load cleaned metadata (for image paths and brands)
        with open(self.data_dir / "cleaned_metadata.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.clean_metadata = list(reader)
        
        # Load text data (for text descriptions)
        self.text_data = pd.read_csv(self.data_dir / "cleaned_watches.csv", low_memory=False)
        
        logger.info(f"Image metadata: {self.image_metadata['total_images']} entries")
        logger.info(f"Text metadata: {self.text_metadata['total_texts']} entries")
        logger.info(f"Cleaned metadata: {len(self.clean_metadata)} entries")
        logger.info(f"Text data: {len(self.text_data)} rows")
    
    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate CLIP embedding for an image"""
        self._load_model()
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().astype('float32')
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate CLIP embedding for text"""
        self._load_model()
        
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().astype('float32')
    
    def retrieve_from_image(self, image_path: str, top_k: int = 5) -> Dict:
        """Retrieve similar watches from an image query
        
        Returns:
            Dict with:
                - detected_brand: most likely brand
                - similar_watches: list of similar watch entries with similarity scores
        """
        logger.info(f"Retrieving similar watches for image: {image_path}")
        
        # Generate query embedding
        query_embedding = self._get_image_embedding(image_path)
        
        # Search image index
        distances, indices = self.image_index.search(query_embedding, k=top_k)
        
        # Map indices to metadata
        similar_watches = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.clean_metadata):
                continue
            
            meta_entry = self.clean_metadata[int(idx)]
            similar_watches.append({
                'brand': meta_entry['brand'],
                'image_path': meta_entry['filtered_image_path'],
                'similarity': float(dist),
                'distance': float(1 - dist)  # Convert to distance for display
            })
        
        # Determine most likely brand (brand of top result)
        detected_brand = similar_watches[0]['brand'] if similar_watches else None
        
        # Count brands in top-k for confidence
        brand_counts = {}
        for watch in similar_watches:
            brand = watch['brand']
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        # Get brand with highest count
        if brand_counts:
            detected_brand = max(brand_counts, key=brand_counts.get)
        
        result = {
            'detected_brand': detected_brand,
            'similar_watches': similar_watches,
            'brand_distribution': brand_counts,
            'query_image': image_path
        }
        
        logger.info(f"Detected brand: {detected_brand}")
        logger.info(f"Found {len(similar_watches)} similar watches")
        
        return result
    
    def retrieve_from_text(self, query: str, brand_filter: Optional[str] = None, top_k: int = 5) -> Dict:
        """Retrieve watches from text query
        
        Args:
            query: Text description to search for
            brand_filter: Optional brand to filter results
            top_k: Number of results to return
        
        Returns:
            Dict with:
                - recommended_brand: most likely brand
                - results: list of matching text entries with similarity scores
        """
        logger.info(f"Retrieving watches for text query: {query}")
        if brand_filter:
            logger.info(f"Brand filter: {brand_filter}")
        
        # Generate query embedding
        query_embedding = self._get_text_embedding(query)
        
        # Search text index
        distances, indices = self.text_index.search(query_embedding, k=top_k * 10)  # Get more, then filter
        
        # Map indices to metadata
        results = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.text_data):
                continue
            
            row = self.text_data.iloc[int(idx)]
            
            # Apply brand filter if specified
            if brand_filter and row['brand'] != brand_filter:
                continue
            
            if len(results) >= top_k:
                break
            
            # Get brand information
            brand_info = row.get('brand', 'Unknown')
            
            results.append({
                'brand': brand_info,
                'model': row.get('model', 'Unknown'),
                'name': row.get('name', str(row.get('brand', ''))),
                'price': row.get('price', 'N/A'),
                'similarity': float(dist),
                'distance': float(1 - dist)
            })
        
        # Determine most likely brand
        if results:
            brand_counts = {}
            for result in results:
                brand = result['brand']
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
            recommended_brand = max(brand_counts, key=brand_counts.get)
        else:
            recommended_brand = None
        
        result = {
            'recommended_brand': recommended_brand,
            'results': results,
            'query': query,
            'brand_filter': brand_filter
        }
        
        logger.info(f"Recommended brand: {recommended_brand}")
        logger.info(f"Found {len(results)} matching watches")
        
        return result
    
    def retrieve_text_for_image(self, image_path: str, top_k: int = 5) -> Dict:
        """Retrieve text descriptions most similar to an image (cross-modal retrieval)"""
        logger.info(f"Cross-modal retrieval: image -> text ({image_path})")
        
        # Generate image embedding
        query_embedding = self._get_image_embedding(image_path)
        
        # Search text index
        distances, indices = self.text_index.search(query_embedding, k=top_k)
        
        # Map indices to metadata
        results = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.text_data):
                continue
            
            row = self.text_data.iloc[int(idx)]
            
            results.append({
                'brand': row.get('brand', 'Unknown'),
                'model': row.get('model', 'Unknown'),
                'name': row.get('name', str(row.get('brand', ''))),
                'price': row.get('price', 'N/A'),
                'similarity': float(dist),
                'distance': float(1 - dist)
            })
        
        result = {
            'image_path': image_path,
            'results': results
        }
        
        logger.info(f"Found {len(results)} text descriptions")
        
        return result
    
    def recommend_brand(self, description: str) -> Dict:
        """Recommend brands based on description
        
        Returns:
            Dict with:
                - rankings: list of (brand, score) tuples sorted by relevance
        """
        logger.info(f"Recommending brands for: {description}")
        
        # Generate query embedding
        query_embedding = self._get_text_embedding(description)
        
        # Search text index (get top 100 for brand aggregation)
        distances, indices = self.text_index.search(query_embedding, k=100)
        
        # Aggregate scores by brand
        brand_scores = {}
        brand_counts = {}
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.text_data):
                continue
            
            row = self.text_data.iloc[int(idx)]
            brand = row.get('brand', 'Unknown')
            
            if brand not in brand_scores:
                brand_scores[brand] = []
                brand_counts[brand] = 0
            
            brand_scores[brand].append(float(dist))
            brand_counts[brand] += 1
        
        # Calculate average score for each brand
        brand_rankings = []
        for brand, scores in brand_scores.items():
            avg_score = np.mean(scores)
            count = brand_counts[brand]
            brand_rankings.append({
                'brand': brand,
                'score': avg_score,
                'count': count,
                'distance': float(1 - avg_score)
            })
        
        # Sort by score (descending)
        brand_rankings.sort(key=lambda x: x['score'], reverse=True)
        
        result = {
            'query': description,
            'rankings': brand_rankings
        }
        
        logger.info(f"Generated rankings for {len(brand_rankings)} brands")
        
        return result
    
    def get_brand_images(self, brand: str, limit: int = 5) -> List[Dict]:
        """Get sample images from a specific brand"""
        brand_images = []
        
        for entry in self.clean_metadata:
            if entry['brand'] == brand:
                brand_images.append({
                    'brand': entry['brand'],
                    'image_path': entry['filtered_image_path']
                })
            
            if len(brand_images) >= limit:
                break
        
        return brand_images

def test_retrieval_system():
    """Test the retrieval system with sample queries"""
    logger.info("=" * 60)
    logger.info("TESTING WATCH RETRIEVAL SYSTEM")
    logger.info("=" * 60)
    
    system = WatchRetrievalSystem()
    
    # Test 1: Brand recommendation
    logger.info("\n--- Test 1: Brand Recommendation ---")
    result = system.recommend_brand("luxury divers watch with rotating bezel")
    print("\nTop 3 Recommended Brands:")
    for i, ranking in enumerate(result['rankings'][:3], 1):
        print(f"{i}. {ranking['brand']}: {ranking['score']:.4f} (distance: {ranking['distance']:.4f})")
    
    # Test 2: Text retrieval
    logger.info("\n--- Test 2: Text Retrieval ---")
    result = system.retrieve_from_text("Rolex Submariner divers watch", top_k=3)
    print(f"\nRecommended Brand: {result['recommended_brand']}")
    print("\nTop 3 Results:")
    for i, watch in enumerate(result['results'], 1):
        print(f"{i}. {watch['brand']} {watch['model']} - {watch['price']} (similarity: {watch['similarity']:.4f})")
    
    # Test 3: Brand-filtered text retrieval
    logger.info("\n--- Test 3: Brand-Filtered Retrieval ---")
    result = system.retrieve_from_text("elegant dress watch", brand_filter="Cartier", top_k=3)
    print("\nTop 3 Cartier Watches:")
    for i, watch in enumerate(result['results'], 1):
        print(f"{i}. {watch['brand']} {watch['model']} - {watch['price']} (similarity: {watch['similarity']:.4f})")
    
    # Test 4: Get brand sample images
    logger.info("\n--- Test 4: Brand Sample Images ---")
    images = system.get_brand_images("Rolex", limit=3)
    print(f"\nSample Rolex images: {len(images)}")
    for img in images:
        print(f"  - {img['image_path']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    test_retrieval_system()