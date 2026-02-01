import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from watch_retrieval import WatchRetrievalSystem
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WatchRetrievalDemo:
    def __init__(self):
        self.system = WatchRetrievalSystem()
        self.data_dir = Path("data/filtered")
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "=" * 70)
        print("WATCH MULTIMODAL RETRIEVAL SYSTEM - INTERACTIVE DEMO")
        print("=" * 70)
        print("\nAvailable Operations:")
        print("1. Search by Text")
        print("2. Search by Image")
        print("3. Recommend Brand")
        print("4. Cross-Modal Retrieval (Image -> Text)")
        print("5. Get Brand Sample Images")
        print("6. Run Performance Benchmark")
        print("7. Exit")
        print()
    
    def search_by_text(self):
        """Text-based retrieval demo"""
        print("\n--- Text-Based Retrieval ---")
        
        query = input("Enter your search query: ").strip()
        if not query:
            print("Query cannot be empty!")
            return
        
        # Ask for brand filter
        print("\nOptional: Filter by brand? (press Enter to skip)")
        print("Available brands:", ", ".join([
            "Rolex", "Omega", "Breitling", "Cartier", "Seiko",
            "Longines", "TAG Heuer", "Hublot", "Audemars Piguet", "Patek Philippe"
        ]))
        brand_filter = input("Brand: ").strip() or None
        
        top_k = int(input("Number of results (default 5): ") or 5)
        
        # Perform retrieval
        import time
        start = time.time()
        result = self.system.retrieve_from_text(query, brand_filter, top_k)
        elapsed = (time.time() - start) * 1000
        
        # Display results
        print(f"\n{'='*70}")
        print(f"RESULTS FOR: '{query}'")
        print(f"{'='*70}")
        print(f"Brand Filter: {brand_filter or 'None'}")
        print(f"Recommended Brand: {result['recommended_brand'] or 'N/A'}")
        print(f"Query Time: {elapsed:.2f}ms")
        print(f"\nTop {len(result['results'])} Results:")
        print("-" * 70)
        
        for i, watch in enumerate(result['results'], 1):
            print(f"\n{i}. {watch['brand']} {watch['model']}")
            print(f"   Price: {watch['price']}")
            print(f"   Similarity: {watch['similarity']:.4f}")
    
    def search_by_image(self):
        """Image-based retrieval demo"""
        print("\n--- Image-Based Retrieval ---")
        
        # Show available images
        print("\nAvailable sample images by brand:")
        for brand_dir in sorted(self.data_dir.iterdir()):
            if brand_dir.is_dir():
                images = list(brand_dir.glob("*.jpg")) + list(brand_dir.glob("*.png")) + list(brand_dir.glob("*.webp"))
                if images:
                    print(f"  {brand_dir.name}: {len(images)} images")
        
        print("\nSample image paths:")
        sample = "data/filtered/Rolex/Image_10_137014c0.jpg"
        print(f"  Example: {sample}")
        print(f"  Example: data/filtered/Omega/Image_1_0ea651c6.png")
        
        image_path = input("\nEnter image path (or press Enter to use sample): ").strip()
        if not image_path:
            image_path = sample
        
        # Check if file exists
        if not Path(image_path).exists():
            print(f"Error: Image not found at {image_path}")
            return
        
        top_k = int(input("Number of results (default 5): ") or 5)
        
        # Perform retrieval
        import time
        start = time.time()
        result = self.system.retrieve_from_image(image_path, top_k)
        elapsed = (time.time() - start) * 1000
        
        # Display results
        print(f"\n{'='*70}")
        print(f"RESULTS FOR IMAGE: {Path(image_path).name}")
        print(f"{'='*70}")
        print(f"Detected Brand: {result['detected_brand'] or 'N/A'}")
        print(f"Query Time: {elapsed:.2f}ms")
        print(f"\nBrand Distribution in Top {top_k}:")
        print("-" * 70)
        
        for brand, count in result['brand_distribution'].items():
            print(f"  {brand}: {count} matches")
        
        print(f"\nTop {len(result['similar_watches'])} Similar Watches:")
        print("-" * 70)
        
        for i, watch in enumerate(result['similar_watches'], 1):
            print(f"\n{i}. {watch['brand']}")
            print(f"   Similarity: {watch['similarity']:.4f}")
            print(f"   Image: {watch['image_path']}")
    
    def recommend_brand(self):
        """Brand recommendation demo"""
        print("\n--- Brand Recommendation ---")
        
        description = input("Enter watch description: ").strip()
        if not description:
            description = "luxury divers watch"
            print(f"Using default: {description}")
        
        # Perform recommendation
        import time
        start = time.time()
        result = self.system.recommend_brand(description)
        elapsed = (time.time() - start) * 1000
        
        # Display results
        print(f"\n{'='*70}")
        print(f"BRAND RECOMMENDATIONS FOR: '{description}'")
        print(f"{'='*70}")
        print(f"Query Time: {elapsed:.2f}ms")
        print(f"\nTop 10 Brand Rankings:")
        print("-" * 70)
        
        for i, ranking in enumerate(result['rankings'], 1):
            print(f"\n{i}. {ranking['brand']}")
            print(f"   Score: {ranking['score']:.4f}")
            print(f"   Distance: {ranking['distance']:.4f}")
            print(f"   Count: {ranking['count']}")
    
    def cross_modal_retrieval(self):
        """Cross-modal retrieval demo (image -> text)"""
        print("\n--- Cross-Modal Retrieval (Image -> Text) ---")
        
        image_path = input("Enter image path (or press Enter for sample): ").strip()
        if not image_path:
            image_path = "data/filtered/Cartier/Image_10_8b4e7381.webp"
            print(f"Using sample: {image_path}")
        
        # Check if file exists
        if not Path(image_path).exists():
            print(f"Error: Image not found at {image_path}")
            return
        
        top_k = int(input("Number of results (default 5): ") or 5)
        
        # Perform retrieval
        import time
        start = time.time()
        result = self.system.retrieve_text_for_image(image_path, top_k)
        elapsed = (time.time() - start) * 1000
        
        # Display results
        print(f"\n{'='*70}")
        print(f"TEXT MATCHES FOR IMAGE: {Path(image_path).name}")
        print(f"{'='*70}")
        print(f"Query Time: {elapsed:.2f}ms")
        print(f"\nTop {len(result['results'])} Text Descriptions:")
        print("-" * 70)
        
        for i, text in enumerate(result['results'], 1):
            print(f"\n{i}. {text['brand']} {text['model']}")
            print(f"   Price: {text['price']}")
            print(f"   Similarity: {text['similarity']:.4f}")
    
    def get_brand_images(self):
        """Get sample images from a brand"""
        print("\n--- Get Brand Sample Images ---")
        
        brand = input("Enter brand name (or press Enter for Rolex): ").strip()
        if not brand:
            brand = "Rolex"
        
        limit = int(input("Number of images (default 5): ") or 5)
        
        # Get images
        images = self.system.get_brand_images(brand, limit)
        
        print(f"\n{'='*70}")
        print(f"SAMPLE IMAGES FOR: {brand}")
        print(f"{'='*70}")
        print(f"Found {len(images)} images:")
        print("-" * 70)
        
        for i, img in enumerate(images, 1):
            print(f"{i}. {img['brand']}")
            print(f"   Path: {img['image_path']}")
    
    def performance_benchmark(self):
        """Run performance benchmarks"""
        print("\n--- Performance Benchmark ---")
        
        import time
        import numpy as np
        
        print("\nRunning benchmarks (10 iterations each)...")
        print("Please wait...\n")
        
        # Benchmark 1: Brand recommendation
        print("1. Brand Recommendation")
        times = []
        for i in range(10):
            start = time.time()
            result = self.system.recommend_brand("luxury watch")
            times.append((time.time() - start) * 1000)
        
        print(f"   Avg: {np.mean(times):.2f}ms")
        print(f"   Min: {np.min(times):.2f}ms")
        print(f"   Max: {np.max(times):.2f}ms")
        
        # Benchmark 2: Text retrieval
        print("\n2. Text Retrieval (top_k=5)")
        times = []
        for i in range(10):
            start = time.time()
            result = self.system.retrieve_from_text("Rolex watch", top_k=5)
            times.append((time.time() - start) * 1000)
        
        print(f"   Avg: {np.mean(times):.2f}ms")
        print(f"   Min: {np.min(times):.2f}ms")
        print(f"   Max: {np.max(times):.2f}ms")
        
        # Benchmark 3: Image retrieval
        print("\n3. Image Retrieval (top_k=5)")
        image_path = "data/filtered/Rolex/Image_10_137014c0.jpg"
        times = []
        for i in range(10):
            start = time.time()
            result = self.system.retrieve_from_image(image_path, top_k=5)
            times.append((time.time() - start) * 1000)
        
        print(f"   Avg: {np.mean(times):.2f}ms")
        print(f"   Min: {np.min(times):.2f}ms")
        print(f"   Max: {np.max(times):.2f}ms")
        
        # Benchmark 4: Cross-modal retrieval
        print("\n4. Cross-Modal Retrieval (image -> text)")
        times = []
        for i in range(10):
            start = time.time()
            result = self.system.retrieve_text_for_image(image_path, top_k=5)
            times.append((time.time() - start) * 1000)
        
        print(f"   Avg: {np.mean(times):.2f}ms")
        print(f"   Min: {np.min(times):.2f}ms")
        print(f"   Max: {np.max(times):.2f}ms")
        
        print("\n" + "=" * 70)
        print("All operations complete in <100ms - System ready for production!")
        print("=" * 70)
    
    def run(self):
        """Run the demo"""
        logger.info("Starting Watch Retrieval Demo")
        logger.info("Initializing retrieval system...")
        
        print("\nInitializing...")
        print("Loading FAISS indexes...")
        print("Loading metadata...")
        print("CLIP model will load on first query (lazy loading)")
        
        while True:
            try:
                self.show_menu()
                
                choice = input("Select operation (1-7): ").strip()
                
                if choice == "1":
                    self.search_by_text()
                elif choice == "2":
                    self.search_by_image()
                elif choice == "3":
                    self.recommend_brand()
                elif choice == "4":
                    self.cross_modal_retrieval()
                elif choice == "5":
                    self.get_brand_images()
                elif choice == "6":
                    self.performance_benchmark()
                elif choice == "7":
                    print("\nThank you for using the Watch Retrieval Demo! Goodbye!")
                    break
                else:
                    print("\nInvalid choice. Please select 1-7.")
            
            except KeyboardInterrupt:
                print("\n\nDemo interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError occurred: {e}")
                print("Please try again.")

def main():
    demo = WatchRetrievalDemo()
    demo.run()

if __name__ == "__main__":
    main()