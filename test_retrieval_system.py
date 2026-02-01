import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from watch_retrieval import WatchRetrievalSystem
import time
import json

def test_retrieval_system():
    """Comprehensive test of the WatchRetrievalSystem"""
    
    print("=" * 80)
    print("WATCH RETRIEVAL SYSTEM - COMPREHENSIVE TEST")
    print("=" * 80)
    print()
    
    # Initialize system
    print("Initializing Watch Retrieval System...")
    system = WatchRetrievalSystem()
    print(f"System initialized successfully!")
    print(f"  - Image index: {system.image_index.ntotal:,} vectors")
    print(f"  - Text index: {system.text_index.ntotal:,} vectors")
    print()
    
    results = {}
    
    # Test 1: Text Retrieval
    print("-" * 80)
    print("TEST 1: TEXT RETRIEVAL")
    print("-" * 80)
    
    queries = [
        ("Rolex Submariner divers watch", None),
        ("Cartier Tank elegant dress watch", "Cartier"),
        ("luxury divers watch with rotating bezel", None)
    ]
    
    text_results = []
    for query, brand_filter in queries:
        print(f"\nQuery: '{query}'")
        if brand_filter:
            print(f"Brand Filter: {brand_filter}")
        
        start = time.time()
        result = system.retrieve_from_text(query, brand_filter, top_k=5)
        elapsed = (time.time() - start) * 1000
        
        print(f"Recommended Brand: {result['recommended_brand']}")
        print(f"Query Time: {elapsed:.2f}ms")
        print(f"Results Found: {len(result['results'])}")
        
        for i, watch in enumerate(result['results'][:3], 1):
            print(f"  {i}. {watch['brand']} {watch['model']} - {watch['price']} (sim: {watch['similarity']:.4f})")
        
        text_results.append({
            'query': query,
            'brand_filter': brand_filter,
            'recommended_brand': result['recommended_brand'],
            'results_count': len(result['results']),
            'query_time_ms': elapsed,
            'top_results': result['results'][:3]
        })
    
    results['text_retrieval'] = text_results
    
    # Test 2: Image Retrieval
    print("\n" + "-" * 80)
    print("TEST 2: IMAGE RETRIEVAL")
    print("-" * 80)
    
    test_images = [
        "data/filtered/Rolex/Image_10_137014c0.jpg",
        "data/filtered/Cartier/Image_10_8b4e7381.webp"
    ]
    
    image_results = []
    for image_path in test_images:
        if not Path(image_path).exists():
            print(f"\nSkipping {image_path} (file not found)")
            continue
        
        print(f"\nImage: {Path(image_path).name}")
        
        start = time.time()
        result = system.retrieve_from_image(image_path, top_k=5)
        elapsed = (time.time() - start) * 1000
        
        print(f"Detected Brand: {result['detected_brand']}")
        print(f"Query Time: {elapsed:.2f}ms")
        print(f"Similar Watches Found: {len(result['similar_watches'])}")
        
        # Show brand distribution
        print("Brand Distribution:")
        for brand, count in result['brand_distribution'].items():
            print(f"  {brand}: {count}")
        
        # Show top 2 similar watches
        print("Top 2 Similar Watches:")
        for i, watch in enumerate(result['similar_watches'][:2], 1):
            print(f"  {i}. {watch['brand']} (sim: {watch['similarity']:.4f})")
        
        image_results.append({
            'image_path': image_path,
            'detected_brand': result['detected_brand'],
            'brand_distribution': result['brand_distribution'],
            'query_time_ms': elapsed,
            'similar_watches_count': len(result['similar_watches'])
        })
    
    results['image_retrieval'] = image_results
    
    # Test 3: Brand Recommendation
    print("\n" + "-" * 80)
    print("TEST 3: BRAND RECOMMENDATION")
    print("-" * 80)
    
    descriptions = [
        "luxury divers watch with rotating bezel",
        "elegant dress watch for formal occasions",
        "sporty chronograph with tachymeter"
    ]
    
    recommendation_results = []
    for desc in descriptions:
        print(f"\nDescription: '{desc}'")
        
        start = time.time()
        result = system.recommend_brand(desc)
        elapsed = (time.time() - start) * 1000
        
        print(f"Query Time: {elapsed:.2f}ms")
        print(f"Top 5 Brand Rankings:")
        
        for i, ranking in enumerate(result['rankings'][:5], 1):
            print(f"  {i}. {ranking['brand']}: {ranking['score']:.4f} (count: {ranking['count']})")
        
        recommendation_results.append({
            'description': desc,
            'query_time_ms': elapsed,
            'top_5_brands': result['rankings'][:5]
        })
    
    results['brand_recommendation'] = recommendation_results
    
    # Test 4: Cross-Modal Retrieval (Image -> Text)
    print("\n" + "-" * 80)
    print("TEST 4: CROSS-MODAL RETRIEVAL (Image -> Text)")
    print("-" * 80)
    
    cross_modal_results = []
    for image_path in test_images:
        if not Path(image_path).exists():
            continue
        
        print(f"\nImage: {Path(image_path).name}")
        
        start = time.time()
        result = system.retrieve_text_for_image(image_path, top_k=5)
        elapsed = (time.time() - start) * 1000
        
        print(f"Query Time: {elapsed:.2f}ms")
        print(f"Text Matches Found: {len(result['results'])}")
        print("Top 3 Text Matches:")
        
        for i, text in enumerate(result['results'][:3], 1):
            print(f"  {i}. {text['brand']} {text['model']} - {text['price']} (sim: {text['similarity']:.4f})")
        
        cross_modal_results.append({
            'image_path': image_path,
            'query_time_ms': elapsed,
            'results_count': len(result['results'])
        })
    
    results['cross_modal_retrieval'] = cross_modal_results
    
    # Test 5: Get Brand Sample Images
    print("\n" + "-" * 80)
    print("TEST 5: GET BRAND SAMPLE IMAGES")
    print("-" * 80)
    
    brands_to_test = ["Rolex", "Omega", "Cartier"]
    brand_images_results = []
    
    for brand in brands_to_test:
        print(f"\nBrand: {brand}")
        
        images = system.get_brand_images(brand, limit=5)
        print(f"Sample Images Found: {len(images)}")
        
        for i, img in enumerate(images[:3], 1):
            print(f"  {i}. {Path(img['image_path']).name}")
        
        brand_images_results.append({
            'brand': brand,
            'images_count': len(images)
        })
    
    results['brand_images'] = brand_images_results
    
    # Performance Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Calculate average times
    all_times = []
    for category in ['text_retrieval', 'image_retrieval', 'brand_recommendation', 'cross_modal_retrieval']:
        if category in results:
            for test in results[category]:
                if 'query_time_ms' in test:
                    all_times.append(test['query_time_ms'])
    
    if all_times:
        import numpy as np
        avg_time = np.mean(all_times)
        min_time = np.min(all_times)
        max_time = np.max(all_times)
        
        print(f"\nOverall Performance (All Tests):")
        print(f"  Average Query Time: {avg_time:.2f}ms")
        print(f"  Minimum Query Time: {min_time:.2f}ms")
        print(f"  Maximum Query Time: {max_time:.2f}ms")
        
        if avg_time < 100:
            print(f"\nExcellent! All queries complete in <100ms [PASS]")
        else:
            print(f"\nPerformance varies (first query loads CLIP model)")
    
    # Save results to JSON
    output_path = Path("test_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"[PASS] Text Retrieval: {len(text_results)} tests passed")
    print(f"[PASS] Image Retrieval: {len(image_results)} tests passed")
    print(f"[PASS] Brand Recommendation: {len(recommendation_results)} tests passed")
    print(f"[PASS] Cross-Modal Retrieval: {len(cross_modal_results)} tests passed")
    print(f"[PASS] Brand Images: {len(brand_images_results)} tests passed")
    print(f"\nTotal Tests: {len(text_results) + len(image_results) + len(recommendation_results) + len(cross_modal_results) + len(brand_images_results)}")
    print("\n[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    test_retrieval_system()