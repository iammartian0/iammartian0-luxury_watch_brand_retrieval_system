import sys
import os
from pathlib import Path

def check_paths():
    """Verify all data paths exists and are accessible"""
    results = []
    
    # Get project root from current working directory
    import os
    PROJECT_ROOT = Path(os.getcwd()).resolve()
    DATA_DIR = PROJECT_ROOT / "data"
    
    results.append(f"Project root: {PROJECT_ROOT}")
    results.append(f"Data directory: {DATA_DIR}")
    results.append(f"Data dir exists: {DATA_DIR.exists()}")
    
    # Check essential files
    essential_files = [
        "data/image_index.faiss",
        "data/text_index.faiss",
        "data/image_embeddings.npy", 
        "data/text_embeddings.npy",
        "data/cleaned_metadata.csv",
        "data/cleaned_watches.csv"
    ]
    
    results.append("\nEssential files:")
    for file_path in essential_files:
        full_path = DATA_DIR / Path(file_path).name
        exists = full_path.exists()
        results.append(f"  {file_path}: {'OK' if exists else 'MISSING'}")
    
    # Check filtered images directory
    filtered_dir = DATA_DIR / "filtered"
    results.append(f"\nImages directory: {filtered_dir}")
    results.append(f"Exists: {filtered_dir.exists()}")
    
    if filtered_dir.exists():
        image_count = len(list(filtered_dir.rglob("*.jpg")) + list(filtered_dir.rglob("*.png")) + list(filtered_dir.rglob("*.webp")))
        results.append(f"Total images: {image_count}")
    
    return "\n".join(results)

if __name__ == "__main__":
    print("=" * 60)
    print("PATH VERIFICATION")
    print("=" * 60)
    print(check_paths())
    print("=" * 60)