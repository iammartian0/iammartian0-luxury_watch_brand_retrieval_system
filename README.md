# Luxury Watch Retrieval System

A CLIP-powered multimodal search engine for luxury watches with both image and text search capabilities.

## 🎯 Features

- **Multimodal Search**: Query by image upload or text description
- **Zero-Shot Learning**: No training required - CLIP embeddings only
- **Fast Retrieval**: Sub-100ms similarity search with FAISS
- **10 Luxury Brands**: Rolex, Omega, Breitling, Cartier, Seiko, Longines, TAG Heuer, Hublot, Audemars Piguet, Patek Philippe
- **Cross-Modal**: Search text with images, find images from text descriptions
- **Brand Detection**: 95% accuracy on clear images
- **Production Demo**: Interactive Streamlit web interface

## 🚀 Live Demo

Try the interactive demo: **[Streamlit Cloud](https://luxurywatchbrandretrievalsystem-fcbhravztrw476ubnwvwev.streamlit.app/)**

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- CUDA-capable GPU (optional, for faster processing)

### Setup

```bash
# Clone repository
git clone https://github.com/iammartian0/luxury_watch_brand_retrieval_system.git
cd luxury_watch_brand_retrieval_system

# Install dependencies
pip install -r requirements.txt

```

## 📊 Dataset

### Images
- **Total Images**: 696 (after cleaning)
- **Brands**: 10 luxury watch brands
- **Format**: JPG, PNG, WEBP
- **Storage**: ~150MB (auto-downloaded from GitHub release)
- **Source**: DuckDuckGo image search, cleaned with CLIP

| Brand | Images |
|-------|--------|
| TAG Heuer | 86 |
| Breitling | 86 |
| Audemars Piguet | 82 |
| Longines | 78 |
| Seiko | 73 |
| Hublot | 73 |
| Omega | 70 |
| Cartier | 68 |
| Patek Philippe | 65 |
| Rolex | 63 |
| **Total** | **696** |

### Text Data
- **Total Records**: 194,301
- **Brands**: 10 luxury watch brands
- **Models**: 515 unique watch models
- **Source**: Kaggle datasets
- **Storage**: ~200MB (included in repository via Git LFS)

## 🛠️ Tech Stack

### Core ML/DL
- **CLIP** (OpenAI/clip-vit-base-patch32) - Vision-language embeddings
- **FAISS** (faiss-cpu) - Vector similarity search
- **PyTorch** (2.5.1) - Deep learning framework
- **Transformers** (Hugging Face) - CLIP model

### Backend & API
- **FastAPI** - REST API framework
- **Uvicorn** - ASGI server
- **Pillow** - Image processing

### Data & Analytics
- **Pandas** - DataFrame operations
- **NumPy** - Numerical computing
- **ImageHash** - Perceptual hashing for deduplication

### Web Interface
- **Streamlit** (1.28+) - Interactive web demo
- **Custom UI** - Dark theme, responsive design

## 📈 Performance Metrics

### Retrieval Speed
| Operation | Average | Min | Max |
|-----------|---------|-----|-----|
| Text Search | 32ms | 28ms | 40ms |
| Image Search | 37ms | 29ms | 95ms |
| Cross-Modal | 50ms | 48ms | 62ms |
| Brand Recommendation | 30ms | 31ms | 50ms |

### Accuracy
- **Brand Detection (Image)**: ~95% for clear images
- **Brand Recommendation**: High relevance (similarity >0.85 for top matches)
- **Text Search**: Accurate semantic matching

### Scalability
- **Indexed Vectors**: 194,997 total (696 images + 194,301 text)
- **Storage**: ~571 MB (embeddings + indexes)
- **Query Time**: <100ms (after model load)

## 🏗️ Project Structure

```
luxury_watch_brand_retrieval_system/
├── README.md                          # This file
├── PROJECT_SUMMARY.md                # Detailed architecture and implementation
├── IMAGES_SETUP_GUIDE.md             # GitHub release setup for images
├── requirements.txt                  # Python dependencies
├── streamlit/
│   ├── app.py                        # Streamlit web demo
│   ├── requirements.txt              # Streamlit dependencies
│   ├── README.md                     # Streamlit deployment guide
│   └── .streamlit/config.toml       # Streamlit configuration
├── src/
│   ├── watch_retrieval.py            # Core retrieval engine
│   ├── watch_retrieval_api.py        # FastAPI REST API
│   ├── generate_embeddings.py        # CLIP embedding generation
│   ├── build_faiss_index.py          # FAISS index creation
│   ├── clean_watch_improved.py       # Data cleaning with CLIP
│   └── download_watch_images.py      # Image scraper
├── data/
│   ├── filtered/                     # 696 cleaned images (auto-downloaded)
│   ├── image_embeddings.npy          # 696 × 512 image embeddings
│   ├── text_embeddings.npy           # 194K × 512 text embeddings (LFS)
│   ├── image_index.faiss             # FAISS image index
│   ├── text_index.faiss              # FAISS text index (LFS)
│   ├── cleaned_metadata.csv          # Image metadata
│   ├── cleaned_watches.csv           # Text dataset
│   └── *_mapping.json                # Metadata mappings
└── test_retrieval_system.py          # Comprehensive test suite
```

## 🎮 Usage Examples

### Python API

```python
from src.watch_retrieval import WatchRetrievalSystem

# Initialize
system = WatchRetrievalSystem()

# Image search
result = system.retrieve_from_image("watch.jpg", top_k=5)
print(f"Detected brand: {result['detected_brand']}")
for watch in result['similar_watches']:
    print(f"{watch['brand']} - similarity: {watch['similarity']:.2%}")

# Text search
result = system.retrieve_from_text("Rolex Submariner", top_k=5)
for watch in result['results']:
    print(f"{watch['brand']} - {watch['price']}")

# Brand recommendation
result = system.recommend_brand("elegant dress watch")
print(f"Top brand: {result['rankings'][0]['brand']}")

# Cross-modal: Image -> Text
result = system.retrieve_text_for_image("watch.jpg", top_k=3)
for text in result['results']:
    print(f"{text['brand']} - {text['name'][:100]}...")
```

### REST API

```bash
# Run FastAPI server
python src/watch_retrieval_api.py

```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_retrieval_system.py
```

### Docker (Optional)

```dockerfile
# See streamlit/README.md for Docker setup
```
## 🔄 System Architecture

### Data Pipeline
1. **Ingestion**: Download images (DuckDuckGo) + text (CSV)
2. **Cleaning**: CLIP zero-shot classification + ImageHash deduplication
3. **Embeddings**: CLIP ViT-B/32 (512-dim vectors)
4. **Indexing**: FAISS IndexFlatIP (cosine similarity)
5. **Retrieval**: Vector search with top-k results

### Search Flow
```
Query (Image/Text)
    ↓
CLIP Model (512-dim embedding)
    ↓
FAISS Index (nearest neighbor search)
    ↓
Results (top-k with similarity scores)
    ↓
Metadata lookup → Display
```