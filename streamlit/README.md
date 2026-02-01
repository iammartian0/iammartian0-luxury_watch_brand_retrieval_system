# Watch Retrieval System - Streamlit Demo

An interactive web demo for the multimodal watch retrieval system using CLIP and FAISS.

## Features

- **Image Search**: Upload a watch image to find similar watches
- **Text Search**: Search using text descriptions with brand filters
- **Real-time Results**: Get results in <100ms
- **Visual Feedback**: Color-coded similarity scores
- **Responsive Design**: Works on desktop and mobile

## Installation

1. **Navigate to streamlit directory:**
   ```bash
   cd streamlit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Demo

### Option 1: Run locally
```bash
streamlit run app.py
```

The demo will open at: http://localhost:8501

### Option 2: Run from project root
```bash
streamlit run streamlit/app.py
```

## Usage

### Image Search
1. Click "Image Search" in the sidebar
2. Upload a watch image (JPG, PNG, WEBP)
3. Click "Find Similar Watches"
4. View top 6 similar watches with similarity scores

### Text Search
1. Click "Text Search" in the sidebar
2. Enter a watch description (e.g., "Rolex Submariner divers watch")
3. Optionally filter by brand
4. Select number of results (3, 5, 7, or 10)
5. Click "Search"
6. View matching watches with prices and similarity scores

## System Requirements

- Python 3.8+
- Streamlit 1.28+
- CLIP model (~300MB download on first run)
- FAISS indexes (~380MB)
- GPU recommended for faster inference

## Performance

- **Image Search**: 50-100ms (after model load)
- **Text Search**: 30-50ms (after model load)
- **First Query**: ~3 seconds (model loading)

## File Structure

```
streamlit/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                 # This file
```

## Troubleshooting

### "Module not found" error
Ensure you're running from the streamlit directory or adjust the Python path:
```bash
cd streamlit
streamlit run app.py
```

### Slow first query
The CLIP model (~300MB) is downloaded and cached on first use. Subsequent queries are much faster.

### Image not found
Make sure you're running from the correct directory so the app can find the `data/filtered/` image folder.

## Features Coming Soon (Optional Enhancements)

- Brand recommendation page
- Cross-modal retrieval (image → text)
- Dashboard with analytics
- Search history
- Save favorites

## Tech Stack

- **Streamlit**: Web UI framework
- **CLIP**: Vision-language model for embeddings
- **FAISS**: Vector similarity search
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library

## License

---

Built with ❤️ using CLIP + FAISS