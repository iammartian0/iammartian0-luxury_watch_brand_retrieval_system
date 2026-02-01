# Quick Start Guide

## Windows (Recommended)

### 1. Activate Virtual Environment
```cmd
venv_gpu\Scripts\activate
```

### 2. Install Streamlit (if not already installed)
```cmd
pip install streamlit
```

### 3. Start the Demo
```cmd
start_streamlit.bat
```

Or manually:
```cmd
streamlit run streamlit/app.py
```

The demo will automatically open in your browser at: **http://localhost:8501**

---

## Linux/Mac

### 1. Activate Virtual Environment
```bash
source venv_gpu/bin/activate
```

### 2. Install Streamlit
```bash
pip install streamlit
```

### 3. Start the Demo
```bash
streamlit run streamlit/app.py
```

---

## Demo Features

### 📸 Image Search
- Upload any watch image (JPG, PNG, WEBP)
- Get top 6 similar watches
- View brand detection accuracy
- Color-coded similarity scores

### 🔍 Text Search  
- Search using natural language
- Filter by brand (10 luxury brands)
- Adjustable number of results
- See prices and similarity scores

---

## Performance Tips

### First Query
- Expect ~3 seconds (CLIP model loading)
- Subsequent queries are <100ms
- Model is cached across sessions

### Optimize Performance
- Keep browser tab open (server keeps running)
- Use text search for faster initial results
- GPU-enabled for fastest inference

---

## Troubleshooting

### Port Already in Use
If port 8501 is busy, use a different port:
```cmd
streamlit run streamlit/app.py --server.port 8502
```

### Module Not Found
Ensure you're in the correct directory:
```cmd
cd C:\Users\varun\CLAUDETWO
streamlit run streamlit/app.py
```

### Images Not Loading
Make sure the `data/filtered/` folder exists and contains images.

---

## Demo Scenarios

### Scenario 1: Find Similar Watch
1. Go to Image Search
2. Upload a watch photo
3. Click "Find Similar Watches"
4. View top 6 matches with similarity scores

### Scenario 2: Text-Based Search
1. Go to Text Search
2. Enter: "Rolex Submariner divers watch"
3. Select brand filter (optional)
4. Click "Search"
5. View matching watches with prices

---

## Recording for Demo

To record your demo for CV/portfolio:

### Windows
1. Press **Win + G** to open Game Bar
2. Click "Record" (or Win + Alt + R)
3. Demonstrate features
4. Stop recording
5. Video saved in: `C:\Users\<username>\Videos\Captures`

### Mac
1. Press **Cmd + Shift + 5** for screenshot selection
2. Or use QuickTime Player for video recording

---

## Sharing Your Demo

### Local Share
- Send someone the localhost URL (if on same network)

### Deploy (Optional)
```bash
streamlit run streamlit/app.py --server.headless true
```

### Streamlit Cloud (Free Hosting)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repository
4. Deploy in 1-click

---

## Customization

### Change Theme
Edit `streamlit/.streamlit/config.toml`

### Add More Brands
Edit `streamlit/app.py` - add to brand_filter selectbox

### Adjust Results Limit
Edit `top_k` options in text_search_page()

---

## Support

For issues or questions:
- Check README.md for full documentation
- View test results: `test_results.json`
- System logs available in console