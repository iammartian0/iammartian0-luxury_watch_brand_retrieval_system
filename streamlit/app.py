import streamlit as st
import sys
import os
import requests
import zipfile
from pathlib import Path

# Get the absolute path to the project root
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

# Add to Python path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

# Also change working directory to project root
os.chdir(PROJECT_ROOT)

# Configuration
IMAGES_RELEASE_URL = "https://github.com/iammartian0/luxury_watch_brand_retrieval_system/releases/download/v1.0-images/watch_images_demo.zip"
IMAGES_DIR = PROJECT_ROOT / "data" / "filtered"
ZIP_FILE = PROJECT_ROOT / "watch_images_demo.zip"

def download_images_if_needed():
    """Download and extract watch images from GitHub release if not present"""
    if IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()):
        return True
    
    try:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.warning("Watch images not found. Downloading from GitHub...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Download zip file
            status_text.text("Downloading images zip (this may take 1-2 minutes)...")
            response = requests.get(IMAGES_RELEASE_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(ZIP_FILE, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(min(progress, 0.7))
            
            # Extract zip file
            status_text.text("Extracting images...")
            progress_bar.progress(0.8)
            
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(IMAGES_DIR.parent)
            
            progress_bar.progress(1.0)
            status_text.text("Images downloaded and extracted successfully!")
            
            # Clean up zip file
            os.remove(ZIP_FILE)
            
            st.success("✅ Watch images loaded! You can now use the demo.")
            return True
            
    except Exception as e:
        st.error(f"Failed to download images: {str(e)}")
        st.info("Please try again later or contact the administrator.")
        return False

# Download images if needed
download_images_if_needed()

@st.cache_resource
def load_retrieval_system():
    """Load the watch retrieval system (cached across sessions)"""
    from watch_retrieval import WatchRetrievalSystem
    
    with st.spinner("Loading Watch Retrieval System... This may take a moment on first load."):
        system = WatchRetrievalSystem()
        return system

def get_image_path(relative_path):
    """Convert relative path to absolute path"""
    # Normalize path separators (handle both \ and / for cross-platform)
    relative_path = relative_path.replace('\\', '/')
    
    # The path from metadata is like "filtered/Omega/Image_34_e6d29b43.jpg"
    # We need to prepend "data/" since that's where it actually is
    base_path = PROJECT_ROOT / "data"
    return base_path / relative_path

def similarity_color(similarity):
    """Return color based on similarity score"""
    if similarity >= 0.85:
        return "#4CAF50"  # Green
    elif similarity >= 0.70:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red

def display_similar_watches(watches, title="Similar Watches"):
    """Display a grid of similar watches"""
    if not watches:
        st.info("No results found.")
        return
    
    st.subheader(title)
    
    # Debug info
    with st.expander("Debug: Paths"):
        st.write(f"Project root: {PROJECT_ROOT}")
        if watches:
            for i, watch in enumerate(watches[:2]):
                path = get_image_path(watch['image_path'])
                st.write(f"{i+1}. Path: {watch['image_path']}")
                st.write(f"   Full: {path}")
                st.write(f"   Exists: {path.exists()}")
    
    # Create columns for results
    cols = st.columns(min(len(watches), 3))
    
    for idx, watch in enumerate(watches):
        col = cols[idx % len(cols)]
        
        with col:
            # Get image path
            img_path = get_image_path(watch['image_path'])
            
            # Display image if it exists
            if img_path.exists():
                st.image(str(img_path), width="stretch")
            else:
                st.warning(f"Image not found: {img_path}")
                # Debug info
                with st.expander("Path details"):
                    st.write(f"Relative path: {watch['image_path']}")
                    st.write(f"Resolved path: {img_path}")
                    st.write(f"Exists: {img_path.exists()}")
            
            # Display watch info
            st.markdown(f"**Brand:** {watch['brand']}")
            
            # Similarity score with color
            similarity = watch['similarity']
            # Clip similarity to [0.0, 1.0] range for Streamlit progress bar
            similarity_clipped = max(0.0, min(1.0, similarity))
            color = similarity_color(similarity_clipped)
            st.markdown(
                f"<div style='background-color: {color}; color: white; padding: 5px; "
                f"border-radius: 5px; text-align: center;'>"
                f"Similarity: {similarity:.2%}</div>",
                unsafe_allow_html=True
            )
            
            # Progress bar (clipped to valid range)
            st.progress(similarity_clipped)
            
            # Image path
            with st.expander("View Details"):
                st.text(f"Path: {watch['image_path']}")
                st.text(f"Distance: {watch['distance']:.4f}")

def display_text_results(results, title="Search Results"):
    """Display text-based search results"""
    if not results:
        st.info("No results found.")
        return
    
    st.subheader(title)
    
    for idx, watch in enumerate(results, 1):
        with st.container():
            # Brand and model
            st.markdown(f"### {idx}. {watch['brand']} - {watch['model']}")
            
            # Similarity score (clip to valid range)
            similarity = watch['similarity']
            similarity_clipped = max(0.0, min(1.0, similarity))
            color = similarity_color(similarity_clipped)
            st.markdown(
                f"<span style='background-color: {color}; color: white; "
                f"padding: 3px 8px; border-radius: 3px;'>"
                f"Similarity: {similarity:.2%}</span> "
                f"<span style='margin-left: 10px;'>Price: {watch['price']}</span>",
                unsafe_allow_html=True
            )
            
            # Progress bar (clipped to valid range)
            st.progress(similarity_clipped)
            
            # Name/details
            if watch['name'] and str(watch['name']) != 'nan':
                with st.expander("Full Description"):
                    st.text(watch['name'])
            
            st.divider()

def display_cross_modal_results(results, title="Cross-Modal Search Results"):
    """Display cross-modal results (image -> text)"""
    if not results:
        st.info("No results found.")
        return

    st.subheader(title)
    
    for idx, watch in enumerate(results, 1):
        with st.container():
            st.markdown(f"### {idx}. {watch['brand']} - {watch['model']}")
            
            similarity = watch['similarity']
            similarity_clipped = max(0.0, min(1.0, similarity))
            color = similarity_color(similarity_clipped)
            
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.markdown(
                    f"<span style='background-color: {color}; color: white; "
                    f"padding: 3px 8px; border-radius: 3px;'>"
                    f"Similarity: {similarity:.2%}</span>",
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(f"**Price:** {watch['price']}")
            
            with col3:
                st.progress(similarity_clipped)
            
            if watch['name'] and str(watch['name']) != 'nan':
                with st.expander("Full Description"):
                    st.text(watch['name'])
            
            st.divider()

def display_brand_recommendations(rankings, title="Brand Recommendations"):
    """Display brand recommendation rankings"""
    if not rankings:
        st.info("No recommendations found.")
        return

    st.subheader(title)

    for idx, brand_rank in enumerate(rankings, 1):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.markdown(f"### {idx}. {brand_rank['brand']}")
            
            with col2:
                score = brand_rank['score']
                score_clipped = max(0.0, min(1.0, score))
                color = similarity_color(score_clipped)
                st.markdown(
                    f"<div style='background-color: {color}; color: white; "
                    f"padding: 5px; border-radius: 5px; text-align: center;'>"
                    f"Score: {score:.2%}</div>",
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(f"**Matches:** {brand_rank['count']}")
            
            with col4:
                st.progress(score_clipped)
            
            st.write(f"Distance: {brand_rank['distance']:.4f}")
            st.divider()

def display_brand_images(images, title="Brand Sample Images"):
    """Display sample images from a brand"""
    if not images:
        st.info("No images found for this brand.")
        return

    st.subheader(title)

    cols = st.columns(min(len(images), 4))

    for idx, img_info in enumerate(images):
        col = cols[idx % len(cols)]
        with col:
            img_path = get_image_path(img_info['image_path'])
            
            if img_path.exists():
                st.image(str(img_path), width="stretch")
            else:
                st.warning("Image not found")
            
            st.markdown(f"**Brand:** {img_info['brand']}")
            with st.expander("View Path"):
                st.text(img_info['image_path'])

def image_search_page(system):
    """Image Search Page"""
    st.header("📸 Image Search")
    st.write("Upload a watch image to find similar watches in our database.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Watch Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP"
    )
    
    if uploaded_file:
        # Display uploaded image
        st.subheader("Uploaded Image")
        st.image(uploaded_file, width="stretch")
        
        # Process button
        if st.button("🔍 Find Similar Watches", type="primary", use_container_width=True):
            with st.spinner("Analyzing image and finding matches..."):
                # Save uploaded file temporarily
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Perform retrieval
                    result = system.retrieve_from_image(tmp_path, top_k=6)
                    
                    # Display detected brand
                    st.success(f"Detected Brand: **{result['detected_brand']}**")
                    
                    # Display brand distribution
                    with st.expander("Brand Distribution in Top 6"):
                        for brand, count in result['brand_distribution'].items():
                            st.write(f"• {brand}: {count}")
                    
                    # Display similar watches
                    display_similar_watches(result['similar_watches'], "Top 6 Similar Watches")
                    
                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    else:
        # Display sample image
        st.info("No image uploaded yet. Try uploading a watch image to get started!")
        
        # Show sample result
        st.subheader("Example Result")
        sample_path = "data/filtered/Rolex/Image_10_137014c0.jpg"
        if Path(sample_path).exists():
            st.image(sample_path, caption="Sample Rolex Image", width="stretch")
            st.write("Upload your own watch image to see similar results!")

def text_search_page(system):
    """Text Search Page"""
    st.header("🔍 Text Search")
    st.write("Search for watches using text descriptions. Specify brand to filter results.")
    
    # Text input
    query = st.text_input(
        "Enter Watch Description",
        placeholder="e.g., 'Rolex Submariner divers watch' or 'elegant dress watch'",
        help="Describe the watch you're looking for"
    )
    
    # Brand filter
    col1, col2 = st.columns([2, 1])
    with col1:
        brand_filter = st.selectbox(
            "Filter by Brand",
            ["All Brands", "Rolex", "Omega", "Breitling", "Cartier", "Seiko",
             "Longines", "TAG Heuer", "Hublot", "Audemars Piguet", "Patek Philippe"]
        )
    
    with col2:
        top_k = st.selectbox(
            "Number of Results",
            options=[3, 5, 7, 10],
            index=1
        )
    
    # Process search
    if st.button("🔎 Search", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a search query.")
            return
        
        with st.spinner("Searching database..."):
            # Convert brand filter
            filter_brand = brand_filter if brand_filter != "All Brands" else None
            
            # Perform retrieval
            result = system.retrieve_from_text(query, brand_filter=filter_brand, top_k=top_k)
            
            # Display recommended brand
            if result['recommended_brand']:
                st.success(f"Recommended Brand: **{result['recommended_brand']}**")
            
            # Display results
            display_text_results(result['results'], f"Top {len(result['results'])} Results")

def cross_modal_page(system):
    """Cross-Modal Retrieval Page (Image -> Text)"""
    st.header("🔄 Cross-Modal Search")
    st.write("Upload a watch image to find the most similar watch descriptions and models from our database.")

    uploaded_file = st.file_uploader(
        "Upload Watch Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP"
    )

    if uploaded_file:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, width="stretch")

        top_k = st.selectbox(
            "Number of Text Results",
            options=[3, 5, 7, 10],
            index=1
        )

        if st.button("🔄 Find Text Descriptions", type="primary", use_container_width=True):
            with st.spinner("Analyzing image and matching text descriptions..."):
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    result = system.retrieve_text_for_image(tmp_path, top_k=top_k)
                    
                    st.success(f"Found **{len(result['results'])}** matching text descriptions")
                    display_cross_modal_results(result['results'], f"Top {len(result['results'])} Similar Watch Descriptions")

                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    else:
        st.info("No image uploaded yet. Upload a watch image to get text descriptions!")

        st.subheader("Example Result")
        sample_path = "data/filtered/Rolex/Image_10_137014c0.jpg"
        if Path(sample_path).exists():
            st.image(sample_path, caption="Sample Rolex Image", width="stretch")
            st.write("Upload your own watch image to get matching descriptions!")

def brand_recommendation_page(system):
    """Brand Recommendation Page"""
    st.header("🏆 Brand Explorer")
    st.write("Describe your ideal watch and we'll recommend the best brands for you.")

    description = st.text_area(
        "Describe Your Ideal Watch",
        placeholder="e.g., 'luxury diver watch with rotating bezel and steel bracelet' or 'elegant dress watch with leather strap'",
        help="Be descriptive for better recommendations"
    )

    top_k = st.selectbox(
        "Number of Brand Recommendations",
        options=[5, 10, 15],
        index=1
    )

    if st.button("🏆 Recommend Brands", type="primary", use_container_width=True):
        if not description:
            st.warning("Please enter a description.")
            return

        with st.spinner("Analyzing description and ranking brands..."):
            result = system.recommend_brand(description)
            
            st.success(f"Found **{len(result['rankings'])}** brand rankings based on your description")
            display_brand_recommendations(result['rankings'][:top_k], f"Top {top_k} Recommended Brands")

    st.divider()
    st.subheader("Example Queries")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sports/Luxury**")
        if st.button("Try: diver watch with rotating bezel"):
            st.session_state.example_query = "luxury diver watch with rotating bezel"
        if st.button("Try: chronograph with tachymeter"):
            st.session_state.example_query = "sporty chronograph with tachymeter"

    with col2:
        st.markdown("**Elegant/Formal**")
        if st.button("Try: elegant dress watch"):
            st.session_state.example_query = "elegant dress watch for formal occasions"
        if st.button("Try: luxury watch with leather strap"):
            st.session_state.example_query = "luxury timepiece with leather strap"

    if "example_query" in st.session_state:
        st.write(f"**Example loaded:** {st.session_state.example_query}")

def brand_browse_page(system):
    """Brand Image Browser Page"""
    st.header("🏢 Brand Gallery")
    st.write("Browse sample watch images from different luxury brands.")

    brand = st.selectbox(
        "Select a Brand",
        ["All Brands", "Rolex", "Omega", "Breitling", "Cartier", "Seiko",
         "Longines", "TAG Heuer", "Hublot", "Audemars Piguet", "Patek Philippe"]
    )

    limit = st.slider(
        "Number of Images to Display",
        min_value=3,
        max_value=12,
        value=6,
        step=1
    )

    if st.button("🏢 Browse Brand", type="primary", use_container_width=True):
        if brand == "All Brands":
            st.info("Please select a specific brand to browse.")
            return

        with st.spinner("Loading brand images..."):
            images = system.get_brand_images(brand, limit=limit)
            display_brand_images(images, f"{brand} Sample Images")

    st.divider()
    st.subheader("Quick Browse")
    quick_brands = ["Rolex", "Omega", "Cartier", "Patek Philippe"]
    cols = st.columns(len(quick_brands))
    for idx, b in enumerate(quick_brands):
        with cols[idx]:
            if st.button(f"Show {b}", key=f"quick_{b}"):
                images = system.get_brand_images(b, limit=6)
                display_brand_images(images, f"{b} Sample Images")

def main():
    """Main application"""
    st.set_page_config(
        page_title="Watch Retriever",
        page_icon="⌚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⌚ Multimodal Watch Retrieval System")
    st.markdown("Find similar watches using image upload, text search, cross-modal retrieval, or explore brands")
    
    # Load system (cached)
    system = load_retrieval_system()
    
    # Display system stats in sidebar
    st.sidebar.subheader("📊 System Stats")
    st.sidebar.write(f"**Index Size:**")
    st.sidebar.write(f"• Images: {system.image_index.ntotal:,}")
    st.sidebar.write(f"• Text Records: {system.text_index.ntotal:,}")
    st.sidebar.write(f"**Device:** {system.data_dir.parent.name}")
    
    st.sidebar.divider()
    
    # Page selection
    page = st.sidebar.radio(
        "Choose Search Method",
        ["📸 Image Search", "🔍 Text Search", "🔄 Cross-Modal", "🏆 Brand Explorer", "🏢 Brand Gallery"],
        label_visibility="collapsed"
    )
    
    # Render selected page
    if page == "📸 Image Search":
        image_search_page(system)
    elif page == "🔍 Text Search":
        text_search_page(system)
    elif page == "🔄 Cross-Modal":
        cross_modal_page(system)
    elif page == "🏆 Brand Explorer":
        brand_recommendation_page(system)
    else:
        brand_browse_page(system)
    
    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #888888; padding: 20px;'>"
        "Built with CLIP + FAISS | "
        "<a href='https://github.com/yourusername/watch-retriever' style='color: #1E88E5;'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()