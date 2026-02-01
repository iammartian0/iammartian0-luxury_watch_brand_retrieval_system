from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import uvicorn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watch_retrieval import WatchRetrievalSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Watch Multimodal Retrieval API",
    description="CLIP-based multimodal retrieval system for luxury watches",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retrieval system (lazy loading)
retrieval_system = None

def get_retrieval_system():
    """Get or initialize the retrieval system"""
    global retrieval_system
    if retrieval_system is None:
        logger.info("Initializing retrieval system...")
        retrieval_system = WatchRetrievalSystem()
        logger.info("Retrieval system initialized")
    return retrieval_system

# Pydantic models for request/response
class TextRetrievalRequest(BaseModel):
    query: str
    brand_filter: Optional[str] = None
    top_k: int = 5

class BrandRecommendationRequest(BaseModel):
    description: str
    top_k: int = 10

class SimilarWatches(BaseModel):
    brand: str
    image_path: str
    similarity: float
    distance: float

class TextWatchResult(BaseModel):
    brand: str
    model: str
    name: str
    price: str
    similarity: float
    distance: float

class ImageRetrievalResponse(BaseModel):
    detected_brand: Optional[str]
    similar_watches: List[SimilarWatches]
    brand_distribution: dict
    query_image: str

class TextRetrievalResponse(BaseModel):
    recommended_brand: Optional[str]
    results: List[TextWatchResult]
    query: str
    brand_filter: Optional[str]

class BrandRecommendationResponse(BaseModel):
    query: str
    rankings: List[dict]

class CrossModalResponse(BaseModel):
    image_path: str
    results: List[TextWatchResult]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    image_index_vectors: int
    text_index_vectors: int

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "name": "Watch Multimodal Retrieval API",
        "version": "1.0.0",
        "endpoints": {
            "predict_from_image": "/predict-from-image",
            "predict_from_text": "/predict-from-text",
            "recommend_brand": "/recommend-brand",
            "retrieve_text_for_image": "/retrieve-text-for-image",
            "health": "/health"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    system = get_retrieval_system()
    
    return HealthResponse(
        status="ok",
        model_loaded=system.model is not None,
        image_index_vectors=system.image_index.ntotal,
        text_index_vectors=system.text_index.ntotal
    )

@app.post("/predict-from-image", response_model=ImageRetrievalResponse)
async def predict_from_image(
    image: UploadFile = File(...),
    top_k: int = 5
):
    """Predict brand and find similar watches from an image
    
    Args:
        image: Uploaded image file
        top_k: Number of similar watches to return
    """
    try:
        system = get_retrieval_system()
        
        # Save uploaded image temporarily
        import tempfile
        import shutil
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image.filename.split('.')[-1]}") as temp_file:
            shutil.copyfileobj(image.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Perform retrieval
            result = system.retrieve_from_image(temp_path, top_k=top_k)
            
            # Convert image path to relative path for response
            query_image_rel = Path(temp_path).name
            
            # Format similar watches
            similar_watches = []
            for watch in result['similar_watches']:
                similar_watches.append(SimilarWatches(
                    brand=watch['brand'],
                    image_path=watch['image_path'],
                    similarity=watch['similarity'],
                    distance=watch['distance']
                ))
            
            return ImageRetrievalResponse(
                detected_brand=result['detected_brand'],
                similar_watches=similar_watches,
                brand_distribution=result['brand_distribution'],
                query_image=query_image_rel
            )
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error in predict_from_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-from-text", response_model=TextRetrievalResponse)
async def predict_from_text(request: TextRetrievalRequest):
    """Retrieve watches from text query
    
    Args:
        query: Text description to search for
        brand_filter: Optional brand to filter results
        top_k: Number of results to return
    """
    try:
        system = get_retrieval_system()
        
        result = system.retrieve_from_text(
            query=request.query,
            brand_filter=request.brand_filter,
            top_k=request.top_k
        )
        
        # Format results
        results = []
        for watch in result['results']:
            results.append(TextWatchResult(
                brand=watch['brand'],
                model=watch['model'],
                name=watch['name'],
                price=watch['price'],
                similarity=watch['similarity'],
                distance=watch['distance']
            ))
        
        return TextRetrievalResponse(
            recommended_brand=result['recommended_brand'],
            results=results,
            query=result['query'],
            brand_filter=result['brand_filter']
        )
    
    except Exception as e:
        logger.error(f"Error in predict_from_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-brand", response_model=BrandRecommendationResponse)
async def recommend_brand(request: BrandRecommendationRequest):
    """Recommend brands based on description
    
    Args:
        description: Text description to analyze
        top_k: Number of brand rankings to return
    """
    try:
        system = get_retrieval_system()
        
        result = system.recommend_brand(description=request.description)
        
        # Limit results to top_k
        rankings = result['rankings'][:request.top_k]
        
        return BrandRecommendationResponse(
            query=result['query'],
            rankings=rankings
        )
    
    except Exception as e:
        logger.error(f"Error in recommend_brand: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve-text-for-image", response_model=CrossModalResponse)
async def retrieve_text_for_image(
    image: UploadFile = File(...),
    top_k: int = 5
):
    """Cross-modal retrieval: find text descriptions similar to an image
    
    Args:
        image: Uploaded image file
        top_k: Number of text results to return
    """
    try:
        system = get_retrieval_system()
        
        # Save uploaded image temporarily
        import tempfile
        import shutil
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image.filename.split('.')[-1]}") as temp_file:
            shutil.copyfileobj(image.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Perform retrieval
            result = system.retrieve_text_for_image(temp_path, top_k=top_k)
            
            # Convert image path to relative path for response
            image_path_rel = Path(temp_path).name
            
            # Format results
            results = []
            for watch in result['results']:
                results.append(TextWatchResult(
                    brand=watch['brand'],
                    model=watch['model'],
                    name=watch['name'],
                    price=watch['price'],
                    similarity=watch['similarity'],
                    distance=watch['distance']
                ))
            
            return CrossModalResponse(
                image_path=image_path_rel,
                results=results
            )
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error in retrieve_text_for_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server"""
    logger.info(f"Starting Watch Retrieval API server...")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "watch_retrieval_api:app",
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    start_server(host="0.0.0.0", port=8000, reload=False)