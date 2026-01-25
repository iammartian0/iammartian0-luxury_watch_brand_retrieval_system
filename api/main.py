from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import base64
import json
from typing import Dict, Any
from src.fusion.multimodal import MultiModalWatchClassifier

app = FastAPI(title="Luxury Watch Classifier API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalWatchClassifier().to(device)
model.eval()  # Set to evaluation mode

# Mock luxury watch classes
WATCH_CLASSES = [
    "Rolex Submariner",
    "Omega Speedmaster",
    "Tag Heuer Carrera",
    "Breitling Navitimer",
    "Patek Philippe Calatrava",
    "Audemars Piguet Royal Oak",
    "Cartier Tank",
    "IWC Portuguese",
    "Jaeger-LeCoultre Reverso",
    "Hublot Big Bang"
]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Luxury Watch Classifier API is running"}


@app.post("/classify")
async def classify_watch(
    image: UploadFile = File(...),
    description: str = None
):
    """
    Classify a luxury watch image with optional text description
    """
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Preprocess image (simplified - you should implement proper preprocessing)
        # This is a placeholder - implement actual preprocessing as needed
        image_tensor = torch.rand((1, 3, 224, 224)).to(device)  # Random tensor for demo

        # Process text description
        if description:
            # This is a placeholder - implement actual text encoding
            input_ids = torch.randint(0, 1000, (1, 128)).to(device)
            attention_mask = torch.ones((1, 128)).to(device)
        else:
            # Use empty description if none provided
            input_ids = torch.zeros((1, 128), dtype=torch.long).to(device)
            attention_mask = torch.zeros((1, 128)).to(device)

        # Get prediction
        with torch.no_grad():
            logits = model(image_tensor, input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()

        return {
            "predicted_class": WATCH_CLASSES[predicted_class_idx],
            "confidence": confidence,
            "all_probabilities": {
                WATCH_CLASSES[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/classes")
async def get_classes():
    """Get list of watch classes"""
    return {"classes": WATCH_CLASSES}


@app.post("/predict-batch")
async def predict_batch(items: list[Dict[str, Any]]):
    """Batch prediction endpoint"""
    results = []

    for item in items:
        # This is a simplified implementation
        # In production, implement proper batch processing
        try:
            image_data = base64.b64decode(item.get("image_base64", ""))
            description = item.get("description", "")

            # Placeholder prediction
            predicted_class = WATCH_CLASSES[torch.randint(0, len(WATCH_CLASSES), (1,)).item()]
            confidence = float(torch.rand(1))

            results.append({
                "predicted_class": predicted_class,
                "confidence": confidence
            })
        except Exception as e:
            results.append({"error": str(e)})

    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)