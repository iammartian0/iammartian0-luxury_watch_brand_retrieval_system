# Luxury Watch Multi-Modal Classifier

A deep learning project that classifies luxury watches using both image and text modalities through PyTorch.

## Project Structure

```
├── data/                 # Dataset and model storage
├── src/
│   ├── image_model/      # CNN model for image classification
│   ├── text_model/       # Transformer model for text processing
│   └── fusion/           # Multi-modal fusion architecture
├── api/                  # FastAPI endpoints
├── docker/               # Docker configuration
├── requirements.txt      # Python dependencies
└── .gitignore           # Git ignore rules
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/train.py
```

### API Server
```bash
python api/main.py
```

## Features

- Image-based watch classification using CNN
- Text-based watch description analysis
- Multi-modal fusion for improved accuracy
- RESTful API for predictions
- Docker support for deployment