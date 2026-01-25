import torch
import torch.nn as nn
from src.image_model.cnn import WatchCNN
from src.text_model.transformer import WatchTextEncoder


class MultiModalWatchClassifier(nn.Module):
    """Multi-modal fusion model combining image and text features"""

    def __init__(self, num_classes=10, image_embedding_dim=2048, text_embedding_dim=768, fusion_dim=512):
        super(MultiModalWatchClassifier, self).__init__()

        # Initialize individual encoders
        self.image_encoder = WatchCNN(num_classes=num_classes)
        self.text_encoder = WatchTextEncoder(num_classes=num_classes)

        # Fusion layers
        self.image_proj = nn.Sequential(
            nn.Linear(image_embedding_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_embedding_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Attention-based fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        """Forward pass through multi-modal model"""
        # Extract features
        image_features = self.image_encoder.extract_features(images)
        _, text_features = self.text_encoder(input_ids, attention_mask)

        # Project features
        image_proj = self.image_proj(image_features)
        text_proj = self.text_proj(text_features)

        # Stack for attention
        stacked_features = torch.stack([image_proj, text_proj], dim=1)

        # Apply attention fusion
        fused_features, _ = self.fusion_attention(
            stacked_features, stacked_features, stacked_features
        )

        # Take weighted sum of attended features
        final_features = torch.mean(fused_features, dim=1)

        # Classification
        logits = self.classifier(final_features)

        return logits