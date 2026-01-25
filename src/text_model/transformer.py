import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class WatchTextEncoder(nn.Module):
    """Transformer model for watch description analysis"""

    def __init__(self, model_name='bert-base-uncased', embedding_dim=768, num_classes=10):
        super(WatchTextEncoder, self).__init__()

        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        """Forward pass through transformer"""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Classification
        logits = self.classifier(cls_embedding)

        return logits, cls_embedding

    def encode_text(self, texts):
        """Encode text inputs"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return encoded['input_ids'], encoded['attention_mask']