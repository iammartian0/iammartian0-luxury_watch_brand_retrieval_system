import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from typing import List, Tuple, Optional


class WatchDataset(Dataset):
    """Dataset for luxury watch multi-modal classification"""

    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        transform=None,
        tokenizer=None,
        max_length: int = 128
    ):
        """
        Args:
            csv_file: Path to the CSV file with annotations
            image_dir: Directory with all the images
            transform: Optional transform to be applied on an image
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length for text
        """
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Verify all image paths exist
        missing_images = []
        for idx, row in self.data_frame.iterrows():
            img_path = os.path.join(self.image_dir, row['image_path'])
            if not os.path.exists(img_path):
                missing_images.append(img_path)

        if missing_images:
            print(f"Warning: {len(missing_images)} images not found")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = os.path.join(self.image_dir, self.data_frame.iloc[idx]['image_path'])
        image = Image.open(img_path).convert('RGB')

        # Get text description
        description = str(self.data_frame.iloc[idx]['description'])

        # Get label
        label = self.data_frame.iloc[idx]['label']

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)

        # Tokenize text if tokenizer is provided
        if self.tokenizer:
            encoding = self.tokenizer(
                description,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'description': description
        }


def create_data_loaders(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str],
    image_dir: str,
    transform,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, validation, and test data loaders"""

    # Create datasets
    train_dataset = WatchDataset(train_csv, image_dir, transform, tokenizer)
    val_dataset = WatchDataset(val_csv, image_dir, transform, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = None
    if test_csv:
        test_dataset = WatchDataset(test_csv, image_dir, transform, tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader


def get_class_weights(csv_file: str) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset"""
    df = pd.read_csv(csv_file)
    class_counts = df['label'].value_counts().sort_index()
    total_samples = len(df)

    # Calculate weights inversely proportional to class frequencies
    weights = total_samples / (len(class_counts) * class_counts)
    weights = weights / weights.sum() * len(class_counts)

    return torch.tensor(weights.values, dtype=torch.float32)