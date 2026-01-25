import torchvision.transforms as transforms
import torch


def get_train_transforms(image_size: tuple = (224, 224)):
    """Data augmentation transforms for training"""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size: tuple = (224, 224)):
    """Transforms for validation/test (no augmentation)"""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_inference_transforms(image_size: tuple = (224, 224)):
    """Transforms for inference"""
    return get_val_transforms(image_size)


class MultiModalTransform:
    """Custom transform for multi-modal inputs"""

    def __init__(self, image_transform, tokenizer=None, max_length=128):
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample):
        image, text = sample['image'], sample['text']

        # Transform image
        if self.image_transform:
            image = self.image_transform(image)

        # Tokenize text if tokenizer is provided
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
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
            'text': text
        }