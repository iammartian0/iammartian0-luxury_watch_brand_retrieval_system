import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.fusion.multimodal import MultiModalWatchClassifier
from src.utils.data_utils import create_data_loaders, get_class_weights
from src.utils.transforms import get_train_transforms, get_val_transforms
from transformers import AutoTokenizer


class Trainer:
    """Advanced trainer class with logging, checkpointing, and metrics"""

    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device
        if self.config['device'] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config['device'])

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['text_model_name'])

        # Setup directories
        self.setup_directories()

        # Initialize model
        self.model = self.build_model()

        # Initialize optimizer and scheduler
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # Initialize loss function with class weights
        self.criterion = self.build_criterion()

        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Training metrics
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def setup_directories(self):
        """Create necessary directories"""
        self.model_save_dir = Path(self.config['paths']['model_save_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])

        for dir_path in [self.model_save_dir, self.log_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def build_model(self):
        """Build and initialize the model"""
        model = MultiModalWatchClassifier(
            num_classes=self.config['model']['num_classes'],
            image_embedding_dim=self.config['model']['image_embedding_dim'],
            text_embedding_dim=self.config['model']['text_embedding_dim'],
            fusion_dim=self.config['model']['fusion_dim']
        ).to(self.device)

        # Initialize weights
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

        return model

    def build_optimizer(self):
        """Build optimizer"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

    def build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config['training']['scheduler']['type'] == 'step_lr':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['scheduler']['step_size'],
                gamma=self.config['training']['scheduler']['gamma']
            )
        else:
            return None

    def build_criterion(self):
        """Build loss function with class weights"""
        try:
            class_weights = get_class_weights(self.config['data']['train_csv'])
            class_weights = class_weights.to(self.device)
            return nn.CrossEntropyLoss(weight=class_weights)
        except:
            return nn.CrossEntropyLoss()

    def create_data_loaders(self):
        """Create data loaders"""
        train_transform = get_train_transforms(tuple(self.config['data']['image_size']))
        val_transform = get_val_transforms(tuple(self.config['data']['image_size']))

        return create_data_loaders(
            train_csv=self.config['data']['train_csv'],
            val_csv=self.config['data']['val_csv'],
            test_csv=self.config['data'].get('test_csv'),
            image_dir=self.config['data']['image_dir'],
            transform=None,  # We'll apply transforms in the dataset
            tokenizer=self.tokenizer,
            batch_size=self.config['training']['batch_size']
        )

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, input_ids, attention_mask)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Accuracy', 100.*correct/total, global_step)

        return running_loss/len(train_loader), 100.*correct/total

    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images, input_ids, attention_mask)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100.*correct/total
        avg_val_loss = val_loss/len(val_loader)

        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', val_acc, epoch)

        # Classification report
        if epoch % 5 == 0:
            report = classification_report(all_labels, all_preds, output_dict=True)
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    self.writer.add_scalar(f'Val/Precision_{class_name}', metrics['precision'], epoch)
                    self.writer.add_scalar(f'Val/Recall_{class_name}', metrics['recall'], epoch)

        return avg_val_loss, val_acc

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(state, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.model_save_dir / 'best_model.pth'
            torch.save(state, best_path)

        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(state, latest_path)

    def train(self):
        """Main training loop"""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create data loaders
        train_loader, val_loader, _ = self.create_data_loaders()

        for epoch in range(self.config['training']['num_epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader, epoch)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Print epoch results
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                print("New best model saved!")
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.config['training']['early_stopping']['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        print(f"\nTraining completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        self.writer.close()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()