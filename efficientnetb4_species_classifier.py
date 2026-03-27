"""
Species Classification - EfficientNet-B4 Transfer Learning

Uses Excel lookup table to assign species labels based on slide names.
Trains with transfer learning from ImageNet pretrained EfficientNet-B4.

EfficientNet-B4 uses compound scaling (depth, width, resolution) for
optimal accuracy/efficiency tradeoff - modern architecture from 2019.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # Paths
    'labels_excel': r'C:\Users\carol\gram stain ai\species_training.xlsx',
    'crops_folder': r'C:\Gram Stain Training Data\detections',
    'model_dir': r'C:\Gram Stain Training Data\models',
    
    # Training
    'image_size': (380, 380),  # EfficientNet-B4 optimal input size
    'batch_size': 16,          # Smaller batch due to larger input size
    'epochs_frozen': 10,       # Phase 1: train final layers only
    'epochs_unfrozen': 20,     # Phase 2: fine-tune deep layers
    'lr_frozen': 1e-3,
    'lr_unfrozen': 1e-5,
    'patience': 10,
}


# ============================================================================
# DATASET
# ============================================================================

class SpeciesDataset(Dataset):
    """Dataset for species classification."""
    
    def __init__(self, image_paths, labels, label_map, image_size=(380, 380), augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.label_map = label_map
        self.image_size = image_size
        self.augment = augment
        
        # ImageNet normalization for pretrained EfficientNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            self.normalize
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.augment:
            img = self.train_transform(img)
        else:
            img = self.val_transform(img)
        
        label = self.label_map[self.labels[idx]]
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label


# ============================================================================
# MODEL
# ============================================================================

class SpeciesEfficientNetB4(nn.Module):
    """EfficientNet-B4 for species classification."""
    
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet-B4
        self.efficientnet = models.efficientnet_b4(weights='IMAGENET1K_V1' if pretrained else None)
        
        # EfficientNet-B4 classifier: 1792 features -> 1000 classes
        num_features = self.efficientnet.classifier[1].in_features  # 1792
        
        # Replace classifier
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.4),  # EfficientNet uses higher dropout
            nn.Linear(num_features, 512),
            nn.SiLU(inplace=True),  # EfficientNet uses SiLU/Swish activation
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)
    
    def freeze_backbone(self):
        """Freeze all feature layers, train only classifier."""
        for param in self.efficientnet.features.parameters():
            param.requires_grad = False
        for param in self.efficientnet.classifier.parameters():
            param.requires_grad = True
        print("Backbone frozen. Training only classifier.")
    
    def unfreeze_deep_layers(self, num_blocks=2):
        """
        Unfreeze last few MBConv blocks for fine-tuning.
        EfficientNet-B4 has 7 main blocks (0-6) in features.
        """
        # Get total number of blocks
        total_blocks = len(self.efficientnet.features)
        
        # Unfreeze last num_blocks
        for i in range(total_blocks - num_blocks, total_blocks):
            for param in self.efficientnet.features[i].parameters():
                param.requires_grad = True
        
        print(f"Unfroze last {num_blocks} MBConv blocks for fine-tuning.")
    
    def unfreeze_all(self):
        for param in self.efficientnet.parameters():
            param.requires_grad = True
        print("All layers unfrozen.")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config):
    """Load data using Excel lookup table."""
    
    df = pd.read_excel(config['labels_excel'])
    print(f"Loaded lookup table with {len(df)} entries")
    
    # Create lookup: slide_name -> species
    lookup = {}
    for _, row in df.iterrows():
        name = str(row['Name']).strip()
        lookup[name] = row['Species'].strip()
    
    print(f"Slides in lookup: {len(lookup)}")
    
    # Find all crops
    crops_folder = config['crops_folder']
    all_images = []
    
    for slide_folder in os.listdir(crops_folder):
        slide_path = os.path.join(crops_folder, slide_folder)
        if not os.path.isdir(slide_path):
            continue
        
        crops_path = os.path.join(slide_path, 'crops')
        if not os.path.exists(crops_path):
            continue
        
        slide_name = slide_folder.strip()
        
        if slide_name not in lookup:
            print(f"  Skipping {slide_name} (not in lookup)")
            continue
        
        species = lookup[slide_name]
        
        try:
            files = os.listdir(crops_path)
            crop_files = [os.path.join(crops_path, f) for f in files 
                         if f.endswith('.png') or f.endswith('.jpg')]
        except Exception as e:
            print(f"  Error reading {crops_path}: {e}")
            continue
        
        for crop_file in crop_files:
            all_images.append({
                'path': crop_file,
                'species': species
            })
        
        print(f"  {slide_name}: {len(crop_files)} crops ({species})")
    
    print(f"\nTotal images: {len(all_images)}")
    
    if len(all_images) == 0:
        raise ValueError("No images found!")
    
    image_paths = [img['path'] for img in all_images]
    labels = [img['species'] for img in all_images]
    
    # Create label map
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_map = {idx: label for label, idx in label_map.items()}
    
    label_counts = Counter(labels)
    print(f"\nClass distribution ({len(unique_labels)} species):")
    for label in unique_labels:
        print(f"  {label}: {label_counts[label]}")
    
    return {
        'image_paths': image_paths,
        'labels': labels,
        'label_map': label_map,
        'reverse_map': reverse_map,
        'num_classes': len(unique_labels)
    }


# ============================================================================
# TRAINER
# ============================================================================

class SpeciesEfficientNetB4Trainer:
    """Training pipeline for EfficientNet-B4 species classifier."""
    
    def __init__(self, config):
        self.config = config
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        os.makedirs(config['model_dir'], exist_ok=True)
        
        self.data = load_data(config)
        self.label_map = self.data['label_map']
        self.reverse_map = self.data['reverse_map']
        self.num_classes = self.data['num_classes']
        
        # Save label mapping
        mapping_path = os.path.join(config['model_dir'], 'species_efficientnetb4_label_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump({'label_map': self.label_map, 'reverse_map': self.reverse_map}, f, indent=2)
        print(f"Saved label mapping to {mapping_path}")
    
    def train(self):
        """Train with simple train/val split."""
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            self.data['image_paths'], self.data['labels'],
            test_size=0.2, stratify=self.data['labels'], random_state=42
        )
        
        print(f"\nTraining samples: {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")
        
        # Datasets
        train_dataset = SpeciesDataset(
            train_paths, train_labels, self.label_map,
            self.config['image_size'], augment=True
        )
        val_dataset = SpeciesDataset(
            val_paths, val_labels, self.label_map,
            self.config['image_size'], augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'],
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'],
                                shuffle=False, num_workers=0, pin_memory=True)
        
        # Model
        model = SpeciesEfficientNetB4(self.num_classes, pretrained=True).to(device)
        
        # Class weights for imbalanced data
        class_counts = Counter(train_labels)
        total = len(train_labels)
        weights = torch.tensor([
            total / (self.num_classes * class_counts[self.reverse_map[i]])
            for i in range(self.num_classes)
        ], dtype=torch.float).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_acc = 0
        best_model_path = None
        
        # =====================================================================
        # PHASE 1: Frozen backbone
        # =====================================================================
        print(f"\n--- Phase 1: Training classifier (frozen backbone) ---")
        model.freeze_backbone()
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.config['lr_frozen'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                          factor=0.5, patience=3)
        
        for epoch in range(self.config['epochs_frozen']):
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            
            scheduler.step(val_acc)
            self._save_history(train_loss, val_loss, train_acc, val_acc)
            
            print(f"Epoch {epoch+1}/{self.config['epochs_frozen']}: "
                  f"Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_path = os.path.join(self.config['model_dir'], 
                                               f'species_efficientnetb4_best_{timestamp}.pth')
                self._save_model(model, best_model_path, val_acc)
                print(f"  ★ Saved best model ({val_acc:.1f}%)")
        
        # =====================================================================
        # PHASE 2: Fine-tune deep layers
        # =====================================================================
        print(f"\n--- Phase 2: Fine-tuning MBConv blocks ---")
        model.unfreeze_deep_layers(num_blocks=3)  # EfficientNet benefits from more unfrozen blocks
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.config['lr_unfrozen'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                          factor=0.5, patience=5)
        
        patience_counter = 0
        total_epochs = self.config['epochs_frozen']
        
        for epoch in range(self.config['epochs_unfrozen']):
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            
            scheduler.step(val_acc)
            self._save_history(train_loss, val_loss, train_acc, val_acc)
            
            print(f"Epoch {total_epochs + epoch + 1}/{total_epochs + self.config['epochs_unfrozen']}: "
                  f"Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                best_model_path = os.path.join(self.config['model_dir'],
                                               f'species_efficientnetb4_best_{timestamp}.pth')
                self._save_model(model, best_model_path, val_acc)
                print(f"  ★ Saved best model ({val_acc:.1f}%)")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {total_epochs + epoch + 1}")
                    break
        
        print(f"\nTraining complete! Best accuracy: {best_acc:.1f}%")
        print(f"Model saved to: {best_model_path}")
        
        # Final evaluation
        self._final_evaluation(model, val_loader, val_labels)
        
        return self.history
    
    def _train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'acc': f"{100.*correct/total:.1f}%"})
        
        return total_loss / len(loader), 100. * correct / total
    
    def _validate(self, model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(loader), 100. * correct / total
    
    def _save_history(self, train_loss, val_loss, train_acc, val_acc):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
    
    def _save_model(self, model, path, val_acc):
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'num_classes': self.num_classes,
            'label_map': self.label_map,
            'reverse_map': self.reverse_map
        }, path)
    
    def _final_evaluation(self, model, loader, true_labels):
        """Print classification report and confusion matrix."""
        model.eval()
        all_preds = []
        
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
        
        # Convert to label indices
        true_indices = [self.label_map[l] for l in true_labels]
        
        print("\nClassification Report:")
        target_names = [self.reverse_map[i] for i in range(self.num_classes)]
        print(classification_report(true_indices, all_preds, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_indices, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Species Confusion Matrix (EfficientNet-B4)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = os.path.join(self.config['model_dir'], 'species_efficientnetb4_confusion_matrix.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
        plt.close()
    
    def plot_history(self):
        """Plot and save training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_title('Accuracy (%)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.config['model_dir'], 'species_efficientnetb4_training_history.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved training history to {save_path}")
        plt.close()


# ============================================================================
# PREDICTOR
# ============================================================================

class SpeciesEfficientNetB4Classifier:
    """Use trained EfficientNet-B4 model for species prediction."""
    
    def __init__(self, model_path, image_size=(380, 380)):
        self.image_size = image_size
        
        checkpoint = torch.load(model_path, map_location=device)
        self.num_classes = checkpoint['num_classes']
        self.label_map = checkpoint['label_map']
        self.reverse_map = {int(k): v for k, v in checkpoint['reverse_map'].items()}
        
        self.model = SpeciesEfficientNetB4(self.num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded EfficientNet-B4 model: {model_path}")
        print(f"Accuracy: {checkpoint['val_acc']:.1f}%")
        print(f"Species: {list(self.label_map.keys())}")
    
    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, 0.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.model(img)
            probs = torch.softmax(outputs, dim=1)
            pred_class = outputs.argmax(1).item()
            confidence = probs[0, pred_class].item()
        
        return self.reverse_map[pred_class], confidence
    
    def predict_folder(self, folder_path, output_csv=None):
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.endswith('.png') or f.endswith('.jpg'):
                    image_files.append(os.path.join(root, f))
        
        print(f"Predicting {len(image_files)} images...")
        
        results = []
        for img_path in tqdm(image_files):
            species, confidence = self.predict(img_path)
            if species:
                results.append({
                    'image': os.path.basename(img_path),
                    'path': img_path,
                    'species': species,
                    'confidence': f"{confidence:.2%}"
                })
        
        df = pd.DataFrame(results)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Saved to {output_csv}")
        
        print(f"\nPrediction summary:")
        print(df['species'].value_counts())
        
        return df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPECIES CLASSIFIER - EfficientNet-B4")
    print("=" * 60)
    
    print("""
USAGE:

1. TRAIN:
   
   from species_efficientnetb4_classifier import SpeciesEfficientNetB4Trainer, CONFIG
   
   trainer = SpeciesEfficientNetB4Trainer(CONFIG)
   trainer.train()
   trainer.plot_history()

2. PREDICT:
   
   from species_efficientnetb4_classifier import SpeciesEfficientNetB4Classifier
   
   classifier = SpeciesEfficientNetB4Classifier('models/species_efficientnetb4_best_XXXXXX.pth')
   species, confidence = classifier.predict('bacteria.png')
   print(f"{species}: {confidence:.1%}")
""")
    
    if os.path.exists(CONFIG['labels_excel']) and os.path.exists(CONFIG['crops_folder']):
        print("\nData found! Ready to train.")
    else:
        print("\nUpdate CONFIG paths before training.")