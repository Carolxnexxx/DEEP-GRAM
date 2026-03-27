"""
Debris Filter

1. Trains a binary classifier (bacteria vs debris) on your labeled data
2. Runs predictions on ALL crops
3. Deletes debris, keeps bacteria

Usage:
    python debris_filter.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # Input
    'labels_csv': r'C:\Gram Stain Training Data\debris_labels.csv',
    'crops_folder': r'C:\Gram Stain Training Data\detections',
    
    # Model output
    'model_dir': r'C:\Gram Stain Training Data\models',
    
    # Training
    'image_size': (64, 64),
    'batch_size': 64,
    'epochs': 30,
    'learning_rate': 1e-3,
    
    # Filtering
    'bacteria_threshold': 0.7,  # Keep if bacteria confidence > this
    'dry_run': False,  # Set to False to actually delete files
}


# ============================================================================
# DATASET
# ============================================================================

class DebrisDataset(Dataset):
    def __init__(self, image_paths, labels, image_size=(64, 64), augment=True):
        self.image_paths = image_paths
        self.labels = labels  # 0 = bacteria, 1 = debris
        self.image_size = image_size
        self.augment = augment
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
        
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
            if np.random.rand() > 0.5:
                angle = np.random.randint(-180, 180)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
        
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, label


# ============================================================================
# MODEL
# ============================================================================

class DebrisCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINING
# ============================================================================

def train_debris_filter(config):
    """Train the debris filter model."""
    
    print("=" * 60)
    print("TRAINING DEBRIS FILTER")
    print("=" * 60)
    
    # Load labels
    df = pd.read_csv(config['labels_csv'])
    print(f"Loaded {len(df)} labeled samples")
    print(f"  Bacteria: {(df['label'] == 'bacteria').sum()}")
    print(f"  Debris: {(df['label'] == 'debris').sum()}")
    
    # Prepare data
    image_paths = df['path'].tolist()
    labels = [0 if l == 'bacteria' else 1 for l in df['label']]
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"\nTrain: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Datasets
    train_dataset = DebrisDataset(train_paths, train_labels, config['image_size'], augment=True)
    val_dataset = DebrisDataset(val_paths, val_labels, config['image_size'], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Model
    model = DebrisCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train
    best_acc = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(config['model_dir'], f'debris_filter_{timestamp}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, model_path)
            print(f"  ★ Saved best model ({val_acc:.1f}%)")
    
    print(f"\nTraining complete! Best accuracy: {best_acc:.1f}%")
    return model_path


# ============================================================================
# FILTERING
# ============================================================================

def filter_debris(config, model_path):
    """Run debris filter on all crops and delete debris."""
    
    print("\n" + "=" * 60)
    print("FILTERING DEBRIS")
    print("=" * 60)
    
    # Load model
    model = DebrisCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model: {model_path}")
    print(f"Model accuracy: {checkpoint['val_acc']:.1f}%")
    
    # Get all crops
    crops_folder = config['crops_folder']
    all_crops = []
    
    print("\nScanning for crops...")
    for slide_folder in os.listdir(crops_folder):
        slide_path = os.path.join(crops_folder, slide_folder)
        if not os.path.isdir(slide_path):
            continue
        
        crops_path = os.path.join(slide_path, 'crops')
        if not os.path.exists(crops_path):
            continue
        
        try:
            files = os.listdir(crops_path)
            for f in files:
                if f.endswith('.png') or f.endswith('.jpg'):
                    all_crops.append(os.path.join(crops_path, f))
        except:
            pass
    
    print(f"Found {len(all_crops):,} crops to filter")
    
    if config['dry_run']:
        print("\n⚠️  DRY RUN MODE - No files will be deleted")
        print("    Set 'dry_run': False to actually delete debris")
    
    # Process in batches
    threshold = config['bacteria_threshold']
    bacteria_count = 0
    debris_count = 0
    debris_to_delete = []
    
    batch_size = 64
    image_size = config['image_size']
    
    for i in tqdm(range(0, len(all_crops), batch_size), desc="Filtering"):
        batch_paths = all_crops[i:i+batch_size]
        
        # Load batch
        images = []
        valid_paths = []
        
        for path in batch_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            images.append(img)
            valid_paths.append(path)
        
        if not images:
            continue
        
        # Predict
        batch_tensor = torch.stack(images).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            bacteria_probs = probs[:, 0].cpu().numpy()  # bacteria = class 0
        
        # Classify - keep if bacteria confidence > threshold
        for path, bacteria_prob in zip(valid_paths, bacteria_probs):
            if bacteria_prob > threshold:
                bacteria_count += 1
            else:
                debris_count += 1
                debris_to_delete.append(path)
    
    print(f"\nResults:")
    print(f"  Bacteria (keep): {bacteria_count:,}")
    print(f"  Debris (delete): {debris_count:,}")
    print(f"  Debris %: {100*debris_count/(bacteria_count+debris_count):.1f}%")
    
    # Delete debris
    if not config['dry_run'] and debris_to_delete:
        print(f"\nDeleting {len(debris_to_delete):,} debris files...")
        
        deleted = 0
        for path in tqdm(debris_to_delete, desc="Deleting"):
            try:
                os.remove(path)
                deleted += 1
            except Exception as e:
                pass
        
        print(f"Deleted {deleted:,} files")
    
    return bacteria_count, debris_count


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEBRIS FILTER")
    print("=" * 60)
    
    # Check if labels exist
    if not os.path.exists(CONFIG['labels_csv']):
        print(f"Labels not found: {CONFIG['labels_csv']}")
        print("Run label_debris.py first!")
        exit()
    
    # Train
    model_path = train_debris_filter(CONFIG)
    
    # Filter
    filter_debris(CONFIG, model_path)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nIf results look good, set 'dry_run': False and run again to delete debris.")