import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import json

try:
    import timm
except ImportError:
    print("Installing timm...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'timm', '--break-system-packages'])
    import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

CONFIG = {

    'labels_excel': r'C:\Users\carol\gram stain ai\species_training.xlsx',
    'crops_folder': r'C:\Gram Stain Training Data\detections',
    'model_dir': r'C:\Gram Stain Training Data\models',
    
    'species_filter': None,
    
    'val_slides': [
        'SA1',
        'SE2',
        'EFS10',
        'EF8',
        'SPY2',
        'SPN6',
        'VS3',
        'CB13',
        'LS9',
        'KP2',
        'ECL7',
        'PA3',
        'AB2',
        'MC1',
        'HI5',
    ],
    
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs_frozen': 5,
    'epochs_unfrozen': 15,
    'lr_frozen': 1e-3,
    'lr_unfrozen': 1e-5,
    'patience': 10,
}

class GramDataset(Dataset):
    
    def __init__(self, image_paths, labels, label_map, image_size=(224, 224), augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.label_map = label_map
        self.image_size = image_size
        self.augment = augment
        
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

class GramViT(nn.Module):
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0
        )
        
        self.num_features = self.vit.embed_dim
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(0.3),
            nn.Linear(self.num_features, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):

        features = self.vit(x)

        return self.classifier(features)
    
    def get_features(self, x):
        return self.vit(x)
    
    def freeze_backbone(self):
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("ViT backbone frozen. Training only classifier.")
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        print("All layers unfrozen for fine-tuning.")
    
    def unfreeze_last_n_blocks(self, n=4):

        for param in self.vit.parameters():
            param.requires_grad = False
        
        for block in self.vit.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        
        for param in self.vit.norm.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        print(f"Unfroze last {n} transformer blocks + classifier.")

def load_data(config):
    
    df = pd.read_excel(config['labels_excel'])
    print(f"Loaded lookup table with {len(df)} entries")
    
    species_filter = config.get('species_filter', None)
    if species_filter:
        print(f"\nFiltering for {len(species_filter)} species:")
        for sp in species_filter:
            print(f"  - {sp}")
    
    lookup = {}
    slide_to_species = {}
    for _, row in df.iterrows():
        name = str(row['Name']).strip()
        gram = row['Gram'].strip().lower()
        species = row['Species'].strip()
        
        if species_filter and species not in species_filter:
            continue
        
        lookup[name] = gram
        slide_to_species[name] = species
    
    print(f"Slides in lookup (after filter): {len(lookup)}")
    
    crops_folder = config['crops_folder']
    slides_data = {}
    
    for slide_folder in os.listdir(crops_folder):
        slide_path = os.path.join(crops_folder, slide_folder)
        if not os.path.isdir(slide_path):
            continue
        
        crops_path = os.path.join(slide_path, 'crops')
        if not os.path.exists(crops_path):
            continue
        
        slide_name = slide_folder.strip()
        
        if slide_name not in lookup:
            continue
        
        gram = lookup[slide_name]
        
        try:
            files = os.listdir(crops_path)
            crop_files = [os.path.join(crops_path, f) for f in files 
                         if f.endswith('.png') or f.endswith('.jpg')]
        except Exception as e:
            print(f"  Error reading {crops_path}: {e}")
            continue
        
        if crop_files:
            slides_data[slide_name] = {
                'gram': gram,
                'crops': crop_files
            }
            print(f"  {slide_name}: {len(crop_files)} crops ({gram})")
    
    print(f"\nTotal slides: {len(slides_data)}")
    
    if len(slides_data) == 0:
        raise ValueError("No slides found!")
    
    config_val_slides = config.get('val_slides', None)
    
    if config_val_slides:

        val_slides = [s for s in config_val_slides if s in slides_data]
        train_slides = [s for s in slides_data.keys() if s not in val_slides]
        
        print("\nUsing specified validation slides:")
        pos_val = sum(1 for s in val_slides if slides_data[s]['gram'] == 'positive')
        neg_val = len(val_slides) - pos_val
        print(f"  {len(val_slides)} slides ({pos_val} positive, {neg_val} negative)")
        
        if len(val_slides) != len(config_val_slides):
            missing = set(config_val_slides) - set(val_slides)
            print(f"  WARNING: Missing slides: {missing}")
    else:

        species_slides = {}
        for _, row in df.iterrows():
            name = str(row['Name']).strip()
            species = row['Species'].strip()
            if name in slides_data:
                if species not in species_slides:
                    species_slides[species] = []
                species_slides[species].append(name)
        
        train_slides = []
        val_slides = []
        
        print("\nAuto-selecting validation slides (1 per species):")
        for species, slides in sorted(species_slides.items()):
            if len(slides) >= 2:
                val_slide = slides[0]
                train_slides.extend(slides[1:])
                val_slides.append(val_slide)
                gram = slides_data[val_slide]['gram']
                print(f"  {species} ({gram}): val={val_slide}")
            else:
                train_slides.extend(slides)
                print(f"  {species}: train only (1 slide)")
    
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    
    for slide_name in train_slides:
        data = slides_data[slide_name]
        train_paths.extend(data['crops'])
        train_labels.extend([data['gram']] * len(data['crops']))
    
    for slide_name in val_slides:
        data = slides_data[slide_name]
        val_paths.extend(data['crops'])
        val_labels.extend([data['gram']] * len(data['crops']))
    
    print(f"\nTotal training images: {len(train_paths)}")
    print(f"Total validation images: {len(val_paths)}")
    
    label_map = {'positive': 0, 'negative': 1}
    reverse_map = {0: 'positive', 1: 'negative'}
    
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    print(f"\nTraining: {train_counts['positive']} positive, {train_counts['negative']} negative")
    print(f"Validation: {val_counts['positive']} positive, {val_counts['negative']} negative")
    
    return {
        'train_paths': train_paths,
        'val_paths': val_paths,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'label_map': label_map,
        'reverse_map': reverse_map,
        'num_classes': 2,
        'train_slides': train_slides,
        'val_slides': val_slides
    }

class GramViTTrainer:
    
    def __init__(self, config):
        self.config = config
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        os.makedirs(config['model_dir'], exist_ok=True)
        
        self.data = load_data(config)
        self.label_map = self.data['label_map']
        self.reverse_map = self.data['reverse_map']
        
        mapping_path = os.path.join(config['model_dir'], 'gram_vit_config.json')
        with open(mapping_path, 'w') as f:
            json.dump({
                'label_map': self.label_map,
                'reverse_map': self.reverse_map,
                'val_slides': self.data['val_slides']
            }, f, indent=2)
        print(f"Saved config to {mapping_path}")
    
    def train(self):
        
        train_paths = self.data['train_paths']
        val_paths = self.data['val_paths']
        train_labels = self.data['train_labels']
        val_labels = self.data['val_labels']
        
        print(f"\nTraining samples: {len(train_paths)} ({len(self.data['train_slides'])} slides)")
        print(f"Validation samples: {len(val_paths)} ({len(self.data['val_slides'])} slides)")
        
        train_dataset = GramDataset(
            train_paths, train_labels, self.label_map,
            self.config['image_size'], augment=True
        )
        val_dataset = GramDataset(
            val_paths, val_labels, self.label_map,
            self.config['image_size'], augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'],
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'],
                                shuffle=False, num_workers=0, pin_memory=True)
        
        model = GramViT(num_classes=2, pretrained=True).to(device)
        
        train_counts = Counter(train_labels)
        total = len(train_labels)
        weights = torch.tensor([
            total / (2 * train_counts['positive']),
            total / (2 * train_counts['negative'])
        ], dtype=torch.float).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_acc = 0
        best_model_path = None
        
        print(f"\n--- Phase 1: Training classifier (frozen ViT) ---")
        model.freeze_backbone()
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config['lr_frozen'], weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['epochs_frozen']
        )
        
        for epoch in range(self.config['epochs_frozen']):
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            
            scheduler.step()
            self._save_history(train_loss, val_loss, train_acc, val_acc)
            
            print(f"Epoch {epoch+1}/{self.config['epochs_frozen']}: "
                  f"Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_path = os.path.join(
                    self.config['model_dir'],
                    f'gram_vit_best_{timestamp}.pth'
                )
                self._save_model(model, best_model_path, val_acc)
                print(f"  \u2605 Saved best model ({val_acc:.1f}%)")
        
        print(f"\n--- Phase 2: Fine-tuning ViT ---")
        model.unfreeze_last_n_blocks(n=4)
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config['lr_unfrozen'], weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['epochs_unfrozen']
        )
        
        patience_counter = 0
        total_epochs = self.config['epochs_frozen']
        
        for epoch in range(self.config['epochs_unfrozen']):
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            
            scheduler.step()
            self._save_history(train_loss, val_loss, train_acc, val_acc)
            
            print(f"Epoch {total_epochs + epoch + 1}/{total_epochs + self.config['epochs_unfrozen']}: "
                  f"Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                best_model_path = os.path.join(
                    self.config['model_dir'],
                    f'gram_vit_best_{timestamp}.pth'
                )
                self._save_model(model, best_model_path, val_acc)
                print(f"  \u2605 Saved best model ({val_acc:.1f}%)")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {total_epochs + epoch + 1}")
                    break
        
        print(f"\nTraining complete! Best accuracy: {best_acc:.1f}%")
        print(f"Model saved to: {best_model_path}")
        
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self._final_evaluation(model, val_loader)
        
        self.plot_history()
        
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
            'label_map': self.label_map,
            'reverse_map': self.reverse_map,
            'val_slides': self.data['val_slides']
        }, path)
    
    def _final_evaluation(self, model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        target_names = ['Gram-positive', 'Gram-negative']
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=target_names))
        
        cm = confusion_matrix(all_labels, all_preds)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Gram Classification Confusion Matrix (%) - ViT')
        plt.tight_layout()
        
        save_path = os.path.join(self.config['model_dir'], 'gram_vit_confusion_matrix.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
        plt.close()
    
    def plot_history(self):
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
        
        save_path = os.path.join(self.config['model_dir'], 'gram_vit_training_history.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved training history to {save_path}")
        plt.close()

class GramViTClassifier:
    
    def __init__(self, model_path, image_size=(224, 224)):
        self.image_size = image_size
        
        checkpoint = torch.load(model_path, map_location=device)
        self.label_map = checkpoint['label_map']
        self.reverse_map = {int(k): v for k, v in checkpoint['reverse_map'].items()}
        
        self.model = GramViT(num_classes=2, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded Gram ViT model: {model_path}")
        print(f"Accuracy: {checkpoint['val_acc']:.1f}%")
    
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
    
    def predict_batch(self, image_paths, batch_size=32):
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None:
                    img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_tensors.append(self.transform(img))
            
            batch = torch.stack(batch_tensors).to(device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(1)
                confs = probs.max(1).values
            
            for j, img_path in enumerate(batch_paths):
                results.append({
                    'image': os.path.basename(img_path),
                    'path': img_path,
                    'gram': self.reverse_map[preds[j].item()],
                    'confidence': confs[j].item()
                })
        
        return pd.DataFrame(results)
