"""
Hierarchical Classification Pipeline

Full routing structure:

GRAM POSITIVE PATH:
  Gram Model → GP Morphology Model → 
    ├── Cocci Clusters → GP Cocci Clusters Species (SA, SE)
    ├── Cocci Chains → GP Cocci Chains Species (SPY, SPN, VS, EFS, EF)
    └── Bacilli → GP Bacilli Species (CB, LS)

GRAM NEGATIVE PATH:
  Gram Model → GN Morphology Model →
    ├── Bacilli → GN Bacilli Species (KP, ECL, PA, AB)
    └── Cocci → GN Cocci Species (MC, HI)

No U-Net segmentation - works directly on pre-extracted crops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import cv2
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from collections import Counter

try:
    import timm
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'timm', '--break-system-packages'])
    import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# ROUTING CONFIGURATION
# ============================================================================

# Species routes (gram + morphology → species list)
SPECIES_ROUTES = {
    'gp_cocci_clusters': {
        'gram': 'positive',
        'morphology': 'cocci_clusters',
        'species': [
            'Staphylococcus aureus',
            'Staphylococcus epidermidis',
        ],
        'description': 'Gram-positive Cocci Clusters (Staphylococci)',
    },
    'gp_cocci_chains': {
        'gram': 'positive',
        'morphology': 'cocci_chains',
        'species': [
            'Streptococcus pyogenes',
            'Streptococcus pneumoniae',
            'Streptococcus mitis',
            'Enterococcus faecalis',
            'Enterococcus faecium',
        ],
        'description': 'Gram-positive Cocci Chains (Strep/Enterococci)',
    },
    'gp_bacilli': {
        'gram': 'positive',
        'morphology': 'bacilli',
        'species': [
            'Corynebacterium striatum',
            'Listeria monocytogenes',
        ],
        'description': 'Gram-positive Bacilli',
    },
    'gn_bacilli': {
        'gram': 'negative',
        'morphology': 'bacilli',
        'species': [
            'Klebsiella pneumoniae',
            'Enterobacter cloacae',
            'Pseudomonas aeruginosa',
            'Acinetobacter baumannii',
        ],
        'description': 'Gram-negative Bacilli',
    },
    'gn_cocci': {
        'gram': 'negative',
        'morphology': 'cocci_clusters',  # Diplococci grouped with clusters
        'species': [
            'Moraxella catarrhalis',
            'Haemophilus influenzae',
        ],
        'description': 'Gram-negative Cocci',
    },
}

# Morphology mappings for each gram type
GP_MORPHOLOGY_MAP = {
    'Cocci Clusters (Staphylococci)': 'cocci_clusters',
    'Cocci Chains (Streptococci)': 'cocci_chains',
    'Cocci Chains (Enterococci)': 'cocci_chains',
    'Bacilli': 'bacilli',
}

GN_MORPHOLOGY_MAP = {
    'Bacilli': 'bacilli',
    'Bacilli (Enterobacterales)': 'bacilli',
    'Bacilli (Non-Enterobacterales)': 'bacilli',
    'Cocci': 'cocci_clusters',  # Diplococci → clusters
}


def get_species_route(gram, morphology):
    """Determine which species route to use based on gram and morphology."""
    gram = gram.lower() if gram else ''
    morphology = morphology.lower().replace(' ', '_') if morphology else ''
    
    for route_name, config in SPECIES_ROUTES.items():
        if config['gram'] == gram and config['morphology'] == morphology:
            return route_name
    
    # Default fallbacks
    if gram == 'positive':
        if 'cluster' in morphology:
            return 'gp_cocci_clusters'
        elif 'chain' in morphology:
            return 'gp_cocci_chains'
        else:
            return 'gp_bacilli'
    else:  # negative
        if 'cocci' in morphology or 'cluster' in morphology:
            return 'gn_cocci'
        else:
            return 'gn_bacilli'


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # Paths
    'labels_excel': r'C:\Users\carol\gram stain ai\species_training.xlsx',
    'crops_folder': r'C:\Gram Stain Training Data\detections',
    'model_dir': r'C:\Gram Stain Training Data\models',
    
    # Level 1: Gram classifier
    'gram_model': r'C:\Gram Stain Training Data\models\Gram\gram_vit_best_20260316_175955.pth',
    
    # Level 2: Morphology classifiers (one per gram type)
    'gp_morphology_model': None,  # GP: cocci_clusters, cocci_chains, bacilli
    'gn_morphology_model': None,  # GN: bacilli, cocci_clusters
    
    # Level 3: Species classifiers (one per gram+morphology combo)
    'species_models': {
        'gp_cocci_clusters': None,
        'gp_cocci_chains': None,
        'gp_bacilli': None,
        'gn_bacilli': None,
        'gn_cocci': None,
    },
    
    # Training settings
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs_frozen': 5,
    'epochs_unfrozen': 15,
    'lr_frozen': 1e-3,
    'lr_unfrozen': 1e-5,
    'patience': 10,
}


# ============================================================================
# DATASET
# ============================================================================

class ClassificationDataset(Dataset):
    """Generic dataset for classification tasks."""
    
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
        return img, torch.tensor(label, dtype=torch.long)


# ============================================================================
# MODELS
# ============================================================================

class ClassifierViT(nn.Module):
    """ViT for classification tasks."""
    
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0
        )
        
        self.num_features = self.vit.embed_dim  # 384
        
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
    
    def freeze_backbone(self):
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("ViT backbone frozen.")
    
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
        print(f"Unfroze last {n} transformer blocks.")


# ============================================================================
# HIERARCHICAL PIPELINE
# ============================================================================

class HierarchicalPipeline:
    """
    Full hierarchical pipeline:
    
    Crop → Gram → Morphology (GP or GN) → Species
    
    Routing:
      Positive → GP Morphology →
        ├── cocci_clusters → gp_cocci_clusters species
        ├── cocci_chains → gp_cocci_chains species
        └── bacilli → gp_bacilli species
      
      Negative → GN Morphology →
        ├── bacilli → gn_bacilli species
        └── cocci_clusters → gn_cocci species
    """
    
    def __init__(self, config):
        self.config = config
        
        # Models
        self.gram_model = None
        self.gram_reverse_map = None
        
        self.gp_morphology_model = None
        self.gp_morphology_reverse_map = None
        
        self.gn_morphology_model = None
        self.gn_morphology_reverse_map = None
        
        self.species_models = {}
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_gram_model(self, model_path):
        """Load gram classifier."""
        
        checkpoint = torch.load(model_path, map_location=device)
        num_classes = checkpoint.get('num_classes', 2)
        
        model = ClassifierViT(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        self.gram_model = model
        self.gram_reverse_map = {int(k): v for k, v in checkpoint['reverse_map'].items()}
        print(f"Loaded Gram model: {checkpoint['val_acc']:.1f}%")
        print(f"  Classes: {list(checkpoint['label_map'].keys())}")
    
    def load_gp_morphology_model(self, model_path):
        """Load Gram-positive morphology classifier."""
        
        checkpoint = torch.load(model_path, map_location=device)
        num_classes = checkpoint.get('num_classes', 3)
        
        model = ClassifierViT(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        self.gp_morphology_model = model
        self.gp_morphology_reverse_map = {int(k): v for k, v in checkpoint['reverse_map'].items()}
        print(f"Loaded GP Morphology model: {checkpoint['val_acc']:.1f}%")
        print(f"  Classes: {list(checkpoint['label_map'].keys())}")
    
    def load_gn_morphology_model(self, model_path):
        """Load Gram-negative morphology classifier."""
        
        checkpoint = torch.load(model_path, map_location=device)
        num_classes = checkpoint.get('num_classes', 2)
        
        model = ClassifierViT(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        self.gn_morphology_model = model
        self.gn_morphology_reverse_map = {int(k): v for k, v in checkpoint['reverse_map'].items()}
        print(f"Loaded GN Morphology model: {checkpoint['val_acc']:.1f}%")
        print(f"  Classes: {list(checkpoint['label_map'].keys())}")
    
    def load_species_model(self, route_name, model_path):
        """Load a species model. Auto-detects ConvNeXt vs ViT."""
        from torchvision import models
        
        checkpoint = torch.load(model_path, map_location=device)
        num_classes = checkpoint['num_classes']
        state_dict = checkpoint['model_state_dict']
        
        # Auto-detect architecture
        first_key = list(state_dict.keys())[0]
        
        if 'convnext' in first_key:
            class SpeciesConvNeXt(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.convnext = models.convnext_small(weights=None)
                    num_features = self.convnext.classifier[2].in_features
                    self.convnext.classifier[2] = nn.Sequential(
                        nn.Dropout(0.4),
                        nn.Linear(num_features, 512),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(512, 256),
                        nn.GELU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, num_classes)
                    )
                def forward(self, x):
                    return self.convnext(x)
            
            model = SpeciesConvNeXt(num_classes)
            arch = "ConvNeXt"
        else:
            model = ClassifierViT(num_classes, pretrained=False)
            arch = "ViT"
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        self.species_models[route_name] = {
            'model': model,
            'label_map': checkpoint['label_map'],
            'reverse_map': {int(k): v for k, v in checkpoint['reverse_map'].items()},
            'val_acc': checkpoint['val_acc']
        }
        print(f"Loaded {route_name} species model ({arch}): {checkpoint['val_acc']:.1f}%")
        print(f"  Species: {list(checkpoint['label_map'].keys())}")
    
    def predict_single(self, image_path_or_array):
        """
        Predict species for a single image through the full pipeline.
        
        Flow: Gram → Morphology (GP or GN) → Species
        """
        
        # Load image
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None:
                return None, 0.0, {}
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB) if image_path_or_array.shape[2] == 3 else image_path_or_array
        
        img_tensor = self.transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Level 1: Gram prediction
            gram_out = self.gram_model(img_tensor)
            gram_probs = torch.softmax(gram_out, dim=1)
            gram_pred = gram_out.argmax(1).item()
            gram_conf = gram_probs[0, gram_pred].item()
            gram = self.gram_reverse_map[gram_pred]
            
            # Level 2: Morphology prediction (based on gram)
            if gram == 'positive':
                if self.gp_morphology_model is None:
                    return None, 0.0, {'error': 'GP morphology model not loaded'}
                
                morph_out = self.gp_morphology_model(img_tensor)
                morph_probs = torch.softmax(morph_out, dim=1)
                morph_pred = morph_out.argmax(1).item()
                morph_conf = morph_probs[0, morph_pred].item()
                morphology = self.gp_morphology_reverse_map[morph_pred]
            else:  # negative
                if self.gn_morphology_model is None:
                    return None, 0.0, {'error': 'GN morphology model not loaded'}
                
                morph_out = self.gn_morphology_model(img_tensor)
                morph_probs = torch.softmax(morph_out, dim=1)
                morph_pred = morph_out.argmax(1).item()
                morph_conf = morph_probs[0, morph_pred].item()
                morphology = self.gn_morphology_reverse_map[morph_pred]
            
            # Determine species route
            route = get_species_route(gram, morphology)
            
            # Level 3: Species prediction
            if route in self.species_models:
                species_model = self.species_models[route]['model']
                species_reverse_map = self.species_models[route]['reverse_map']
                
                species_out = species_model(img_tensor)
                species_probs = torch.softmax(species_out, dim=1)
                species_pred = species_out.argmax(1).item()
                species_conf = species_probs[0, species_pred].item()
                species = species_reverse_map[species_pred]
            else:
                species = f"No model for route: {route}"
                species_conf = 0.0
        
        details = {
            'gram': gram,
            'gram_confidence': gram_conf,
            'morphology': morphology,
            'morphology_confidence': morph_conf,
            'route': route,
            'species': species,
            'species_confidence': species_conf
        }
        
        return species, species_conf, details
    
    def predict_batch(self, image_paths, batch_size=32):
        """Predict species for multiple images."""
        
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
            batch_paths = image_paths[i:i+batch_size]
            
            for img_path in batch_paths:
                species, conf, details = self.predict_single(img_path)
                results.append({
                    'image': os.path.basename(img_path),
                    'path': img_path,
                    'species': species,
                    'confidence': conf,
                    **details
                })
        
        return pd.DataFrame(results)
    
    def predict_slide(self, slide_crops_folder, aggregation='vote', save_path=None):
        """
        Predict species for a slide by aggregating crop predictions.
        
        Args:
            slide_crops_folder: Path to folder containing crop images
            aggregation: 'vote' (majority vote) or 'confidence' (weighted)
            save_path: Optional path to save crop-level results CSV
        
        Returns:
            final_species, confidence, summary dict
        """
        
        if not os.path.exists(slide_crops_folder):
            return None, 0.0, {}
        
        crops = [os.path.join(slide_crops_folder, f) for f in os.listdir(slide_crops_folder)
                 if f.endswith('.png') or f.endswith('.jpg')]
        
        if len(crops) == 0:
            return None, 0.0, {}
        
        print(f"  Classifying {len(crops)} crops...")
        start_time = datetime.now()
        
        # Collect predictions
        all_preds = []
        gram_votes = Counter()
        morph_votes = Counter()
        
        for crop_path in tqdm(crops, desc="Processing", leave=False):
            species, conf, details = self.predict_single(crop_path)
            all_preds.append({
                'image': os.path.basename(crop_path),
                'path': crop_path,
                'species': species,
                'confidence': conf,
                **details
            })
            gram_votes[details['gram']] += 1
            morph_votes[details['morphology']] += 1
        
        # Aggregate at slide level
        slide_gram = gram_votes.most_common(1)[0][0]
        slide_morph = morph_votes.most_common(1)[0][0]
        slide_route = get_species_route(slide_gram, slide_morph)
        
        # Filter to predictions from the slide-level route
        route_preds = [p for p in all_preds if p.get('route') == slide_route]
        
        if len(route_preds) == 0:
            route_preds = all_preds
        
        if aggregation == 'vote':
            species_votes = Counter(p['species'] for p in route_preds)
            final_species = species_votes.most_common(1)[0][0]
            vote_count = species_votes.most_common(1)[0][1]
            confidence = vote_count / len(route_preds)
        else:
            species_scores = {}
            for p in route_preds:
                sp = p['species']
                if sp not in species_scores:
                    species_scores[sp] = 0
                species_scores[sp] += p['confidence']
            
            final_species = max(species_scores, key=species_scores.get)
            confidence = species_scores[final_species] / sum(species_scores.values())
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        summary = {
            'total_crops': len(crops),
            'slide_gram': slide_gram,
            'slide_morphology': slide_morph,
            'slide_route': slide_route,
            'gram_distribution': dict(gram_votes),
            'morphology_distribution': dict(morph_votes),
            'species_distribution': dict(Counter(p['species'] for p in route_preds)),
            'processing_time_seconds': elapsed
        }
        
        # Save crop-level results if requested
        if save_path:
            df_crops = pd.DataFrame(all_preds)
            df_crops['slide_prediction'] = final_species
            df_crops['slide_confidence'] = confidence
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            df_crops.to_csv(save_path, index=False)
            print(f"  Saved crop results to: {save_path}")
        
        return final_species, confidence, summary, all_preds


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_pipeline(config, val_slides, gram_model_path, gp_morph_path, gn_morph_path, species_model_paths, 
                      output_dir=None, save_crop_results=True):
    """
    Evaluate the full hierarchical pipeline.
    
    Args:
        config: CONFIG dict with paths
        val_slides: List of slide names to evaluate (e.g., ['SA1', 'SE2', 'KP2'])
        gram_model_path: Path to gram classifier model
        gp_morph_path: Path to GP morphology classifier model
        gn_morph_path: Path to GN morphology classifier model
        species_model_paths: Dict mapping route_name -> model_path
        output_dir: Directory to save results (default: config['model_dir']/evaluation_results)
        save_crop_results: Whether to save crop-level predictions for each slide
    
    Returns:
        df_slides: DataFrame with slide-level results
        df_all_crops: DataFrame with all crop-level predictions
        accuracy: Overall slide-level accuracy
    """
    
    print("=" * 60)
    print("HIERARCHICAL PIPELINE EVALUATION")
    print("Gram → Morphology → Species")
    print("=" * 60)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(config['model_dir'], 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load pipeline
    pipeline = HierarchicalPipeline(config)
    pipeline.load_gram_model(gram_model_path)
    pipeline.load_gp_morphology_model(gp_morph_path)
    pipeline.load_gn_morphology_model(gn_morph_path)
    
    for route_name, model_path in species_model_paths.items():
        if model_path and os.path.exists(model_path):
            pipeline.load_species_model(route_name, model_path)
    
    # Load ground truth
    df_labels = pd.read_excel(config['labels_excel'])
    lookup = {}
    for _, row in df_labels.iterrows():
        name = str(row['Name']).strip()
        lookup[name] = {
            'species': row['Species'].strip(),
            'gram': row['Gram'].strip().lower(),
            'grouping': row['Grouping'].strip()
        }
    
    print(f"\nValidation slides: {len(val_slides)}")
    for slide in val_slides:
        if slide in lookup:
            print(f"  {slide}: {lookup[slide]['species']}")
        else:
            print(f"  {slide}: NOT FOUND IN LABELS")
    
    # Evaluate each slide
    slide_results = []
    all_crop_results = []
    
    for slide_name in val_slides:
        if slide_name not in lookup:
            print(f"\n  Warning: {slide_name} not in lookup, skipping")
            continue
        
        crops_path = os.path.join(config['crops_folder'], slide_name, 'crops')
        if not os.path.exists(crops_path):
            print(f"\n  Warning: {crops_path} not found, skipping")
            continue
        
        true_info = lookup[slide_name]
        
        print(f"\nEvaluating {slide_name} (true: {true_info['species']})...")
        
        # Save crop results for this slide if requested
        crop_save_path = None
        if save_crop_results:
            crop_save_path = os.path.join(output_dir, f'{slide_name}_crops_{timestamp}.csv')
        
        pred_species, confidence, summary, crop_preds = pipeline.predict_slide(
            crops_path, 
            aggregation='vote',
            save_path=crop_save_path
        )
        
        correct = pred_species == true_info['species']
        
        # Add slide info to crop predictions
        for pred in crop_preds:
            pred['slide'] = slide_name
            pred['true_species'] = true_info['species']
            pred['true_gram'] = true_info['gram']
            pred['crop_correct'] = pred['species'] == true_info['species']
        all_crop_results.extend(crop_preds)
        
        slide_results.append({
            'slide': slide_name,
            'true_species': true_info['species'],
            'true_gram': true_info['gram'],
            'pred_species': pred_species,
            'pred_gram': summary.get('slide_gram', ''),
            'pred_morphology': summary.get('slide_morphology', ''),
            'route': summary.get('slide_route', ''),
            'confidence': confidence,
            'correct': correct,
            'total_crops': summary.get('total_crops', 0),
            'species_distribution': str(summary.get('species_distribution', {})),
            'gram_distribution': str(summary.get('gram_distribution', {})),
            'morphology_distribution': str(summary.get('morphology_distribution', {})),
        })
        
        status = "✓" if correct else "✗"
        print(f"  {status} Predicted: {pred_species} ({confidence:.1%})")
        print(f"    Gram: {summary.get('slide_gram')}, Morph: {summary.get('slide_morphology')}")
        print(f"    Route: {summary.get('slide_route')}")
    
    # Create DataFrames
    df_slides = pd.DataFrame(slide_results)
    df_all_crops = pd.DataFrame(all_crop_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    total = len(df_slides)
    correct = df_slides['correct'].sum()
    accuracy = 100 * correct / total if total > 0 else 0
    
    print(f"\nOverall Slide-Level Accuracy: {correct}/{total} = {accuracy:.1f}%")
    
    # Gram accuracy
    gram_correct = (df_slides['true_gram'] == df_slides['pred_gram']).sum()
    print(f"Gram Accuracy: {gram_correct}/{total} = {100*gram_correct/total:.1f}%")
    
    # Per-route accuracy
    print("\nPer-Route Species Accuracy:")
    for route in sorted(df_slides['route'].unique()):
        route_df = df_slides[df_slides['route'] == route]
        rt_correct = route_df['correct'].sum()
        rt_total = len(route_df)
        print(f"  {route}: {rt_correct}/{rt_total} = {100*rt_correct/rt_total:.1f}%")
    
    # Per-species accuracy
    print("\nPer-Species Accuracy:")
    for species in sorted(df_slides['true_species'].unique()):
        species_df = df_slides[df_slides['true_species'] == species]
        sp_correct = species_df['correct'].sum()
        sp_total = len(species_df)
        print(f"  {species}: {sp_correct}/{sp_total} = {100*sp_correct/sp_total:.1f}%")
    
    # Crop-level accuracy
    if len(df_all_crops) > 0:
        crop_correct = df_all_crops['crop_correct'].sum()
        crop_total = len(df_all_crops)
        print(f"\nCrop-Level Accuracy: {crop_correct}/{crop_total} = {100*crop_correct/crop_total:.1f}%")
    
    # Save results
    slide_csv_path = os.path.join(output_dir, f'slide_results_{timestamp}.csv')
    df_slides.to_csv(slide_csv_path, index=False)
    print(f"\nSaved slide results to: {slide_csv_path}")
    
    all_crops_csv_path = os.path.join(output_dir, f'all_crop_results_{timestamp}.csv')
    df_all_crops.to_csv(all_crops_csv_path, index=False)
    print(f"Saved all crop results to: {all_crops_csv_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, f'evaluation_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("HIERARCHICAL PIPELINE EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Validation slides: {len(val_slides)}\n")
        f.write(f"Total crops evaluated: {len(df_all_crops)}\n\n")
        f.write(f"Slide-Level Accuracy: {correct}/{total} = {accuracy:.1f}%\n")
        f.write(f"Gram Accuracy: {gram_correct}/{total} = {100*gram_correct/total:.1f}%\n")
        if len(df_all_crops) > 0:
            f.write(f"Crop-Level Accuracy: {crop_correct}/{crop_total} = {100*crop_correct/crop_total:.1f}%\n")
        f.write("\nValidation Slides:\n")
        for slide in val_slides:
            f.write(f"  - {slide}\n")
        f.write("\nModel Paths:\n")
        f.write(f"  Gram: {gram_model_path}\n")
        f.write(f"  GP Morphology: {gp_morph_path}\n")
        f.write(f"  GN Morphology: {gn_morph_path}\n")
        for route, path in species_model_paths.items():
            f.write(f"  {route}: {path}\n")
    print(f"Saved summary to: {summary_path}")
    
    # Generate plots
    plot_confusion_matrices(df_slides, df_all_crops, output_dir, timestamp)
    plot_roc_curves(df_all_crops, output_dir, timestamp)
    
    return df_slides, df_all_crops, accuracy


def plot_confusion_matrices(df_slides, df_all_crops, output_dir, timestamp):
    """
    Generate confusion matrices for:
    1. Slide-level species predictions
    2. Crop-level species predictions
    3. Gram predictions
    4. Morphology predictions
    """
    
    print("\n" + "=" * 60)
    print("GENERATING CONFUSION MATRICES")
    print("=" * 60)
    
    # 1. Slide-level species confusion matrix
    if len(df_slides) > 0:
        true_species = df_slides['true_species'].values
        pred_species = df_slides['pred_species'].values
        
        # Get all unique species (union of true and predicted)
        all_species = sorted(set(true_species) | set(pred_species))
        
        # Create confusion matrix
        cm = confusion_matrix(true_species, pred_species, labels=all_species)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_species, yticklabels=all_species, ax=ax)
        
        ax.set_xlabel('Predicted Species', fontsize=12)
        ax.set_ylabel('True Species', fontsize=12)
        ax.set_title('Slide-Level Species Confusion Matrix', fontsize=14)
        
        # Rotate labels for readability
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'confusion_matrix_slide_species_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved slide species confusion matrix to: {save_path}")
        plt.close()
    
    # 2. Crop-level species confusion matrix
    if len(df_all_crops) > 0:
        true_species = df_all_crops['true_species'].values
        pred_species = df_all_crops['species'].values
        
        all_species = sorted(set(true_species) | set(pred_species))
        
        cm = confusion_matrix(true_species, pred_species, labels=all_species)
        
        # Normalize by row (percentage)
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_species, yticklabels=all_species, ax=axes[0])
        axes[0].set_xlabel('Predicted Species', fontsize=12)
        axes[0].set_ylabel('True Species', fontsize=12)
        axes[0].set_title('Crop-Level Species Confusion Matrix (Counts)', fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)
        
        # Percentages
        annot_percent = np.array([[f'{cm_percent[i, j]:.0f}%' if cm_percent[i, j] >= 1 else '' 
                                   for j in range(len(all_species))] 
                                  for i in range(len(all_species))])
        
        sns.heatmap(cm_percent, annot=annot_percent, fmt='', cmap='Blues',
                    xticklabels=all_species, yticklabels=all_species, ax=axes[1],
                    vmin=0, vmax=100)
        axes[1].set_xlabel('Predicted Species', fontsize=12)
        axes[1].set_ylabel('True Species', fontsize=12)
        axes[1].set_title('Crop-Level Species Confusion Matrix (Row %)', fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'confusion_matrix_crop_species_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved crop species confusion matrix to: {save_path}")
        plt.close()
    
    # 3. Gram confusion matrix
    if len(df_all_crops) > 0 and 'true_gram' in df_all_crops.columns and 'gram' in df_all_crops.columns:
        true_gram = df_all_crops['true_gram'].values
        pred_gram = df_all_crops['gram'].values
        
        gram_labels = ['positive', 'negative']
        cm = confusion_matrix(true_gram, pred_gram, labels=gram_labels)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Annotations with count and percentage
        annot = np.array([[f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)' 
                          for j in range(2)] for i in range(2)])
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=['Gram+', 'Gram-'], yticklabels=['Gram+', 'Gram-'], ax=ax)
        
        ax.set_xlabel('Predicted Gram', fontsize=12)
        ax.set_ylabel('True Gram', fontsize=12)
        ax.set_title('Gram Classification Confusion Matrix (Crop-Level)', fontsize=14)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'confusion_matrix_gram_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved gram confusion matrix to: {save_path}")
        plt.close()
    
    # 4. Morphology confusion matrix (separate for GP and GN)
    if len(df_all_crops) > 0 and 'morphology' in df_all_crops.columns:
        # We need to infer true morphology from species
        # Create morphology lookup
        species_to_morph = {
            'Staphylococcus aureus': 'cocci_clusters',
            'Staphylococcus epidermidis': 'cocci_clusters',
            'Streptococcus pyogenes': 'cocci_chains',
            'Streptococcus pneumoniae': 'cocci_chains',
            'Streptococcus mitis': 'cocci_chains',
            'Enterococcus faecalis': 'cocci_chains',
            'Enterococcus faecium': 'cocci_chains',
            'Corynebacterium striatum': 'bacilli',
            'Listeria monocytogenes': 'bacilli',
            'Klebsiella pneumoniae': 'bacilli',
            'Enterobacter cloacae': 'bacilli',
            'Pseudomonas aeruginosa': 'bacilli',
            'Acinetobacter baumannii': 'bacilli',
            'Moraxella catarrhalis': 'cocci_clusters',
            'Haemophilus influenzae': 'bacilli',
        }
        
        df_all_crops['true_morphology'] = df_all_crops['true_species'].map(species_to_morph)
        
        true_morph = df_all_crops['true_morphology'].dropna().values
        pred_morph = df_all_crops.loc[df_all_crops['true_morphology'].notna(), 'morphology'].values
        
        if len(true_morph) > 0:
            morph_labels = sorted(set(true_morph) | set(pred_morph))
            cm = confusion_matrix(true_morph, pred_morph, labels=morph_labels)
            
            cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            annot = np.array([[f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)' 
                              for j in range(len(morph_labels))] for i in range(len(morph_labels))])
            
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                        xticklabels=morph_labels, yticklabels=morph_labels, ax=ax)
            
            ax.set_xlabel('Predicted Morphology', fontsize=12)
            ax.set_ylabel('True Morphology', fontsize=12)
            ax.set_title('Morphology Classification Confusion Matrix (Crop-Level)', fontsize=14)
            
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'confusion_matrix_morphology_{timestamp}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved morphology confusion matrix to: {save_path}")
            plt.close()


def plot_roc_curves(df_all_crops, output_dir, timestamp):
    """
    Generate ROC curves for:
    1. Gram classification (binary)
    2. Species classification (multi-class, one-vs-rest)
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    print("\n" + "=" * 60)
    print("GENERATING ROC CURVES")
    print("=" * 60)
    
    if len(df_all_crops) == 0:
        print("No crop data available for ROC curves")
        return
    
    # 1. Gram ROC curve (binary)
    if 'true_gram' in df_all_crops.columns and 'gram_confidence' in df_all_crops.columns:
        # For binary classification: positive = 1, negative = 0
        y_true = (df_all_crops['true_gram'] == 'positive').astype(int).values
        
        # Confidence for positive class
        # If predicted gram is positive, use confidence; if negative, use 1-confidence
        y_scores = []
        for _, row in df_all_crops.iterrows():
            if row['gram'] == 'positive':
                y_scores.append(row['gram_confidence'])
            else:
                y_scores.append(1 - row['gram_confidence'])
        y_scores = np.array(y_scores)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
        
        # Find optimal threshold (Youden's J)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        ax.scatter([fpr[best_idx]], [tpr[best_idx]], color='red', s=100, zorder=5,
                   label=f'Optimal (t={thresholds[best_idx]:.2f})')
        
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title(f'Gram Classification ROC Curve (AUC = {roc_auc:.3f})', fontsize=14)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'roc_curve_gram_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved gram ROC curve to: {save_path}")
        print(f"  Gram AUC: {roc_auc:.4f}")
        plt.close()
    
    # 2. Species ROC curves (multi-class, one-vs-rest)
    if 'true_species' in df_all_crops.columns and 'species_confidence' in df_all_crops.columns:
        # Get all unique species
        all_species = sorted(df_all_crops['true_species'].unique())
        n_classes = len(all_species)
        
        # Binarize true labels
        y_true = df_all_crops['true_species'].values
        y_true_bin = label_binarize(y_true, classes=all_species)
        
        # For each sample, create score vector
        # Score for predicted species = confidence, others = 0
        y_scores = np.zeros((len(df_all_crops), n_classes))
        for i, (_, row) in enumerate(df_all_crops.iterrows()):
            pred_species = row['species']
            if pred_species in all_species:
                pred_idx = all_species.index(pred_species)
                y_scores[i, pred_idx] = row['species_confidence']
        
        # Compute ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i, species in enumerate(all_species):
            fpr[species], tpr[species], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[species] = auc(fpr[species], tpr[species])
        
        # Compute micro-average
        fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        
        # Compute macro-average
        all_fpr = np.unique(np.concatenate([fpr[sp] for sp in all_species]))
        mean_tpr = np.zeros_like(all_fpr)
        for species in all_species:
            mean_tpr += np.interp(all_fpr, fpr[species], tpr[species])
        mean_tpr /= n_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
        
        # Plot all ROC curves
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot micro and macro averages
        ax.plot(fpr['micro'], tpr['micro'],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=3)
        
        ax.plot(fpr['macro'], tpr['macro'],
                label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
                color='navy', linestyle=':', linewidth=3)
        
        # Plot individual species curves
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
        for i, species in enumerate(all_species):
            # Shorten species name for legend
            short_name = species.split()[0][0] + '. ' + species.split()[1] if len(species.split()) > 1 else species
            ax.plot(fpr[species], tpr[species], color=colors[i], linewidth=1.5,
                    label=f'{short_name} (AUC = {roc_auc[species]:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'Species Classification ROC Curves (One-vs-Rest)\nMacro AUC = {roc_auc["macro"]:.3f}', fontsize=14)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'roc_curve_species_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved species ROC curves to: {save_path}")
        print(f"  Species Macro AUC: {roc_auc['macro']:.4f}")
        print(f"  Species Micro AUC: {roc_auc['micro']:.4f}")
        plt.close()
        
        # Also save per-species AUC to CSV
        auc_df = pd.DataFrame([
            {'species': sp, 'auc': roc_auc[sp]} for sp in all_species
        ])
        auc_df = auc_df.sort_values('auc', ascending=False)
        auc_csv_path = os.path.join(output_dir, f'species_auc_scores_{timestamp}.csv')
        auc_df.to_csv(auc_csv_path, index=False)
        print(f"Saved per-species AUC scores to: {auc_csv_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HIERARCHICAL CLASSIFICATION PIPELINE")
    print("Gram → Morphology → Species")
    print("=" * 60)
    
    print("""
PIPELINE STRUCTURE:

  ┌─────────────────────────────────────────────────────────────┐
  │                         CROP                                │
  └─────────────────────────┬───────────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │  GRAM MODEL   │
                    └───────┬───────┘
                            │
           ┌────────────────┴────────────────┐
           │                                 │
    ┌──────▼──────┐                   ┌──────▼──────┐
    │  POSITIVE   │                   │  NEGATIVE   │
    └──────┬──────┘                   └──────┬──────┘
           │                                 │
    ┌──────▼──────┐                   ┌──────▼──────┐
    │ GP MORPH    │                   │ GN MORPH    │
    │ MODEL       │                   │ MODEL       │
    └──────┬──────┘                   └──────┬──────┘
           │                                 │
     ┌─────┼─────┐                     ┌─────┴─────┐
     │     │     │                     │           │
  ┌──▼──┐┌─▼─┐┌──▼──┐               ┌──▼──┐     ┌──▼──┐
  │Clust││Chn││Bacil│               │Bacil│     │Cocci│
  └──┬──┘└─┬─┘└──┬──┘               └──┬──┘     └──┬──┘
     │     │     │                     │           │
  ┌──▼──┐┌─▼─┐┌──▼──┐               ┌──▼──┐     ┌──▼──┐
  │SA,SE││SPY││CB,LS│               │KP,EC│     │MC,HI│
  │     ││SPN│|     │               │PA,AB│     │     │
  └─────┘│VS │└─────┘               └─────┘     └─────┘
         │EFS│
         │EF │
         └───┘

USAGE:

1. LOAD AND USE PIPELINE:
   
   from hierarchical_pipeline import HierarchicalPipeline, CONFIG
   
   pipeline = HierarchicalPipeline(CONFIG)
   
   # Load all models
   pipeline.load_gram_model('models/gram_vit_best_XXX.pth')
   pipeline.load_gp_morphology_model('models/gp_morphology_vit_best_XXX.pth')
   pipeline.load_gn_morphology_model('models/gn_morphology_vit_best_XXX.pth')
   
   pipeline.load_species_model('gp_cocci_clusters', 'models/gp_cocci_clusters_best_XXX.pth')
   pipeline.load_species_model('gp_cocci_chains', 'models/gp_cocci_chains_best_XXX.pth')
   pipeline.load_species_model('gp_bacilli', 'models/gp_bacilli_best_XXX.pth')
   pipeline.load_species_model('gn_bacilli', 'models/gn_bacilli_best_XXX.pth')
   pipeline.load_species_model('gn_cocci', 'models/gn_cocci_best_XXX.pth')
   
   # Single crop prediction
   species, conf, details = pipeline.predict_single('crop.png')
   print(f"Species: {species} ({conf:.1%})")
   print(f"Gram: {details['gram']}, Morphology: {details['morphology']}")
   print(f"Route: {details['route']}")
   
   # Slide-level prediction with CSV output
   species, conf, summary, crop_preds = pipeline.predict_slide(
       'slide/crops/',
       aggregation='vote',
       save_path='results/slide_crops.csv'  # Optional: saves crop-level results
   )
   print(f"Slide: {species} ({conf:.1%})")

2. EVALUATE PIPELINE WITH SELECTED VALIDATION SLIDES:
   
   from hierarchical_pipeline import evaluate_pipeline, CONFIG
   
   # SELECT YOUR VALIDATION SLIDES HERE
   val_slides = [
       # Gram-positive Cocci Clusters
       'SA1', 'SE2',
       # Gram-positive Cocci Chains  
       'EFS10', 'EF8', 'SPY2', 'SPN6', 'VS3',
       # Gram-positive Bacilli
       'CB13', 'LS9',
       # Gram-negative Bacilli
       'KP2', 'ECL7', 'PA3', 'AB2',
       # Gram-negative Cocci
       'MC1', 'HI5',
   ]
   
   species_models = {
       'gp_cocci_clusters': r'C:\\...\\gp_cocci_clusters_best_XXX.pth',
       'gp_cocci_chains': r'C:\\...\\gp_cocci_chains_best_XXX.pth',
       'gp_bacilli': r'C:\\...\\gp_bacilli_best_XXX.pth',
       'gn_bacilli': r'C:\\...\\gn_bacilli_best_XXX.pth',
       'gn_cocci': r'C:\\...\\gn_cocci_best_XXX.pth',
   }
   
   df_slides, df_crops, accuracy = evaluate_pipeline(
       CONFIG,
       val_slides=val_slides,
       gram_model_path=r'C:\\...\\gram_vit_best_XXX.pth',
       gp_morph_path=r'C:\\...\\gp_morphology_vit_best_XXX.pth',
       gn_morph_path=r'C:\\...\\gn_morphology_vit_best_XXX.pth',
       species_model_paths=species_models,
       output_dir=r'C:\\Gram Stain Training Data\\models\\evaluation_results',
       save_crop_results=True  # Save individual crop predictions per slide
   )
   
   # Results saved to:
   #   evaluation_results/slide_results_YYYYMMDD_HHMMSS.csv
   #   evaluation_results/all_crop_results_YYYYMMDD_HHMMSS.csv
   #   evaluation_results/SA1_crops_YYYYMMDD_HHMMSS.csv (per-slide)
   #   evaluation_results/evaluation_summary_YYYYMMDD_HHMMSS.txt

MODELS NEEDED:
  - 1 Gram model (positive/negative)
  - 1 GP Morphology model (cocci_clusters, cocci_chains, bacilli)
  - 1 GN Morphology model (bacilli, cocci_clusters)
  - 5 Species models (gp_cocci_clusters, gp_cocci_chains, gp_bacilli, gn_bacilli, gn_cocci)
  
  TOTAL: 8 models

OUTPUT FILES:
  - slide_results_XXX.csv: One row per slide with predictions
  - all_crop_results_XXX.csv: All crops from all slides combined
  - {slide}_crops_XXX.csv: Individual crop predictions per slide
  - evaluation_summary_XXX.txt: Text summary with accuracies
""")