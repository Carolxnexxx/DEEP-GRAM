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


SPECIES_ROUTES = {
    'gp_cocci_clusters': {
        'gram': 'positive',
        'morphology': 'cocci_clusters',
        'species': ['Staphylococcus aureus', 'Staphylococcus epidermidis'],
        'description': 'Gram-positive Cocci Clusters (Staphylococci)',
    },
    'gp_cocci_chains': {
        'gram': 'positive',
        'morphology': 'cocci_chains',
        'species': ['Streptococcus pyogenes', 'Streptococcus pneumoniae', 'Streptococcus mitis',
                    'Enterococcus faecalis', 'Enterococcus faecium'],
        'description': 'Gram-positive Cocci Chains (Strep/Enterococci)',
    },
    'gp_bacilli': {
        'gram': 'positive',
        'morphology': 'bacilli',
        'species': ['Corynebacterium striatum', 'Listeria monocytogenes'],
        'description': 'Gram-positive Bacilli',
    },
    'gn_bacilli': {
        'gram': 'negative',
        'morphology': 'bacilli',
        'species': ['Klebsiella pneumoniae', 'Enterobacter cloacae', 'Pseudomonas aeruginosa'],
        'description': 'Gram-negative Bacilli',
    },
    'gn_cocci': {
        'gram': 'negative',
        'morphology': 'cocci_clusters',
        'species': ['Moraxella catarrhalis'],
        'description': 'Gram-negative Cocci',
    },
}


def get_species_route(gram, morphology):
    gram = gram.lower() if gram else ''
    morphology = morphology.lower().replace(' ', '_') if morphology else ''

    for route_name, config in SPECIES_ROUTES.items():
        if config['gram'] == gram and config['morphology'] == morphology:
            return route_name

    if gram == 'positive':
        if 'cluster' in morphology:
            return 'gp_cocci_clusters'
        elif 'chain' in morphology:
            return 'gp_cocci_chains'
        else:
            return 'gp_bacilli'
    else:
        if 'cocci' in morphology or 'cluster' in morphology:
            return 'gn_cocci'
        else:
            return 'gn_bacilli'


CONFIG = {
    'labels_excel': r'C:\Users\carol\gram stain ai\species_training.xlsx',
    'crops_folder': r'C:\Gram Stain Training Data\detections',
    'model_dir': r'C:\Gram Stain Training Data\models',

    'gram_model': r"C:\Gram Stain Training Data\models\gram updated\gram_vit_best_20260409_212633.pth",
    'gp_morphology_model': r"C:\Gram Stain Training Data\models\gram pos\morphology_vit_best_20260319_193821.pth",
    'gn_morphology_model': r"C:\Gram Stain Training Data\models\gram neg updated\gram - morphology_vit_best_20260408_163816.pth",

    'species_models': {
        'gp_cocci_clusters': r"C:\Gram Stain Training Data\models\Gram Pos Cocci Clusters\gram_pos_clusters.pth",
        'gp_cocci_chains': r"C:\Gram Stain Training Data\models\Gram Pos Cocci Chains\gram_pos_cocci_chains_best_20260315_124115.pth",
        'gp_bacilli': r"C:\Gram Stain Training Data\models\Gram Pos Bacilli\gram_pos_bacilli_best_20260316_093529.pth",
        'gn_bacilli': r"C:\Gram Stain Training Data\models\gram neg bacilli updated\gram - bacilli.pth",
        'gn_cocci': r"C:\Gram Stain Training Data\models\Gram Neg Cocci\gram_neg_cocci.pth",
    },

    'species_thresholds': {
        'gp_cocci_clusters': {'Staphylococcus aureus': 0.1790},
    },

    'image_size': (224, 224),
    'batch_size': 32,
    'epochs_frozen': 5,
    'epochs_unfrozen': 15,
    'lr_frozen': 1e-3,
    'lr_unfrozen': 1e-5,
    'patience': 10,
}


class ClassifierViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
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


class HierarchicalPipeline:
    def __init__(self, config):
        self.config = config

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
        if not os.path.exists(model_path):
            print(f"WARNING: Gram model not found: {model_path}")
            return

        checkpoint = torch.load(model_path, map_location=device)
        num_classes = checkpoint.get('num_classes', 2)

        model = ClassifierViT(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        self.gram_model = model
        self.gram_reverse_map = {int(k): v for k, v in checkpoint['reverse_map'].items()}
        print(f"Loaded Gram model: {checkpoint.get('val_acc', 'N/A')}%")
        print(f"  Classes: {list(checkpoint['label_map'].keys())}")

    def load_gp_morphology_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"WARNING: GP Morphology model not found: {model_path}")
            return

        checkpoint = torch.load(model_path, map_location=device)
        num_classes = checkpoint.get('num_classes', 3)

        model = ClassifierViT(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        self.gp_morphology_model = model
        self.gp_morphology_reverse_map = {int(k): v for k, v in checkpoint['reverse_map'].items()}
        print(f"Loaded GP Morphology model: {checkpoint.get('val_acc', 'N/A')}%")
        print(f"  Classes: {list(checkpoint['label_map'].keys())}")

    def load_gn_morphology_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"WARNING: GN Morphology model not found: {model_path}")
            return

        checkpoint = torch.load(model_path, map_location=device)
        num_classes = checkpoint.get('num_classes', 2)

        model = ClassifierViT(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        self.gn_morphology_model = model
        self.gn_morphology_reverse_map = {int(k): v for k, v in checkpoint['reverse_map'].items()}
        print(f"Loaded GN Morphology model: {checkpoint.get('val_acc', 'N/A')}%")
        print(f"  Classes: {list(checkpoint['label_map'].keys())}")

    def load_species_model(self, route_name, model_path):
        from torchvision import models

        if not os.path.exists(model_path):
            print(f"WARNING: Species model not found for {route_name}: {model_path}")
            return

        checkpoint = torch.load(model_path, map_location=device)
        num_classes = checkpoint['num_classes']
        state_dict = checkpoint['model_state_dict']

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
            'val_acc': checkpoint.get('val_acc', 'N/A')
        }
        print(f"Loaded {route_name} species model ({arch}): {checkpoint.get('val_acc', 'N/A')}%")
        print(f"  Species: {list(checkpoint['label_map'].keys())}")

    def predict_single(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None:
                return None, 0.0, {}
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path_or_array
            if len(img.shape) == 3 and img.shape[2] == 3:
                if img.dtype == np.uint8:
                    pass
                else:
                    img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = self.transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            gram_out = self.gram_model(img_tensor)
            gram_probs = torch.softmax(gram_out, dim=1)
            gram_pred = gram_out.argmax(1).item()
            gram_conf = gram_probs[0, gram_pred].item()
            gram = self.gram_reverse_map[gram_pred]

            if gram == 'positive':
                if self.gp_morphology_model is None:
                    return None, 0.0, {'error': 'GP morphology model not loaded'}

                morph_out = self.gp_morphology_model(img_tensor)
                morph_probs = torch.softmax(morph_out, dim=1)
                morph_pred = morph_out.argmax(1).item()
                morph_conf = morph_probs[0, morph_pred].item()
                morphology = self.gp_morphology_reverse_map[morph_pred]
            else:
                if self.gn_morphology_model is None:
                    return None, 0.0, {'error': 'GN morphology model not loaded'}

                morph_out = self.gn_morphology_model(img_tensor)
                morph_probs = torch.softmax(morph_out, dim=1)
                morph_pred = morph_out.argmax(1).item()
                morph_conf = morph_probs[0, morph_pred].item()
                morphology = self.gn_morphology_reverse_map[morph_pred]

            route = get_species_route(gram, morphology)

            if route in self.species_models:
                species_model = self.species_models[route]['model']
                species_reverse_map = self.species_models[route]['reverse_map']
                species_label_map = self.species_models[route]['label_map']

                species_out = species_model(img_tensor)
                species_probs = torch.softmax(species_out, dim=1)

                thresholds = self.config.get('species_thresholds', {}).get(route, {})

                if len(species_reverse_map) == 2 and thresholds:
                    threshold_class = list(thresholds.keys())[0]
                    threshold_value = thresholds[threshold_class]
                    threshold_idx = species_label_map[threshold_class]

                    prob_threshold_class = species_probs[0, threshold_idx].item()

                    if prob_threshold_class >= threshold_value:
                        species_pred = threshold_idx
                    else:
                        species_pred = 1 - threshold_idx

                    species_conf = species_probs[0, species_pred].item()
                    species = species_reverse_map[species_pred]
                else:
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
        if not os.path.exists(slide_crops_folder):
            return None, 0.0, {}, []

        crops = [os.path.join(slide_crops_folder, f) for f in os.listdir(slide_crops_folder)
                 if f.lower().endswith('.png') or f.lower().endswith('.jpg')]

        if len(crops) == 0:
            return None, 0.0, {}, []

        print(f"  Classifying {len(crops)} crops (each independently routed)...")
        start_time = datetime.now()

        all_preds = []
        gram_votes = Counter()
        morph_votes = Counter()
        route_votes = Counter()

        for crop_path in tqdm(crops, desc="Processing", leave=False):
            species, conf, details = self.predict_single(crop_path)

            if species is None:
                continue

            all_preds.append({
                'image': os.path.basename(crop_path),
                'path': crop_path,
                'species': species,
                'confidence': conf,
                **details
            })
            gram_votes[details['gram']] += 1
            morph_votes[details['morphology']] += 1
            route_votes[details['route']] += 1

        if len(all_preds) == 0:
            return None, 0.0, {}, []

        if aggregation == 'vote':
            species_votes = Counter(p['species'] for p in all_preds)
            final_species = species_votes.most_common(1)[0][0]
            vote_count = species_votes.most_common(1)[0][1]
            confidence = vote_count / len(all_preds)
        else:
            species_scores = {}
            for p in all_preds:
                sp = p['species']
                if sp not in species_scores:
                    species_scores[sp] = 0
                species_scores[sp] += p['confidence']

            final_species = max(species_scores, key=species_scores.get)
            confidence = species_scores[final_species] / sum(species_scores.values())

        elapsed = (datetime.now() - start_time).total_seconds()

        slide_gram = gram_votes.most_common(1)[0][0] if gram_votes else 'unknown'
        slide_morphology = morph_votes.most_common(1)[0][0] if morph_votes else 'unknown'

        summary = {
            'total_crops': len(crops),
            'valid_crops': len(all_preds),
            'predicted_gram': slide_gram,
            'predicted_morphology': slide_morphology,
            'gram_distribution': dict(gram_votes),
            'morphology_distribution': dict(morph_votes),
            'route_distribution': dict(route_votes),
            'species_distribution': dict(Counter(p['species'] for p in all_preds)),
            'processing_time_seconds': elapsed
        }

        if save_path:
            df_crops = pd.DataFrame(all_preds)
            df_crops['slide_prediction'] = final_species
            df_crops['slide_confidence'] = confidence
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df_crops.to_csv(save_path, index=False)
            print(f"  Saved crop results to: {save_path}")

        return final_species, confidence, summary, all_preds


def evaluate_pipeline(config, val_slides, gram_model_path, gp_morph_path, gn_morph_path, species_model_paths,
                      output_dir=None, save_crop_results=True):
    print("=" * 60)
    print("HIERARCHICAL PIPELINE EVALUATION")
    print("Each crop independently routed: Gram → Morphology → Species")
    print("=" * 60)

    if output_dir is None:
        output_dir = os.path.join(config['model_dir'], 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pipeline = HierarchicalPipeline(config)
    pipeline.load_gram_model(gram_model_path)
    pipeline.load_gp_morphology_model(gp_morph_path)
    pipeline.load_gn_morphology_model(gn_morph_path)

    for route_name, model_path in species_model_paths.items():
        if model_path and os.path.exists(model_path):
            pipeline.load_species_model(route_name, model_path)

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

        crop_save_path = None
        if save_crop_results:
            crop_save_path = os.path.join(output_dir, f'{slide_name}_crops_{timestamp}.csv')

        pred_species, confidence, summary, crop_preds = pipeline.predict_slide(
            crops_path,
            aggregation='vote',
            save_path=crop_save_path
        )

        correct = pred_species == true_info['species']

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
            'pred_gram': summary.get('predicted_gram', ''),
            'pred_morphology': summary.get('predicted_morphology', ''),
            'confidence': confidence,
            'correct': correct,
            'total_crops': summary.get('total_crops', 0),
            'route_distribution': str(summary.get('route_distribution', {})),
            'species_distribution': str(summary.get('species_distribution', {})),
            'gram_distribution': str(summary.get('gram_distribution', {})),
            'morphology_distribution': str(summary.get('morphology_distribution', {})),
        })

        status = "✓" if correct else "✗"
        print(f"  {status} Predicted: {pred_species} ({confidence:.1%})")
        print(f"    Gram: {summary.get('predicted_gram')} | Morphology: {summary.get('predicted_morphology')}")
        print(f"    Routes used: {summary.get('route_distribution')}")

    df_slides = pd.DataFrame(slide_results)
    df_all_crops = pd.DataFrame(all_crop_results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    total = len(df_slides)
    correct = df_slides['correct'].sum()
    accuracy = 100 * correct / total if total > 0 else 0

    print(f"\nOverall Slide-Level Accuracy: {correct}/{total} = {accuracy:.1f}%")

    print("\nPer-Species Accuracy:")
    for species in sorted(df_slides['true_species'].unique()):
        species_df = df_slides[df_slides['true_species'] == species]
        sp_correct = species_df['correct'].sum()
        sp_total = len(species_df)
        print(f"  {species}: {sp_correct}/{sp_total} = {100*sp_correct/sp_total:.1f}%")

    if len(df_all_crops) > 0:
        crop_correct = df_all_crops['crop_correct'].sum()
        crop_total = len(df_all_crops)
        print(f"\nCrop-Level Accuracy: {crop_correct}/{crop_total} = {100*crop_correct/crop_total:.1f}%")

    slide_csv_path = os.path.join(output_dir, f'slide_results_{timestamp}.csv')
    df_slides.to_csv(slide_csv_path, index=False)
    print(f"\nSaved slide results to: {slide_csv_path}")

    all_crops_csv_path = os.path.join(output_dir, f'all_crop_results_{timestamp}.csv')
    df_all_crops.to_csv(all_crops_csv_path, index=False)
    print(f"Saved all crop results to: {all_crops_csv_path}")

    generate_confusion_matrices(df_slides, df_all_crops, output_dir, timestamp)

    return df_slides, df_all_crops, accuracy


def generate_confusion_matrices(df_slides, df_all_crops, output_dir, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING CONFUSION MATRICES")
    print("=" * 60)

    def plot_confusion_matrix(cm, labels, title, save_path, figsize=None):
        n_classes = len(labels)

        if figsize is None:
            figsize = (max(8, n_classes * 1.2), max(6, n_classes * 1.0))

        cm_pct = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10) * 100

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

        annot_size = 16 if n_classes <= 4 else 12 if n_classes <= 8 else 9
        label_size = 14 if n_classes <= 6 else 11
        title_size = 16 if n_classes <= 6 else 14

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[0],
                    annot_kws={'size': annot_size, 'fontweight': 'bold'})
        axes[0].set_xlabel('Predicted', fontsize=label_size, fontweight='bold')
        axes[0].set_ylabel('True', fontsize=label_size, fontweight='bold')
        axes[0].set_title(f'{title} (Counts)', fontsize=title_size, fontweight='bold')
        axes[0].tick_params(axis='both', labelsize=label_size - 2)
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
        plt.setp(axes[0].get_yticklabels(), rotation=0)

        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[1],
                    annot_kws={'size': annot_size, 'fontweight': 'bold'},
                    vmin=0, vmax=100)
        axes[1].set_xlabel('Predicted', fontsize=label_size, fontweight='bold')
        axes[1].set_ylabel('True', fontsize=label_size, fontweight='bold')
        axes[1].set_title(f'{title} (Row %)', fontsize=title_size, fontweight='bold')
        axes[1].tick_params(axis='both', labelsize=label_size - 2)
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        plt.setp(axes[1].get_yticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {save_path}")

    if len(df_slides) > 0 and 'true_species' in df_slides.columns and 'pred_species' in df_slides.columns:
        true_species = df_slides['true_species'].values
        pred_species = df_slides['pred_species'].values

        all_species = sorted(set(true_species) | set(pred_species))
        cm = confusion_matrix(true_species, pred_species, labels=all_species)

        cm_df = pd.DataFrame(cm, index=all_species, columns=all_species)
        cm_df.index.name = 'true_species'
        cm_df.columns.name = 'pred_species'

        csv_path = os.path.join(output_dir, f'confusion_matrix_species_slide_{timestamp}.csv')
        cm_df.to_csv(csv_path)
        print(f"Saved slide-level species confusion matrix CSV: {csv_path}")

        png_path = os.path.join(output_dir, f'confusion_matrix_species_slide_{timestamp}.png')
        plot_confusion_matrix(cm, all_species, 'Slide-Level Species', png_path)

        print("\nSlide-Level Species Confusion Matrix:")
        print(cm_df.to_string())

    if len(df_all_crops) > 0 and 'true_species' in df_all_crops.columns and 'species' in df_all_crops.columns:
        true_species = df_all_crops['true_species'].values
        pred_species = df_all_crops['species'].values

        all_species = sorted(set(true_species) | set(pred_species))
        cm = confusion_matrix(true_species, pred_species, labels=all_species)

        cm_df = pd.DataFrame(cm, index=all_species, columns=all_species)
        cm_df.index.name = 'true_species'
        cm_df.columns.name = 'pred_species'

        csv_path = os.path.join(output_dir, f'confusion_matrix_species_crop_{timestamp}.csv')
        cm_df.to_csv(csv_path)
        print(f"\nSaved crop-level species confusion matrix CSV: {csv_path}")

        cm_pct = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10) * 100
        cm_pct_df = pd.DataFrame(cm_pct, index=all_species, columns=all_species)
        cm_pct_df.index.name = 'true_species'
        cm_pct_df.columns.name = 'pred_species'

        csv_path_pct = os.path.join(output_dir, f'confusion_matrix_species_crop_percent_{timestamp}.csv')
        cm_pct_df.round(2).to_csv(csv_path_pct)
        print(f"Saved crop-level species confusion matrix (%) CSV: {csv_path_pct}")

        png_path = os.path.join(output_dir, f'confusion_matrix_species_crop_{timestamp}.png')
        plot_confusion_matrix(cm, all_species, 'Crop-Level Species', png_path)

        print("\nCrop-Level Species Confusion Matrix (counts):")
        print(cm_df.to_string())

    if len(df_all_crops) > 0 and 'true_gram' in df_all_crops.columns and 'gram' in df_all_crops.columns:
        true_gram = df_all_crops['true_gram'].values
        pred_gram = df_all_crops['gram'].values

        gram_labels = sorted(set(true_gram) | set(pred_gram))
        cm = confusion_matrix(true_gram, pred_gram, labels=gram_labels)

        cm_df = pd.DataFrame(cm, index=gram_labels, columns=gram_labels)
        cm_df.index.name = 'true_gram'
        cm_df.columns.name = 'pred_gram'

        csv_path = os.path.join(output_dir, f'confusion_matrix_gram_{timestamp}.csv')
        cm_df.to_csv(csv_path)
        print(f"\nSaved gram confusion matrix CSV: {csv_path}")

        png_path = os.path.join(output_dir, f'confusion_matrix_gram_{timestamp}.png')
        plot_confusion_matrix(cm, gram_labels, 'Gram Classification', png_path)

        gram_correct = (df_all_crops['true_gram'] == df_all_crops['gram']).sum()
        gram_total = len(df_all_crops)

        print("\nGram Confusion Matrix:")
        print(cm_df.to_string())
        print(f"Gram Accuracy: {gram_correct}/{gram_total} = {100*gram_correct/gram_total:.1f}%")

    if len(df_all_crops) > 0 and 'morphology' in df_all_crops.columns:
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
            'Escherichia coli': 'bacilli',
            'Stenotrophomonas maltophilia': 'bacilli',
            'Neisseria meningitidis': 'cocci_clusters',
        }

        df_all_crops['true_morphology'] = df_all_crops['true_species'].map(species_to_morph)

        df_morph = df_all_crops[df_all_crops['true_morphology'].notna()].copy()

        if len(df_morph) > 0:
            true_morph = df_morph['true_morphology'].values
            pred_morph = df_morph['morphology'].values

            morph_labels = sorted(set(true_morph) | set(pred_morph))
            cm = confusion_matrix(true_morph, pred_morph, labels=morph_labels)

            cm_df = pd.DataFrame(cm, index=morph_labels, columns=morph_labels)
            cm_df.index.name = 'true_morphology'
            cm_df.columns.name = 'pred_morphology'

            csv_path = os.path.join(output_dir, f'confusion_matrix_morphology_{timestamp}.csv')
            cm_df.to_csv(csv_path)
            print(f"\nSaved morphology confusion matrix CSV: {csv_path}")

            png_path = os.path.join(output_dir, f'confusion_matrix_morphology_{timestamp}.png')
            plot_confusion_matrix(cm, morph_labels, 'Morphology Classification', png_path)

            morph_correct = (df_morph['true_morphology'] == df_morph['morphology']).sum()
            morph_total = len(df_morph)

            print("\nMorphology Confusion Matrix:")
            print(cm_df.to_string())
            print(f"Morphology Accuracy: {morph_correct}/{morph_total} = {100*morph_correct/morph_total:.1f}%")

    if len(df_all_crops) > 0 and 'route' in df_all_crops.columns:
        def get_true_route(row):
            gram = row.get('true_gram', '')
            morph = row.get('true_morphology', '')
            if pd.isna(gram) or pd.isna(morph):
                return None
            return get_species_route(gram, morph)

        df_all_crops['true_route'] = df_all_crops.apply(get_true_route, axis=1)

        df_route = df_all_crops[df_all_crops['true_route'].notna()].copy()

        if len(df_route) > 0:
            true_route = df_route['true_route'].values
            pred_route = df_route['route'].values

            route_labels = sorted(set(true_route) | set(pred_route))
            cm = confusion_matrix(true_route, pred_route, labels=route_labels)

            cm_df = pd.DataFrame(cm, index=route_labels, columns=route_labels)
            cm_df.index.name = 'true_route'
            cm_df.columns.name = 'pred_route'

            csv_path = os.path.join(output_dir, f'confusion_matrix_route_{timestamp}.csv')
            cm_df.to_csv(csv_path)
            print(f"\nSaved route confusion matrix CSV: {csv_path}")

            png_path = os.path.join(output_dir, f'confusion_matrix_route_{timestamp}.png')
            plot_confusion_matrix(cm, route_labels, 'Route Classification', png_path)

            route_correct = (df_route['true_route'] == df_route['route']).sum()
            route_total = len(df_route)

            print("\nRoute Confusion Matrix:")
            print(cm_df.to_string())
            print(f"Route Accuracy: {route_correct}/{route_total} = {100*route_correct/route_total:.1f}%")

    print("\n" + "=" * 60)
    print("CONFUSION MATRICES COMPLETE")
    print("=" * 60)
