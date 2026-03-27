"""
Quick Labeling Tool for Bacteria vs Debris

Shows random crop images. Press keys to label:
  'b' = bacteria (keep)
  'd' = debris/background (delete)
  'u' = undo last
  'q' = quit and save

Labels are saved to a CSV file for training a debris filter.
"""

import os
import cv2
import random
import pandas as pd
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # Folder containing all crop subfolders
    'crops_folder': r'C:\Gram Stain Training Data\detections',
    
    # Output CSV for labels
    'output_csv': r'C:\Gram Stain Training Data\debris_labels.csv',
    
    # Number of images to label per session
    'num_to_label': 1000,
    
    # Display size (crops will be scaled for easier viewing)
    'display_size': 256,
    
    # Random seed for reproducibility
    'random_seed': 42,
}


# ============================================================================
# LABELING TOOL
# ============================================================================

def get_all_crops(crops_folder):
    """Get list of all crop image paths."""
    print("Scanning for crops...")
    
    all_crops = []
    
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
        except Exception as e:
            print(f"  Error reading {crops_path}: {e}")
    
    print(f"Found {len(all_crops):,} total crops")
    return all_crops


def load_existing_labels(csv_path):
    """Load any existing labels to avoid re-labeling."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        labeled = set(df['path'].tolist())
        print(f"Loaded {len(labeled)} existing labels")
        return df.to_dict('records'), labeled
    return [], set()


def save_labels(labels, csv_path):
    """Save labels to CSV."""
    df = pd.DataFrame(labels)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(labels)} labels to {csv_path}")


def run_labeling(config):
    """Main labeling loop."""
    
    # Get all crops
    all_crops = get_all_crops(config['crops_folder'])
    
    if not all_crops:
        print("No crops found!")
        return
    
    # Load existing labels
    labels, already_labeled = load_existing_labels(config['output_csv'])
    
    # Filter out already labeled
    unlabeled = [c for c in all_crops if c not in already_labeled]
    print(f"Unlabeled crops: {len(unlabeled):,}")
    
    if not unlabeled:
        print("All crops have been labeled!")
        return
    
    # Random sample
    random.seed(config['random_seed'])
    random.shuffle(unlabeled)
    to_label = unlabeled[:config['num_to_label']]
    
    print(f"\nLabeling {len(to_label)} images...")
    print("=" * 50)
    print("Controls:")
    print("  'b' = bacteria (keep)")
    print("  'd' = debris/background (delete)")
    print("  'u' = undo last")
    print("  'q' = quit and save")
    print("=" * 50)
    
    display_size = config['display_size']
    session_labels = []
    idx = 0
    
    cv2.namedWindow('Labeling', cv2.WINDOW_AUTOSIZE)
    
    while idx < len(to_label):
        img_path = to_label[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue
        
        # Use original size, but ensure minimum display size for readability
        img_h, img_w = img.shape[:2]
        min_display = 200
        
        if img_h < min_display or img_w < min_display:
            # Scale up small images for easier viewing (bicubic for smooth scaling)
            scale = max(min_display / img_h, min_display / img_w)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img_display = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            # Show at original size
            img_display = img.copy()
        
        # Add label count to image
        bacteria_count = sum(1 for l in session_labels if l['label'] == 'bacteria')
        debris_count = sum(1 for l in session_labels if l['label'] == 'debris')
        
        # Add text overlay
        overlay = img_display.copy()
        h, w = overlay.shape[:2]
        cv2.putText(overlay, f"[{idx+1}/{len(to_label)}]", (5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, f"B:{bacteria_count} D:{debris_count}", (5, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(overlay, "b=bacteria d=debris u=undo q=quit", (5, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Labeling', overlay)
        
        # Wait for key
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('b'):
            # Bacteria
            session_labels.append({
                'path': img_path,
                'label': 'bacteria',
                'filename': os.path.basename(img_path)
            })
            idx += 1
            
        elif key == ord('d'):
            # Debris
            session_labels.append({
                'path': img_path,
                'label': 'debris',
                'filename': os.path.basename(img_path)
            })
            idx += 1
            
        elif key == ord('u'):
            # Undo
            if session_labels:
                removed = session_labels.pop()
                idx -= 1
                print(f"  Undid: {removed['label']}")
            
        elif key == ord('q'):
            # Quit
            print("\nQuitting...")
            break
    
    cv2.destroyAllWindows()
    
    # Merge with existing labels and save
    all_labels = labels + session_labels
    save_labels(all_labels, config['output_csv'])
    
    # Summary
    bacteria_count = sum(1 for l in all_labels if l['label'] == 'bacteria')
    debris_count = sum(1 for l in all_labels if l['label'] == 'debris')
    
    print(f"\n" + "=" * 50)
    print(f"SESSION COMPLETE")
    print(f"=" * 50)
    print(f"This session: {len(session_labels)} labeled")
    print(f"Total labeled: {len(all_labels)}")
    print(f"  Bacteria: {bacteria_count}")
    print(f"  Debris: {debris_count}")
    print(f"\nLabels saved to: {config['output_csv']}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("BACTERIA vs DEBRIS LABELING TOOL")
    print("=" * 50)
    
    run_labeling(CONFIG)