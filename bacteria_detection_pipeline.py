"""
Bacteria Detection Pipeline — Large TIFF to Bounding Boxes

Processes whole slide images tile-by-tile:
1. Loads tiles from large TIFF using zarr
2. Runs U-Net prediction on each tile
3. Extracts bounding boxes around detected bacteria
4. Saves crops for classification
5. Outputs CSV with all detections

Usage:
    python bacteria_detection_pipeline.py
"""

import numpy as np
import cv2
import tifffile
import zarr
import os
import glob
import json
import csv
import torch
import torch.nn as nn
import shutil
from datetime import datetime
from tqdm import tqdm
from skimage import measure, morphology

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def check_storage_gb(path='C:\\'):
    """Check available storage in GB."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)  # Convert to GB


# ============================================================================
# U-NET MODEL
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters_base=64):
        super().__init__()
        
        self.enc1 = DoubleConv(in_channels, filters_base, dropout=0.1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(filters_base, filters_base*2, dropout=0.2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(filters_base*2, filters_base*4, dropout=0.2)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(filters_base*4, filters_base*8, dropout=0.3)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(filters_base*8, filters_base*16, dropout=0.3)
        
        self.up4 = nn.ConvTranspose2d(filters_base*16, filters_base*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(filters_base*16, filters_base*8, dropout=0.2)
        self.up3 = nn.ConvTranspose2d(filters_base*8, filters_base*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(filters_base*8, filters_base*4, dropout=0.2)
        self.up2 = nn.ConvTranspose2d(filters_base*4, filters_base*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(filters_base*4, filters_base*2, dropout=0.1)
        self.up1 = nn.ConvTranspose2d(filters_base*2, filters_base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(filters_base*2, filters_base)
        
        self.out_conv = nn.Conv2d(filters_base, out_channels, kernel_size=1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


# ============================================================================
# DEBRIS FILTER CNN
# ============================================================================

class DebrisCNN(nn.Module):
    """CNN for debris vs bacteria classification."""
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


def load_debris_filter(model_path):
    """Load trained debris filter model."""
    print(f"Loading debris filter from {model_path}...")
    model = DebrisCNN()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    if 'val_acc' in checkpoint:
        print(f"  Debris filter accuracy: {checkpoint['val_acc']:.1f}%")
    return model


def filter_debris_in_folder(crops_folder, debris_model, threshold=0.7):
    """Filter debris from a crops folder, delete debris files."""
    
    # Get all crops
    all_crops = []
    try:
        files = os.listdir(crops_folder)
        all_crops = [os.path.join(crops_folder, f) for f in files 
                     if f.endswith('.png') or f.endswith('.jpg')]
    except:
        return 0, 0
    
    if not all_crops:
        return 0, 0
    
    bacteria_count = 0
    debris_count = 0
    debris_to_delete = []
    
    batch_size = 64
    image_size = (64, 64)
    
    for i in range(0, len(all_crops), batch_size):
        batch_paths = all_crops[i:i+batch_size]
        
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
        
        batch_tensor = torch.stack(images).to(device)
        
        with torch.no_grad():
            outputs = debris_model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            bacteria_probs = probs[:, 0].cpu().numpy()
        
        for path, bacteria_prob in zip(valid_paths, bacteria_probs):
            if bacteria_prob > threshold:
                bacteria_count += 1
            else:
                debris_count += 1
                debris_to_delete.append(path)
    
    # Delete debris
    for path in debris_to_delete:
        try:
            os.remove(path)
        except:
            pass
    
    return bacteria_count, debris_count


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # Model
    'model_path': r"C:\Gram Stain Training Data\models\unet_best_20260216_200753.pth",
    'model_input_size': (1024, 1024),
    
    # Input — folder containing all .tif slides
    'slides_folder': r"D:\Gram Stain Slides",
    'tile_size': 2048,
    
    # Slides to skip (add filenames here)
    'skip_slides': [
        # 'AB1.tif',
        # 'AB2.tif',
    ],
    
    # Or process only specific slides (leave empty to process all)
    # UNDERSAMPLED SPECIES - Re-process to get more crops
    'only_slides': [
        # Neisseria meningitidis (1,500 crops - need ~30k)
        'NS4.tif', 'NS5.tif', 'NS9.tif',
        # Pseudomonas aeruginosa (2,950 crops - need ~30k)
        'PA2.tif', 'PA3.tif', 'PA6.tif', 'PA7.tif', 'PA10.tif',
        # Haemophilus influenzae (3,479 crops - need ~30k)
        'HI4.tif', 'HI5.tif', 'HI6.tif', 'HI7.tif', 'HI8.tif',
        # Stenotrophomonas maltophilia (4,915 crops - need ~30k)
        'SM1.tif', 'SM4.tif', 'SM5.tif', 'SM6.tif', 'SM8.tif', 'SM9.tif',
        # Klebsiella pneumoniae (6,125 crops - need ~30k)
        'KP2.tif', 'KP3.tif', 'KP4.tif', 'KP5.tif', 'KP6.tif', 'KP8.tif', 'KP10.tif',
        # Moraxella catarrhalis (8,192 crops - need ~30k)
        'MC1.tif', 'MC2.tif', 'MC4.tif', 'MC6.tif', 'MC7.tif', 'MC8.tif', 'MC9.tif', 'MC10.tif',
        # Escherichia coli (8,418 crops - need ~30k)
        'EC2.tif', 'EC3.tif', 'EC4.tif', 'EC6.tif', 'EC7.tif',
        # Enterobacter cloacae (17,492 crops - need ~30k)
        'ECL1.tif', 'ECL2.tif', 'ECL3.tif', 'ECL4.tif', 'ECL5.tif', 'ECL7.tif', 'ECL8.tif',
    ],
    
    # Output
    'output_folder': r'C:\Gram Stain Training Data\detections',
    
    # Detection parameters
    'threshold': 0.95,           # Prediction confidence threshold
    'min_area': 5,             # Minimum bacteria area in pixels
    'max_area': 5000,           # Maximum bacteria area (filter large artifacts)
    'min_circularity': 0.2,     # Filter elongated debris
    'crop_padding': 15,         # Padding around each detection
    
    # Auto-skip empty tiles
    'skip_empty_tiles': True,   # Skip tiles that are mostly white background
    'white_threshold': 240,     # Pixel value above this is "white"
    'white_fraction': 0.85,     # Skip if more than this fraction is white
    
    # Storage management
    'save_crops': True,         # Save individual bacteria crops
    'save_overlays': True,      # Save overlay images with bounding boxes
    'save_masks': False,        # Save prediction masks
    'max_crops_per_slide': 50000,   # Increased to get more crops for undersampled species
    'max_overlays_per_slide': 10,   # Randomly select this many overlays to save
    'min_storage_gb': 20,       # Stop processing if storage falls below this
    'delete_after_processing': False,  # DELETE original .tif after processing (DANGEROUS!)
    'overwrite_existing': True,  # Delete existing crops before re-processing
    
    # Debris filtering (runs after detection)
    'filter_debris': True,      # Run debris filter after saving crops
    'debris_model_path': r"C:\Gram Stain Training Data\models\debris_filter_20260227_180942.pth",  # UPDATE THIS
    'debris_threshold': 0.7,    # Keep if bacteria confidence > this
    
    # Processing options
    'process_all_tiles': True,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model(model_path):
    """Load trained U-Net model."""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model = UNet(in_channels=3, out_channels=1, filters_base=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if 'val_precision' in checkpoint:
        print(f"  Val precision: {checkpoint['val_precision']:.4f}")
        print(f"  Val recall: {checkpoint['val_recall']:.4f}")
    
    return model


def open_slide(slide_path, tile_size):
    """Open large TIFF slide."""
    print(f"Opening {slide_path}...")
    tif = tifffile.TiffFile(slide_path)
    series = tif.series[0]
    
    if hasattr(series, 'levels') and len(series.levels) > 1:
        shape = series.levels[0].shape
        store = tif.aszarr(level=0)
    else:
        shape = series.shape
        store = tif.aszarr()
    
    zarray = zarr.open(store, mode='r')
    
    # Navigate into zarr group structure to find the actual array
    # Some TIFFs have nested groups like {'0': array, '1': array, ...}
    while hasattr(zarray, 'keys') and not hasattr(zarray, 'shape'):
        keys = list(zarray.keys())
        if keys:
            # Use '0' for highest resolution, or first available key
            if '0' in keys:
                zarray = zarray['0']
            else:
                zarray = zarray[keys[0]]
        else:
            break
    
    # Verify we have an array now
    if not hasattr(zarray, 'shape'):
        raise ValueError(f"Could not find array in zarr structure. Keys found: {list(zarray.keys()) if hasattr(zarray, 'keys') else 'none'}")
    
    height, width = zarray.shape[0], zarray.shape[1]
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    
    print(f"  Size: {width} x {height}")
    print(f"  Grid: {tiles_x} x {tiles_y} = {tiles_x * tiles_y} tiles")
    
    return {
        'tif': tif,
        'zarray': zarray,
        'width': width,
        'height': height,
        'tiles_x': tiles_x,
        'tiles_y': tiles_y,
        'tile_size': tile_size,
    }


def get_tile(slide, tx, ty):
    """Extract a single tile."""
    ts = slide['tile_size']
    x0 = tx * ts
    y0 = ty * ts
    x1 = min(x0 + ts, slide['width'])
    y1 = min(y0 + ts, slide['height'])
    
    tile = np.array(slide['zarray'][y0:y1, x0:x1])
    
    if len(tile.shape) == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
    elif tile.shape[2] == 4:
        tile = cv2.cvtColor(tile, cv2.COLOR_RGBA2RGB)
    
    return tile.astype(np.uint8), (x0, y0, x1, y1)


def is_empty_tile(tile_rgb, white_threshold=240, white_fraction=0.85):
    """
    Check if tile is mostly empty (white background or black border).
    
    Returns True if tile should be skipped.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    
    # Count pixels that are "white" (above threshold)
    white_pixels = np.sum(gray > white_threshold)
    total_pixels = gray.size
    fraction_white = white_pixels / total_pixels
    
    # Count pixels that are "black" (near zero)
    black_pixels = np.sum(gray < 15)
    fraction_black = black_pixels / total_pixels
    
    # Skip if mostly white OR mostly black
    return fraction_white > white_fraction or fraction_black > 0.5


def predict_tile(model, tile_rgb, input_size, threshold):
    """Run U-Net prediction on a tile."""
    tile_h, tile_w = tile_rgb.shape[:2]
    
    # Resize for model
    tile_resized = cv2.resize(tile_rgb, input_size)
    tile_norm = tile_resized.astype(np.float32) / 255.0
    tile_tensor = torch.from_numpy(tile_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        pred = model(tile_tensor)
    
    prob_map = pred.squeeze().cpu().numpy()
    
    # Resize back to tile size
    prob_map_full = cv2.resize(prob_map, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
    
    # Apply threshold
    mask = (prob_map_full > threshold).astype(np.uint8)
    
    return mask, prob_map_full


def extract_detections(tile_rgb, mask, tile_coords, config):
    """
    Extract bounding boxes and properties from mask.
    
    Returns list of detection dictionaries.
    """
    x0_tile, y0_tile, _, _ = tile_coords
    
    # Remove small objects
    mask_cleaned = morphology.remove_small_objects(
        mask.astype(bool), 
        min_size=config['min_area'],
        connectivity=1
    ).astype(np.uint8)
    
    # Label connected components
    labeled = measure.label(mask_cleaned)
    regions = measure.regionprops(labeled)
    
    detections = []
    
    for region in regions:
        area = region.area
        
        # Filter by area
        if area < config['min_area'] or area > config['max_area']:
            continue
        
        # Filter by circularity
        perimeter = region.perimeter
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < config['min_circularity']:
                continue
        else:
            circularity = 0
        
        # Get bounding box (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = region.bbox
        
        # Add padding
        pad = config['crop_padding']
        min_row = max(0, min_row - pad)
        min_col = max(0, min_col - pad)
        max_row = min(tile_rgb.shape[0], max_row + pad)
        max_col = min(tile_rgb.shape[1], max_col + pad)
        
        # Crop from original image
        crop = tile_rgb[min_row:max_row, min_col:max_col].copy()
        
        # Calculate global coordinates
        global_x = x0_tile + min_col
        global_y = y0_tile + min_row
        
        # Shape hint based on eccentricity
        eccentricity = region.eccentricity
        if eccentricity < 0.6:
            shape_hint = "cocci"
        elif eccentricity > 0.85:
            shape_hint = "bacilli"
        else:
            shape_hint = "coccobacilli"
        
        # Get axis lengths (handle both old and new skimage API)
        if hasattr(region, 'axis_major_length'):
            major_axis = region.axis_major_length
            minor_axis = region.axis_minor_length
        else:
            major_axis = region.major_axis_length
            minor_axis = region.minor_axis_length
        
        detection = {
            'bbox_local': (min_col, min_row, max_col, max_row),  # x1, y1, x2, y2 in tile
            'bbox_global': (global_x, global_y, global_x + (max_col - min_col), global_y + (max_row - min_row)),
            'centroid_local': (region.centroid[1], region.centroid[0]),  # x, y in tile
            'centroid_global': (x0_tile + region.centroid[1], y0_tile + region.centroid[0]),
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'eccentricity': eccentricity,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'shape_hint': shape_hint,
            'crop': crop,
        }
        
        detections.append(detection)
    
    return detections


def draw_bounding_boxes(tile_rgb, detections, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on tile image."""
    overlay = tile_rgb.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox_local']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        
        # Add label
        label = f"{det['shape_hint'][:1].upper()} {det['area']}"
        cv2.putText(overlay, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return overlay


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_detection_pipeline(config, slide_path=None):
    """Run the complete detection pipeline for a single slide."""
    
    # Use provided slide_path or fall back to config
    if slide_path is None:
        slide_path = config.get('slide_path')
    
    if slide_path is None:
        print("ERROR: No slide_path provided!")
        return []
    
    # Check storage
    min_storage = config.get('min_storage_gb', 20)
    free_gb = check_storage_gb()
    if free_gb < min_storage:
        print(f"WARNING: Only {free_gb:.1f} GB free. Minimum is {min_storage} GB. Stopping.")
        return []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output folders — use slide name as subfolder
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    output_folder = os.path.join(config['output_folder'], slide_name)
    crops_folder = os.path.join(output_folder, 'crops')
    overlays_folder = os.path.join(output_folder, 'overlays')
    masks_folder = os.path.join(output_folder, 'masks')
    
    # Delete existing crops if overwrite is enabled
    if config.get('overwrite_existing', False):
        if os.path.exists(crops_folder):
            existing_count = len([f for f in os.listdir(crops_folder) if f.endswith('.png') or f.endswith('.jpg')])
            print(f"  Deleting {existing_count} existing crops...")
            shutil.rmtree(crops_folder)
        if os.path.exists(overlays_folder):
            shutil.rmtree(overlays_folder)
        if os.path.exists(masks_folder):
            shutil.rmtree(masks_folder)
    
    os.makedirs(output_folder, exist_ok=True)
    if config.get('save_crops', True):
        os.makedirs(crops_folder, exist_ok=True)
    if config.get('save_overlays', False):
        os.makedirs(overlays_folder, exist_ok=True)
    if config.get('save_masks', False):
        os.makedirs(masks_folder, exist_ok=True)
    
    # Load model
    model = load_model(config['model_path'])
    
    # Open slide
    slide = open_slide(slide_path, config['tile_size'])
    
    # Determine tiles to process
    if config.get('process_all_tiles', True):
        tiles_to_process = [
            (tx, ty) 
            for ty in range(slide['tiles_y']) 
            for tx in range(slide['tiles_x'])
        ]
    else:
        tiles_to_process = config.get('tiles_to_process', [])
    
    if not tiles_to_process:
        print("\nERROR: No tiles specified!")
        return []
    
    print(f"\nProcessing {slide_name}...")
    print(f"Tiles: {len(tiles_to_process)}")
    print(f"Threshold: {config['threshold']}")
    print(f"Storage free: {free_gb:.1f} GB")
    
    # Process tiles
    all_detections = []
    detection_id = 0
    tiles_skipped = 0
    tiles_with_detections = 0
    max_crops = config.get('max_crops_per_slide', 10000)
    max_overlays = config.get('max_overlays_per_slide', 10)
    
    # Collect all valid crops first, then randomly select
    all_valid_crops = []
    
    # Collect tiles with detections for overlay saving
    tiles_for_overlay = []
    
    for tx, ty in tqdm(tiles_to_process, desc=f"Processing {slide_name}"):
        # Check storage periodically
        if len(all_valid_crops) % 5000 == 0 and len(all_valid_crops) > 0:
            free_gb = check_storage_gb()
            if free_gb < min_storage:
                print(f"\nStorage low ({free_gb:.1f} GB). Stopping early.")
                break
        
        # Get tile
        tile_rgb, tile_coords = get_tile(slide, tx, ty)
        
        # Skip empty tiles
        if config.get('skip_empty_tiles', True):
            if is_empty_tile(tile_rgb, 
                           config.get('white_threshold', 240),
                           config.get('white_fraction', 0.85)):
                tiles_skipped += 1
                continue
        
        # Predict
        mask, prob_map = predict_tile(
            model, tile_rgb, 
            config['model_input_size'], 
            config['threshold']
        )
        
        # Extract detections
        detections = extract_detections(tile_rgb, mask, tile_coords, config)
        
        # Collect valid crops
        if config.get('save_crops', True) and detections:
            for det in detections:
                crop = det['crop']
                
                # Skip empty crops
                if crop.size == 0:
                    continue
                
                # Skip crops that are too dark (likely artifacts/background)
                mean_brightness = np.mean(crop)
                if mean_brightness < 50:
                    continue
                
                # Skip crops with black regions (edge of slide)
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                black_pixels = np.sum(gray_crop < 15)
                total_pixels = gray_crop.size
                if black_pixels / total_pixels > 0.1:
                    continue
                
                # Skip crops that are mostly white
                if mean_brightness > 240:
                    continue
                
                # Store valid crop info for later saving
                all_valid_crops.append({
                    'crop': crop.copy(),
                    'tile': (tx, ty),
                    'det': {k: v for k, v in det.items() if k != 'crop'}
                })
        
        # Collect tile for potential overlay (save tile data, not the image yet)
        if config.get('save_overlays', False) and detections:
            tiles_for_overlay.append({
                'tile_rgb': tile_rgb.copy(),
                'detections': detections,
                'tx': tx,
                'ty': ty
            })
        
        # Save mask
        if config.get('save_masks', False):
            mask_path = os.path.join(masks_folder, f"tile_{tx}_{ty}_mask.png")
            cv2.imwrite(mask_path, mask * 255)
        
        # Add to all detections (remove crop array to save memory)
        for det in detections:
            det_copy = {k: v for k, v in det.items() if k != 'crop'}
            all_detections.append(det_copy)
        
        if detections:
            tiles_with_detections += 1
    
    # Close slide
    slide['tif'].close()
    
    # Randomly select crops to save
    print(f"\nFound {len(all_valid_crops)} valid crops, selecting {min(max_crops, len(all_valid_crops))} to save...")
    
    if len(all_valid_crops) > max_crops:
        import random
        random.seed(42)  # For reproducibility
        selected_crops = random.sample(all_valid_crops, max_crops)
    else:
        selected_crops = all_valid_crops
    
    # Save selected crops
    crops_saved = 0
    for i, crop_info in enumerate(tqdm(selected_crops, desc="Saving crops")):
        crop = crop_info['crop']
        tx, ty = crop_info['tile']
        det = crop_info['det']
        
        crop_filename = f"{slide_name}_{i:05d}.png"
        crop_path = os.path.join(crops_folder, crop_filename)
        cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        
        det['crop_path'] = crop_path
        det['detection_id'] = i
        det['tile'] = (tx, ty)
        det['crop_width'] = crop.shape[1]
        det['crop_height'] = crop.shape[0]
        crops_saved += 1
    
    # Clear memory
    del all_valid_crops
    del selected_crops
    
    # Randomly select and save overlays
    if config.get('save_overlays', False) and tiles_for_overlay:
        import random
        random.seed(43)  # Different seed from crops
        
        num_overlays = min(max_overlays, len(tiles_for_overlay))
        print(f"Found {len(tiles_for_overlay)} tiles with detections, saving {num_overlays} overlays...")
        
        selected_overlays = random.sample(tiles_for_overlay, num_overlays)
        
        for overlay_info in selected_overlays:
            tile_rgb = overlay_info['tile_rgb']
            detections = overlay_info['detections']
            tx = overlay_info['tx']
            ty = overlay_info['ty']
            
            overlay = draw_bounding_boxes(tile_rgb, detections)
            overlay_path = os.path.join(overlays_folder, f"tile_{tx}_{ty}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        del tiles_for_overlay
        del selected_overlays
    
    # Save CSV with all detections
    csv_path = os.path.join(output_folder, f'detections_{timestamp}.csv')
    
    if all_detections:
        fieldnames = [
            'detection_id', 'tile_x', 'tile_y',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'crop_width', 'crop_height',
            'centroid_x', 'centroid_y',
            'area', 'circularity', 'eccentricity',
            'major_axis', 'minor_axis', 'shape_hint',
            'crop_path',
            'gram', 'species', 'notes'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for det in all_detections:
                row = {
                    'detection_id': det.get('detection_id', ''),
                    'tile_x': det.get('tile', ('', ''))[0],
                    'tile_y': det.get('tile', ('', ''))[1],
                    'bbox_x1': det['bbox_global'][0],
                    'bbox_y1': det['bbox_global'][1],
                    'bbox_x2': det['bbox_global'][2],
                    'bbox_y2': det['bbox_global'][3],
                    'crop_width': det.get('crop_width', ''),
                    'crop_height': det.get('crop_height', ''),
                    'centroid_x': f"{det['centroid_global'][0]:.1f}",
                    'centroid_y': f"{det['centroid_global'][1]:.1f}",
                    'area': det['area'],
                    'circularity': f"{det['circularity']:.3f}",
                    'eccentricity': f"{det['eccentricity']:.3f}",
                    'major_axis': f"{det['major_axis']:.1f}",
                    'minor_axis': f"{det['minor_axis']:.1f}",
                    'shape_hint': det['shape_hint'],
                    'crop_path': det.get('crop_path', ''),
                    'gram': '',
                    'species': '',
                    'notes': ''
                }
                writer.writerow(row)
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'slide_path': slide_path,
        'slide_name': slide_name,
        'model_path': config['model_path'],
        'threshold': config['threshold'],
        'tiles_total': len(tiles_to_process),
        'tiles_skipped': tiles_skipped,
        'tiles_processed': len(tiles_to_process) - tiles_skipped,
        'tiles_with_detections': tiles_with_detections,
        'total_detections': len(all_detections),
        'crops_saved': crops_saved,
        'shape_distribution': {
            'cocci': sum(1 for d in all_detections if d['shape_hint'] == 'cocci'),
            'bacilli': sum(1 for d in all_detections if d['shape_hint'] == 'bacilli'),
            'coccobacilli': sum(1 for d in all_detections if d['shape_hint'] == 'coccobacilli'),
        },
    }
    
    summary_path = os.path.join(output_folder, f'summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{slide_name} complete:")
    print(f"  Tiles processed: {len(tiles_to_process) - tiles_skipped}")
    print(f"  Bacteria detected: {len(all_detections)}")
    print(f"  Crops saved: {crops_saved}")
    
    # Run debris filter if enabled
    if config.get('filter_debris', False) and config.get('debris_model_path'):
        debris_model_path = config['debris_model_path']
        if os.path.exists(debris_model_path):
            print(f"\n  Running debris filter...")
            debris_model = load_debris_filter(debris_model_path)
            threshold = config.get('debris_threshold', 0.7)
            bacteria_kept, debris_deleted = filter_debris_in_folder(crops_folder, debris_model, threshold)
            print(f"  Debris filter: kept {bacteria_kept:,}, deleted {debris_deleted:,} ({100*debris_deleted/(bacteria_kept+debris_deleted+0.001):.1f}% debris)")
            crops_saved = bacteria_kept  # Update count
        else:
            print(f"  WARNING: Debris model not found: {debris_model_path}")
            print(f"  Skipping debris filter.")
    
    return all_detections


def run_all_slides(config):
    """Process all .tif slides in the slides_folder."""
    
    slides_folder = config['slides_folder']
    slide_files = glob.glob(os.path.join(slides_folder, '*.tif'))
    
    if not slide_files:
        print(f"No .tif files found in {slides_folder}")
        return
    
    # Filter slides based on skip_slides and only_slides
    skip_slides = config.get('skip_slides', [])
    only_slides = config.get('only_slides', [])
    
    filtered_slides = []
    for slide_path in slide_files:
        slide_name = os.path.basename(slide_path)
        
        # Skip if in skip list
        if slide_name in skip_slides:
            print(f"Skipping (in skip_slides): {slide_name}")
            continue
        
        # If only_slides is specified, only process those
        if only_slides and slide_name not in only_slides:
            print(f"Skipping (not in only_slides): {slide_name}")
            continue
        
        filtered_slides.append(slide_path)
    
    if not filtered_slides:
        print("No slides to process after filtering!")
        return
    
    print(f"="*60)
    print(f"BATCH PROCESSING: {len(filtered_slides)} slides")
    print(f"="*60)
    print(f"Slides folder: {slides_folder}")
    print(f"Output folder: {config['output_folder']}")
    print(f"Storage free: {check_storage_gb():.1f} GB")
    print(f"Min storage: {config.get('min_storage_gb', 20)} GB")
    if skip_slides:
        print(f"Skipping: {skip_slides}")
    if only_slides:
        print(f"Only processing: {only_slides}")
    print()
    
    total_detections = 0
    total_crops = 0
    processed = 0
    
    for i, slide_path in enumerate(filtered_slides, 1):
        slide_name = os.path.basename(slide_path)
        print(f"\n{'='*60}")
        print(f"[{i}/{len(filtered_slides)}] {slide_name}")
        print(f"{'='*60}")
        
        # Check storage before each slide
        free_gb = check_storage_gb()
        if free_gb < config.get('min_storage_gb', 20):
            print(f"Storage too low ({free_gb:.1f} GB). Stopping batch.")
            break
        
        detections = run_detection_pipeline(config, slide_path=slide_path)
        total_detections += len(detections)
        total_crops += sum(1 for d in detections if d.get('crop_path'))
        processed += 1
        
        # Delete original slide after processing to save space
        if config.get('delete_after_processing', False):
            try:
                slide_size_gb = os.path.getsize(slide_path) / (1024 ** 3)
                os.remove(slide_path)
                print(f"  DELETED {slide_name} ({slide_size_gb:.2f} GB freed)")
            except Exception as e:
                print(f"  WARNING: Could not delete {slide_name}: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Slides processed: {processed}")
    print(f"Total detections: {total_detections}")
    print(f"Total crops saved: {total_crops}")
    print(f"Storage remaining: {check_storage_gb():.1f} GB")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("BACTERIA DETECTION PIPELINE")
    print("="*60)
    
    run_all_slides(CONFIG)