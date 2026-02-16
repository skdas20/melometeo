"""
RGB-D Salient Object Detection - Batch Inference on NLPR Test Set
Processes multiple test images and computes overall metrics
"""

import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from PIL import Image
from tqdm import tqdm
import json
import time

# ============ CUSTOM LAYERS (must match training) ============
@keras.utils.register_keras_serializable(package="RGBD_SOD")
class InvertGate(layers.Layer):
    """Computes 1.0 - x (for gated fusion)"""
    def call(self, x):
        return 1.0 - x

# ============ MODEL BUILDING BLOCKS ============
def conv_block(x, filters, prefix):
    x = layers.Conv2D(filters, 3, padding='same', name=f"{prefix}_c1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.ReLU(name=f"{prefix}_r1")(x)
    x = layers.Conv2D(filters, 3, padding='same', name=f"{prefix}_c2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.ReLU(name=f"{prefix}_r2")(x)
    return x

def encoder_stage(x, filters, prefix):
    feat = conv_block(x, filters, prefix)
    down = layers.MaxPooling2D(2, name=f"{prefix}_pool")(feat)
    return down, feat

def depth_attention(rgb_feat, depth_feat, filters, prefix):
    sp = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name=f"{prefix}_sp")(depth_feat)
    ch = layers.GlobalAveragePooling2D(name=f"{prefix}_gap")(depth_feat)
    ch = layers.Dense(filters // 4, activation='relu', name=f"{prefix}_d1")(ch)
    ch = layers.Dense(filters, activation='sigmoid', name=f"{prefix}_d2")(ch)
    ch = layers.Reshape((1, 1, filters), name=f"{prefix}_rs")(ch)
    att = layers.Multiply(name=f"{prefix}_spm")([rgb_feat, sp])
    att = layers.Multiply(name=f"{prefix}_chm")([att, ch])
    out = layers.Add(name=f"{prefix}_add")([rgb_feat, att])
    return out

def gated_fusion(rgb_feat, depth_feat, filters, prefix):
    cat = layers.Concatenate(name=f"{prefix}_cat")([rgb_feat, depth_feat])
    gate = layers.Conv2D(filters, 1, padding='same', name=f"{prefix}_gc")(cat)
    gate = layers.BatchNormalization(name=f"{prefix}_gbn")(gate)
    gate = layers.Activation('sigmoid', name=f"{prefix}_gs")(gate)
    inv = InvertGate(name=f"{prefix}_inv")(gate)
    r = layers.Multiply(name=f"{prefix}_rm")([rgb_feat, gate])
    d = layers.Multiply(name=f"{prefix}_dm")([depth_feat, inv])
    return layers.Add(name=f"{prefix}_fuse")([r, d])

def decoder_stage(x, skip, filters, prefix):
    x = layers.UpSampling2D(2, interpolation='bilinear', name=f"{prefix}_up")(x)
    x = layers.Concatenate(name=f"{prefix}_cat")([x, skip])
    x = conv_block(x, filters, prefix)
    return x

def build_model(img_size=224):
    """Build the EXACT model architecture used in training"""
    rgb_in = layers.Input((img_size, img_size, 3), name='rgb_input')
    dep_in = layers.Input((img_size, img_size, 1), name='depth_input')

    # RGB encoder: 48→96→192→384
    r1, rs1 = encoder_stage(rgb_in, 48, "re1")
    r2, rs2 = encoder_stage(r1, 96, "re2")
    r3, rs3 = encoder_stage(r2, 192, "re3")
    r4, rs4 = encoder_stage(r3, 384, "re4")

    # Depth encoder: 24→48→96→192
    d1, ds1 = encoder_stage(dep_in, 24, "de1")
    d2, ds2 = encoder_stage(d1, 48, "de2")
    d3, ds3 = encoder_stage(d2, 96, "de3")
    d4, ds4 = encoder_stage(d3, 192, "de4")

    # Project depth to match RGB channels
    ds1p = layers.Conv2D(48, 1, padding='same', name="dp1")(ds1)
    ds2p = layers.Conv2D(96, 1, padding='same', name="dp2")(ds2)
    ds3p = layers.Conv2D(192, 1, padding='same', name="dp3")(ds3)
    ds4p = layers.Conv2D(384, 1, padding='same', name="dp4")(ds4)

    # Multi-scale depth-aware fusion
    f1 = gated_fusion(depth_attention(rs1, ds1p, 48, "da1"), ds1p, 48, "gf1")
    f2 = gated_fusion(depth_attention(rs2, ds2p, 96, "da2"), ds2p, 96, "gf2")
    f3 = gated_fusion(depth_attention(rs3, ds3p, 192, "da3"), ds3p, 192, "gf3")
    f4 = gated_fusion(depth_attention(rs4, ds4p, 384, "da4"), ds4p, 384, "gf4")

    # Bottleneck
    d4p_bot = layers.Conv2D(384, 1, padding='same', name="dp_bot")(d4)
    bot = gated_fusion(r4, d4p_bot, 384, "gf_bot")
    bot = conv_block(bot, 384, "bot")

    # Decoder
    x = decoder_stage(bot, f4, 192, "dec4")
    x = decoder_stage(x, f3, 96, "dec3")
    x = decoder_stage(x, f2, 48, "dec2")
    x = decoder_stage(x, f1, 24, "dec1")

    # Output
    out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)

    return models.Model([rgb_in, dep_in], out, name='RGBD_SOD')

# ============ METRICS COMPUTATION ============
def compute_metrics(pred_binary, gt_binary):
    """Compute IoU, F-measure, MAE"""
    # IoU
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / (union + 1e-7)
    
    # Precision, Recall, F-measure
    tp = intersection
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    precision = tp / (pred_sum + 1e-7)
    recall = tp / (gt_sum + 1e-7)
    f_measure = (1.3 * precision * recall) / (0.3 * precision + recall + 1e-7)
    
    # MAE
    mae = np.abs(pred_binary.astype(float) - gt_binary.astype(float)).mean()
    
    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f_measure': float(f_measure),
        'mae': float(mae)
    }

# ============ MAIN BATCH INFERENCE ============
def main():
    print("="*70)
    print("RGB-D SOD - BATCH INFERENCE ON NLPR TEST SET")
    print("="*70)
    
    # Configuration
    NLPR_DIR = "/workspaces/melometeo/datasets/train_data/NLPR"
    MODEL_WEIGHTS = "../models/rgbd_sod.weights.h5"
    OUTPUT_DIR = "./nlpr_test_results"
    IMG_SIZE = 224
    
    # DATASET NOTE: Official NLPR has 1000 images, but we only have 700 (incomplete dataset)
    # We'll use a portion for testing (not used in training split)
    TEST_SPLIT = 0.3  # Use 30% for testing (210 images)
    SAVE_OUTPUTS = False  # Set to False to skip saving individual images (faster)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all RGB images
    print(f"\n[1/5] Loading NLPR dataset...")
    all_rgb_files = sorted(glob.glob(os.path.join(NLPR_DIR, "*.jpg")))
    total_available = len(all_rgb_files)
    num_test = int(total_available * TEST_SPLIT)
    
    print(f"  • NLPR images available: {total_available} (Note: Official NLPR has 1000)")
    print(f"  • Test split: {TEST_SPLIT*100:.0f}% = {num_test} images")
    
    # Use last N% images as test set (simulating held-out test data)
    test_rgb_files = all_rgb_files[-num_test:]
    print(f"  • Test range: {os.path.basename(test_rgb_files[0])} to {os.path.basename(test_rgb_files[-1])}")
    
    # Build corresponding depth and GT paths
    test_triplets = []
    for rgb_path in test_rgb_files:
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        depth_path = os.path.join(NLPR_DIR, f"{base_name}_Depth.bmp")
        gt_path = os.path.join(NLPR_DIR, f"{base_name}_GT.png")
        
        if os.path.exists(depth_path) and os.path.exists(gt_path):
            test_triplets.append((rgb_path, depth_path, gt_path))
    
    print(f"  ✓ Loaded {len(test_triplets)} valid test triplets (RGB + Depth + GT)")
    
    # Load model
    print(f"\n[2/5] Building and loading model...")
    print(f"  • Building dual-branch encoder-decoder with DAM + Gated Fusion...")
    model = build_model(img_size=IMG_SIZE)
    print(f"  • Model parameters: {model.count_params():,}")
    print(f"  • Loading weights from: {MODEL_WEIGHTS}")
    model.load_weights(MODEL_WEIGHTS)
    print(f"  ✓ Model ready!")
    
    # Run batch inference
    print(f"\n[3/5] Running inference on {len(test_triplets)} test images...")
    print(f"  • Output directory: {OUTPUT_DIR}")
    print(f"  • Saving individual outputs: {SAVE_OUTPUTS}")
    
    all_metrics = []
    failed_count = 0
    start_time = time.time()
    
    for idx, (rgb_path, depth_path, gt_path) in enumerate(tqdm(test_triplets, desc="Processing")):
        try:
            # Load and preprocess
            rgb = np.array(Image.open(rgb_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR))
            depth = np.array(Image.open(depth_path).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR))
            gt = np.array(Image.open(gt_path).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR))
            
            # Normalize
            rgb_norm = rgb.astype(np.float32) / 255.0
            depth_norm = depth.astype(np.float32) / 255.0
            gt_norm = (gt > 127).astype(np.uint8)  # Binary threshold
            
            # Prepare batch
            rgb_batch = np.expand_dims(rgb_norm, axis=0)
            depth_batch = np.expand_dims(np.expand_dims(depth_norm, axis=-1), axis=0)
            
            # Inference
            pred = model.predict([rgb_batch, depth_batch], verbose=0)[0, :, :, 0]
            pred_binary = (pred > 0.5).astype(np.uint8)
            
            # Compute metrics
            metrics = compute_metrics(pred_binary, gt_norm)
            metrics['filename'] = os.path.basename(rgb_path)
            all_metrics.append(metrics)
            
            # Save output (optional)
            if SAVE_OUTPUTS:
                base_name = os.path.splitext(os.path.basename(rgb_path))[0]
                output_path = os.path.join(OUTPUT_DIR, f"{base_name}_pred.png")
                pred_uint8 = (pred * 255).astype(np.uint8)
                Image.fromarray(pred_uint8).save(output_path)
        
        except Exception as e:
            print(f"\n  ✗ Failed on {os.path.basename(rgb_path)}: {e}")
            failed_count += 1
            continue
    
    elapsed = time.time() - start_time
    print(f"\n  ✓ Inference complete!")
    print(f"  • Processed: {len(all_metrics)} images")
    print(f"  • Failed: {failed_count} images")
    print(f"  • Time: {elapsed:.1f}s ({elapsed/len(test_triplets):.2f}s per image)")
    
    # Compute statistics
    print(f"\n[4/5] Computing overall statistics...")
    
    iou_scores = [m['iou'] for m in all_metrics]
    f_scores = [m['f_measure'] for m in all_metrics]
    mae_scores = [m['mae'] for m in all_metrics]
    precision_scores = [m['precision'] for m in all_metrics]
    recall_scores = [m['recall'] for m in all_metrics]
    
    stats = {
        'dataset': 'NLPR',
        'dataset_note': f'Incomplete dataset: {total_available}/1000 images available',
        'num_test_images': len(all_metrics),
        'test_split_percentage': TEST_SPLIT * 100,
        'model_parameters': model.count_params(),
        'inference_time_per_image': elapsed / len(test_triplets),
        'metrics': {
            'iou': {
                'mean': float(np.mean(iou_scores)),
                'std': float(np.std(iou_scores)),
                'median': float(np.median(iou_scores)),
                'min': float(np.min(iou_scores)),
                'max': float(np.max(iou_scores))
            },
            'f_measure': {
                'mean': float(np.mean(f_scores)),
                'std': float(np.std(f_scores)),
                'median': float(np.median(f_scores)),
                'min': float(np.min(f_scores)),
                'max': float(np.max(f_scores))
            },
            'mae': {
                'mean': float(np.mean(mae_scores)),
                'std': float(np.std(mae_scores)),
                'median': float(np.median(mae_scores)),
                'min': float(np.min(mae_scores)),
                'max': float(np.max(mae_scores))
            },
            'precision': {
                'mean': float(np.mean(precision_scores)),
                'std': float(np.std(precision_scores))
            },
            'recall': {
                'mean': float(np.mean(recall_scores)),
                'std': float(np.std(recall_scores))
            }
        },
        'per_image_results': all_metrics
    }
    
    # Save results
    print(f"\n[5/5] Saving results...")
    results_file = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved detailed results to: {results_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH INFERENCE RESULTS - NLPR TEST SET")
    print("="*70)
    print(f"\nDataset: NLPR ({num_test}/{total_available} images, {TEST_SPLIT*100:.0f}% test split)")
    print(f"Note: Official NLPR has 1000 images, we have {total_available} (incomplete dataset)")
    print(f"Model: Dual-Branch RGB-D SOD with DAM + Gated Fusion")
    print(f"Parameters: {model.count_params():,}")
    print(f"\nTest Images Processed: {len(all_metrics)}/{len(test_triplets)}")
    print(f"Average Inference Time: {elapsed/len(test_triplets):.2f}s per image")
    
    print(f"\n{'Metric':<15} {'Mean':<12} {'Std':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
    print("-"*70)
    print(f"{'IoU':<15} {stats['metrics']['iou']['mean']:<12.4f} {stats['metrics']['iou']['std']:<12.4f} {stats['metrics']['iou']['median']:<12.4f} {stats['metrics']['iou']['min']:<12.4f} {stats['metrics']['iou']['max']:<12.4f}")
    print(f"{'F-measure':<15} {stats['metrics']['f_measure']['mean']:<12.4f} {stats['metrics']['f_measure']['std']:<12.4f} {stats['metrics']['f_measure']['median']:<12.4f} {stats['metrics']['f_measure']['min']:<12.4f} {stats['metrics']['f_measure']['max']:<12.4f}")
    print(f"{'MAE':<15} {stats['metrics']['mae']['mean']:<12.4f} {stats['metrics']['mae']['std']:<12.4f} {stats['metrics']['mae']['median']:<12.4f} {stats['metrics']['mae']['min']:<12.4f} {stats['metrics']['mae']['max']:<12.4f}")
    print(f"{'Precision':<15} {stats['metrics']['precision']['mean']:<12.4f} {stats['metrics']['precision']['std']:<12.4f}")
    print(f"{'Recall':<15} {stats['metrics']['recall']['mean']:<12.4f} {stats['metrics']['recall']['std']:<12.4f}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {OUTPUT_DIR}/")
    if SAVE_OUTPUTS:
        print(f"  • {len(all_metrics)} prediction images (*_pred.png)")
    print(f"  • test_results.json (detailed metrics)")
    print("="*70)

if __name__ == "__main__":
    main()
