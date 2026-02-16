"""
RGB-D Salient Object Detection - Inference Script
Demonstrates model inference on a single test sample
Uses the EXACT architecture from training (DAM + Gated Fusion)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ============ CUSTOM LAYERS (must match training) ============
@keras.utils.register_keras_serializable(package="RGBD_SOD")
class InvertGate(layers.Layer):
    """Computes 1.0 - x (for gated fusion)"""
    def call(self, x):
        return 1.0 - x

# ============ MODEL BUILDING BLOCKS (copied from training) ============
def conv_block(x, filters, prefix):
    """Conv-BN-ReLU-Conv-BN-ReLU block"""
    x = layers.Conv2D(filters, 3, padding='same', name=f"{prefix}_c1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.ReLU(name=f"{prefix}_r1")(x)
    x = layers.Conv2D(filters, 3, padding='same', name=f"{prefix}_c2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.ReLU(name=f"{prefix}_r2")(x)
    return x

def encoder_stage(x, filters, prefix):
    """Encoder stage with downsampling"""
    feat = conv_block(x, filters, prefix)
    down = layers.MaxPooling2D(2, name=f"{prefix}_pool")(feat)
    return down, feat

def depth_attention(rgb_feat, depth_feat, filters, prefix):
    """Depth Awareness Module (DAM) - spatial + channel attention"""
    # Spatial attention
    sp = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name=f"{prefix}_sp")(depth_feat)
    # Channel attention
    ch = layers.GlobalAveragePooling2D(name=f"{prefix}_gap")(depth_feat)
    ch = layers.Dense(filters // 4, activation='relu', name=f"{prefix}_d1")(ch)
    ch = layers.Dense(filters, activation='sigmoid', name=f"{prefix}_d2")(ch)
    ch = layers.Reshape((1, 1, filters), name=f"{prefix}_rs")(ch)
    # Apply attention
    att = layers.Multiply(name=f"{prefix}_spm")([rgb_feat, sp])
    att = layers.Multiply(name=f"{prefix}_chm")([att, ch])
    out = layers.Add(name=f"{prefix}_add")([rgb_feat, att])
    return out

def gated_fusion(rgb_feat, depth_feat, filters, prefix):
    """Complementary Gated Fusion (CGF)"""
    cat = layers.Concatenate(name=f"{prefix}_cat")([rgb_feat, depth_feat])
    gate = layers.Conv2D(filters, 1, padding='same', name=f"{prefix}_gc")(cat)
    gate = layers.BatchNormalization(name=f"{prefix}_gbn")(gate)
    gate = layers.Activation('sigmoid', name=f"{prefix}_gs")(gate)
    inv = InvertGate(name=f"{prefix}_inv")(gate)
    r = layers.Multiply(name=f"{prefix}_rm")([rgb_feat, gate])
    d = layers.Multiply(name=f"{prefix}_dm")([depth_feat, inv])
    return layers.Add(name=f"{prefix}_fuse")([r, d])

def decoder_stage(x, skip, filters, prefix):
    """Decoder stage with skip connections"""
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

    # Multi-scale depth-aware fusion: DAM + Gated Fusion
    f1 = gated_fusion(depth_attention(rs1, ds1p, 48, "da1"), ds1p, 48, "gf1")
    f2 = gated_fusion(depth_attention(rs2, ds2p, 96, "da2"), ds2p, 96, "gf2")
    f3 = gated_fusion(depth_attention(rs3, ds3p, 192, "da3"), ds3p, 192, "gf3")
    f4 = gated_fusion(depth_attention(rs4, ds4p, 384, "da4"), ds4p, 384, "gf4")

    # Bottleneck
    d4p_bot = layers.Conv2D(384, 1, padding='same', name="dp_bot")(d4)
    bot = gated_fusion(r4, d4p_bot, 384, "gf_bot")
    bot = conv_block(bot, 384, "bot")

    # Decoder with fused skip connections
    x = decoder_stage(bot, f4, 192, "dec4")
    x = decoder_stage(x, f3, 96, "dec3")
    x = decoder_stage(x, f2, 48, "dec2")
    x = decoder_stage(x, f1, 24, "dec1")

    # Output
    out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)

    return models.Model([rgb_in, dep_in], out, name='RGBD_SOD')

# Paths
MODEL_WEIGHTS_PATH = "../models/rgbd_sod.weights.h5"
RGB_PATH = "input_rgb.jpg"
DEPTH_PATH = "input_depth.bmp"
GT_PATH = "ground_truth.png"
IMG_SIZE = 224

print("="*70)
print("RGB-D SALIENT OBJECT DETECTION - INFERENCE")
print("="*70)

# Load images
print("\n[1/4] Loading images...")
rgb = np.array(Image.open(RGB_PATH).convert('RGB'), dtype=np.uint8)
original_rgb = rgb.copy()

depth = np.array(Image.open(DEPTH_PATH).convert('L'), dtype=np.uint8)
original_depth = depth.copy()

gt = np.array(Image.open(GT_PATH).convert('L'), dtype=np.uint8)

print(f"  ✓ RGB shape: {rgb.shape}")
print(f"  ✓ Depth shape: {depth.shape}")
print(f"  ✓ GT shape: {gt.shape}")

# Resize to model input size (224x224)
rgb = np.array(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)
depth = np.array(Image.fromarray(depth).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)

# Normalize
rgb_normalized = rgb.astype(np.float32) / 255.0
depth_normalized = depth.astype(np.float32) / 255.0

# Add batch dimension
rgb_batch = np.expand_dims(rgb_normalized, axis=0)
depth_batch = np.expand_dims(depth_normalized, axis=0)
depth_batch = np.expand_dims(depth_batch, axis=-1)  # RGB expects 3 channels for depth input

print(f"  ✓ Preprocessed RGB batch: {rgb_batch.shape}")
print(f"  ✓ Preprocessed Depth batch: {depth_batch.shape}")

# Load model
print("\n[2/4] Building model with correct architecture and loading weights...")
print("  • Building dual-branch encoder-decoder with DAM + Gated Fusion...")
model = build_model(img_size=IMG_SIZE)
print(f"  • Model architecture: {model.count_params():,} parameters")

print("  • Loading trained weights...")
model.load_weights(MODEL_WEIGHTS_PATH)
print(f"  ✓ Model ready for inference with correct architecture!")

# Run inference
print("\n[3/4] Running inference on test sample...")
output_saliency = model.predict([rgb_batch, depth_batch], verbose=0)
output_saliency = output_saliency[0, :, :, 0]  # Remove batch and channel dims
print(f"  ✓ Output saliency map shape: {output_saliency.shape}")
print(f"  ✓ Saliency range: [{output_saliency.min():.4f}, {output_saliency.max():.4f}]")

# Save output
print("\n[4/4] Saving results...")
output_saliency_uint8 = (output_saliency * 255).astype(np.uint8)
Image.fromarray(output_saliency_uint8).save("output_saliency.png")
print(f"  ✓ Saved: output_saliency.png")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

fig.suptitle('RGB-D Salient Object Detection - Inference Result\nInput: NJU2K Dataset (Sample 000003)', 
             fontsize=14, fontweight='bold', y=0.98)

# Row 1: Input images
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(original_rgb)
ax1.set_title('Input RGB (Original)', fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(original_depth, cmap='gray')
ax2.set_title('Input Depth (Original)', fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(np.array(Image.fromarray(original_rgb).resize((224, 224), Image.Resampling.BILINEAR)))
ax3.set_title('RGB (224×224)', fontweight='bold')
ax3.axis('off')

ax4 = fig.add_subplot(gs[0, 3])
depth_resized = np.array(Image.fromarray(original_depth).resize((224, 224), Image.Resampling.BILINEAR))
ax4.imshow(depth_resized, cmap='gray')
ax4.set_title('Depth (224×224)', fontweight='bold')
ax4.axis('off')

# Row 2: Model output and GT
ax5 = fig.add_subplot(gs[1, 0])
ax5.imshow(output_saliency, cmap='hot')
ax5.set_title('Model Output\n(Saliency Map)', fontweight='bold')
ax5.axis('off')
cbar1 = plt.colorbar(ax5.imshow(output_saliency, cmap='hot'), ax=ax5, fraction=0.046, pad=0.04)
cbar1.set_label('Saliency', fontsize=9)

ax6 = fig.add_subplot(gs[1, 1])
gt_resized = np.array(Image.fromarray(gt).resize((224, 224), Image.Resampling.BILINEAR))
ax6.imshow(gt_resized, cmap='gray')
ax6.set_title('Ground Truth', fontweight='bold')
ax6.axis('off')

ax7 = fig.add_subplot(gs[1, 2])
output_binary = (output_saliency > 0.5).astype(np.uint8) * 255
ax7.imshow(output_binary, cmap='gray')
ax7.set_title('Output (Threshold=0.5)', fontweight='bold')
ax7.axis('off')

ax8 = fig.add_subplot(gs[1, 3])
# IoU calculation
intersection = np.logical_and(output_binary, gt_resized)
union = np.logical_or(output_binary, gt_resized)
iou = np.sum(intersection) / (np.sum(union) + 1e-6)
ax8.text(0.5, 0.7, f'Inference IoU\n(vs Ground Truth)\n\n{iou:.4f}', 
         ha='center', va='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.8))
ax8.axis('off')
ax8.set_xlim(0, 1)
ax8.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('inference_visualization.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: inference_visualization.png")
plt.close()

# Summary
print("\n" + "="*70)
print("INFERENCE COMPLETE!")
print("="*70)
print("\nGenerated files in /workspaces/melomoteo/test/:")
print("  • input_rgb.jpg              - Original RGB image")
print("  • input_depth.bmp            - Original depth map")
print("  • ground_truth.png           - Ground truth saliency annotation")
print("  • output_saliency.png        - Model predicted saliency map")
print("  • inference_visualization.png - Complete visualization")
print(f"\nPerformance Metrics:")
print(f"  • Saliency range: [{output_saliency.min():.4f}, {output_saliency.max():.4f}]")
print(f"  • Binary mask IoU vs GT: {iou:.4f}")
print("="*70)
