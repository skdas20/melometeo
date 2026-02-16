"""
RGB-D Salient Object Detection — Fresh Training Script
=======================================================
ARCHITECTURE CREDITS:
  - DS-Net (IEEE TIP 2022): Depth Awareness Module (DAM), Gated Fusion
  - SaliencyGAN (IEEE TII 2019): Multi-scale encoder-decoder structure
  - ALTERED: Removed VGG-16 backbone, removed GAN training, simplified to supervised learning

KEY CHANGES FROM PAPERS:
  ✗ DS-Net's 3-stage semi-supervised training → Direct supervised training
  ✗ DS-Net's teacher-student framework → Single model
  ✗ SaliencyGAN's adversarial training → Standard supervised
  ✗ VGG-16 pretrained backbone → Custom lightweight CNN (9.1M params)
  ✓ Kept: DAM, Gated Fusion, Multi-scale architecture

Architecture: Dual-Branch Encoder-Decoder with Depth-Aware Gated Fusion
Datasets: NLPR (700) + NJU2K (1500) = 2200 samples
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
import json
import time

# ============ CONFIG ============
# [ALTERED from DS-Net & SaliencyGAN]
# DS-Net used: 256×256 or 320×320 input size
# SaliencyGAN used: 256×256 input size
# Our choice: 224×224 (smaller for CPU efficiency)
NLPR_DIR = "/workspaces/melometeo/datasets/train_data/NLPR"
NJU2K_DIR = "/workspaces/melometeo/datasets/train_data/NJU2K"
IMG_SIZE = 224          # ALTERED: Papers use 256+, we use 224 for CPU speed
BATCH_SIZE = 4          # ALTERED: Papers use 8-16 on GPU, we use 4 on CPU
EPOCHS = 50             # SIMILAR: Both papers use 20-50 epochs
LEARNING_RATE = 1e-4    # SIMILAR: DS-Net uses 1e-5, SaliencyGAN uses 1e-4
MODEL_SAVE_PATH = "/workspaces/melometeo/models/rgbd_sod_model.keras"
WEIGHTS_SAVE_PATH = "/workspaces/melometeo/models/rgbd_sod.weights.h5"
HISTORY_PATH = "/workspaces/melometeo/training/history.json"

os.makedirs("/workspaces/melometeo/models", exist_ok=True)

# Set memory growth and threading for CPU optimization
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

print("=" * 60)
print("RGB-D SALIENT OBJECT DETECTION — FRESH TRAINING")
print("DS-Net + SaliencyGAN Inspired Architecture")
print(f"Device: CPU ({os.cpu_count()} cores)")
print("=" * 60)

# ============ DATA COLLECTION ============
# [ALTERED from DS-Net]
# DS-Net: Uses labeled + unlabeled data in 3-stage semi-supervised training
# Our approach: Only use labeled RGB-D-GT triplets (simpler, fully supervised)
def collect_triplets():
    """Collect (RGB, Depth, GT) file path triplets from both datasets.
    
    ALTERED: DS-Net uses both labeled and unlabeled RGB. We only use labeled RGB-D pairs.
    """
    triplets = []

    # NLPR: {id}.jpg, {id}_Depth.bmp, {id}_GT.png
    for rgb in sorted(glob.glob(os.path.join(NLPR_DIR, "*.jpg"))):
        base = os.path.splitext(os.path.basename(rgb))[0]
        depth = os.path.join(NLPR_DIR, base + "_Depth.bmp")
        gt = os.path.join(NLPR_DIR, base + "_GT.png")
        if os.path.exists(depth) and os.path.exists(gt):
            triplets.append((rgb, depth, gt))

    # NJU2K: {id}_left.jpg, {id}_left_Depth.bmp, {id}_left_GT.png
    for rgb in sorted(glob.glob(os.path.join(NJU2K_DIR, "*.jpg"))):
        base = os.path.splitext(os.path.basename(rgb))[0]
        depth = os.path.join(NJU2K_DIR, base + "_Depth.bmp")
        gt = os.path.join(NJU2K_DIR, base + "_GT.png")
        if os.path.exists(depth) and os.path.exists(gt):
            triplets.append((rgb, depth, gt))

    return triplets

all_triplets = collect_triplets()
train_t, test_t = train_test_split(all_triplets, test_size=0.2, random_state=42)
train_t, val_t = train_test_split(train_t, test_size=0.15, random_state=42)
print(f"\nDataset: {len(all_triplets)} total | Train: {len(train_t)} | Val: {len(val_t)} | Test: {len(test_t)}")

# ============ DATA PIPELINE ============
# [STANDARD - Similar to both papers]
# Both DS-Net and SaliencyGAN use similar data loading pipelines
def load_triplet(rgb_path, depth_path, gt_path):
    # RGB
    rgb = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_jpeg(rgb, channels=3)
    rgb = tf.image.resize(rgb, [IMG_SIZE, IMG_SIZE])
    rgb = tf.cast(rgb, tf.float32) / 255.0

    # Depth (BMP — may be 1ch or 3ch)
    depth = tf.io.read_file(depth_path)
    depth = tf.image.decode_bmp(depth)
    depth = tf.cast(depth, tf.float32)
    depth = tf.cond(
        tf.equal(tf.shape(depth)[-1], 3),
        lambda: tf.image.rgb_to_grayscale(depth),
        lambda: depth
    )
    depth = tf.image.resize(depth, [IMG_SIZE, IMG_SIZE])
    depth = depth / 255.0

    # Ground truth
    gt = tf.io.read_file(gt_path)
    gt = tf.image.decode_png(gt, channels=1)
    gt = tf.image.resize(gt, [IMG_SIZE, IMG_SIZE])
    gt = tf.cast(gt, tf.float32) / 255.0
    gt = tf.where(gt > 0.5, 1.0, 0.0)

    return (rgb, depth), gt

def augment(inputs, gt):
    """Data augmentation.
    
    [STANDARD - Both papers use similar augmentation]
    DS-Net & SaliencyGAN: Horizontal flip, random brightness/contrast
    Our approach: Same basic augmentation
    """
    rgb, depth = inputs
    if tf.random.uniform(()) > 0.5:
        rgb = tf.image.flip_left_right(rgb)
        depth = tf.image.flip_left_right(depth)
        gt = tf.image.flip_left_right(gt)
    rgb = tf.image.random_brightness(rgb, 0.1)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    return (rgb, depth), gt

def make_dataset(triplets, training=True):
    rgbs, depths, gts = zip(*triplets)
    ds = tf.data.Dataset.from_tensor_slices((list(rgbs), list(depths), list(gts)))
    ds = ds.map(load_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(400)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

print("Building data pipelines...")
train_ds = make_dataset(train_t, training=True)
val_ds = make_dataset(val_t, training=False)
test_ds = make_dataset(test_t, training=False)

# ============ CUSTOM LAYERS (serializable, no Lambda) ============
# [CUSTOM - Implementation detail for TensorFlow/Keras]
# Both papers don't specify this (they use PyTorch)
# We need this for proper Keras model serialization
@keras.utils.register_keras_serializable(package="RGBD_SOD")
class InvertGate(layers.Layer):
    """Computes 1.0 - x (replaces Lambda for serialization safety).
    
    CUSTOM: Required for Keras model saving. Papers use PyTorch (no such issue).
    """
    def call(self, x):
        return 1.0 - x

# ============ MODEL BUILDING BLOCKS ============
# [FROM: SaliencyGAN - Basic conv block structure]
# SaliencyGAN uses Conv-BN-ReLU blocks throughout encoder-decoder
# DS-Net uses similar blocks but within VGG-16 backbone
def conv_block(x, filters, prefix):
    """Basic convolutional block: Conv-BN-ReLU-Conv-BN-ReLU.
    
    FROM: SaliencyGAN (standard encoder-decoder building block)
    ALTERED: Filter counts customized (48→96→192→384 instead of VGG's 64→128→256→512)
    """
    x = layers.Conv2D(filters, 3, padding='same', name=f"{prefix}_c1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.ReLU(name=f"{prefix}_r1")(x)
    x = layers.Conv2D(filters, 3, padding='same', name=f"{prefix}_c2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.ReLU(name=f"{prefix}_r2")(x)
    return x

def encoder_stage(x, filters, prefix):
    """Encoder stage with feature extraction and downsampling.
    
    FROM: SaliencyGAN (multi-scale feature extraction with skip connections)
    SIMILAR: Both papers use progressive downsampling to extract multi-scale features
    """
    feat = conv_block(x, filters, prefix)
    down = layers.MaxPooling2D(2, name=f"{prefix}_pool")(feat)
    return down, feat

def depth_attention(rgb_feat, depth_feat, filters, prefix):
    """Depth Awareness Module (DAM) - applies depth-guided attention to RGB features.
    
    ✓ FROM: DS-Net (Core contribution - Depth Awareness Module)
    DS-Net paper Section 3.2: "Depth-Aware Module consists of spatial attention 
    and channel attention to adaptively select depth-related features."
    
    IMPLEMENTATION:
    - Spatial attention: Conv(depth) → Sigmoid → Multiply with RGB features
    - Channel attention: GAP(depth) → FC → Sigmoid → Multiply with RGB features
    - Residual connection: Add attended features back to original RGB
    
    SAME as DS-Net: Core mechanism preserved
    """
    # Spatial attention from depth
    sp = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name=f"{prefix}_sp")(depth_feat)
    # Channel attention from depth
    ch = layers.GlobalAveragePooling2D(name=f"{prefix}_gap")(depth_feat)
    ch = layers.Dense(filters // 4, activation='relu', name=f"{prefix}_d1")(ch)
    ch = layers.Dense(filters, activation='sigmoid', name=f"{prefix}_d2")(ch)
    ch = layers.Reshape((1, 1, filters), name=f"{prefix}_rs")(ch)
    # Apply
    att = layers.Multiply(name=f"{prefix}_spm")([rgb_feat, sp])
    att = layers.Multiply(name=f"{prefix}_chm")([att, ch])
    out = layers.Add(name=f"{prefix}_add")([rgb_feat, att])
    return out

def gated_fusion(rgb_feat, depth_feat, filters, prefix):
    """Complementary Gated Fusion (CGF) - learns how to balance RGB vs Depth.
    
    ✓ FROM: DS-Net (Core contribution - Section 3.3)
    DS-Net paper: "Complementary Gated Fusion adaptively integrates RGB and depth 
    features by learning a spatial gate: Output = RGB × gate + Depth × (1 - gate)"
    
    MATH: For each spatial location (i,j):
      gate(i,j) = σ(Conv([RGB; Depth]))  ← Learned from concatenated features
      output(i,j) = RGB(i,j) × gate(i,j) + Depth(i,j) × (1 - gate(i,j))
    
    WHY THIS WORKS:
    - Gate learns where RGB is more reliable (gate→1) vs where depth is better (gate→0)
    - Automatically balances modalities based on scene content
    
    SAME as DS-Net: Core gating mechanism preserved
    """
    cat = layers.Concatenate(name=f"{prefix}_cat")([rgb_feat, depth_feat])
    gate = layers.Conv2D(filters, 1, padding='same', name=f"{prefix}_gc")(cat)
    gate = layers.BatchNormalization(name=f"{prefix}_gbn")(gate)
    gate = layers.Activation('sigmoid', name=f"{prefix}_gs")(gate)
    inv = InvertGate(name=f"{prefix}_inv")(gate)  # Computes (1 - gate)
    r = layers.Multiply(name=f"{prefix}_rm")([rgb_feat, gate])
    d = layers.Multiply(name=f"{prefix}_dm")([depth_feat, inv])
    return layers.Add(name=f"{prefix}_fuse")([r, d])

def decoder_stage(x, skip, filters, prefix):
    """Decoder stage with skip connections.
    
    FROM: SaliencyGAN (U-Net style decoder with skip connections)
    SIMILAR: Both papers use progressive upsampling + skip connections
    DS-Net: Has similar decoder structure
    SaliencyGAN: Emphasizes multi-scale feature fusion in decoder
    
    Our approach: Standard U-Net decoder (same as both papers)
    """
    x = layers.UpSampling2D(2, interpolation='bilinear', name=f"{prefix}_up")(x)
    x = layers.Concatenate(name=f"{prefix}_cat")([x, skip])
    x = conv_block(x, filters, prefix)
    return x

# ============ BUILD MODEL ============
# [ARCHITECTURE: Combination of DS-Net + SaliencyGAN concepts with alterations]
def build_model():
    """Build the complete RGB-D SOD model.
    
    OVERALL STRUCTURE:
    ✓ FROM DS-Net: Dual-branch (RGB + Depth) encoder, DAM, Gated Fusion
    ✓ FROM SaliencyGAN: Multi-scale encoder-decoder with skip connections
    ✗ REMOVED from DS-Net: VGG-16 backbone, teacher-student, semi-supervised training
    ✗ REMOVED from SaliencyGAN: GAN discriminator, adversarial loss
    ✓ ALTERED: Custom lightweight encoder (48→96→192→384 vs VGG 64→128→256→512)
    """
    rgb_in = layers.Input((IMG_SIZE, IMG_SIZE, 3), name='rgb_input')
    dep_in = layers.Input((IMG_SIZE, IMG_SIZE, 1), name='depth_input')

    # RGB encoder
    # ALTERED: DS-Net uses VGG-16 (64→128→256→512), we use custom (48→96→192→384)
    # WHY: Lighter model for CPU training, still captures multi-scale features
    r1, rs1 = encoder_stage(rgb_in, 48, "re1")    # 112×112 - Stage 1
    r2, rs2 = encoder_stage(r1, 96, "re2")         # 56×56   - Stage 2
    r3, rs3 = encoder_stage(r2, 192, "re3")        # 28×28   - Stage 3
    r4, rs4 = encoder_stage(r3, 384, "re4")        # 14×14   - Stage 4

    # Depth encoder (lighter than RGB)
    # ALTERED: DS-Net depth encoder has same capacity as RGB, we use lighter (24→48→96→192)
    # WHY: Depth is less complex than RGB, can use fewer filters
    d1, ds1 = encoder_stage(dep_in, 24, "de1")   # Half of RGB filters
    d2, ds2 = encoder_stage(d1, 48, "de2")
    d3, ds3 = encoder_stage(d2, 96, "de3")
    d4, ds4 = encoder_stage(d3, 192, "de4")

    # Project depth features to match RGB channel dimensions
    # FROM DS-Net: Uses 1×1 conv to match feature dimensions before fusion
    ds1p = layers.Conv2D(48, 1, padding='same', name="dp1")(ds1)
    ds2p = layers.Conv2D(96, 1, padding='same', name="dp2")(ds2)
    ds3p = layers.Conv2D(192, 1, padding='same', name="dp3")(ds3)
    ds4p = layers.Conv2D(384, 1, padding='same', name="dp4")(ds4)

    # Multi-scale depth-aware fusion at each encoder stage
    # ✓ FROM DS-Net: Apply DAM + Gated Fusion at each scale
    # DS-Net paper Section 3.4: "Features from each scale are fused using DAM and CGF"
    # Pipeline at each scale: RGB → DAM(with depth guidance) → Gated Fusion(RGB + Depth)
    f1 = gated_fusion(depth_attention(rs1, ds1p, 48, "da1"), ds1p, 48, "gf1")  # 112×112
    f2 = gated_fusion(depth_attention(rs2, ds2p, 96, "da2"), ds2p, 96, "gf2")  # 56×56
    f3 = gated_fusion(depth_attention(rs3, ds3p, 192, "da3"), ds3p, 192, "gf3") # 28×28
    f4 = gated_fusion(depth_attention(rs4, ds4p, 384, "da4"), ds4p, 384, "gf4") # 14×14

    # Bottleneck (deepest layer)
    # SIMILAR to both papers: Bottleneck processes lowest-resolution, highest-semantic features
    d4p_bot = layers.Conv2D(384, 1, padding='same', name="dp_bot")(d4)
    bot = gated_fusion(r4, d4p_bot, 384, "gf_bot")  # Fuse RGB + Depth at bottleneck
    bot = conv_block(bot, 384, "bot")                 # Additional processing

    # Decoder - Progressive upsampling with skip connections
    # ✓ FROM SaliencyGAN: U-Net style decoder with multi-scale skip connections
    # ✓ FROM DS-Net: Skip connections use FUSED features (f1, f2, f3, f4) not raw RGB
    # KEY: We concatenate with fused features, not separate RGB/Depth (DS-Net innovation)
    x = decoder_stage(bot, f4, 192, "dec4")  # 14→28, skip from fused f4
    x = decoder_stage(x, f3, 96, "dec3")      # 28→56, skip from fused f3
    x = decoder_stage(x, f2, 48, "dec2")      # 56→112, skip from fused f2
    x = decoder_stage(x, f1, 24, "dec1")      # 112→224, skip from fused f1

    # Output head - Single-channel saliency map
    # STANDARD for both papers: 1×1 conv + sigmoid → Binary saliency map [0,1]
    out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)

    return models.Model([rgb_in, dep_in], out, name='RGBD_SOD')

# ============ LOSS & METRICS ============
# [ALTERED from both papers]
def bce_iou_loss(y_true, y_pred):
    """Combined BCE + IoU loss.
    
    DS-Net uses: Binary Cross-Entropy (BCE) only
    SaliencyGAN uses: Adversarial loss + BCE + Perceptual loss (complex)
    
    ✓ ALTERED - Our approach: BCE + IoU loss (simpler than SaliencyGAN, better than DS-Net)
    WHY IoU loss:
    - BCE: Good for pixel-wise accuracy
    - IoU: Good for region-level accuracy (directly optimizes IoU metric)
    - Combined: Best of both worlds
    """
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - inter
    iou = tf.reduce_mean((inter + 1e-7) / (union + 1e-7))
    return bce + (1.0 - iou)  # Minimize: BCE + (1-IoU) ≡ Maximize: IoU while minimizing BCE

def iou_metric(y_true, y_pred):
    """Intersection over Union metric.
    STANDARD: Both papers use IoU as evaluation metric
    """
    p = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * p, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + p, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + 1e-7) / (union + 1e-7))

def f_measure(y_true, y_pred):
    """F-measure (weighted F1 score, β²=0.3).
    STANDARD: Both papers use F-measure as evaluation metric
    Formula: F_β = ((1+β²) × Precision × Recall) / (β² × Precision + Recall)
    """
    p = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * p, axis=[1, 2, 3])
    prec = tp / (tf.reduce_sum(p, axis=[1, 2, 3]) + 1e-7)
    rec = tp / (tf.reduce_sum(y_true, axis=[1, 2, 3]) + 1e-7)
    return tf.reduce_mean((1.3 * prec * rec) / (0.3 * prec + rec + 1e-7))

def mae_metric(y_true, y_pred):
    """Mean Absolute Error.
    STANDARD: Both papers use MAE as evaluation metric
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# ============ BUILD, COMPILE, TRAIN ============
print("\nBuilding model...")
model = build_model()
total_params = model.count_params()
print(f"Total parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB)")
model.summary(print_fn=lambda x: None)  # suppress verbose summary

# Resume from saved weights if available
RESUME_EPOCH = 0
if os.path.exists(WEIGHTS_SAVE_PATH):
    print(f"Loading weights from {WEIGHTS_SAVE_PATH}...")
    model.load_weights(WEIGHTS_SAVE_PATH)
    RESUME_EPOCH = 26  # Best checkpoint was at epoch 26
    print(f"Weights loaded. Resuming from epoch {RESUME_EPOCH + 1}")

# Compile model
# ALTERED: DS-Net uses SGD, SaliencyGAN uses Adam (for both G and D)
# Our approach: Adam optimizer (simpler, works well)
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),  # FROM SaliencyGAN
    loss=bce_iou_loss,        # ALTERED: Our custom BCE+IoU loss
    metrics=['accuracy', iou_metric, f_measure, mae_metric]  # STANDARD metrics
)

# Training callbacks
# STANDARD: Similar to both papers (early stopping, LR reduction, checkpointing)
cbs = [
    callbacks.EarlyStopping(monitor='val_iou_metric', patience=10,
                            restore_best_weights=True, mode='max', verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,  # Both papers use LR decay
                                patience=5, min_lr=1e-7, verbose=1),
    callbacks.ModelCheckpoint(WEIGHTS_SAVE_PATH, monitor='val_iou_metric',
                              save_best_only=True, save_weights_only=True,
                              mode='max', verbose=1),
]

print(f"\n{'='*60}")
if RESUME_EPOCH > 0:
    print(f"RESUMING TRAINING from epoch {RESUME_EPOCH + 1} — up to {EPOCHS}")
else:
    print(f"STARTING TRAINING — {EPOCHS} epochs, batch={BATCH_SIZE}")
print(f"{'='*60}\n")

t0 = time.time()
history = model.fit(train_ds, epochs=EPOCHS, initial_epoch=RESUME_EPOCH,
                    validation_data=val_ds, callbacks=cbs, verbose=1)
elapsed = time.time() - t0
print(f"\nTraining done in {elapsed/60:.1f} min ({len(history.history['loss'])} epochs)")

# ============ EVALUATE ============
print(f"\n{'='*60}")
print("TEST SET EVALUATION")
print(f"{'='*60}")
results = model.evaluate(test_ds, verbose=1)
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

# Save full model (native keras format — no Lambda issues)
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved: {MODEL_SAVE_PATH}")
print(f"Weights saved: {WEIGHTS_SAVE_PATH}")

# Save history
h = {k: [float(v) for v in vs] for k, vs in history.history.items()}
h['metadata'] = {
    'arch': 'Dual-Branch Encoder-Decoder + DAM + Gated Fusion',
    'inspired_by': 'DS-Net (IEEE TIP 2022) + SaliencyGAN (IEEE TII 2019)',
    'datasets': 'NLPR(700) + NJU2K(1500)',
    'splits': f'train={len(train_t)}, val={len(val_t)}, test={len(test_t)}',
    'params': total_params,
    'epochs_run': len(history.history['loss']),
    'time_min': round(elapsed / 60, 1),
    'test_results': {n: round(float(v), 4) for n, v in zip(model.metrics_names, results)},
}
with open(HISTORY_PATH, 'w') as f:
    json.dump(h, f, indent=2)

print(f"\n{'='*60}")
print("COMPLETE!")
print(f"  IoU:  {results[model.metrics_names.index('iou_metric')]:.4f}")
print(f"  F-m:  {results[model.metrics_names.index('f_measure')]:.4f}")
print(f"  MAE:  {results[model.metrics_names.index('mae_metric')]:.4f}")
print(f"{'='*60}")
