"""
RGB-D Salient Object Detection — Fresh Training Script
=======================================================
Inspired by DS-Net (IEEE TIP 2022) & SaliencyGAN (IEEE TII 2019)

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
NLPR_DIR = "/workspaces/melometeo/datasets/train_data/NLPR"
NJU2K_DIR = "/workspaces/melometeo/datasets/train_data/NJU2K"
IMG_SIZE = 224          # Slightly smaller for faster CPU training
BATCH_SIZE = 4          # Small batch for CPU + 16GB RAM
EPOCHS = 50
LEARNING_RATE = 1e-4
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
def collect_triplets():
    """Collect (RGB, Depth, GT) file path triplets from both datasets."""
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
@keras.utils.register_keras_serializable(package="RGBD_SOD")
class InvertGate(layers.Layer):
    """Computes 1.0 - x (replaces Lambda for serialization safety)."""
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
    """DS-Net inspired Depth Awareness Module: spatial + channel attention."""
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
    """DS-Net inspired Complementary Gated Fusion."""
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

# ============ BUILD MODEL ============
def build_model():
    rgb_in = layers.Input((IMG_SIZE, IMG_SIZE, 3), name='rgb_input')
    dep_in = layers.Input((IMG_SIZE, IMG_SIZE, 1), name='depth_input')

    # RGB encoder
    r1, rs1 = encoder_stage(rgb_in, 48, "re1")    # 112
    r2, rs2 = encoder_stage(r1, 96, "re2")         # 56
    r3, rs3 = encoder_stage(r2, 192, "re3")        # 28
    r4, rs4 = encoder_stage(r3, 384, "re4")        # 14

    # Depth encoder (lighter)
    d1, ds1 = encoder_stage(dep_in, 24, "de1")
    d2, ds2 = encoder_stage(d1, 48, "de2")
    d3, ds3 = encoder_stage(d2, 96, "de3")
    d4, ds4 = encoder_stage(d3, 192, "de4")

    # Project depth to match RGB channels
    ds1p = layers.Conv2D(48, 1, padding='same', name="dp1")(ds1)
    ds2p = layers.Conv2D(96, 1, padding='same', name="dp2")(ds2)
    ds3p = layers.Conv2D(192, 1, padding='same', name="dp3")(ds3)
    ds4p = layers.Conv2D(384, 1, padding='same', name="dp4")(ds4)

    # Depth-aware fusion at each skip-connection scale
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

    out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)

    return models.Model([rgb_in, dep_in], out, name='RGBD_SOD')

# ============ LOSS & METRICS ============
def bce_iou_loss(y_true, y_pred):
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - inter
    iou = tf.reduce_mean((inter + 1e-7) / (union + 1e-7))
    return bce + (1.0 - iou)

def iou_metric(y_true, y_pred):
    p = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * p, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + p, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + 1e-7) / (union + 1e-7))

def f_measure(y_true, y_pred):
    p = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * p, axis=[1, 2, 3])
    prec = tp / (tf.reduce_sum(p, axis=[1, 2, 3]) + 1e-7)
    rec = tp / (tf.reduce_sum(y_true, axis=[1, 2, 3]) + 1e-7)
    return tf.reduce_mean((1.3 * prec * rec) / (0.3 * prec + rec + 1e-7))

def mae_metric(y_true, y_pred):
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

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=bce_iou_loss,
    metrics=['accuracy', iou_metric, f_measure, mae_metric]
)

cbs = [
    callbacks.EarlyStopping(monitor='val_iou_metric', patience=10,
                            restore_best_weights=True, mode='max', verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
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
