"""
Generate training visualizations and model architecture diagram
for RGB-D Salient Object Detection model
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Load training stats
with open('training_stats.json', 'r') as f:
    stats = json.load(f)

epochs = [s['epoch'] for s in stats]
train_loss = [s['loss'] for s in stats]
val_loss = [s['val_loss'] for s in stats]
train_acc = [s['accuracy'] for s in stats]
val_acc = [s['val_accuracy'] for s in stats]
train_iou = [s['iou_metric'] for s in stats]
val_iou = [s['val_iou_metric'] for s in stats]
train_f1 = [s['f_measure'] for s in stats]
val_f1 = [s['val_f_measure'] for s in stats]
train_mae = [s['mae_metric'] for s in stats]
val_mae = [s['val_mae_metric'] for s in stats]
learning_rate = [s['learning_rate'] for s in stats]

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'train': '#2E86AB', 'val': '#A23B72'}

print("Generating training visualizations...")

# ============================================================================
# 1. LOSS CURVES
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, train_loss, label='Training Loss', color=colors['train'], linewidth=2)
ax.plot(epochs, val_loss, label='Validation Loss', color=colors['val'], linewidth=2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss (BCE + IoU)', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation Loss over Epochs', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
# Add LR reduction markers
lr_changes = []
for i in range(1, len(learning_rate)):
    if learning_rate[i] < learning_rate[i-1]:
        lr_changes.append(epochs[i])
for ep in lr_changes:
    ax.axvline(x=ep, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(ep, ax.get_ylim()[1]*0.95, f'LR→{learning_rate[epochs.index(ep)]:.0e}', 
            rotation=90, verticalalignment='top', fontsize=9, color='orange')
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: loss_curve.png")
plt.close()

# ============================================================================
# 2. ACCURACY CURVES
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, train_acc, label='Training Accuracy', color=colors['train'], linewidth=2)
ax.plot(epochs, val_acc, label='Validation Accuracy', color=colors['val'], linewidth=2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation Accuracy over Epochs', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.75, 1.0])
for ep in lr_changes:
    ax.axvline(x=ep, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
plt.tight_layout()
plt.savefig('accuracy_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: accuracy_curve.png")
plt.close()

# ============================================================================
# 3. IoU CURVES
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, train_iou, label='Training IoU', color=colors['train'], linewidth=2)
ax.plot(epochs, val_iou, label='Validation IoU', color=colors['val'], linewidth=2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('IoU (Intersection over Union)', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation IoU over Epochs', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
# Mark best val IoU
best_val_iou_idx = val_iou.index(max(val_iou))
ax.scatter(epochs[best_val_iou_idx], val_iou[best_val_iou_idx], 
           color='red', s=150, zorder=5, marker='*', edgecolors='black', linewidths=1.5)
ax.annotate(f'Best: {max(val_iou):.4f}\n(Epoch {epochs[best_val_iou_idx]})', 
            xy=(epochs[best_val_iou_idx], val_iou[best_val_iou_idx]),
            xytext=(epochs[best_val_iou_idx]+3, val_iou[best_val_iou_idx]-0.05),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5))
for ep in lr_changes:
    ax.axvline(x=ep, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
plt.tight_layout()
plt.savefig('iou_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: iou_curve.png")
plt.close()

# ============================================================================
# 4. F-MEASURE CURVES
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, train_f1, label='Training F-measure', color=colors['train'], linewidth=2)
ax.plot(epochs, val_f1, label='Validation F-measure', color=colors['val'], linewidth=2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('F-measure (F1 Score)', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation F-measure over Epochs', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
# Mark best val F-measure
best_val_f1_idx = val_f1.index(max(val_f1))
ax.scatter(epochs[best_val_f1_idx], val_f1[best_val_f1_idx], 
           color='red', s=150, zorder=5, marker='*', edgecolors='black', linewidths=1.5)
ax.annotate(f'Best: {max(val_f1):.4f}\n(Epoch {epochs[best_val_f1_idx]})', 
            xy=(epochs[best_val_f1_idx], val_f1[best_val_f1_idx]),
            xytext=(epochs[best_val_f1_idx]+3, val_f1[best_val_f1_idx]-0.05),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5))
for ep in lr_changes:
    ax.axvline(x=ep, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
plt.tight_layout()
plt.savefig('fmeasure_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: fmeasure_curve.png")
plt.close()

# ============================================================================
# 5. MAE CURVES
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, train_mae, label='Training MAE', color=colors['train'], linewidth=2)
ax.plot(epochs, val_mae, label='Validation MAE', color=colors['val'], linewidth=2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation MAE over Epochs', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
# Mark best (lowest) val MAE
best_val_mae_idx = val_mae.index(min(val_mae))
ax.scatter(epochs[best_val_mae_idx], val_mae[best_val_mae_idx], 
           color='red', s=150, zorder=5, marker='*', edgecolors='black', linewidths=1.5)
ax.annotate(f'Best: {min(val_mae):.4f}\n(Epoch {epochs[best_val_mae_idx]})', 
            xy=(epochs[best_val_mae_idx], val_mae[best_val_mae_idx]),
            xytext=(epochs[best_val_mae_idx]+3, val_mae[best_val_mae_idx]+0.015),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5))
for ep in lr_changes:
    ax.axvline(x=ep, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
plt.tight_layout()
plt.savefig('mae_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mae_curve.png")
plt.close()

# ============================================================================
# 6. LEARNING RATE SCHEDULE
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, learning_rate, color='#F18F01', linewidth=3, marker='o', markersize=4)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
for ep in lr_changes:
    ax.axvline(x=ep, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(ep, ax.get_ylim()[1]*0.5, f'Reduce\n@Epoch {ep}', 
            rotation=0, verticalalignment='center', horizontalalignment='center',
            fontsize=9, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('learning_rate_schedule.png', dpi=300, bbox_inches='tight')
print("✓ Saved: learning_rate_schedule.png")
plt.close()

# ============================================================================
# 7. COMBINED METRICS OVERVIEW
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold', y=0.995)

# Loss
axes[0, 0].plot(epochs, train_loss, label='Train', color=colors['train'], linewidth=2)
axes[0, 0].plot(epochs, val_loss, label='Val', color=colors['val'], linewidth=2)
axes[0, 0].set_title('Loss (BCE + IoU)', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(epochs, train_acc, label='Train', color=colors['train'], linewidth=2)
axes[0, 1].plot(epochs, val_acc, label='Val', color=colors['val'], linewidth=2)
axes[0, 1].set_title('Accuracy', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# IoU
axes[1, 0].plot(epochs, train_iou, label='Train', color=colors['train'], linewidth=2)
axes[1, 0].plot(epochs, val_iou, label='Val', color=colors['val'], linewidth=2)
axes[1, 0].set_title('IoU (Intersection over Union)', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('IoU')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# F-measure
axes[1, 1].plot(epochs, train_f1, label='Train', color=colors['train'], linewidth=2)
axes[1, 1].plot(epochs, val_f1, label='Val', color=colors['val'], linewidth=2)
axes[1, 1].set_title('F-measure (F1 Score)', fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('F-measure')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metrics_overview.png")
plt.close()

print("\n" + "="*70)
print("TRAINING GRAPHS COMPLETE!")
print("="*70)

# ============================================================================
# 8. MODEL ARCHITECTURE DIAGRAM
# ============================================================================
print("\nGenerating model architecture diagram...")

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Title
ax.text(10, 13.5, 'RGB-D Salient Object Detection Architecture', 
        ha='center', va='top', fontsize=18, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))

# Helper function to draw boxes
def draw_box(ax, x, y, w, h, text, color='lightblue', fontsize=9):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                          boxstyle="round,pad=0.1", 
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', multialignment='center')

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->', lw=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color=color, linewidth=lw,
                           connectionstyle="arc3,rad=0", mutation_scale=20)
    ax.add_patch(arrow)

# Input layer (top)
draw_box(ax, 6, 12, 2, 0.8, 'RGB Input\n224×224×3', '#90EE90', 10)
draw_box(ax, 14, 12, 2, 0.8, 'Depth Input\n224×224×1', '#87CEEB', 10)

# RGB Encoder branch (left)
y_pos = 10.5
stages_rgb = [
    ('Conv 48\n112×112', '#98D8C8'),
    ('Conv 96\n56×56', '#7FB3D5'),
    ('Conv 192\n28×28', '#6495ED'),
    ('Conv 384\n14×14', '#4169E1')
]
for i, (text, color) in enumerate(stages_rgb):
    draw_box(ax, 3.5, y_pos - i*1.8, 2.2, 0.9, text, color, 8)
    if i == 0:
        draw_arrow(ax, 6, 11.6, 3.5, y_pos + 0.45)
    else:
        draw_arrow(ax, 3.5, y_pos + 1.35, 3.5, y_pos + 0.45)
        
rgb_encoder_bottoms = [y_pos - i*1.8 for i in range(4)]

# Depth Encoder branch (right)
y_pos = 10.5
stages_depth = [
    ('Conv 24\n112×112', '#FFB6C1'),
    ('Conv 48\n56×56', '#FF69B4'),
    ('Conv 96\n28×28', '#FF1493'),
    ('Conv 192\n14×14', '#C71585')
]
for i, (text, color) in enumerate(stages_depth):
    draw_box(ax, 16.5, y_pos - i*1.8, 2.2, 0.9, text, color, 8)
    if i == 0:
        draw_arrow(ax, 14, 11.6, 16.5, y_pos + 0.45)
    else:
        draw_arrow(ax, 16.5, y_pos + 1.35, 16.5, y_pos + 0.45)
        
depth_encoder_bottoms = [y_pos - i*1.8 for i in range(4)]

# Depth Projection layers (between encoders and fusion)
for i in range(4):
    y = rgb_encoder_bottoms[i]
    draw_box(ax, 13.5, y, 1.5, 0.6, 'Depth\nProj', '#FFE4B5', 7)
    draw_arrow(ax, 16.5 - 1.1, y, 13.5 + 0.75, y, '#FF1493')

# Depth Attention Module (DAM) boxes
for i in range(4):
    y = rgb_encoder_bottoms[i]
    draw_box(ax, 11.5, y, 1.4, 0.6, 'DAM', '#FFDAB9', 7)
    draw_arrow(ax, 13.5 - 0.75, y, 11.5 + 0.7, y)

# Gated Fusion boxes
for i in range(4):
    y = rgb_encoder_bottoms[i]
    draw_box(ax, 8.5, y, 1.8, 0.7, 'Gated\nFusion', '#FFD700', 8)
    # Arrow from RGB
    draw_arrow(ax, 3.5 + 1.1, y, 8.5 - 0.9, y, '#4169E1')
    # Arrow from DAM
    draw_arrow(ax, 11.5 - 0.7, y, 8.5 + 0.9, y, '#FFDAB9')

# Bottleneck
bottleneck_y = rgb_encoder_bottoms[3] - 1.5
draw_box(ax, 8.5, bottleneck_y, 2.5, 0.8, 'Bottleneck\nConv 192', '#FF6347', 9)
draw_arrow(ax, 8.5, rgb_encoder_bottoms[3] - 0.45, 8.5, bottleneck_y + 0.4)

# Decoder stages
decoder_stages = [
    ('UpConv 192\n28×28', '#FFA07A'),
    ('UpConv 96\n56×56', '#FFDEAD'),
    ('UpConv 48\n112×112', '#F0E68C'),
    ('UpConv 24\n224×224', '#FFFFE0')
]
y_pos = bottleneck_y - 1.2
for i, (text, color) in enumerate(decoder_stages):
    y = y_pos - i*1.5
    draw_box(ax, 8.5, y, 2.2, 0.8, text, color, 8)
    if i == 0:
        draw_arrow(ax, 8.5, bottleneck_y - 0.4, 8.5, y + 0.4)
    else:
        draw_arrow(ax, 8.5, y + 1.1, 8.5, y + 0.4)
    
    # Skip connections from Gated Fusion
    if i < 4:
        skip_y = rgb_encoder_bottoms[3-i]
        draw_arrow(ax, 8.5 - 0.9, skip_y - 0.35, 8.5 - 0.9, y + 0.2, 'green', '->', 2)

# Output head
output_y = y - 1.2
draw_box(ax, 8.5, output_y, 2.5, 0.8, 'Conv 1×1 + Sigmoid\nOutput 224×224×1', '#32CD32', 9)
draw_arrow(ax, 8.5, y - 0.4, 8.5, output_y + 0.4)

# Add legend/annotations
legend_x = 1.5
legend_y = 2
ax.text(legend_x, legend_y + 2, 'Architecture Details:', fontsize=11, fontweight='bold')
details = [
    '• Dual-Branch Encoder (RGB + Depth)',
    '• Multi-Scale Feature Extraction',
    '• Depth-Aware Module (DAM): Spatial + Channel Attention',
    '• Gated Fusion: Learned RGB-Depth Combination',
    '• Decoder: Progressive Upsampling with Skip Connections',
    '• Total Parameters: 9.1M',
    '• Loss: BCE + IoU Loss',
    '• Datasets: NLPR (700) + NJU2K (1500)'
]
for i, detail in enumerate(details):
    ax.text(legend_x, legend_y - i*0.3, detail, fontsize=8)

# Color legend
legend_x = 17
legend_y_start = 2
ax.text(legend_x, legend_y_start + 0.3, 'Component Colors:', fontsize=10, fontweight='bold')
legend_items = [
    ('RGB Encoder', '#6495ED'),
    ('Depth Encoder', '#FF1493'),
    ('Fusion Modules', '#FFD700'),
    ('Decoder', '#FFA07A'),
    ('Output', '#32CD32')
]
for i, (label, color) in enumerate(legend_items):
    box = mpatches.Rectangle((legend_x - 0.3, legend_y_start - i*0.35 - 0.1), 0.25, 0.2, 
                             facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(box)
    ax.text(legend_x + 0.05, legend_y_start - i*0.35, label, fontsize=8, va='center')

plt.tight_layout()
plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_architecture.png")
plt.close()

print("\n" + "="*70)
print("MODEL ARCHITECTURE DIAGRAM COMPLETE!")
print("="*70)

# Summary
print("\n" + "="*70)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*70)
print("\nGenerated files:")
print("  1. loss_curve.png              - Training and validation loss")
print("  2. accuracy_curve.png          - Training and validation accuracy")
print("  3. iou_curve.png              - IoU metric with best point marked")
print("  4. fmeasure_curve.png         - F-measure with best point marked")
print("  5. mae_curve.png              - MAE with best point marked")
print("  6. learning_rate_schedule.png  - LR changes over epochs")
print("  7. metrics_overview.png        - Combined 2×2 metrics dashboard")
print("  8. model_architecture.png      - Complete model architecture diagram")
print("\nBest Results:")
print(f"  • Val IoU:       {max(val_iou):.4f} @ Epoch {epochs[val_iou.index(max(val_iou))]}")
print(f"  • Val F-measure: {max(val_f1):.4f} @ Epoch {epochs[val_f1.index(max(val_f1))]}")
print(f"  • Val Accuracy:  {max(val_acc):.4f} @ Epoch {epochs[val_acc.index(max(val_acc))]}")
print(f"  • Val MAE:       {min(val_mae):.4f} @ Epoch {epochs[val_mae.index(min(val_mae))]}")
print("="*70)
