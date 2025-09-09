import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Update this dictionary with the paths to your metric files
files = {
    'Weight = 0 (Baseline)': R'D:\Researches\SR\SS-TSR\experiments\tsrn_2025-09-08_15-06-46\weight_0\metrics_weight_0.csv',
    'Weight = 0.0001': R'D:\Researches\SR\SS-TSR\experiments\tsrn_2025-09-08_15-06-46\weight_0.001\metrics_weight_001.csv',
    'Weight = 0.001': R'D:\Researches\SR\SS-TSR\experiments\tsrn_2025-09-08_15-06-46\weight_0.0001\metrics_weight_0001.csv'
}

# --- Plotting ---
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Validation OCR Loss (The Primary Metric)
for label, path in files.items():
    df = pd.read_csv(path)
    sns.lineplot(ax=ax1, x='epoch', y='val_ocr_loss', data=df, label=label, marker='o', markersize=4, alpha=0.8)

ax1.set_title('Validation OCR Loss (Lower is Better)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend()
ax1.grid(True, which='both', linestyle='--')


# Plot 2: Validation Image Loss (The Trade-off)
for label, path in files.items():
    df = pd.read_csv(path)
    sns.lineplot(ax=ax2, x='epoch', y='val_img_loss', data=df, label=label, marker='o', markersize=4, alpha=0.8)

ax2.set_title('Validation Image Loss (Lower is Better)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend()
ax2.grid(True, which='both', linestyle='--')

fig.suptitle('Ablation Study: The Effect of OCR Loss Weight', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
