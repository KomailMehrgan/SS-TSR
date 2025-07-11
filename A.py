import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Define alphabet and 10 words using only A, B, C ===
alphabet = ['A', 'B', 'C']
eos = 'EOS'
class_labels = ['A', 'B', 'C', eos]
class_indices = {c: i for i, c in enumerate(class_labels)}

# 10 toy ABC words (length = 4)
words = ["ACAB", "BACA", "ABAC", "CABA", "AABC", "CCBA", "ABCB", "BCAC", "CABC", "BAAC"]
num_samples = len(words)
seq_len = 4
num_classes = len(class_labels)

# === Step 2: Create HR vectors (mostly one-hot), and noisy versions for MyMethod & TPGSR ===
def make_vector(char, strength=0.9):
    """Return a softmax-like vector with most weight on the target class."""
    vec = np.ones(num_classes) * ((1 - strength) / (num_classes - 1))
    vec[class_indices[char]] = strength
    return vec

# Generate all probabilities
probs_hr = np.zeros((num_samples, seq_len, num_classes))
probs_my = np.zeros_like(probs_hr)
probs_tpgsr = np.zeros_like(probs_hr)

for i, word in enumerate(words):
    for j, char in enumerate(word):
        probs_hr[i, j] = make_vector(char, strength=0.95)       # HR: very confident
        probs_my[i, j] = make_vector(char, strength=0.85)       # MyMethod: fairly confident
        probs_tpgsr[i, j] = make_vector(char, strength=0.7)     # TPGSR: weaker

# === Step 3: KL Divergence ===
def kl_divergence(p, q):
    p = np.clip(p, 1e-6, 1)
    q = np.clip(q, 1e-6, 1)
    return np.sum(p * np.log(p / q), axis=-1)  # shape: [num_samples, seq_len]

kl_my = kl_divergence(probs_hr, probs_my)
kl_tpgsr = kl_divergence(probs_hr, probs_tpgsr)

# === Step 4: Plotting ===
def plot_kl_heatmap(kl_matrix, title):
    plt.figure(figsize=(8, 5))
    sns.heatmap(kl_matrix, cmap="magma", annot=True, fmt=".2f", cbar=True,
                xticklabels=[f"Char {i+1}" for i in range(seq_len)],
                yticklabels=[f"{i}: {words[i]}" for i in range(num_samples)])
    plt.title(title)
    plt.xlabel("Character Position")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.show()

# === Step 5: Show Heatmaps ===
plot_kl_heatmap(kl_my, "KL Divergence: MyMethod vs HR")
plot_kl_heatmap(kl_tpgsr, "KL Divergence: TPGSR vs HR")
