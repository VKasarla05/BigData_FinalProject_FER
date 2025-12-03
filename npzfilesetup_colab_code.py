# ============================================================
# PREPROCESSING + EDA + NPZ EXPORT
# ============================================================

!pip install pillow numpy matplotlib seaborn scikit-learn --quiet

import os, zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
from google.colab import files
# ============================================================
# 1. SET SAVE DIRECTORY
# ============================================================
SAVE_DIR = "/content/FinalProject_P1_Output"
EDA_DIR  = f"{SAVE_DIR}/EDA"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)
print("Files will be saved to:", SAVE_DIR)
# ============================================================
# 2. UPLOAD ZIP AND EXTRACT
# ============================================================
uploaded = files.upload()
zip_filename = list(uploaded.keys())[0]
extract_path = "/content/dataset"
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_filename, 'r') as z:
    z.extractall(extract_path)
print("Extracted to:", extract_path)
# ============================================================
# 3. FIND TRAIN/TEST FOLDERS
# ============================================================
def find_folder(name, base="/content/dataset"):
    for root, dirs, files in os.walk(base):
        if name in dirs:
            return os.path.join(root, name)
    return None
train_path = find_folder("train")
test_path  = find_folder("test")
if not train_path or not test_path:
    raise Exception("❌ train/test folders not found!")
print("Train:", train_path)
print("Test: ", test_path)
# ============================================================
# 4. EMOTIONS + PARAMETERS
# ============================================================
EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]
LABEL_MAP = {e:i for i,e in enumerate(EMOTIONS)}
IMG_SIZE = 48
# ============================================================
# 5. LOAD IMAGES
# ============================================================
def load_images(path):
    X, y = [], []
    for emo in EMOTIONS:
        folder = os.path.join(path, emo)
        if not os.path.exists(folder):
            print("Missing folder:", folder)
            continue
        for file in os.listdir(folder):
            if file.lower().endswith(("jpg","jpeg","png")):
                fpath = os.path.join(folder, file)
                try:
                    img = Image.open(fpath).convert("L")
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                    X.append(np.asarray(img) / 255.0)
                    y.append(LABEL_MAP[emo])
                except:
                    print("Bad file:", fpath)
    return np.array(X), np.array(y)

X_train_raw, y_train_raw = load_images(train_path)
X_test, y_test = load_images(test_path)
# ============================================================
# 6. TRAIN/VAL SPLITS (70/15/15)
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(X_train_raw, y_train_raw, test_size=0.25,random_state=42, stratify=y_train_raw)
X_val, X_extra, y_val, y_extra = train_test_split(X_temp, y_temp, test_size=0.40,random_state=42, stratify=y_temp)
# ============================================================
# 7. SAVING SPLITS
# ============================================================
np.savez_compressed(f"{SAVE_DIR}/train.npz", X=X_train, y=y_train)
np.savez_compressed(f"{SAVE_DIR}/val.npz",   X=X_val,   y=y_val)
np.savez_compressed(f"{SAVE_DIR}/test.npz",  X=X_test,  y=y_test)
# ============================================================
# 8. EDA VISUALIZATIONS
# ============================================================
import matplotlib.colors as mcolors
# Pick a palette of 7 distinct colors for the 7 emotions
emotion_colors = ['#FF6B6B', '#8E44AD', '#3498DB', '#F1C40F', '#2ECC71', '#E67E22', '#1ABC9C']

def plot_class_dist(y, name):
    counts = Counter(y)
    values = [counts.get(i,0) for i in range(len(EMOTIONS))]

    plt.figure(figsize=(8,5))
    plt.bar(EMOTIONS, values, color=emotion_colors)
    plt.title(f"{name} Class Distribution", fontsize=14)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/{name}_ClassDist.png")
    plt.show()
def sample_images(X, y, name):
    plt.figure(figsize=(7,7))
    idxs = np.random.choice(len(X), 9, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(3,3,i+1)
        plt.imshow(X[idx], cmap='gray')
        plt.title(EMOTIONS[y[idx]])
        plt.axis("off")
    plt.suptitle(f"{name} Sample Images", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/{name}_Samples.png")
    plt.show()
def pixel_histogram(X, name):
    plt.figure(figsize=(8,5))
    plt.hist(X.reshape(-1), bins=50, color="#3498DB", alpha=0.8)
    plt.title(f"{name} Pixel Intensity Histogram", fontsize=14)
    plt.xlabel("Pixel Value (0–1)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/{name}_Histogram.png")
    plt.show()
def boxplot_intensity(X, name):
    flat = X.reshape(len(X), -1)
    plt.figure(figsize=(8,5))
    sns.boxplot(data=flat[:, ::500], palette="Set2")
    plt.title(f"{name} Pixel Intensity Boxplot", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{EDA_DIR}/{name}_Boxplot.png")
    plt.show()
# ============================================================
# 9. RUN FULL EDA
# ============================================================
plot_class_dist(y_train, "Training")
plot_class_dist(y_val,   "Validation")
plot_class_dist(y_test,  "Testing")
sample_images(X_train, y_train, "Training")
sample_images(X_val,   y_val,   "Validation")
pixel_histogram(X_train, "Training")
pixel_histogram(X_test,  "Testing")
boxplot_intensity(X_train, "Training")
# ============================================================
# 10. SUMMARY
# ============================================================
print("\n================ FINAL SUMMARY ================")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)
print("EDA saved to:", EDA_DIR)
print("==============================================")
