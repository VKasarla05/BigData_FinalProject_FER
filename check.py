# ===============================================================
# PERSON 4 – Final Evaluation + Interpretability Suite
#   - Grad-CAM
#   - Grad-CAM++
#   - LRP-style gradient saliency
#   - Occlusion sensitivity
#   - SmoothGrad + SmoothGrad-CAM
#   - SHAP (if available)
# ===============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Optional SHAP – handled safely
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ---------------------------------------------------------------
# 1. PATHS / CONSTANTS
# ---------------------------------------------------------------
BASE = "/home/sat3812/Final_project"
NPZ_PATH = f"{BASE}/Dataset/npz"
OUTPUT = f"{BASE}/Output_P4"
MODEL_PATH = f"{BASE}/Output_P3/mobilenetv2finetuned.h5"

os.makedirs(OUTPUT, exist_ok=True)
print("Saving Person4 outputs to:", OUTPUT)

IMG_SIZE = 96
NUM_CLASSES = 7

EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# ---------------------------------------------------------------
# 2. LOAD DATA (NPZ) AND PREPROCESS FOR MOBILENETV2
#    – same preprocessing as Person2/Person3
# ---------------------------------------------------------------
print("Loading NPZ datasets...")
train = np.load(f"{NPZ_PATH}/train.npz")
val   = np.load(f"{NPZ_PATH}/val.npz")
test  = np.load(f"{NPZ_PATH}/test.npz")

X_train, y_train = train["X"], train["y"]
X_val,   y_val   = val["X"],   val["y"]
X_test,  y_test  = test["X"],  test["y"]

print("Raw shapes:")
print("  Train:", X_train.shape, "| Labels:", len(y_train))
print("  Val:  ", X_val.shape,   "| Labels:", len(y_val))
print("  Test: ", X_test.shape,  "| Labels:", len(y_test))

def preprocess_for_mobilenet(X):
    X = np.repeat(X[..., np.newaxis], 3, axis=-1)   # (N,48,48) -> (N,48,48,3)
    X = tf.image.resize(X, (IMG_SIZE, IMG_SIZE)).numpy().astype("float32")
    return X

X_train = preprocess_for_mobilenet(X_train)
X_val   = preprocess_for_mobilenet(X_val)
X_test  = preprocess_for_mobilenet(X_test)

print("Preprocessed test shape:", X_test.shape)

# ---------------------------------------------------------------
# 3. LOAD FINE-TUNED MODEL
# ---------------------------------------------------------------
print("Loading Person3 fine-tuned model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

with open(os.path.join(OUTPUT, "person4_model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# ---------------------------------------------------------------
# 4. EVALUATE MODEL ON TEST SET
# ---------------------------------------------------------------
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

with open(os.path.join(OUTPUT, "person4_test_metrics.txt"), "w") as f:
    f.write(f"Test accuracy: {test_acc:.4f}\n")
    f.write(f"Test loss: {test_loss:.4f}\n")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

report = classification_report(
    y_test,
    y_pred,
    target_names=[EMOTION_MAP[i] for i in range(NUM_CLASSES)]
)
print(report)

with open(os.path.join(OUTPUT, "person4_classification_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[EMOTION_MAP[i] for i in range(NUM_CLASSES)],
            yticklabels=[EMOTION_MAP[i] for i in range(NUM_CLASSES)])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Person4")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "confusion_matrix.png"))
plt.close()

print("Classification report and confusion matrix saved.")

# ---------------------------------------------------------------
# 5. BUILD GRAD MODEL (FOR CAM METHODS)
# ---------------------------------------------------------------
# Use last convolutional feature layer before GAP
LAST_CONV_LAYER_NAME = "out_relu"   # from your model summary

last_conv_layer = model.get_layer(LAST_CONV_LAYER_NAME)
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)

# ---------------------------------------------------------------
# 6. COMMON UTILS
# ---------------------------------------------------------------
def normalize_heatmap(cam):
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., np.newaxis], (IMG_SIZE, IMG_SIZE)).numpy()
    cam = cam.reshape((IMG_SIZE, IMG_SIZE))
    return cam

def overlay_heatmap_on_image(cam, img):
    img_norm = img.astype("float32") / 255.0
    heatmap_color = plt.cm.jet(cam)[:, :, :3]  # RGB
    overlay = 0.5 * heatmap_color + 0.5 * img_norm
    return overlay

def save_overlay(cam, img, title, filename_prefix):
    cam_norm = normalize_heatmap(cam)
    overlay = overlay_heatmap_on_image(cam_norm, img)

    # Raw heatmap
    plt.figure(figsize=(4, 4))
    plt.imshow(cam_norm, cmap="jet")
    plt.title(f"{title} Heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, f"{filename_prefix}_heatmap.png"))
    plt.close()

    # Overlay
    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.title(f"{title} Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, f"{filename_prefix}_overlay.png"))
    plt.close()

# ---------------------------------------------------------------
# 7. GRAD-CAM
# ---------------------------------------------------------------
def grad_cam(img, class_idx):
    img_tensor = tf.expand_dims(img, axis=0)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)[0]          # (h,w,c)
    conv_output = conv_output[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))         # (c,)
    cam = tf.zeros(conv_output.shape[:2], dtype=tf.float32)

    for k, w in enumerate(weights):
        cam += w * conv_output[:, :, k]

    return cam.numpy()

# ---------------------------------------------------------------
# 8. GRAD-CAM++
# ---------------------------------------------------------------
def grad_cam_plus(img, class_idx):
    img_tensor = tf.expand_dims(img, axis=0)

    with tf.GradientTape(persistent=True) as tape:
        conv_output, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)           # (1,h,w,c)
    conv_output = conv_output

    first = grads
    second = grads ** 2
    third = grads ** 3

    global_sum = tf.reduce_sum(conv_output * third, axis=(1, 2), keepdims=True)
    alpha_num = second
    alpha_denom = 2.0 * second + global_sum
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))

    alpha = alpha_num / alpha_denom
    positive_grads = tf.nn.relu(first)

    weights = tf.reduce_sum(alpha * positive_grads, axis=(1, 2))    # (1,c)
    cam = tf.reduce_sum(weights[..., tf.newaxis, tf.newaxis] * conv_output, axis=-1)
    cam = cam[0]  # (h,w)

    del tape
    return cam.numpy()

# ---------------------------------------------------------------
# 9. LRP-STYLE GRADIENT SALIENCY (gradient * input)
# ---------------------------------------------------------------
def lrp_style_saliency(img, class_idx):
    img_tensor = tf.expand_dims(img, axis=0)
    img_tensor = tf.cast(img_tensor, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, img_tensor)[0]     # (96,96,3)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)  # (96,96)
    saliency = saliency.numpy()
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency

# ---------------------------------------------------------------
# 10. OCCLUSION SENSITIVITY
# ---------------------------------------------------------------
def occlusion_sensitivity(img, true_label, patch_size=8, stride=4):
    img = img.astype("float32")
    h, w, _ = img.shape

    base_prob = model.predict(img[None, ...], verbose=0)[0, true_label]

    heatmap = np.zeros((h, w), dtype="float32")

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = img.copy()
            occluded[y:y+patch_size, x:x+patch_size, :] = 0.0

            prob = model.predict(occluded[None, ...], verbose=0)[0, true_label]
            diff = base_prob - prob
            heatmap[y:y+patch_size, x:x+patch_size] = diff

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap

# ---------------------------------------------------------------
# 11. SMOOTHGRAD + SMOOTHGRAD-CAM
# ---------------------------------------------------------------
def smoothgrad_saliency(img, class_idx, n_samples=20, sigma=0.1):
    img = img.astype("float32")
    h, w, c = img.shape
    accum = np.zeros((h, w), dtype="float32")

    for _ in range(n_samples):
        noise = np.random.normal(0, sigma * 255.0, size=img.shape).astype("float32")
        noisy_img = np.clip(img + noise, 0, 255)

        img_tensor = tf.expand_dims(noisy_img, axis=0)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            preds = model(img_tensor)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, img_tensor)[0]
        grads = tf.reduce_max(tf.abs(grads), axis=-1).numpy()
        accum += grads

    saliency = accum / n_samples
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency

def smoothgrad_cam(img, class_idx, n_samples=15, sigma=0.1):
    img = img.astype("float32")
    h, w, _ = img.shape
    accum = np.zeros((h, w), dtype="float32")

    for _ in range(n_samples):
        noise = np.random.normal(0, sigma * 255.0, size=img.shape).astype("float32")
        noisy_img = np.clip(img + noise, 0, 255)
        cam = grad_cam(noisy_img, class_idx)
        accum += cam

    cam_mean = accum / n_samples
    return cam_mean

# ---------------------------------------------------------------
# 12. SHAP FOR CNN (IF AVAILABLE)
# ---------------------------------------------------------------
def run_shap_analysis():
    if not HAS_SHAP:
        print("SHAP not installed; skipping SHAP analysis.")
        return

    print("Running SHAP DeepExplainer on a small subset...")
    background = X_test[:50]
    test_samples = X_test[50:60]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)

    np.save(os.path.join(OUTPUT, "shap_values.npy"), shap_values)
    print("SHAP values saved to shap_values.npy")

# ---------------------------------------------------------------
# 13. RUN INTERPRETABILITY METHODS ON SELECTED SAMPLES
# ---------------------------------------------------------------
# choose one sample per class (first occurrence)
indices_per_class = []
for c in range(NUM_CLASSES):
    idxs = np.where(y_test == c)[0]
    if len(idxs) > 0:
        indices_per_class.append(idxs[0])

print("Selected indices per class for interpretability:", indices_per_class)

for idx in indices_per_class:
    img = X_test[idx]
    label = y_test[idx]
    class_name = EMOTION_MAP[label]
    base_name = f"class_{label}_idx_{idx}"

    # Grad-CAM
    cam = grad_cam(img, label)
    save_overlay(cam, img, f"Grad-CAM {class_name}", f"gradcam_{base_name}")

    # Grad-CAM++
    cam_pp = grad_cam_plus(img, label)
    save_overlay(cam_pp, img, f"Grad-CAM++ {class_name}", f"gradcampp_{base_name}")

    # LRP-style saliency
    sal = lrp_style_saliency(img, label)
    plt.figure(figsize=(4, 4))
    plt.imshow(sal, cmap="hot")
    plt.title(f"LRP-style Saliency {class_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, f"lrp_style_{base_name}.png"))
    plt.close()

    # SmoothGrad saliency
    sg_sal = smoothgrad_saliency(img, label, n_samples=20, sigma=0.1)
    plt.figure(figsize=(4, 4))
    plt.imshow(sg_sal, cmap="hot")
    plt.title(f"SmoothGrad Saliency {class_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, f"smoothgrad_{base_name}.png"))
    plt.close()

    # SmoothGrad-CAM
    sg_cam = smoothgrad_cam(img, label, n_samples=15, sigma=0.1)
    save_overlay(sg_cam, img, f"SmoothGrad-CAM {class_name}", f"smoothgrad_cam_{base_name}")

# Occlusion sensitivity on one representative correctly classified sample
correct_indices = np.where(y_test == y_pred)[0]
if len(correct_indices) > 0:
    occl_idx = int(correct_indices[0])
    img = X_test[occl_idx]
    label = y_test[occl_idx]
    class_name = EMOTION_MAP[label]

    occ_heat = occlusion_sensitivity(img, label, patch_size=8, stride=4)
    plt.figure(figsize=(4, 4))
    plt.imshow(occ_heat, cmap="jet")
    plt.title(f"Occlusion Sensitivity {class_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, f"occlusion_idx_{occl_idx}.png"))
    plt.close()

    overlay_occ = overlay_heatmap_on_image(occ_heat, img)
    plt.figure(figsize=(4, 4))
    plt.imshow(overlay_occ)
    plt.title(f"Occlusion Overlay {class_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, f"occlusion_overlay_idx_{occl_idx}.png"))
    plt.close()

# Run SHAP (if available)
run_shap_analysis()

print("Person4 processing completed. All metrics and interpretability outputs saved.")
