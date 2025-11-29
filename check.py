import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ===============================================================
# PATHS
# ===============================================================
BASE = "/home/sat3812/Final_project"
NPZ = f"{BASE}/Dataset/npz"
OUTPUT = f"{BASE}/Output_4"
MODEL_PATH = f"{BASE}/Output_P3/mobilenetv2finetuned.h5"

os.makedirs(OUTPUT, exist_ok=True)
print("Saving Person4 outputs to:", OUTPUT)

# ===============================================================
# LOAD NPZ FILES
# ===============================================================
test_npz = np.load(f"{NPZ}/test.npz")
X_test = test_npz["X"]
y_test = test_npz["y"]

IMG_SIZE = 96

def preprocess(x):
    x = np.repeat(x[..., np.newaxis], 3, axis=-1)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy()
    return x

X_test = preprocess(X_test)
print("Final test shape:", X_test.shape)

# ===============================================================
# LOAD PERSON3 MODEL
# ===============================================================
print("Loading Person3 fine-tuned model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.\n")

# ===============================================================
# Evaluate
# ===============================================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

# ===============================================================
# Predict
# ===============================================================
preds = model.predict(X_test)
y_pred = preds.argmax(axis=1)

# Save classification report
report = classification_report(y_test, y_pred)
with open(f"{OUTPUT}/classification_report.txt", "w") as f:
    f.write(report)

print(report)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Person4")
plt.tight_layout()
plt.savefig(f"{OUTPUT}/confusion_matrix.png")
plt.close()

# ===============================================================
# GRAD-CAM
# Correct last conv layer: "out_relu"
# ===============================================================
LAST_CONV = "out_relu"
last_conv_layer = model.get_layer(LAST_CONV)
grad_model = tf.keras.models.Model(
    [model.inputs],
    [last_conv_layer.output, model.output]
)

def generate_gradcam(img, true_label, idx):
    img_tensor = tf.expand_dims(img, axis=0)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        class_channel = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)

    plt.figure(figsize=(4,4))
    plt.imshow(heatmap, cmap="jet")
    plt.title(f"Grad-CAM class {np.argmax(preds)}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/gradcam_class_{idx}.png")
    plt.close()


# Generate Grad-CAM for first 7 classes
print("Generating Grad-CAM heatmaps...")
for i in range(7):
    sample_idx = np.where(y_test == i)[0][0]
    generate_gradcam(X_test[sample_idx], y_test[sample_idx], i)

print("Grad-CAM saved.")

# ===============================================================
# SAVE 10 MISCLASSIFIED SAMPLES
# ===============================================================
wrong = np.where(y_pred != y_test)[0]
wrong = wrong[:10]

for n, idx in enumerate(wrong):
    plt.imshow(X_test[idx].astype("uint8"))
    plt.title(f"True={y_test[idx]}  Pred={y_pred[idx]}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/misclassified_{n}.png")
    plt.close()

print("Misclassified samples saved.")
print("Person4 processing completed.")
