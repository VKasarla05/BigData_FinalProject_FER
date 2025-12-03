# ===============================================================
# Final Evaluation + Interpretability 
# ===============================================================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------
BASE = "/home/sat3812/Final_project"
NPZ_PATH = f"{BASE}/Dataset/npz"
OUTPUT = f"{BASE}/Output4"
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
# LOAD NPZ FILES
# ---------------------------------------------------------------
train = np.load(f"{NPZ_PATH}/train.npz")
val   = np.load(f"{NPZ_PATH}/val.npz")
test  = np.load(f"{NPZ_PATH}/test.npz")
X_test, y_test = test["X"], test["y"]
# ---------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------
def preprocess(x):
    x = np.repeat(x[..., np.newaxis], 3, axis=-1)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE)).numpy().astype("float32")
    return x
X_test = preprocess(X_test)
print("Final test shape:", X_test.shape)
# ---------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------
print("Loading Person3 fine-tuned model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")
# Save model summary
with open(os.path.join(OUTPUT, "person4_model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
# ---------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

with open(os.path.join(OUTPUT, "person4_test_metrics.txt"), "w") as f:
    f.write(f"Test accuracy: {test_acc}\n")
    f.write(f"Test loss: {test_loss}\n")

y_pred = np.argmax(model.predict(X_test), axis=1)
report = classification_report(y_test, y_pred)

with open(os.path.join(OUTPUT, "person4_classification_report.txt"), "w") as f:
    f.write(report)

print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix – Person4")
plt.savefig(os.path.join(OUTPUT, "confusion_matrix.png"))
plt.close()

# ---------------------------------------------------------------
# SELECT CORRECT LAST CONV LAYER
# ---------------------------------------------------------------
LAST_CONV_LAYER_NAME = "Conv_1"   
last_conv_layer = model.get_layer(LAST_CONV_LAYER_NAME)

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)

# ---------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------
def normalize(cam):
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., np.newaxis], (IMG_SIZE, IMG_SIZE)).numpy()
    return cam.reshape((IMG_SIZE, IMG_SIZE))

def overlay(cam, img):
    img_norm = img.astype("float32") / 255.0
    cam_color = plt.cm.jet(cam)[:, :, :3]
    return 0.5 * cam_color + 0.5 * img_norm

def save(cam, img, title, name):
    cam_norm = normalize(cam)
    heat = overlay(cam_norm, img)

    plt.figure(figsize=(4, 4))
    plt.imshow(cam_norm, cmap="jet")
    plt.axis("off")
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT, f"{name}_heatmap.png"))
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.imshow(heat)
    plt.axis("off")
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT, f"{name}_overlay.png"))
    plt.close()

# ---------------------------------------------------------------
# GRAD-CAM
# ---------------------------------------------------------------
def grad_cam(img, class_idx):
    img_tensor = tf.expand_dims(img, 0)

    with tf.GradientTape() as tape:
        conv, preds = grad_model(img_tensor)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv)[0]
    conv = conv[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.zeros(conv.shape[:2])

    for i, w in enumerate(weights):
        cam += w * conv[:, :, i]

    return cam.numpy()

# ---------------------------------------------------------------
# GRAD-CAM++ 
# ---------------------------------------------------------------
def grad_cam_plus(img, class_idx):
    img_tensor = tf.expand_dims(img, 0)

    with tf.GradientTape(persistent=True) as tape:
        conv, preds = grad_model(img_tensor)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv)
    conv = conv[0]
    grads = grads[0]

    first = grads
    second = grads ** 2
    third = grads ** 3

    global_sum = tf.reduce_sum(conv * third, axis=(0, 1))

    alpha_num = second
    alpha_den = 2.0 * second + global_sum

    alpha = alpha_num / (alpha_den + 1e-10)
    weights = tf.reduce_sum(alpha * tf.nn.relu(first), axis=(0, 1))

    cam = tf.reduce_sum(weights * conv, axis=-1)
    return cam.numpy()

# ---------------------------------------------------------------
# RUNNING ON ONE SAMPLE PER CLASS
# ---------------------------------------------------------------
indices = []
for c in range(NUM_CLASSES):
    idxs = np.where(y_test == c)[0]
    if len(idxs):
        indices.append(idxs[0])

print("Selected indices:", indices)

for idx in indices:
    img = X_test[idx]
    label = y_test[idx]
    name = f"class_{label}_{idx}"

    cam = grad_cam(img, label)
    save(cam, img, f"Grad-CAM {EMOTION_MAP[label]}", f"gradcam_{name}")

    campp = grad_cam_plus(img, label)
    save(campp, img, f"Grad-CAM++ {EMOTION_MAP[label]}", f"gradcampp_{name}")

print("Person4 done.")
