# ===============================================================
# PERSON 4 — Model Interpretability & Final Evaluation
# ===============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ===============================================================
# 1. SET OUTPUT DIRECTORY
# ===============================================================
OUTPUT_DIR = "/home/sat3812/Final_project/Output_4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Saving Person4 outputs to: {OUTPUT_DIR}")

# ===============================================================
# 2. LOAD NPZ FILES
# ===============================================================
npz_path = "/home/sat3812/Final_project/Dataset/npz"

train = np.load(f"{npz_path}/train.npz")
val   = np.load(f"{npz_path}/val.npz")
test  = np.load(f"{npz_path}/test.npz")

X_train, y_train = train["X"], train["y"]
X_val, y_val     = val["X"],   val["y"]
X_test, y_test   = test["X"], test["y"]

print("Loaded NPZ files")
print("Shapes:")
print("Train:", X_train.shape, "| Labels:", len(y_train))
print("Val:  ", X_val.shape,   "| Labels:", len(y_val))
print("Test: ", X_test.shape,  "| Labels:", len(y_test))

# Add channel dimension for CNN
X_test = X_test[..., np.newaxis]

# ===============================================================
# 3. LOAD PERSON-3 FINE-TUNED MODEL
# ===============================================================
model_path = "/home/sat3812/Final_project/Output_3/mobilenetv2_person3_finetuned.h5"
print(f"\nLoading Person3 model from: {model_path}")

model = tf.keras.models.load_model(model_path)
print("✔ Model loaded successfully!")

# ===============================================================
# 4. EVALUATE ON TEST SET
# ===============================================================
print("\nEvaluating on Test Set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("\nTest accuracy:", test_acc)
print("Test loss:", test_loss)

# ===============================================================
# 5. PREDICT ON TEST SET
# ===============================================================
print("\nRunning predictions on test set...")
pred_probs = model.predict(X_test, verbose=0)
preds = np.argmax(pred_probs, axis=1)

# ===============================================================
# 6. SAVE CLASSIFICATION REPORT
# ===============================================================
report = classification_report(y_test, preds)
report_path = f"{OUTPUT_DIR}/classification_report.txt"

with open(report_path, "w") as f:
    f.write(report)

print(f"Classification report saved to: {report_path}")

# ===============================================================
# 7. SAVE CONFUSION MATRIX PLOT
# ===============================================================
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

print("Confusion matrix saved.")

# ===============================================================
# 8. GRAD-CAM IMPLEMENTATION
# ===============================================================
def make_gradcam_heatmap(img, model, last_conv_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img, axis=0))
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# ===============================================================
# 9. GET LAST CONV LAYER NAME
# ===============================================================
last_conv_name = None
for layer in reversed(model.layers):
    if len(layer.output_shape) == 4:  # Conv layer
        last_conv_name = layer.name
        break

if last_conv_name is None:
    raise ValueError("❌ No 4D Conv layer found for Grad-CAM.")

print(f"\n✔ Last Conv Layer found: {last_conv_name}")

# ===============================================================
# 10. GENERATE GRAD-CAM HEATMAPS FOR ALL CLASSES
# ===============================================================
print("\nGenerating Grad-CAM heatmaps...")

classes = np.unique(y_test)

for cls in classes:
    idx = np.where(y_test == cls)[0][0]   # first sample of each class

    img = (X_test[idx] * 255).astype("uint8")   # FIXED: no .numpy()
    heatmap = make_gradcam_heatmap(img, model, last_conv_name)

    # Save heatmap
    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap="jet")
    plt.title(f"Grad-CAM — Class {cls}")
    plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}/gradcam_class_{cls}.png")
    plt.close()

print("✔ Grad-CAM heatmaps saved.")

# ===============================================================
# 11. SAVE SAMPLE MISCLASSIFIED IMAGES
# ===============================================================
misclassified = np.where(preds != y_test)[0][:10]

print("\nSaving misclassified samples...")

for i, idx in enumerate(misclassified):
    img = (X_test[idx] * 255).astype("uint8")   # FIXED

    plt.figure(figsize=(3, 3))
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"True: {y_test[idx]} | Pred: {preds[idx]}")
    plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}/misclassified_{i}.png")
    plt.close()

print("✔ Misclassified samples saved.")

# ===============================================================
# 12. DONE
# ===============================================================
print("\n🎉 PERSON 4 COMPLETE — All outputs saved successfully!")
