# ==============================================================
## Transfer Learning Model
# ==============================================================
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pyspark.sql import SparkSession

def main():
    # ==============================================================
    # 1. START SPARK SESSION
    # ==============================================================
    spark = SparkSession.builder \
        .appName("TransferLearning") \
        .master("spark://192.168.13.134:7077") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    print("Spark session started")
    print("Spark master:", spark.sparkContext.master)
    print("Spark workers:", spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos().length)

    # ==============================================================
    # 2. OUTPUT DIRECTORY
    # ==============================================================
    output_dir = "/home/sat3812/Final_project/Output_2"
    os.makedirs(output_dir, exist_ok=True)
    print("Saving Person2 outputs to:", output_dir)

    # ==============================================================
    # 3. LOAD NPZ FILES
    # ==============================================================
    train = np.load("/home/sat3812/Final_project/Dataset/npz/train.npz")
    val   = np.load("/home/sat3812/Final_project/Dataset/npz/val.npz")
    test  = np.load("/home/sat3812/Final_project/Dataset/npz/test.npz")

    X_train, y_train = train["X"], train["y"]
    X_val, y_val     = val["X"],   val["y"]
    X_test, y_test   = test["X"],  test["y"]

    print("Loaded NPZ:")
    print("Train:", X_train.shape, "Labels:", len(y_train))
    print("Val:  ", X_val.shape,   "Labels:", len(y_val))
    print("Test: ", X_test.shape,  "Labels:", len(y_test))

    # ==============================================================
    # 4. PREPROCESSING
    # ==============================================================
    IMG_SIZE = 96
    def prep_images(X):
        X = np.repeat(X[..., np.newaxis], 3, axis=-1)
        X = tf.image.resize(X, (IMG_SIZE, IMG_SIZE)).numpy()
        return X
    print("Preprocessing...")
    X_train = prep_images(X_train)
    X_val   = prep_images(X_val)
    X_test  = prep_images(X_test)
    print("Final image shape:", X_train.shape)

    # ==============================================================
    # 5. BUILD TRANSFER LEARNING MODEL
    # ==============================================================
    print("Building MobileNetV2 model...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(7, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    # Save model summary
    with open(os.path.join(output_dir, "mobilenet_model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # ==============================================================
    # 6. TRAIN MODEL
    # ==============================================================
    print("Training MobileNetV2...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=12,
        batch_size=64,
        verbose=2
    )
    # ==============================================================
    # 7. SAVE TRAINING PLOTS
    # ==============================================================
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("MobileNetV2 Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.close()
    
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("MobileNetV2 Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    # ==============================================================
    # 8. TEST EVALUATION
    # ==============================================================
    print("Evaluating on test set...")
    y_pred = model.predict(X_test, verbose=2).argmax(axis=1)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("MobileNetV2 Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Save model
    model.save(os.path.join(output_dir, "mobilenetv2P2.h5"))
    print("Saved Person2 model.")

    # ==============================================================
    # 9. STOP SPARK SESSION
    # ==============================================================
    spark.stop()
    print("Spark session stopped")
    print("Person2 completed successfully.")


if __name__ == "__main__":
    # Disable GPU to prevent CUDA errors
    try:
        tf.config.set_visible_devices([], "GPU")
    except:
        pass

    main()
