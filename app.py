# =============================================
# 🧠 Deep Learning Project: CNN (AlexNet)
# - Split otomatis: train 80%, test 20%, validasi 20% dari train
# - Augmentasi hanya di training
# - Evaluasi: akurasi, presisi, recall, f1, ROC/AUC, computation time
# - UI pakai Streamlit
# - ✅ Tambahan: Pilih model manual + tampilkan tabel evaluasi
# =============================================

import os, time, glob, io, datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, accuracy_score
)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

st.set_page_config(page_title="AlexNet Classifier", layout="wide")

# ======================================================
# AlexNet Architecture
# ======================================================
def alexnet_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ======================================================
# Helper
# ======================================================
def list_images_with_labels(root_dir):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    rows = []
    for c in classes:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            for path in glob.glob(os.path.join(root_dir, c, ext)):
                rows.append((path, c))
    return pd.DataFrame(rows, columns=["filepath", "label"]), classes

def compute_metrics(y_true, y_pred, y_prob, classes):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    lb = LabelBinarizer()
    y_true_bin = lb.fit(classes).transform(y_true)
    try:
        auc_macro = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc_macro = np.nan
    return acc, prec, rec, f1, auc_macro

def get_model_list():
    os.makedirs("models", exist_ok=True)
    files = [f for f in os.listdir("models") if f.endswith(".h5")]
    files.sort(reverse=True)
    return files

def get_latest_model_path():
    model_dir = "models"
    files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".h5")]
    if not files:
        return None
    return max(files, key=os.path.getctime)

# ======================================================
# Sidebar
# ======================================================
st.sidebar.title("⚙️ Konfigurasi")
data_dir = st.sidebar.text_input("Folder dataset", value="data")
img_size = st.sidebar.number_input("Ukuran gambar", 100, 512, 227)
batch_size = st.sidebar.number_input("Batch size", 4, 128, 32)
epochs = st.sidebar.number_input("Epochs", 1, 200, 50)
learning_rate = st.sidebar.number_input("Learning rate", 1e-5, 1e-1, 0.0005, format="%.5f")
augment = st.sidebar.checkbox("Gunakan Augmentasi (flip, zoom, shear, rotasi 20°)", value=True)
model_files = get_model_list()
selected_model = st.sidebar.selectbox("📦 Pilih model (.h5)", ["(terbaru otomatis)"] + model_files)

st.title("🍛 Deep Learning: CNN AlexNet – Klasifikasi Makanan Padang")

if "train_time" not in st.session_state:
    st.session_state["train_time"] = 0.0

# ======================================================
# Load Data
# ======================================================
if not os.path.isdir(data_dir):
    st.warning("Pastikan folder dataset benar dan berisi subfolder per kelas.")
else:
    df, classes = list_images_with_labels(data_dir)
    if df.empty:
        st.error("Tidak ada gambar ditemukan!")
    else:
        st.success(f"Ditemukan {len(df)} gambar dari {len(classes)} kelas: {', '.join(classes)}")
        df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        df_train, df_val = train_test_split(df_train, test_size=0.2, stratify=df_train['label'], random_state=42)
        st.info(f"Split → Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

        gen_args = dict(rescale=1./255)
        if augment:
            gen_args.update(dict(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True))
        train_gen = ImageDataGenerator(**gen_args)
        val_gen = ImageDataGenerator(rescale=1./255)
        test_gen = ImageDataGenerator(rescale=1./255)

        train_data = train_gen.flow_from_dataframe(df_train, x_col='filepath', y_col='label',
            target_size=(img_size, img_size), class_mode='categorical', batch_size=batch_size, shuffle=True, seed=42)
        val_data = val_gen.flow_from_dataframe(df_val, x_col='filepath', y_col='label',
            target_size=(img_size, img_size), class_mode='categorical', batch_size=batch_size, shuffle=False)
        test_data = test_gen.flow_from_dataframe(df_test, x_col='filepath', y_col='label',
            target_size=(img_size, img_size), class_mode='categorical', batch_size=batch_size, shuffle=False)

        # ======================================================
        # Training
        # ======================================================
        num_classes = len(classes)
        input_shape = (img_size, img_size, 3)
        model = alexnet_model(input_shape, num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="categorical_crossentropy", metrics=["accuracy"])

        if st.button("🚀 Mulai Training Model"):
            os.makedirs("models", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            best_model_path = f"models/best_model_{timestamp}.h5"
            cb = [
                ModelCheckpoint(best_model_path, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1),
                EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1)
            ]
            st.write("Training sedang berjalan...")
            t0 = time.time()
            hist = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=cb)
            t1 = time.time()
            st.session_state["train_time"] = round(t1 - t0, 2)
            st.success(f"Training selesai! Model terbaik: `{best_model_path}`")

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(hist.history["accuracy"], label="Train Acc")
            ax[0].plot(hist.history["val_accuracy"], label="Val Acc")
            ax[0].set_title("Accuracy"); ax[0].legend()
            ax[1].plot(hist.history["loss"], label="Train Loss")
            ax[1].plot(hist.history["val_loss"], label="Val Loss")
            ax[1].set_title("Loss"); ax[1].legend()
            st.pyplot(fig, clear_figure=True)

        # ======================================================
        # Evaluasi + Tabel Hasil
        # ======================================================
        if st.button("📊 Evaluasi Model di Data Testing"):
            model_path = f"models/{selected_model}" if selected_model != "(terbaru otomatis)" else get_latest_model_path()
            if not model_path:
                st.error("Model belum ditemukan.")
            else:
                st.write(f"Menggunakan model: `{os.path.basename(model_path)}`")
                start_eval = time.time()
                model = tf.keras.models.load_model(model_path)
                y_prob = model.predict(test_data, verbose=0)
                y_pred_idx = np.argmax(y_prob, axis=1)
                y_true_idx = test_data.classes
                idx2class = {v: k for k, v in test_data.class_indices.items()}
                y_pred = [idx2class[i] for i in y_pred_idx]
                y_true = [idx2class[i] for i in y_true_idx]
                end_eval = time.time()

                acc, prec, rec, f1, auc = compute_metrics(y_true, y_pred, y_prob, classes)
                test_time = round(end_eval - start_eval, 2)
                total_time = round(st.session_state.get("train_time", 0.0) + test_time, 2)

                eval_df = pd.DataFrame([{
                    "Arsitektur CNN": "AlexNet",
                    "Akurasi": f"{acc:.4f}",
                    "Recall": f"{rec:.4f}",
                    "Presisi": f"{prec:.4f}",
                    "F1-Score": f"{f1:.4f}",
                    "ROC/AUC": f"{auc:.4f}",
                    "Computation Time (detik)": total_time
                }])
                st.subheader("📋 Hasil Evaluasi Lengkap")
                st.dataframe(eval_df, use_container_width=True)

        # ======================================================
        # Prediksi Gambar Tunggal
        # ======================================================
        st.subheader("🔍 Uji Prediksi Gambar Tunggal")
        uploaded = st.file_uploader("Upload 1 gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            model_path = f"models/{selected_model}" if selected_model != "(terbaru otomatis)" else get_latest_model_path()
            if not model_path:
                st.error("Model belum ditemukan.")
            else:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Gambar yang di-upload", width=300)
                image = image.resize((img_size, img_size))
                img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
                model = tf.keras.models.load_model(model_path)
                preds = model.predict(img_array, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                pred_class = classes[pred_idx]
                confidence = preds[pred_idx] * 100
                st.success(f"🧠 Hasil Prediksi: **{pred_class}**")
                st.metric("Kepercayaan", f"{confidence:.2f}%")
                prob_df = pd.DataFrame({"Kelas": classes, "Probabilitas": preds * 100}).sort_values("Probabilitas", ascending=False)
                st.dataframe(prob_df.style.format({"Probabilitas": "{:.2f}%"}), use_container_width=True)
