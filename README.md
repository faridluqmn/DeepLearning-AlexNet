🧠 AlexNet Food Classification – Deep Learning Project

Project ini merupakan implementasi arsitektur Convolutional Neural Network (CNN) AlexNet untuk tugas Deep Learning, dengan studi kasus Klasifikasi Makanan Padang.
Aplikasi dilengkapi dengan train–test–validation split otomatis, augmentasi data, evaluasi lengkap, dan UI interaktif menggunakan Streamlit.

📂 Fitur Utama
- Arsitektur CNN AlexNet (konfigurasi asli diadaptasi ke TensorFlow/Keras).
- Split dataset otomatis:
  80% train → dari train di-split lagi 20% jadi validation
  20% test
- Augmentasi data:
  shear, zoom, rotation, horizontal flip
- Evaluasi lengkap:
  Accuracy
  Precision
  Recall
  F1-Score
  ROC/AUC
  Computation Time
- Menyimpan model terbaik otomatis (ModelCheckpoint)
- Early Stopping untuk mencegah overfitting
- Pilihan load model terbaru atau pilih manual
- Prediksi gambar tunggal
- UI berbasis Streamlit

📂 Struktur Dataset
Dataset harus berada dalam folder bernama `dataset/`

🚀 Cara Menjalankan

1. Clone / Download Repo
2. Install Dependency
Jika tidak punya requirements.txt, kamu bisa install manual:
pip install streamlit tensorflow pandas numpy scikit-learn pillow matplotlib
3. Jalankan Aplikasi
streamlit run app.py

🛠 Teknologi yang Digunakan
Python
TensorFlow / Keras
Streamlit
Pandas
NumPy
scikit-learn
Matplotlib
Pillow
