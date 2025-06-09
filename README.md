# vehicle_deeplearning
Vehicle Deep Learning: Car 🚗 or Motorcycle 🏍️ Classifier Selamat datang di repository Vehicle Deep Learning! 🎉 Proyek ini berfokus pengembangan model deep learning Convolutional Neural Network (CNN) untuk mengklasifikasikan gambar kendaraan sebagai mobil 🚗 atau motor 🏍️. Model ini dilatih menggunakan dataset yang tersimpan pada drive

#Download dan Upload dataset pada drive
https://tinyurl.com/Vechiledataset

#Buat Notepad Collab baru, copy paste semua code dibawah pada noptepad collab baru tersebut
https://colab.research.google.com/drive/1Z8ioYPY8L-Qj4MUf1SM2IwE9k9l2JIW8?usp=sharing

Struktur Kode dan Fungsinya 🛠️
1. Menghubungkan Google Drive 📂
from google.colab import drive
drive.mount('/content/drive')
Fungsi: Menghubungkan Google Drive ke Colab untuk mengakses dataset di direktori /content/drive. Anda akan diminta token otorisasi untuk mengakses Drive.

2. Instalasi Gradio 🌐
!pip install gradio
Fungsi: Menginstal library Gradio di Colab untuk membuat antarmuka web interaktif guna menguji klasifikasi gambar.

3. Impor Library 📚
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import gradio as gr
import numpy as np
from PIL import Image
import os
Fungsi:
tensorflow: Library untuk membangun dan melatih model deep learning.
ImageDataGenerator: Memproses dan menormalkan gambar (misalnya, mengubah skala piksel).
Sequential: Membuat model CNN dengan layer berurutan.
Conv2D, MaxPooling2D, Flatten, Dense, Dropout: Layer untuk ekstraksi fitur, reduksi dimensi, dan klasifikasi.
gradio: Membuat antarmuka pengguna interaktif.
numpy: Operasi array numerik.
PIL.Image: Memproses gambar.
os: Mengelola file dan direktori.

4. Menentukan Direktori Dataset 📂
train_dir = '/content/drive/MyDrive/vechile_dataset/train'
validation_dir = '/content/drive/MyDrive/vechile_dataset/validation'
test_dir = '/content/drive/MyDrive/vechile_dataset/test'
Fungsi: Menentukan lokasi dataset di Google Drive:
train_dir: Data pelatihan.
validation_dir: Data validasi.
test_dir: Data pengujian.

5. Preprocessing Gambar 🖼️
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
Fungsi: Membuat objek ImageDataGenerator untuk menormalkan nilai piksel gambar dari [0, 255] menjadi [0, 1].

6. Membuat Generator Data 📊
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
Fungsi:
Membuat generator untuk memuat data pelatihan dan validasi.
target_size=(150, 150): Mengubah ukuran gambar menjadi 150x150 piksel.
batch_size=32: Memproses 32 gambar per batch.
class_mode='binary': Klasifikasi biner (mobil 🚗 = 0, motor 🏍️ = 1).

7. Menghitung Jumlah Gambar 🔢
def count_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len([file for file in files if file.endswith('.jpg') or file.endswith('.png')])
    return count

num_train_images = count_images(train_dir)
num_validation_images = count_images(validation_dir)
num_test_images = count_images(test_dir)
print(f"Jumlah gambar di train set: {num_train_images}")
print(f"Jumlah gambar di validation set: {num_validation_images}")
print(f"Jumlah gambar di test set: {num_test_images}")
Fungsi:
Fungsi count_images: Menghitung jumlah file gambar (.jpg atau .png) di direktori.
Menampilkan jumlah gambar di set pelatihan, validasi, dan pengujian untuk verifikasi.

8. Menghitung Batch 📏
batch_size = 32
steps_per_epoch = num_train_images // batch_size
validation_steps = num_validation_images // batch_size
Fungsi: Menghitung jumlah batch per epoch untuk pelatihan (steps_per_epoch) dan validasi (validation_steps) berdasarkan jumlah gambar dan ukuran batch (32).

9. Membangun Model CNN 🧠
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
Fungsi:
Membuat model CNN berurutan:
Conv2D: Layer konvolusi (32, 64, 128 filter) untuk ekstraksi fitur gambar.
MaxPooling2D: Mengurangi dimensi gambar untuk efisiensi.
Flatten: Mengubah data 2D menjadi 1D.
Dense(512): Layer fully connected untuk klasifikasi.
Dropout(0.5): Mencegah overfitting dengan mematikan 50% neuron.
Dense(1, activation='sigmoid'): Output untuk klasifikasi biner (mobil 🚗 atau motor 🏍️).

10. Kompilasi Model ⚙️
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
Fungsi:
Mengatur model dengan:
loss='binary_crossentropy': Fungsi kerugian untuk klasifikasi biner.
optimizer='Adam': Optimizer untuk pelatihan efisien.
metrics=['accuracy']: Melacak akurasi selama pelatihan.

11. Melatih Model 🏋️
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps
)
Fungsi: Melatih model selama 10 epoch menggunakan data pelatihan dan validasi.

12. Menampilkan Riwayat Pelatihan 📈
print(history.history)
Fungsi: Menampilkan metrik pelatihan (loss dan akurasi) untuk setiap epoch.

13. Fungsi Klasifikasi Gambar 🖼️
def classify_image(img):
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return 'Car' if prediction < 0.5 else 'Motorcycle'
Fungsi:
Mengubah ukuran gambar menjadi 150x150 piksel.
Menormalkan nilai piksel ke [0, 1].
Menambahkan dimensi batch untuk prediksi.
Memprediksi kelas gambar (mobil 🚗 jika < 0.5, motor 🏍️ jika ≥ 0.5).

14. Antarmuka Gradio 🌐
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='pil'),
    outputs="text",
    title="Vehicle Classifier",
    description="Upload a vehicle image (car or motorcycle) to classify it using a CNN.",
    allow_flagging="never"
)
interface.launch()
Fungsi:
Membuat antarmuka web dengan Gradio untuk mengunggah dan mengklasifikasikan gambar.
Menampilkan hasil klasifikasi sebagai teks (‘Car’ 🚗 atau ‘Motorcycle’ 🏍️).
Menonaktifkan fitur flagging untuk antarmuka.
