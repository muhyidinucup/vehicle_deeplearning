# vehicle_deeplearning
Vehicle Deep Learning: Car 🚗 or Motorcycle 🏍️ Classifier Selamat datang di repository Vehicle Deep Learning! 🎉 Proyek ini berfokus pengembangan model deep learning Convolutional Neural Network (CNN) untuk mengklasifikasikan gambar kendaraan sebagai mobil 🚗 atau motor 🏍️. Model ini dilatih menggunakan dataset yang tersimpan pada drive

#Download dan Upload dataset pada drive
https://tinyurl.com/Vechiledataset

#Buat Notepad Collab baru, copy paste semua code dibawah pada noptepad collab baru tersebut
https://colab.research.google.com/drive/1Z8ioYPY8L-Qj4MUf1SM2IwE9k9l2JIW8?usp=sharing

#berikut code vechile_clasiffier dan fungsinya
from google.colab import drive
drive.mount('/content/drive')
Menghubungkan Google Drive ke direktori /content/drive di Colab. Ini akan meminta otorisasi (token) untuk mengakses Drive Anda.

!pip install gradio
Menginstal library gradio di lingkungan Colab menggunakan perintah shell (!). Gradio digunakan untuk membuat antarmuka web interaktif.

import tensorflow as tf
Mengimpor ImageDataGenerator untuk memproses dan augmentasi gambar secara otomatis (misalnya, rescaling, rotasi, dll.).

from tensorflow.keras.models import Sequential
Mengimpor Sequential, kelas untuk membuat model berurutan (layer demi layer) di Keras.

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
Mengimpor layer spesifik dari Keras:
Conv2D: Layer konvolusi untuk ekstraksi fitur gambar.
MaxPooling2D: Layer pooling untuk mengurangi dimensi spasial.
Flatten: Mengubah data 2D menjadi 1D.
Dense: Layer fully connected untuk klasifikasi.
Dropout: Mencegah overfitting dengan mematikan beberapa neuron secara acak.

import gradio as gr
Mengimpor gradio untuk membuat antarmuka pengguna interaktif.

import numpy as np
Mengimpor numpy untuk operasi array numerik.

from PIL import Image
Mengimpor Image dari PIL (Python Imaging Library) untuk memproses gambar.

import os
Mengimpor os untuk operasi sistem file, seperti menghitung jumlah file di direktori.

train_dir = '/content/drive/MyDrive/vechile_dataset/train'
validation_dir = '/content/drive/MyDrive/vechile_dataset/validation'
test_dir = '/content/drive/MyDrive/vechile_dataset/test'
Mendefinisikan path ke direktori dataset di Google Drive:
train_dir: Direktori data pelatihan.
validation_dir: Direktori data validasi.
test_dir: Direktori data pengujian.

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
Membuat objek ImageDataGenerator untuk data pelatihan dan validasi. Parameter rescale=1./255 menormalkan nilai piksel gambar dari [0, 255] menjadi [0, 1].

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')
Membuat generator untuk data pelatihan:

train_dir: Direktori data pelatihan.
target_size=(150, 150): Mengubah ukuran gambar menjadi 150x150 piksel.
batch_size=32: Jumlah gambar per batch untuk pelatihan.
class_mode='binary': Mode klasifikasi biner (dua kelas: mobil atau motor).

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')
Membuat generator untuk data validasi, dengan parameter serupa seperti train_generator.

def count_images(directory):
Mendefinisikan fungsi count_images untuk menghitung jumlah file gambar (.jpg atau .png) di direktori tertentu.

count = 0
Inisialisasi variabel count untuk menghitung jumlah gambar.

for root, dirs, files in os.walk(directory):
Menggunakan os.walk untuk menelusuri semua file dan subdirektori di directory.

count += len([file for file in files if file.endswith('.jpg') or file.endswith('.png')])
Menghitung file dengan ekstensi .jpg atau .png dan menambahkannya ke count.

return count
Mengembalikan total jumlah gambar.

num_train_images = count_images(train_dir)
num_validation_images = count_images(validation_dir)
num_test_images = count_images(test_dir)
Menghitung jumlah gambar di direktori pelatihan, validasi, dan pengujian menggunakan fungsi count_images.

print(f"Jumlah gambar di train set: {num_train_images}")
print(f"Jumlah gambar di validation set: {num_validation_images}")
print(f"Jumlah gambar di test set: {num_test_images}")
Mencetak jumlah gambar di setiap set untuk memverifikasi data.

batch_size = 32
Mendefinisikan ukuran batch (32) untuk digunakan dalam perhitungan berikutnya.

steps_per_epoch = num_train_images // batch_size
validation_steps = num_validation_images // batch_size
Menghitung:
steps_per_epoch: Jumlah batch per epoch untuk pelatihan (total gambar pelatihan dibagi batch size).
validation_steps: Jumlah batch untuk validasi.

model = Sequential([
Membuat model Sequential dengan layer-layer berikut:

Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
Layer konvolusi pertama: 32 filter ukuran 3x3, aktivasi ReLU, menerima input gambar 150x150 piksel dengan 3 kanal (RGB).

MaxPooling2D(2, 2),
Layer pooling pertama: Mengurangi dimensi spasial dengan faktor 2 menggunakan window 2x2.

Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
Layer konvolusi kedua (64 filter) dan pooling kedua, meningkatkan ekstraksi fitur dan mengurangi dimensi.

Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
Layer konvolusi ketiga (128 filter) dan pooling ketiga, untuk fitur yang lebih kompleks.

Flatten(),
Mengubah output 2D dari layer sebelumnya menjadi vektor 1D untuk input ke layer dense.

Dense(512, activation='relu'),
Layer fully connected dengan 512 neuron dan aktivasi ReLU.

Dropout(0.5),
Layer dropout: Mematikan 50% neuron secara acak untuk mencegah overfitting.

Dense(1, activation='sigmoid')])
Layer output: 1 neuron dengan aktivasi sigmoid untuk klasifikasi biner (0 untuk mobil, 1 untuk motor).

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
Mengompilasi model:
loss='binary_crossentropy': Fungsi kerugian untuk klasifikasi biner.
optimizer='Adam': Optimizer Adam untuk mempercepat gradient descent.
metrics=['accuracy']: Melacak akurasi selama pelatihan.

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps
)
Melatih model:
train_generator: Data pelatihan.
steps_per_epoch: Jumlah batch per epoch.
epochs=10: Melatih selama 10 epoch.
validation_data: Data validasi.
validation_steps: Jumlah batch validasi per epoch.

print(history.history)
Mencetak riwayat pelatihan (loss dan akurasi untuk pelatihan dan validasi per epoch).

def classify_image(img):
Mendefinisikan fungsi classify_image untuk memprediksi kelas gambar (mobil atau motor).

img = img.resize((150, 150))
Mengubah ukuran gambar input menjadi 150x150 piksel menggunakan PIL.

img = np.array(img) / 255.0
Mengonversi gambar ke array NumPy dan menormalkan nilai piksel ke [0, 1].

img = np.expand_dims(img, axis=0)
Menambahkan dimensi batch (axis=0) untuk membuat input kompatibel dengan model (bentuk: [1, 150, 150, 3]).

prediction = model.predict(img)
Memprediksi kelas gambar menggunakan model yang telah dilatih.

return 'Car' if prediction < 0.5 else 'Motorcycle'
Mengembalikan label 'Car' jika prediksi < 0.5, atau 'Motorcycle' jika >= 0.5 (berdasarkan sigmoid).

interface = gr.Interface(
Membuat antarmuka Gradio untuk klasifikasi gambar.

fn=classify_image,
Fungsi yang akan dipanggil untuk memproses input (yaitu classify_image).

inputs=gr.Image(type='pil'),
Input antarmuka berupa gambar dalam format PIL.

outputs="text",
Output berupa teks (label: 'Car' atau 'Motorcycle').

title="Vehicle Classifier",
    description="Upload a vehicle image (car or motorcycle) to classify it using a CNN.",
Judul dan deskripsi antarmuka Gradio.

allow_flagging="never"
Menonaktifkan fitur "flagging" di Gradio (untuk mencegah pengguna menandai output).

interface.launch()
Meluncurkan antarmuka Gradio, menghasilkan URL untuk mengakses antarmuka web interaktif.
