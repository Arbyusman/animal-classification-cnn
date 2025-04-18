# -*- coding: utf-8 -*-
"""notebook.ipynb

# Proyek Klasifikasi Gambar: Animal (cats,dogs,elephants,pandas,zebra)

## Import Semua Packages/Library yang Digunakan
"""

# Standard library
import os
import shutil
import zipfile
import random
from pathlib import Path
from glob import glob

# External library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Google Colab specific
from google.colab import drive, files

# Downloading utilities
import gdown

# Machine learning libraries
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import InputLayer, Conv2D, SeparableConv2D, MaxPooling2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

# Scikit-learn
from sklearn.model_selection import train_test_split

# TensorFlow.js
!pip install tensorflowjs
import tensorflowjs as tfjs

#Requirements File Generation
!pip3 freeze > requirements.txt

"""## Data Preparation

### Data Loading
"""

zip_url = 'https://drive.google.com/uc?id=10bE32dY4WGUZ4sq02qL_tj8CXgdoJHTc&export=download'
zip_path = './dataset.zip'

dataset_dir = "./dataset"

if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)
os.makedirs(dataset_dir, exist_ok=True)

print("Mengunduh dataset dari Google Drive...")
gdown.download(zip_url, zip_path, quiet=False)

print("Mengekstrak dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./")

os.remove(zip_path)
print("Dataset telah diekstrak ke:", dataset_dir)

classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
print(f"Class Names: {classes}")

"""### Show Plot"""

def get_all_images(directory):
    image_extensions = ['.jpg', '.jpeg', '.png']
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                images.append(os.path.join(root, file))
    return images

data = {}
for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        all_images = get_all_images(class_dir)
        data[class_name] = all_images

labels = []
for class_name, images in data.items():
    labels.extend([class_name] * len(images))
    print(f"Gambar {class_name} memiliki {len(images)} gambar.")

total_images = sum(len(images) for images in data.values())
print(f"Total {total_images} gambar.")

# plot distribusi gambar
plt.figure(figsize=(6, 6))
sns.set_style("darkgrid")
plot_data = sns.countplot(x=labels, hue=labels, palette="pastel", dodge=False, legend=False)
plot_data.set_title("Distribusi Gambar per Kelas")
plot_data.set_xlabel("Kelas")
plot_data.set_ylabel("Jumlah Gambar")
plt.xticks(rotation=45)
plt.show()

"""### Data Preprocessing

#### Split Dataset
"""

data_dir = os.path.join('./animals')
os.makedirs(dataset_dir, exist_ok=True)

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'validation')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    print(f"Memproses direktori: {class_dir}")
    if os.path.isdir(class_dir):
        images = get_all_images(class_dir)

        if len(images) == 0:
            print(f"Direktori {class_name} kosong, dilewati.")
            continue

        # Split dataset menjadi train, validation, dan test
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
        val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        for img in train_images:
            src_path = img
            dest_path = os.path.join(train_dir, class_name, os.path.basename(img))
            shutil.move(src_path, dest_path)

        for img in val_images:
            src_path = img
            dest_path = os.path.join(val_dir, class_name, os.path.basename(img))
            shutil.move(src_path, dest_path)

        for img in test_images:
            src_path = img
            dest_path = os.path.join(test_dir, class_name, os.path.basename(img))
            shutil.move(src_path, dest_path)

shutil.rmtree(dataset_dir)
print("Pemisahan data selesai.")

"""## Modelling"""

model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(512, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(512, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),

    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),

    Dense(256, activation='relu'),
    BatchNormalization(),

    Dense(5, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(train_dir)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

class accuracyAchieved(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nAkurasi telah mencapai >95%! Menghentikan pelatihan.")
            self.model.stop_training = True

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
    ModelCheckpoint("./model/best_model/best_model.keras",monitor="val_accuracy",save_best_only=True, mode="max",verbose=1),
    accuracyAchieved(),
]

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=55,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks,
)

test_loss, test_acc = model.evaluate(test_generator)

print("Training Accuracy:", history.history['accuracy'][-1])
print("Testing Accuracy:", test_acc)

"""## Evaluasi dan Visualisasi"""

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()


plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='lower right')
plt.show()

"""## Konversi Model"""

saved_model = "model/saved_model"
tflite = "model/tflite"
tfj = "model/tfjs_model"

os.makedirs(saved_model, exist_ok=True)
os.makedirs(tflite, exist_ok=True)
os.makedirs(tfj, exist_ok=True)

#SavedModel
tf.saved_model.save(model, saved_model)

# TF-Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(tflite, "model.tflite"), 'wb') as f:
    f.write(tflite_model)


# TFJS
tfjs.converters.save_keras_model(model, tfj)

"""## Inference (Optional)"""

def preprocess_image(image_path):
    img = Image.open(image_path).resize((150, 150))
    img = img.convert('RGB')
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img
def predict_image(interpreter, image_path):
    img = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = tf.nn.softmax(output_data).numpy()[0]
    predicted_class_index = np.argmax(probabilities)
    predicted_class = classes[predicted_class_index]
    return predicted_class, probabilities

interpreter = tf.lite.Interpreter(model_path="./model/tflite/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_paths = [
    './animals/test/cats/00144-200124454.png',
    './animals/test/dogs/00715-3846168877.png',
    './animals/test/elephants/Elephant_160.jpg',
    './animals/test/pandas/66c17390-493e-412f-a441-dd9d134917db.jpg',
    './animals/test/zebra/Zebra_103.jpg',
]

for image_path in image_paths:
    predicted_class, probabilities = predict_image(interpreter, image_path)

    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.show()

    print(f"Image: {os.path.basename(image_path)}")
    print("Probabilities for each class:")
    for class_name, prob in zip(classes, probabilities):
        print(f"{class_name}: {prob:.4f}")
    print("\n" + "=" * 50 + "\n")