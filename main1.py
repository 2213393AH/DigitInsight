import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.architecture()
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
from google.colab import drive
drive.mount('/content/drive')
!cp "/content/drive/MyDrive/data/test/6.jpeg" "/content"
import cv2
import numpy as np
image = cv2.imread('test.png',cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = 255-image
pred = model.predict(np.array([image]))
print(pred.argmax())
!pip install tensorflow==2.12.0
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Assuming your images are 128x128 pixels
IMG_WIDTH = 128
IMG_HEIGHT = 128

# Define the model architecture
model = Sequential([
  Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
  MaxPooling2D(2, 2),
  Conv2D(32, (3, 3), activation='relu'),
  MaxPooling2D(2, 2),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D(2, 2),
  Flatten(),
  Dense(512, activation='relu'),
  Dense(1, activation='sigmoid')  # Assuming binary classification (healthy vs diseased)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Assuming 'train_images' and 'train_labels' are your training data
model.fit(x_train, y_train, epochs=10)
