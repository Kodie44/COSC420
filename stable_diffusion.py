#Note the following code was run from google colab

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/Colab Notebooks')

import tensorflow as tf
from load_smallnorb import load_smallnorb
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = load_smallnorb()

# Normalize the input data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape labels
train_labels = train_images[:, :, :, 0:1]
test_labels = test_images[:, :, :, :0:1]

# Define inputs
inputs = Input((96, 96, 2))

# Define the U-NET model

# Contracting path
conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)

# Expanding path
up4 = UpSampling2D(size=(2, 2))(conv3)
up4 = Conv2D(64, 3, activation='relu', padding='same')(up4)
merge4 = concatenate([conv2, up4], axis=3)

up5 = UpSampling2D(size=(2, 2))(merge4)
up5 = Conv2D(32, 3, activation='relu', padding='same')(up5)
merge5 = concatenate([conv1, up5], axis=3)

outputs = Conv2D(1, 1, activation='linear')(merge5)

model = Model(inputs=[inputs], outputs=[outputs])
model.summary()

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(optimizer='adam', loss='mse')

# Fit the model on the training data
model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_data=(test_images, test_labels))

# Predict masks on test images
pred_masks = model.predict(test_images)

# Binarize the predicted masks
pred_masks[pred_masks >= 0.5] = 1
pred_masks[pred_masks < 0.5] = 0

# Plot the sample image and the predicted segmentation mask
n_images = 4  # number of images to plot
fig, axs = plt.subplots(n_images, 2, figsize=(10, 5 * n_images))

for i in range(n_images):
    # Plot original image
    axs[i, 0].imshow(test_images[i, :, :, 0], cmap='gray')
    axs[i, 0].set_title('Sample Image {}'.format(i + 1))
    axs[i, 0].axis('off')

    # Plot predicted mask
    axs[i, 1].imshow(pred_masks[i, :, :, 0], cmap='gray')
    axs[i, 1].set_title('Predicted Mask {}'.format(i + 1))
    axs[i, 1].axis('off')

plt.show()