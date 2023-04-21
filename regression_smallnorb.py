import PyQt5
from load_smallnorb import load_smallnorb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle, gzip
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load the SmallNorb dataset
(train_images, train_labels), (test_images, test_labels) = load_smallnorb()

# Get the class labels
train_labels = train_labels[:, 2]
test_labels = test_labels[:, 2]

# Data pre-processing - normalisation
train_images = train_images / 255
test_images = test_images / 255

# Initial model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu", input_shape=(96, 96, 2)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    #tf.keras.layers.Dense(64, activation='relu', input_shape=(96, 96, 2)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Print summary of model before running
model.summary()

# Using MSE as loss function as regression problem and adam to begin with (switched later to SGD)
model.compile(optimizer='SGD',
              loss='mse',
              metrics=['mse'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), batch_size=32)


# Show diagram of network
tf.keras.utils.plot_model(model, to_file="my_model.png", show_shapes=True)

# Get the MSE values from the training history
mse = history.history['loss']

# Plot the train and validation MSE over epochs
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

