from load_smallnorb import load_smallnorb
import numpy as np
import tensorflow as tf
# import show_methods
import matplotlib.pyplot as plt
import os
import pickle, gzip
import os
import show_methods

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the SmallNorb dataset
(train_images, train_labels), (test_images, test_labels) = load_smallnorb()

category_labels = ['animal', 'human', 'airplane', 'truck', 'car']
num_classes = 5

# Get the class labels
train_labels = train_labels[:, 2]
test_labels = test_labels[:, 2]

# Data pre-processing - normalisation
train_images = train_images / 255
test_images = test_images / 255

# Create model load from keras
model = tf.keras.Sequential(
    [
        # First convolutional layer
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu",
                               input_shape=(96, 96, 2)),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Second convolutional layer
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Third convolutional layer
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Fully connected classification
        # Pass in single vector
        tf.keras.layers.Flatten(),

        # Hidden layer
        tf.keras.layers.Dense(128, activation="relu"),

        # Add Dropout regularization
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ]
)

# Print summary of model before running
model.summary()

# Complie the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Show model architecture
tf.keras.utils.plot_model(model, to_file="my_model.png", show_shapes=True)

# Gather history
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
