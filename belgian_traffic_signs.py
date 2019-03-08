import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


ROOT_PATH = "./"
train_data_directory = os.path.join(ROOT_PATH, "BelgiumTSC_Training/Training")
test_data_directory = os.path.join(ROOT_PATH, "BelgiumTSC_Testing/Testing")

images, labels = load_data(train_data_directory)
images = np.asarray(images)
labels = np.asarray(labels)

test_images, test_labels = load_data(train_data_directory)
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)

print("""Images array dimensions: %i
Images array size: %i
Images array memory size: %i MB
Labels array dimensions: %i
Labels array size: %i
Labels distinct count: %i"""
% (images.ndim, images.size, images.nbytes / 1024, labels.ndim, labels.size, len(set(labels))))

# Rescale the images in the `images` array
images = [skimage.transform.resize(image, (28, 28)) for image in images]

def plot_images(images, grayscale = False):
    #### plot sample images, labels and how many are there of each
    # Get the unique labels
    unique_labels = set(labels)

    # Initialize the figure
    plt.figure(figsize=(15, 15))

    # Set a counter
    i = 1

    # For each unique label,
    for label in unique_labels:
        # You pick the first image for each label
        image = images[list(labels).index(label)]
        # Define 64 subplots
        plt.subplot(8, 8, i)
        # Don't include axes
        plt.axis('off')
        # Add a title to each subplot
        plt.title("Label {0} ({1})".format(label, list(labels).count(label)))
        # Add 1 to the counter
        i += 1
        # And you plot this first image
        if grayscale:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)

    # Show the plot
    plt.show()

plot_images(images)

# Rescale the images in the `images` array
images28 = [skimage.transform.resize(image, (28, 28)) for image in images]
plot_images(images28)

# Convert `images28` to an array
images28 = np.array(images28)
# Convert `images28` to grayscale
images28 = skimage.color.rgb2gray(images28)
plot_images(images28, True)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(1000, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.softmax)
])

print("\nCompiling model...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nTraining model...")
model.fit(images28, labels, epochs=10)

# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28 for i in sample_indexes]
sample_labels = [labels for i in sample_indexes]

print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(images28, labels)

print('\nTest accuracy:', test_acc)
