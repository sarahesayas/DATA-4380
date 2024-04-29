# dataloaders.py

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil

def load_data():
    # Select classes and discard extra classes
    classes = ['pizza', 'not_pizza']

    # Create directories for train and test data
    train_dir = "train_data"
    test_dir = "test_data"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy a subset of images for each class to train and test directories
    for class_name in classes:
        class_dir = os.path.join("pizza_not_pizza", class_name)
        all_images = os.listdir(class_dir)
        selected_images = np.random.choice(all_images, size=min(100, len(all_images)), replace=False)
        train_images, test_images = train_test_split(selected_images, test_size=0.2, random_state=42)

        for image in train_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(train_dir, class_name, image)
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            shutil.copy(src, dst)

        for image in test_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(test_dir, class_name, image)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
            shutil.copy(src, dst)

    # Define data generators for train and test data
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')

    # Load test data
    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary')

    return train_generator, test_generator

def visualize_images(generator, num_images=5):
    """
    Visualizes a few sample images from the data generator.
    
    Parameters:
        generator (DirectoryIterator): Data generator for loading images.
        num_images (int): Number of images to visualize for each class. Default is 5.
    """
    class_labels = {v: k for k, v in generator.class_indices.items()}
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        for j in range(2):  # Number of classes (binary)
            plt.subplot(2, num_images, i + 1 + j * num_images)
            batch = next(generator)
            image = batch[0][0]
            label = class_labels[int(batch[1][0])]
            plt.imshow(image)
            plt.title(label)
            plt.axis('off')
    plt.show()
