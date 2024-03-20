import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

def load_cifar10_data():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    return (train_images, train_labels), (test_images, test_labels)

def preprocess_image(image_data):
    # Resize the image to match the input shape (32, 32, 3)
    resized_image = cv2.resize(image_data, (32, 32))
    # Normalize the image data (assuming pixel values range from 0 to 255)
    normalized_image = resized_image.astype('float32') / 255.0
    return normalized_image

def normalize_images(images):
    return images.astype('float32') / 255.0

def display_sample_images(images, labels, class_names):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        idx = np.random.choice(np.where(labels == i)[0])
        plt.subplot(2, 5, i+1)
        plt.imshow(images[idx])
        plt.title(class_names[i])
        plt.axis('off')
    plt.show()

class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] >= 0.95:
            print("\nTraining Stopped. Accuracy achieved 95%.")
            self.model.stop_training = True

def train_model(model, train_images, train_labels, test_images, test_labels, callbacks):
    history = model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels), callbacks=callbacks)
    return history

def print_accuracy(history):
    train_accuracy = history.history['accuracy'][-1]
    test_accuracy = history.history['val_accuracy'][-1]
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

# def save_training_results(history, filepath):
#     with open(filepath, 'w') as file:
#         file.write("Epoch\tAccuracy\tError\n")
#         for epoch, acc in enumerate(history.history['accuracy']):
#             error = 1 - acc
#             file.write(f"{epoch+1}\t{acc:.4f}\t{error:.4f}\n")

def save_training_results(history, filepath):
    with open(filepath, 'w') as file:
        file.write("Epoch\tTrain_Accuracy\tTrain_Error\tVal_Accuracy\tVal_Error\n")
        for epoch in range(len(history.history['accuracy'])):
            train_acc = history.history['accuracy'][epoch]
            train_error = 1 - train_acc
            val_acc = history.history['val_accuracy'][epoch]
            val_error = 1 - val_acc
            file.write(f"{epoch+1}\t{train_acc:.4f}\t{train_error:.4f}\t{val_acc:.4f}\t{val_error:.4f}\n")

def save_model(model, filepath):
    model.save(filepath)

def load_saved_model(filepath):
    return load_model(filepath)

