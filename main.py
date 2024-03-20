from functions import load_cifar10_data, display_sample_images, normalize_images, AccuracyCallback, train_model, save_model, load_saved_model, save_training_results, print_accuracy
from model import create_vgg19_model
from tensorflow.keras.optimizers import Adam

(train_images, train_labels), (test_images, test_labels) = load_cifar10_data()
print(f"Train images shape: {train_images.shape}, Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}, Test labels shape: {test_labels.shape}")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
display_sample_images(train_images, train_labels, class_names)

train_images = normalize_images(train_images)
test_images = normalize_images(test_images)

input_shape = train_images[0].shape
num_classes = len(class_names)

model = create_vgg19_model(input_shape, num_classes)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Define callbacks
callbacks = [AccuracyCallback()]

# Train the model
history = train_model(model, train_images, train_labels, test_images, test_labels, callbacks)

# Save the training results to a log file
save_training_results(history, 'training_results.log')

# Print the train and test accuracy
print_accuracy(history)

# Save the trained model
save_model(model, 'vgg19_model.h5')

# Load the saved model and use it for predictions
loaded_model = load_saved_model('vgg19_model.h5')

