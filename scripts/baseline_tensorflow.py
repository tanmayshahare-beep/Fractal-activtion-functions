import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import idx2numpy

# Configure GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'Using GPU: {gpus}')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print('No GPU available, using CPU')


def load_mnist_local(dataset_dir):
    """Load MNIST from local idx files."""
    # Load training data
    train_images = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 'train-images.idx3-ubyte')
    )
    train_labels = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 'train-labels.idx1-ubyte')
    )
    
    # Load test data
    test_images = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 't10k-images.idx3-ubyte')
    )
    test_labels = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 't10k-labels.idx1-ubyte')
    )
    
    return (train_images, train_labels), (test_images, test_labels)


# 1. Load and preprocess MNIST from local dataset
dataset_dir = r'.\MNIST dataset'
(x_train, y_train), (x_test, y_test) = load_mnist_local(dataset_dir)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

# Normalize pixel values to [0,1] and flatten
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Keep labels as integers for sparse_categorical_crossentropy
# (no need for one-hot encoding)

# 2. Build the model
model = keras.Sequential([
    layers.Dense(128, activation='sigmoid', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train
print("\n=== TensorFlow Baseline Model ===")
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.1,
                    verbose=1)

# 5. Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\nTest accuracy: {test_acc:.4f}')
