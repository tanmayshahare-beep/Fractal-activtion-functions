import tensorflow as tf
from tensorflow.keras import layers, models
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


# ----------------------------
# 1. Custom Maxout layer
# ----------------------------
class MaxoutLayer(layers.Layer):
    def __init__(self, num_units, k=2, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units  # number of output neurons
        self.k = k                   # pieces per neuron

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Weight shape: (input_dim, num_units, k)
        # Use He initialization for better gradient flow
        self.w = self.add_weight(
            shape=(input_dim, self.num_units, self.k),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            name='w'
        )
        self.b = self.add_weight(
            shape=(self.num_units, self.k),
            initializer='zeros',
            trainable=True,
            name='b'
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, input_dim)
        # Compute all linear combinations: (batch, num_units, k)
        z = tf.tensordot(inputs, self.w, axes=[[1], [0]]) + self.b
        # Max over k: (batch, num_units)
        return tf.reduce_max(z, axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_units)


# ----------------------------
# 2. Build the model
# ----------------------------
def create_maxout_model(k=4):
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, use_bias=False),          # no bias here (bias in Maxout)
        layers.BatchNormalization(),                 # batch norm for stability
        MaxoutLayer(num_units=128, k=k),            # branching activation
        layers.Dense(10, activation='softmax')
    ])
    return model


# ----------------------------
# 3. Load MNIST from local dataset
# ----------------------------
dataset_dir = r'.\MNIST dataset'
(x_train, y_train), (x_test, y_test) = load_mnist_local(dataset_dir)

print(f"\n=== TensorFlow Maxout Model (k=4) ===")
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


model = create_maxout_model(k=4)

# Use Adam with lower learning rate for stability
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ----------------------------
# 4. Train & evaluate
# ----------------------------
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.1,
                    verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\nTest accuracy: {test_acc:.4f}')
