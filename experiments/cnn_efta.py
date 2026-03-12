"""
CNN with Exponential Fractal Tree Activation (EFTA)

This script implements a Convolutional Neural Network using EFTA as the activation function.
EFTA replaces linear branches in FTA with learnable exponential functions of the form:
    f(x) = α * (exp(β*x) - 1)  if x < 0
    f(x) = γ * x                if x >= 0

The exponential branches are combined hierarchically via max through a tree structure.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import idx2numpy
import time
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure GPU / CUDA
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'\n✓ CUDA GPU Enabled: {len(gpus)} device(s)')
        for i, gpu in enumerate(gpus):
            print(f'  GPU {i}: {tf.config.experimental.get_device_details(gpu)}')
    except RuntimeError as e:
        print(f'GPU configuration error: {e}')
else:
    print('\n✗ No GPU available, using CPU')


# ----------------------------
# Exponential Fractal Tree Activation Layer
# ----------------------------
class ExponentialFTA(layers.Layer):
    """
    Exponential Fractal Tree Activation: each leaf is a learnable exponential function
    of the form: f(x) = α * (exp(β*x) - 1) for x < 0, and f(x) = γ * x for x >= 0.
    Leaves are combined via max through a tree of given depth and branching factor.

    Args:
        num_units: number of output neurons/channels
        depth: number of levels in the tree (depth=1 is standard Maxout)
        branch_factor: number of children per internal node
    """
    def __init__(self, num_units, depth=2, branch_factor=2, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.num_leaves = branch_factor ** depth

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Parameters for each leaf: (alpha, beta, gamma) per output unit
        # Shape: (num_leaves, num_units, 3)
        # Initialize to identity-like: alpha=1, beta=1, gamma=1
        self.leaf_params = self.add_weight(
            shape=(self.num_leaves, self.num_units, 3),
            initializer=tf.keras.initializers.Constant([1.0, 1.0, 1.0]),
            trainable=True,
            name='leaf_params'
        )
        super().build(input_shape)

    def _leaf_function(self, x, alpha, beta, gamma):
        """
        Apply exponential branch to a single leaf.
        x: (batch, ..., num_units) - input to this leaf
        alpha, beta, gamma: scalars (or broadcastable to num_units)
        """
        # Split into negative and positive parts for stability
        neg_mask = tf.cast(x < 0, tf.float32)
        pos_mask = tf.cast(x >= 0, tf.float32)

        # Negative part: α * (exp(β * x) - 1)
        # Use tf.exp with clipping to avoid overflow
        neg_part = alpha * (tf.exp(tf.clip_by_value(beta * x, -10, 10)) - 1.0)

        # Positive part: γ * x
        pos_part = gamma * x

        return neg_mask * neg_part + pos_mask * pos_part

    def call(self, inputs):
        """
        Apply EFTA to inputs. Handles both 2D (batch, features) and 4D (batch, h, w, channels).
        """
        # Determine input shape and flatten spatial dimensions if needed
        if len(inputs.shape) == 4:
            # Convolutional: (batch, height, width, channels)
            batch_size = tf.shape(inputs)[0]
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            channels = inputs.shape[-1]

            # Flatten spatial dimensions to apply leaf functions per pixel
            flat_inputs = tf.reshape(inputs, [batch_size, height * width, channels])
        else:
            # Dense: (batch, features)
            flat_inputs = inputs
            batch_size = tf.shape(flat_inputs)[0]
            height = width = None

        # Compute leaf values for all leaves
        leaf_values_list = []
        for leaf_idx in range(self.num_leaves):
            alpha = self.leaf_params[leaf_idx, :, 0]  # (num_units,)
            beta = self.leaf_params[leaf_idx, :, 1]   # (num_units,)
            gamma = self.leaf_params[leaf_idx, :, 2]  # (num_units,)

            # Apply leaf function
            leaf_val = self._leaf_function(flat_inputs, alpha, beta, gamma)
            leaf_values_list.append(leaf_val)

        # Stack leaves: (num_leaves, batch, N, num_units) where N = spatial elements
        leaf_values = tf.stack(leaf_values_list, axis=0)

        # Reshape for convolutional input
        if len(inputs.shape) == 4:
            leaf_values = tf.reshape(leaf_values,
                [self.num_leaves, batch_size, height, width, self.num_units])

        # Recursive max combination through tree
        current = leaf_values
        for level in range(self.depth):
            if len(inputs.shape) == 4:
                # 4D: (num_leaves, batch, height, width, num_units)
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = tf.reshape(current,
                    [num_nodes, self.branch_factor, batch_size, height, width, self.num_units])
                current = tf.reduce_max(current, axis=1)
            else:
                # 2D: (num_leaves, batch, num_units)
                batch_size = tf.shape(current)[1]
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = tf.reshape(current,
                    [num_nodes, self.branch_factor, batch_size, self.num_units])
                current = tf.reduce_max(current, axis=1)

        # After all levels: squeeze to final shape
        if len(inputs.shape) == 4:
            return tf.reshape(current, [batch_size, height, width, self.num_units])
        else:
            return tf.squeeze(current, axis=0)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_units,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_units': self.num_units,
            'depth': self.depth,
            'branch_factor': self.branch_factor,
        })
        return config


# ----------------------------
# Fractal Tree Activation Layer (for comparison)
# ----------------------------
class FractalTreeActivation(layers.Layer):
    """
    Standard FTA with linear branches for comparison.
    """
    def __init__(self, num_units, depth=2, branch_factor=2, share_weights=False, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.share_weights = share_weights
        self.num_leaves = branch_factor ** depth

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self.share_weights:
            self.leaf_weights = []
            self.leaf_biases = []
            for d in range(self.depth):
                w = self.add_weight(
                    shape=(input_dim, self.num_units),
                    initializer=tf.keras.initializers.HeNormal(),
                    trainable=True, name=f'leaf_weights_depth_{d}')
                b = self.add_weight(
                    shape=(self.num_units,),
                    initializer='zeros',
                    trainable=True, name=f'leaf_bias_depth_{d}')
                self.leaf_weights.append(w)
                self.leaf_biases.append(b)
        else:
            self.leaf_weights = self.add_weight(
                shape=(self.num_leaves, input_dim, self.num_units),
                initializer=tf.keras.initializers.HeNormal(),
                trainable=True, name='leaf_weights')
            self.leaf_biases = self.add_weight(
                shape=(self.num_leaves, self.num_units),
                initializer='zeros',
                trainable=True, name='leaf_biases')
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs.shape) == 4:
            batch_size = tf.shape(inputs)[0]
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            flat_inputs = tf.reshape(inputs, [batch_size, height * width, -1])

            if self.share_weights:
                leaf_values = []
                for _ in range(self.num_leaves):
                    leaf_val = tf.matmul(flat_inputs, self.leaf_weights[0]) + self.leaf_biases[0]
                    leaf_values.append(leaf_val)
                leaf_values = tf.stack(leaf_values, axis=0)
            else:
                leaf_values = tf.stack([
                    tf.matmul(flat_inputs, self.leaf_weights[i]) + self.leaf_biases[i]
                    for i in range(self.num_leaves)
                ], axis=0)

            leaf_values = tf.reshape(leaf_values,
                [self.num_leaves, batch_size, height, width, self.num_units])

            current = leaf_values
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = tf.reshape(current,
                    [num_nodes, self.branch_factor, batch_size, height, width, self.num_units])
                current = tf.reduce_max(current, axis=1)

            return tf.reshape(current, [batch_size, height, width, self.num_units])
        else:
            if self.share_weights:
                leaf_values = []
                for _ in range(self.num_leaves):
                    leaf_val = tf.matmul(inputs, self.leaf_weights[0]) + self.leaf_biases[0]
                    leaf_values.append(leaf_val)
                leaf_values = tf.stack(leaf_values, axis=0)
            else:
                leaf_values = tf.stack([
                    tf.matmul(inputs, self.leaf_weights[i]) + self.leaf_biases[i]
                    for i in range(self.num_leaves)
                ], axis=0)

            current = leaf_values
            for level in range(self.depth):
                batch_size = tf.shape(current)[1]
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = tf.reshape(current,
                    [num_nodes, self.branch_factor, batch_size, self.num_units])
                current = tf.reduce_max(current, axis=1)

            return tf.squeeze(current, axis=0)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_units,)


# ----------------------------
# Maxout Layer (for comparison)
# ----------------------------
class MaxoutLayer(layers.Layer):
    def __init__(self, num_units, k=2, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.k = k

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.num_units, self.k),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True, name='w')
        self.b = self.add_weight(
            shape=(self.num_units, self.k),
            initializer='zeros',
            trainable=True, name='b')
        super().build(input_shape)

    def call(self, inputs):
        z = tf.tensordot(inputs, self.w, axes=[[1], [0]]) + self.b
        return tf.reduce_max(z, axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_units)


# ----------------------------
# Data Loading
# ----------------------------
def load_mnist_local(dataset_dir):
    """Load MNIST from local idx files."""
    train_images = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 'train-images.idx3-ubyte'))
    train_labels = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 'train-labels.idx1-ubyte'))
    test_images = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 't10k-images.idx3-ubyte'))
    test_labels = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 't10k-labels.idx1-ubyte'))
    return (train_images, train_labels), (test_images, test_labels)


def load_fashion_mnist_local(dataset_dir):
    """Load Fashion-MNIST from local idx files."""
    if os.path.exists(os.path.join(dataset_dir, 'train-images.idx3-ubyte')):
        train_images_path = os.path.join(dataset_dir, 'train-images.idx3-ubyte')
        train_labels_path = os.path.join(dataset_dir, 'train-labels.idx1-ubyte')
        test_images_path = os.path.join(dataset_dir, 't10k-images.idx3-ubyte')
        test_labels_path = os.path.join(dataset_dir, 't10k-labels.idx1-ubyte')
    else:
        train_images_path = os.path.join(dataset_dir, 'train-images-idx3-ubyte')
        train_labels_path = os.path.join(dataset_dir, 'train-labels-idx1-ubyte')
        test_images_path = os.path.join(dataset_dir, 't10k-images-idx3-ubyte')
        test_labels_path = os.path.join(dataset_dir, 't10k-labels-idx1-ubyte')

    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    test_images = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)
    return (train_images, train_labels), (test_images, test_labels)


# ----------------------------
# Model Architectures
# ----------------------------
def create_cnn_baseline(activation='relu', input_shape=(28, 28, 1), num_classes=10):
    """
    Standard CNN with configurable activation.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_cnn_fta(depth=2, branch_factor=2, input_shape=(28, 28, 1), num_classes=10):
    """
    CNN with Fractal Tree Activation (linear branches).
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = FractalTreeActivation(num_units=32, depth=depth, branch_factor=branch_factor)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = FractalTreeActivation(num_units=64, depth=depth, branch_factor=branch_factor)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Dense
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = FractalTreeActivation(num_units=128, depth=depth, branch_factor=branch_factor)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)


def create_cnn_efta(depth=2, branch_factor=2, input_shape=(28, 28, 1), num_classes=10):
    """
    CNN with Exponential Fractal Tree Activation (EFTA).
    Each leaf uses: f(x) = α*(exp(β*x)-1) for x<0, γ*x for x>=0
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = ExponentialFTA(num_units=32, depth=depth, branch_factor=branch_factor)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = ExponentialFTA(num_units=64, depth=depth, branch_factor=branch_factor)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Dense
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = ExponentialFTA(num_units=128, depth=depth, branch_factor=branch_factor)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)


def create_cnn_maxout(k=4, input_shape=(28, 28, 1), num_classes=10):
    """
    CNN with Maxout activation for comparison.
    """
    def maxout_wrapper(inputs, filters, k):
        x = layers.Conv2D(filters * k, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Reshape((-1, filters, k))(x)
        x = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
        x = layers.Reshape((-1, filters))(x)
        return x

    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = maxout_wrapper(inputs, 32, k)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = maxout_wrapper(x, 64, k)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Dense
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)


# ----------------------------
# Training
# ----------------------------
def train_model(model, x_train, y_train, x_val, y_val, x_test, y_test, epochs=30,
                model_name="Model", use_augmentation=True):
    """Train model with optional data augmentation."""

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    print(f"Parameters: {model.count_params():,}")

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # Data augmentation
    if use_augmentation:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        datagen.fit(x_train)
        train_gen = datagen.flow(x_train, y_train, batch_size=128)

        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    return {
        'name': model_name,
        'history': history,
        'test_acc': test_acc * 100,
        'test_loss': test_loss,
        'params': model.count_params(),
        'epochs_trained': len(history.history['loss'])
    }


# ----------------------------
# Visualization
# ----------------------------
def plot_results(results, dataset_name="MNIST"):
    """Plot training results comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    # 1. Test accuracy bar chart
    ax = axes[0, 0]
    test_accs = [r['test_acc'] for r in results.values()]
    bars = ax.bar(range(len(names)), test_accs, color=colors)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'{dataset_name} - Final Test Accuracy')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(min(test_accs) - 2, max(test_accs) + 2)

    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=8)

    # 2. Validation accuracy curves
    ax = axes[0, 1]
    for i, (name, result) in enumerate(results.items()):
        val_acc = [acc * 100 for acc in result['history'].history['val_accuracy']]
        ax.plot(val_acc, color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Over Training')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Validation loss curves
    ax = axes[1, 0]
    for i, (name, result) in enumerate(results.items()):
        val_loss = result['history'].history['val_loss']
        ax.plot(val_loss, color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Over Training')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Accuracy vs Parameters
    ax = axes[1, 1]
    params = [r['params'] for r in results.values()]
    ax.scatter(params, test_accs, s=150, c=colors)
    for i, name in enumerate(names):
        ax.annotate(name, (params[i], test_accs[i]), fontsize=7,
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Parameter Efficiency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cnn_efta_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to 'cnn_efta_comparison.png'")
    plt.show()


# ----------------------------
# Print Results Table
# ----------------------------
def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n" + "="*90)
    print("CNN ACTIVATION COMPARISON RESULTS".center(90))
    print("="*90)
    print(f"{'Model':<25} | {'Test Acc':<12} | {'Test Loss':<12} | {'Params':<12} | {'Epochs':<8}")
    print("-"*90)

    for name, result in results.items():
        print(f"{name:<25} | {result['test_acc']:>10.2f}% | {result['test_loss']:>10.4f} | "
              f"{result['params']:>10,} | {result['epochs_trained']:>6}")

    print("="*90)

    # Find best
    best = max(results, key=lambda x: results[x]['test_acc'])
    print(f"\n🏆 Best Model: {best} with {results[best]['test_acc']:.2f}% test accuracy")

    # Compare EFTA vs FTA
    efta_models = [k for k in results.keys() if 'EFTA' in k]
    fta_models = [k for k in results.keys() if 'FTA' in k and 'EFTA' not in k]

    if efta_models and fta_models:
        print("\n📊 EFTA vs FTA Comparison:")
        for efta_name in efta_models:
            # Find matching FTA configuration
            base_name = efta_name.replace('EFTA', 'FTA')
            if base_name in fta_models:
                efta_acc = results[efta_name]['test_acc']
                fta_acc = results[base_name]['test_acc']
                diff = efta_acc - fta_acc
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"  {efta_name} vs {base_name}: {symbol} {abs(diff):.2f}%")

    print("="*90)


# ----------------------------
# Main Script
# ----------------------------
def run_cnn_efta_comparison(mnist_dir, fashion_mnist_dir=None, epochs=30, dataset='mnist'):
    """Run CNN comparison with EFTA on MNIST or Fashion-MNIST."""
    print("="*70)
    print(f"CNN with Exponential Fractal Tree Activation (EFTA) on {dataset.upper()}")
    print("="*70)

    # Load data
    if dataset == 'fashion' and fashion_mnist_dir:
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist_local(fashion_mnist_dir)
        dataset_name = "Fashion-MNIST"
    else:
        (x_train, y_train), (x_test, y_test) = load_mnist_local(mnist_dir)
        dataset_name = "MNIST"

    print(f"\nDataset: {dataset_name}")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    # Preprocess
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Split validation from training
    val_split = int(0.9 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train_sub, y_train_sub = x_train[:val_split], y_train[:val_split]

    # Models to compare
    models_config = {
        'CNN + ReLU': lambda: create_cnn_baseline(activation='relu'),
        'CNN + LeakyReLU': lambda: create_cnn_baseline(activation='leaky_relu'),
        'CNN + Maxout (k=4)': lambda: create_cnn_maxout(k=4),
        'CNN + FTA (d=2,k=2)': lambda: create_cnn_fta(depth=2, branch_factor=2),
        'CNN + FTA (d=3,k=2)': lambda: create_cnn_fta(depth=3, branch_factor=2),
        'CNN + FTA (d=2,k=3)': lambda: create_cnn_fta(depth=2, branch_factor=3),
        'CNN + EFTA (d=2,k=2)': lambda: create_cnn_efta(depth=2, branch_factor=2),
        'CNN + EFTA (d=3,k=2)': lambda: create_cnn_efta(depth=3, branch_factor=2),
        'CNN + EFTA (d=2,k=3)': lambda: create_cnn_efta(depth=2, branch_factor=3),
    }

    results = {}

    for name, model_fn in models_config.items():
        try:
            model = model_fn()
            result = train_model(model, x_train_sub, y_train_sub, x_val, y_val, x_test, y_test,
                                epochs=epochs, model_name=name, use_augmentation=True)
            results[name] = result
        except Exception as e:
            print(f"\n[ERROR] Training {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    print_comparison_table(results)

    # Plot results
    try:
        plot_results(results, dataset_name)
    except Exception as e:
        print(f"\nCould not generate plots: {e}")
        print("Install matplotlib: pip install matplotlib")

    # Save results
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
                   for k, v in results.items()}
    save_path = 'cnn_efta_results.npy'
    np.save(save_path, save_results, allow_pickle=True)
    print(f"\nResults saved to: {save_path}")

    return results


if __name__ == '__main__':
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mnist_dir = os.path.join(script_dir, 'MNIST dataset')
    fashion_mnist_dir = os.path.join(script_dir, 'fashionMNIST')

    print(f"Script directory: {script_dir}")
    print(f"MNIST directory: {mnist_dir}")
    print(f"Fashion-MNIST directory: {fashion_mnist_dir}")

    # Run on MNIST by default
    results = run_cnn_efta_comparison(mnist_dir, fashion_mnist_dir, epochs=30, dataset='mnist')

    # Optional: Run on Fashion-MNIST
    # results = run_cnn_efta_comparison(mnist_dir, fashion_mnist_dir, epochs=30, dataset='fashion')
