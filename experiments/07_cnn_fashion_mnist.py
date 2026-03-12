"""
CNN with Fractal Tree Activation for Fashion-MNIST

This script implements a Convolutional Neural Network using FTA as the activation function
for the Fashion-MNIST dataset. The FTA replaces standard activations like ReLU.
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
    print('  To enable CUDA:')
    print('  1. Install NVIDIA GPU drivers')
    print('  2. Install tensorflow-gpu or tensorflow with CUDA support')
    print('  3. Ensure CUDA Toolkit and cuDNN are installed')


# ----------------------------
# Fractal Tree Activation Layer
# ----------------------------
class FractalTreeActivation(layers.Layer):
    """
    Fractal Tree Activation: a tree of linear transforms combined via max.
    
    Args:
        num_units: number of output neurons/channels
        depth: number of levels in the tree (depth=1 is standard Maxout)
        branch_factor: number of children per internal node
        share_weights: if True, all nodes at the same depth share weights
    """
    def __init__(self, num_units, depth=2, branch_factor=2, share_weights=False, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.share_weights = share_weights
        self.num_leaves = branch_factor ** depth

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Number of input channels
        
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
        """
        Apply FTA to feature maps.
        Works with both 2D (batch, features) and 4D (batch, h, w, channels) inputs.
        """
        if len(inputs.shape) == 4:
            # Convolutional: (batch, height, width, channels)
            batch_size = tf.shape(inputs)[0]
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            
            # Flatten spatial dimensions
            inputs_flat = tf.reshape(inputs, [batch_size, height * width, -1])
            
            if self.share_weights:
                leaf_values = []
                for _ in range(self.num_leaves):
                    leaf_val = tf.matmul(inputs_flat, self.leaf_weights[0]) + self.leaf_biases[0]
                    leaf_values.append(leaf_val)
                leaf_values = tf.stack(leaf_values, axis=0)
            else:
                leaf_values = tf.stack([
                    tf.matmul(inputs_flat, self.leaf_weights[i]) + self.leaf_biases[i]
                    for i in range(self.num_leaves)
                ], axis=0)
            
            # Reshape back: (num_leaves, batch, height, width, num_units)
            leaf_values = tf.reshape(leaf_values, 
                [self.num_leaves, batch_size, height, width, self.num_units])
            
            # Combine through tree
            current = leaf_values
            for level in range(self.depth):
                num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
                current = tf.reshape(current, 
                    [num_nodes, self.branch_factor, batch_size, height, width, self.num_units])
                current = tf.reduce_max(current, axis=1)
            
            # Reshape to (batch, height, width, num_units)
            return tf.reshape(current, [batch_size, height, width, self.num_units])
        
        else:
            # Dense: (batch, features)
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
                current = tf.reshape(current, [num_nodes, self.branch_factor, batch_size, self.num_units])
                current = tf.reduce_max(current, axis=1)
            
            return tf.squeeze(current, axis=0)


# ----------------------------
# Data Loading
# ----------------------------
def load_fashion_mnist_local(dataset_dir):
    """Load Fashion-MNIST from local idx files."""
    # Try both naming conventions
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
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_cnn_fta(depth=2, branch_factor=2, input_shape=(28, 28, 1), num_classes=10, 
                   use_fta_dense=True):
    """
    CNN with Fractal Tree Activation replacing standard activations.
    
    Args:
        depth: FTA tree depth
        branch_factor: FTA branching factor
        use_fta_dense: Also use FTA in dense layers (or just conv layers)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Block 1 - Conv with FTA
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = FractalTreeActivation(num_units=32, depth=depth, branch_factor=branch_factor)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2 - Conv with FTA
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = FractalTreeActivation(num_units=64, depth=depth, branch_factor=branch_factor)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    
    if use_fta_dense:
        x = FractalTreeActivation(num_units=128, depth=depth, branch_factor=branch_factor)(x)
    else:
        x = layers.Activation('relu')(x)
    
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)


def create_cnn_maxout(k=4, input_shape=(28, 28, 1), num_classes=10):
    """
    CNN with Maxout activation for comparison.
    """
    def maxout_wrapper(inputs, filters, k):
        # Maxout through concatenation + max pooling
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
    x = maxout_wrapper(tf.expand_dims(tf.expand_dims(x, 1), 1), 128, k)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)


# ----------------------------
# Training
# ----------------------------
def train_model(model, x_train, y_train, x_test, y_test, epochs=30, model_name="Model",
                use_augmentation=True):
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
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
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
            validation_data=(x_test, y_test),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
    
    # Evaluate
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
def plot_results(results):
    """Plot training results comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # 1. Test accuracy bar chart
    ax = axes[0, 0]
    test_accs = [r['test_acc'] for r in results.values()]
    bars = ax.bar(range(len(names)), test_accs, color=colors)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Final Test Accuracy Comparison')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(80, 95)
    
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Validation accuracy curves
    ax = axes[0, 1]
    for i, (name, result) in enumerate(results.items()):
        val_acc = [acc * 100 for acc in result['history'].history['val_accuracy']]
        ax.plot(val_acc, color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Over Training')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. Validation loss curves
    ax = axes[1, 0]
    for i, (name, result) in enumerate(results.items()):
        val_loss = result['history'].history['val_loss']
        ax.plot(val_loss, color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Over Training')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Accuracy vs Parameters
    ax = axes[1, 1]
    params = [r['params'] for r in results.values()]
    ax.scatter(params, test_accs, s=150, c=colors)
    for i, name in enumerate(names):
        ax.annotate(name, (params[i], test_accs[i]), fontsize=8,
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Parameter Efficiency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_fashion_mnist_results.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to 'cnn_fashion_mnist_results.png'")
    plt.show()


# ----------------------------
# Main
# ----------------------------
def run_cnn_comparison(fashion_mnist_dir, epochs=30):
    """Run CNN comparison on Fashion-MNIST."""
    print("="*70)
    print("CNN with FTA on Fashion-MNIST")
    print("="*70)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist_local(fashion_mnist_dir)
    
    print(f"\nDataset: Fashion-MNIST")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Preprocess
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Class names for reference
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Models to compare
    models_config = {
        'CNN + ReLU': lambda: create_cnn_baseline(activation='relu'),
        'CNN + LeakyReLU': lambda: create_cnn_baseline(activation='leaky_relu'),
        'CNN + Maxout (k=4)': lambda: create_cnn_maxout(k=4),
        'CNN + FTA (d=2,k=2)': lambda: create_cnn_fta(depth=2, branch_factor=2),
        'CNN + FTA (d=3,k=2)': lambda: create_cnn_fta(depth=3, branch_factor=2),
        'CNN + FTA (d=2,k=3)': lambda: create_cnn_fta(depth=2, branch_factor=3),
    }
    
    results = {}
    
    for name, model_fn in models_config.items():
        try:
            model = model_fn()
            result = train_model(model, x_train, y_train, x_test, y_test,
                                epochs=epochs, model_name=name)
            results[name] = result
        except Exception as e:
            print(f"\n[ERROR] Training {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("CNN COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<25} | {'Test Acc':<10} | {'Params':<12} | {'Epochs':<8}")
    print("-"*80)
    
    for name, result in results.items():
        print(f"{name:<25} | {result['test_acc']:>8.2f}% | {result['params']:>10,} | {result['epochs_trained']:>6}")
    
    print("="*80)
    
    # Find best
    best = max(results, key=lambda x: results[x]['test_acc'])
    print(f"\n🏆 Best Model: {best} with {results[best]['test_acc']:.2f}% test accuracy")
    
    # Plot
    plot_results(results)
    
    # Save results
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
                   for k, v in results.items()}
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'cnn_fashion_mnist_results.npy')
    np.save(save_path, save_results, allow_pickle=True)
    print(f"\nResults saved to: {save_path}")
    
    return results


if __name__ == '__main__':
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    fashion_mnist_dir = os.path.join(project_dir, 'fashionMNIST')
    
    print(f"Project directory: {project_dir}")
    print(f"Fashion-MNIST directory: {fashion_mnist_dir}")
    
    results = run_cnn_comparison(fashion_mnist_dir, epochs=30)
