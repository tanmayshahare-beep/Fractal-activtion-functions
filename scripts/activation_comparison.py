import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import idx2numpy
import time
import matplotlib.pyplot as plt

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
print()


def load_mnist_local(dataset_dir):
    """Load MNIST from local idx files."""
    train_images = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 'train-images.idx3-ubyte')
    )
    train_labels = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 'train-labels.idx1-ubyte')
    )
    
    test_images = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 't10k-images.idx3-ubyte')
    )
    test_labels = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 't10k-labels.idx1-ubyte')
    )
    
    return (train_images, train_labels), (test_images, test_labels)


# ----------------------------
# Custom Activation Functions
# ----------------------------
def swish(x):
    """Swish / SiLU activation: x * sigmoid(x)"""
    return x * tf.sigmoid(x)


def mish(x):
    """Mish activation: x * tanh(softplus(x))"""
    return x * tf.tanh(tf.nn.softplus(x))


# ----------------------------
# Maxout Layer
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
        z = tf.tensordot(inputs, self.w, axes=[[1], [0]]) + self.b
        return tf.reduce_max(z, axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_units)


# ----------------------------
# Fractal Tree Activation (FTA) Layer
# ----------------------------
class FractalTreeActivation(layers.Layer):
    """
    Fractal Tree Activation: a tree of linear transforms combined via max.
    
    This creates a hierarchical/maxout-like activation where each neuron
    has an internal tree of linear transformations.
    
    Args:
        num_units: number of output neurons
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
        input_dim = input_shape[-1]
        
        if self.share_weights:
            # One set of weights per depth level
            self.leaf_weights = []
            self.leaf_biases = []
            for d in range(self.depth):
                w = self.add_weight(
                    shape=(input_dim, self.num_units),
                    initializer=tf.keras.initializers.HeNormal(),
                    trainable=True,
                    name=f'leaf_weights_depth_{d}'
                )
                b = self.add_weight(
                    shape=(self.num_units,),
                    initializer='zeros',
                    trainable=True,
                    name=f'leaf_bias_depth_{d}'
                )
                self.leaf_weights.append(w)
                self.leaf_biases.append(b)
        else:
            # Independent weights for each leaf
            self.leaf_weights = self.add_weight(
                shape=(self.num_leaves, input_dim, self.num_units),
                initializer=tf.keras.initializers.HeNormal(),
                trainable=True,
                name='leaf_weights'
            )
            self.leaf_biases = self.add_weight(
                shape=(self.num_leaves, self.num_units),
                initializer='zeros',
                trainable=True,
                name='leaf_biases'
            )
        
        super().build(input_shape)

    def call(self, inputs):
        """
        Compute fractal tree activation.
        
        Args:
            inputs: tensor of shape (batch, input_dim)
        Returns:
            tensor of shape (batch, num_units)
        """
        batch_size = tf.shape(inputs)[0]
        
        if self.share_weights:
            # Compute leaf values using shared weights
            # Each leaf corresponds to a path through the tree
            leaf_values = []
            for leaf_idx in range(self.num_leaves):
                # Determine the path for this leaf
                path = []
                temp = leaf_idx
                for _ in range(self.depth):
                    path.append(temp % self.branch_factor)
                    temp //= self.branch_factor
                
                # Compute weighted sum along path (simplified: just use first depth weight)
                # For proper sharing, we'd combine weights at each level
                leaf_val = tf.matmul(inputs, self.leaf_weights[0]) + self.leaf_biases[0]
                leaf_values.append(leaf_val)
            
            # Stack: (num_leaves, batch, num_units)
            leaf_values = tf.stack(leaf_values, axis=0)
        else:
            # Independent weights: compute all leaf values
            # leaf_values shape: (batch, num_leaves, num_units)
            leaf_values = tf.einsum('bi,lnu->blu', inputs, self.leaf_weights) + self.leaf_biases[:, tf.newaxis, :]
            leaf_values = tf.transpose(leaf_values, [1, 0, 2])  # (num_leaves, batch, num_units)
        
        # Recursively combine using max
        current = leaf_values  # (num_leaves, batch, num_units)
        
        for level in range(self.depth):
            # Reshape to group children
            # current shape: (num_nodes * branch_factor, batch, num_units)
            num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
            current = tf.reshape(current, [num_nodes, self.branch_factor, -1, self.num_units])
            # Take max over children: (num_nodes, batch, num_units)
            current = tf.reduce_max(current, axis=1)
        
        # Final shape: (1, batch, num_units) -> (batch, num_units)
        return tf.squeeze(current, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_units)


# ----------------------------
# Model Factory
# ----------------------------
def create_model(activation='relu', activation_params=None, input_shape=(784,)):
    """
    Create a 2-layer MLP with specified activation function.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    if activation == 'maxout':
        model.add(layers.Dense(128, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(MaxoutLayer(num_units=128, k=activation_params.get('k', 4)))
    elif activation == 'fta':
        model.add(layers.Dense(128, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(FractalTreeActivation(
            num_units=128,
            depth=activation_params.get('depth', 2),
            branch_factor=activation_params.get('branch_factor', 2),
            share_weights=activation_params.get('share_weights', False)
        ))
    elif activation == 'prelu':
        model.add(layers.Dense(128, use_bias=False))
        model.add(layers.PReLU(**(activation_params or {})))
    elif activation == 'leaky_relu':
        model.add(layers.Dense(128))
        model.add(layers.LeakyReLU(**(activation_params or {})))
    elif callable(activation):
        model.add(layers.Dense(128))
        model.add(layers.Activation(activation))
    else:
        model.add(layers.Dense(128, activation=activation))
    
    model.add(layers.Dense(10, activation='softmax'))
    return model


# ----------------------------
# Training and Evaluation
# ----------------------------
def train_and_evaluate(model, x_train, y_train, x_val, y_val, epochs=10, model_name="Model"):
    """Train a model and return metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    print(f"Parameters: {model.count_params():,}")
    print()
    
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val)
    )
    
    total_time = time.time() - start_time
    avg_epoch_time = total_time / epochs
    
    return {
        'name': model_name,
        'history': history,
        'best_val_acc': max(history.history['val_accuracy']) * 100,
        'final_val_acc': history.history['val_accuracy'][-1] * 100,
        'final_train_acc': history.history['accuracy'][-1] * 100,
        'val_loss_curve': history.history['val_loss'],
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'params': model.count_params()
    }


# ----------------------------
# Plot Comparison
# ----------------------------
def plot_comparison(results):
    """Plot comparison of all activation functions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Validation Accuracy over epochs
    ax = axes[0, 0]
    for i, (name, result) in enumerate(results.items()):
        val_acc = [acc * 100 for acc in result['history'].history['val_accuracy']]
        ax.plot(val_acc, color=colors[i], label=name, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Validation Loss over epochs
    ax = axes[0, 1]
    for i, (name, result) in enumerate(results.items()):
        ax.plot(result['val_loss_curve'], color=colors[i], label=name, linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Best Validation Accuracy (bar chart)
    ax = axes[1, 0]
    names = list(results.keys())
    best_accs = [r['best_val_acc'] for r in results.values()]
    bars = ax.bar(names, best_accs, color=colors)
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title('Best Validation Accuracy Achieved')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, best_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # Training Time per Epoch
    ax = axes[1, 1]
    epoch_times = [r['avg_epoch_time'] for r in results.values()]
    bars = ax.bar(names, epoch_times, color=colors)
    ax.set_ylabel('Avg Time per Epoch (s)')
    ax.set_title('Training Speed Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, t in zip(bars, epoch_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{t:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to 'activation_comparison.png'")
    plt.show()


# ----------------------------
# Print Comparison Table
# ----------------------------
def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n" + "="*90)
    print("ACTIVATION FUNCTION COMPARISON RESULTS".center(90))
    print("="*90)
    print(f"{'Activation':<15} | {'Best Val Acc':<12} | {'Final Val Acc':<12} | {'Final Train Acc':<15} | {'Time/Epoch':<10} | {'Params':<10}")
    print("-"*90)
    
    for name, result in results.items():
        print(f"{name:<15} | {result['best_val_acc']:>10.2f}% | {result['final_val_acc']:>10.2f}% | "
              f"{result['final_train_acc']:>13.2f}% | {result['avg_epoch_time']:>8.2f}s | {result['params']:>8,}")
    
    print("="*90)
    
    best_name = max(results, key=lambda x: results[x]['best_val_acc'])
    best_result = results[best_name]
    
    print(f"\n🏆 Best Performer: {best_name} with {best_result['best_val_acc']:.2f}% validation accuracy")
    
    print("\n📊 Analysis:")
    baseline_acc = results.get('ReLU', list(results.values())[0])['best_val_acc']
    for name, result in results.items():
        diff = result['best_val_acc'] - baseline_acc
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"  {name}: {symbol} {abs(diff):.2f}% vs ReLU baseline")
    print("="*90)


# ----------------------------
# Main Comparison Script
# ----------------------------
def run_comparison(epochs=10):
    """Run comparison of all activation functions including Fractal Tree Activation."""
    print("="*60)
    print("Activation Function Comparison on MNIST (with Fractal Tree Activation)")
    print("="*60)
    
    dataset_dir = r'.\MNIST dataset'
    (x_train, y_train), (x_test, y_test) = load_mnist_local(dataset_dir)
    
    print(f"\nTraining samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    val_split = int(0.9 * len(x_train))
    x_val = x_train[val_split:]
    y_val = y_train[val_split:]
    x_train_sub = x_train[:val_split]
    y_train_sub = y_train[:val_split]
    
    # Define activations to test - including FTA variants
    activations = {
        'ReLU': ('relu', {}),
        'Leaky ReLU': ('leaky_relu', {'alpha': 0.01}),
        'ELU': ('elu', {'alpha': 1.0}),
        'Swish': (swish, {}),
        'Mish': (mish, {}),
        'GELU': ('gelu', {}),
        'PReLU': ('prelu', {}),
        'Maxout (k=4)': ('maxout', {'k': 4}),
        'Sigmoid': ('sigmoid', {}),
        # Fractal Tree Activation variants
        'FTA (d=2,k=2)': ('fta', {'depth': 2, 'branch_factor': 2}),
        'FTA (d=3,k=2)': ('fta', {'depth': 3, 'branch_factor': 2}),
        'FTA (d=2,k=3)': ('fta', {'depth': 2, 'branch_factor': 3}),
    }
    
    results = {}
    
    for name, (act_fn, params) in activations.items():
        try:
            model = create_model(act_fn, params)
            result = train_and_evaluate(
                model, x_train_sub, y_train_sub, x_val, y_val,
                epochs=epochs, model_name=name
            )
            results[name] = result
        except Exception as e:
            print(f"\n❌ Error training {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print_comparison_table(results)
    
    try:
        plot_comparison(results)
    except Exception as e:
        print(f"\nCould not generate plots: {e}")
        print("Install matplotlib: pip install matplotlib")
    
    return results


if __name__ == '__main__':
    results = run_comparison(epochs=10)
