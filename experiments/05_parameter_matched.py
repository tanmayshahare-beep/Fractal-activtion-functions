"""
Parameter-Matched Comparison

Compare FTA with parameter-matched Maxout models for fair comparison.
FTA (d=4,k=2) has many params - create Maxout with similar params.
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
    except RuntimeError as e:
        print(f'GPU configuration error: {e}')
else:
    print('\n✗ No GPU available, using CPU')


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
            trainable=True, name='w'
        )
        self.b = self.add_weight(
            shape=(self.num_units, self.k),
            initializer='zeros',
            trainable=True, name='b'
        )
        super().build(input_shape)

    def call(self, inputs):
        z = tf.tensordot(inputs, self.w, axes=[[1], [0]]) + self.b
        return tf.reduce_max(z, axis=2)


class FractalTreeActivation(layers.Layer):
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
                w = self.add_weight(shape=(input_dim, self.num_units),
                    initializer=tf.keras.initializers.HeNormal(), trainable=True,
                    name=f'leaf_weights_depth_{d}')
                b = self.add_weight(shape=(self.num_units,),
                    initializer='zeros', trainable=True, name=f'leaf_bias_depth_{d}')
                self.leaf_weights.append(w)
                self.leaf_biases.append(b)
        else:
            self.leaf_weights = self.add_weight(
                shape=(self.num_leaves, input_dim, self.num_units),
                initializer=tf.keras.initializers.HeNormal(), trainable=True, name='leaf_weights')
            self.leaf_biases = self.add_weight(
                shape=(self.num_leaves, self.num_units),
                initializer='zeros', trainable=True, name='leaf_biases')
        super().build(input_shape)

    def call(self, inputs):
        if self.share_weights:
            leaf_values = [tf.matmul(inputs, self.leaf_weights[0]) + self.leaf_biases[0]
                          for _ in range(self.num_leaves)]
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


def load_mnist_local(dataset_dir):
    train_images = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 'train-images-idx3-ubyte')
        if not os.path.exists(os.path.join(dataset_dir, 'train-images.idx3-ubyte'))
        else os.path.join(dataset_dir, 'train-images.idx3-ubyte'))
    train_labels = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 'train-labels-idx1-ubyte')
        if not os.path.exists(os.path.join(dataset_dir, 'train-labels.idx1-ubyte'))
        else os.path.join(dataset_dir, 'train-labels.idx1-ubyte'))
    test_images = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 't10k-images-idx3-ubyte')
        if not os.path.exists(os.path.join(dataset_dir, 't10k-images.idx3-ubyte'))
        else os.path.join(dataset_dir, 't10k-images.idx3-ubyte'))
    test_labels = idx2numpy.convert_from_file(
        os.path.join(dataset_dir, 't10k-labels-idx1-ubyte')
        if not os.path.exists(os.path.join(dataset_dir, 't10k-labels.idx1-ubyte'))
        else os.path.join(dataset_dir, 't10k-labels.idx1-ubyte'))
    return (train_images, train_labels), (test_images, test_labels)


def create_maxout_model(k=4, hidden_units=128):
    """Create Maxout model with specified k and hidden units."""
    return models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(hidden_units, use_bias=False),
        layers.BatchNormalization(),
        MaxoutLayer(num_units=hidden_units, k=k),
        layers.Dense(10, activation='softmax')
    ])


def create_fta_model(depth=2, branch_factor=2, hidden_units=128, share_weights=False):
    """Create FTA model with specified parameters."""
    return models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(hidden_units, use_bias=False),
        layers.BatchNormalization(),
        FractalTreeActivation(num_units=hidden_units, depth=depth, 
                             branch_factor=branch_factor, share_weights=share_weights),
        layers.Dense(10, activation='softmax')
    ])


def calculate_params():
    """Calculate parameter counts for different configurations."""
    # Formula for Maxout: input*hidden*k + hidden*k (weights + bias)
    # Formula for FTA: leaves*input*hidden + leaves*hidden (weights + bias)
    
    input_dim = 784
    hidden = 128
    
    configs = []
    
    # Maxout variants
    for k in [2, 4, 8, 16]:
        params = input_dim * hidden * k + hidden * k  # Dense + Maxout
        configs.append(('Maxout', f'k={k}', params))
    
    # FTA variants
    for depth in [1, 2, 3, 4]:
        for branch in [2, 3, 4]:
            leaves = branch ** depth
            params = leaves * input_dim * hidden + leaves * hidden  # FTA weights + bias
            configs.append(('FTA', f'd={depth},k={branch}', params))
    
    # FTA with weight sharing
    for depth in [2, 3, 4]:
        for branch in [2, 3]:
            leaves = branch ** depth
            # With sharing: depth*input*hidden + depth*hidden
            params = depth * input_dim * hidden + depth * hidden
            configs.append(('FTA (shared)', f'd={depth},k={branch}', params))
    
    return configs


def train_and_evaluate(model, x_train, y_train, x_val, y_val, x_test, y_test,
                       epochs=10, model_name="Model"):
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    print(f"Parameters: {model.count_params():,}")
    
    start_time = time.time()
    history = model.fit(x_train, y_train, batch_size=64, epochs=epochs,
        verbose=1, validation_data=(x_val, y_val))
    total_time = time.time() - start_time
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    return {
        'name': model_name,
        'history': history,
        'best_val_acc': max(history.history['val_accuracy']) * 100,
        'test_acc': test_acc * 100,
        'params': model.count_params(),
        'training_time': total_time
    }


def plot_parameter_matched_results(results, param_groups):
    """Plot parameter-matched comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy vs Parameters scatter
    ax = axes[0, 0]
    for name, result in results.items():
        color = 'steelblue' if 'FTA' in name else 'coral'
        ax.scatter(result['params'], result['best_val_acc'], s=100, c=color, alpha=0.7)
        ax.annotate(name, (result['params'], result['best_val_acc']), 
                   fontsize=8, xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title('Accuracy vs Model Size')
    ax.grid(True, alpha=0.3)
    
    # 2. Bar chart: matched pairs comparison
    ax = axes[0, 1]
    
    matched_pairs = [
        ('Maxout k=4', 'FTA d=2,k=2'),
        ('Maxout k=8', 'FTA d=3,k=2'),
        ('Maxout k=16', 'FTA d=4,k=2'),
    ]
    
    x = np.arange(len(matched_pairs))
    width = 0.35
    
    maxout_accs = []
    fta_accs = []
    
    for maxout_name, fta_name in matched_pairs:
        if maxout_name in results:
            maxout_accs.append(results[maxout_name]['best_val_acc'])
        else:
            maxout_accs.append(0)
        if fta_name in results:
            fta_accs.append(results[fta_name]['best_val_acc'])
        else:
            fta_accs.append(0)
    
    bars1 = ax.bar(x - width/2, maxout_accs, width, label='Maxout', color='coral')
    bars2 = ax.bar(x + width/2, fta_accs, width, label='FTA', color='steelblue')
    
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title('Parameter-Matched Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Maxout {m[7:]}\nvs\nFTA {f[4:]}' for m, f in matched_pairs], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Efficiency plot (accuracy per million params)
    ax = axes[1, 0]
    names = list(results.keys())
    efficiency = [r['best_val_acc'] / (r['params'] / 1e6) for r in results.values()]
    colors = ['steelblue' if 'FTA' in n else 'coral' for n in names]
    
    bars = ax.bar(range(len(names)), efficiency, color=colors)
    ax.set_ylabel('Accuracy per Million Parameters')
    ax.set_title('Parameter Efficiency')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Training time comparison
    ax = axes[1, 1]
    times = [r['training_time'] for r in results.values()]
    colors = ['steelblue' if 'FTA' in n else 'coral' for n in names]
    
    bars = ax.bar(range(len(names)), times, color=colors)
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Speed Comparison')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('parameter_matched_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to 'parameter_matched_comparison.png'")
    plt.show()


def run_parameter_matched_comparison(dataset_dir, epochs=10):
    """Run parameter-matched comparison."""
    print("="*70)
    print("PARAMETER-MATCHED COMPARISON: FTA vs Maxout")
    print("="*70)
    
    # First, show parameter calculations
    print("\n📊 PARAMETER COUNTS:")
    print("-"*50)
    param_configs = calculate_params()
    for model_type, config, params in param_configs[:15]:  # Show first 15
        print(f"{model_type:15} {config:15} {params:>10,} params")
    print("-"*50)
    
    (x_train, y_train), (x_test, y_test) = load_mnist_local(dataset_dir)
    
    print(f"\nDataset: MNIST")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    val_split = int(0.9 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train_sub, y_train_sub = x_train[:val_split], y_train[:val_split]
    
    # Parameter-matched configurations
    configs = {
        # ~100k params
        'Maxout k=2': lambda: create_maxout_model(k=2, hidden_units=128),
        'FTA d=1,k=2': lambda: create_fta_model(depth=1, branch_factor=2),
        
        # ~200k params
        'Maxout k=4': lambda: create_maxout_model(k=4, hidden_units=128),
        'FTA d=2,k=2': lambda: create_fta_model(depth=2, branch_factor=2),
        'FTA d=1,k=4': lambda: create_fta_model(depth=1, branch_factor=4),
        
        # ~400k params
        'Maxout k=8': lambda: create_maxout_model(k=8, hidden_units=128),
        'FTA d=3,k=2': lambda: create_fta_model(depth=3, branch_factor=2),
        'FTA d=2,k=4': lambda: create_fta_model(depth=2, branch_factor=4),
        
        # ~800k params
        'Maxout k=16': lambda: create_maxout_model(k=16, hidden_units=128),
        'FTA d=4,k=2': lambda: create_fta_model(depth=4, branch_factor=2),
        'FTA d=3,k=3': lambda: create_fta_model(depth=3, branch_factor=3),
        
        # FTA with weight sharing (more efficient)
        'FTA d=3,k=2 (shared)': lambda: create_fta_model(depth=3, branch_factor=2, share_weights=True),
        'FTA d=4,k=2 (shared)': lambda: create_fta_model(depth=4, branch_factor=2, share_weights=True),
    }
    
    results = {}
    
    for name, model_fn in configs.items():
        try:
            model = model_fn()
            result = train_and_evaluate(model, x_train_sub, y_train_sub, x_val, y_val,
                                        x_test, y_test, epochs=epochs, model_name=name)
            results[name] = result
        except Exception as e:
            print(f"\n[ERROR] Training {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary table
    print("\n" + "="*90)
    print("PARAMETER-MATCHED COMPARISON RESULTS")
    print("="*90)
    print(f"{'Model':<25} | {'Best Val':<10} | {'Test Acc':<10} | {'Time (s)':<10} | {'Params':<12}")
    print("-"*90)
    
    for name, result in results.items():
        print(f"{name:<25} | {result['best_val_acc']:>8.2f}% | {result['test_acc']:>8.2f}% | "
              f"{result['training_time']:>8.1f} | {result['params']:>10,}")
    
    print("="*90)
    
    # Find best in each parameter class
    print("\n📊 PARAMETER-CLASS WINNERS:")
    param_classes = [
        ('~100k', ['Maxout k=2', 'FTA d=1,k=2']),
        ('~200k', ['Maxout k=4', 'FTA d=2,k=2', 'FTA d=1,k=4']),
        ('~400k', ['Maxout k=8', 'FTA d=3,k=2', 'FTA d=2,k=4']),
        ('~800k', ['Maxout k=16', 'FTA d=4,k=2', 'FTA d=3,k=3']),
    ]
    
    for class_name, models in param_classes:
        valid_models = [m for m in models if m in results]
        if valid_models:
            best = max(valid_models, key=lambda x: results[x]['best_val_acc'])
            print(f"  {class_name}: {best} ({results[best]['best_val_acc']:.2f}%)")
    
    # Plot results
    plot_parameter_matched_results(results, param_classes)
    
    # Save results
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'history'} 
                   for k, v in results.items()}
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'parameter_matched_results.npy')
    np.save(save_path, save_results, allow_pickle=True)
    print(f"\nResults saved to: {save_path}")
    
    return results


if __name__ == '__main__':
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Parent directory is the main project folder
    project_dir = os.path.dirname(script_dir)
    # MNIST dataset path
    dataset_dir = os.path.join(project_dir, 'MNIST dataset')
    
    print(f"Project directory: {project_dir}")
    print(f"Dataset directory: {dataset_dir}")
    
    results = run_parameter_matched_comparison(dataset_dir, epochs=10)
