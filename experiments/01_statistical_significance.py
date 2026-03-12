"""
Statistical Significance Testing

Run each activation function 5 times with different random seeds,
report mean ± standard deviation for all metrics.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import idx2numpy
import time
from datetime import datetime

# Suppress GPU messages
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


def load_mnist_local(dataset_dir):
    """Load MNIST from local idx files."""
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


def create_model(activation='relu', activation_params=None, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = models.Sequential()
    model.add(layers.Input(shape=(784,)))
    
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
        model.add(layers.PReLU())
    elif activation == 'leaky_relu':
        model.add(layers.Dense(128))
        model.add(layers.LeakyReLU(alpha=0.01))
    elif callable(activation):
        model.add(layers.Dense(128))
        model.add(layers.Activation(activation))
    else:
        model.add(layers.Dense(128, activation=activation))
    
    model.add(layers.Dense(10, activation='softmax'))
    return model


def train_single_run(model, x_train, y_train, x_val, y_val, epochs=10):
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(x_train, y_train, batch_size=64, epochs=epochs,
        verbose=0, validation_data=(x_val, y_val))
    
    return {
        'best_val_acc': max(history.history['val_accuracy']) * 100,
        'final_val_acc': history.history['val_accuracy'][-1] * 100,
        'final_train_acc': history.history['accuracy'][-1] * 100,
    }


def run_statistical_test(dataset_dir, n_runs=5, epochs=10):
    """Run statistical significance testing."""
    print("="*70)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*70)
    print(f"Running {n_runs} trials with different random seeds\n")
    
    (x_train, y_train), (x_test, y_test) = load_mnist_local(dataset_dir)
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # 90/10 split for validation
    val_split = int(0.9 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train_sub, y_train_sub = x_train[:val_split], y_train[:val_split]
    
    # Configurations to test
    configs = {
        'ReLU': ('relu', {}),
        'Leaky ReLU': ('leaky_relu', {}),
        'ELU': ('elu', {}),
        'Swish': (lambda x: x * tf.sigmoid(x), {}),
        'GELU': ('gelu', {}),
        'PReLU': ('prelu', {}),
        'Maxout (k=4)': ('maxout', {'k': 4}),
        'FTA (d=2,k=2)': ('fta', {'depth': 2, 'branch_factor': 2}),
        'FTA (d=3,k=2)': ('fta', {'depth': 3, 'branch_factor': 2}),
    }
    
    all_results = {}
    
    for name, (act_fn, params) in configs.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        runs_data = []
        for run in range(n_runs):
            seed = 42 + run * 100
            tf.random.set_seed(seed)
            np.random.seed(seed)
            
            model = create_model(act_fn, params, seed=seed)
            result = train_single_run(model, x_train_sub, y_train_sub, x_val, y_val, epochs)
            runs_data.append(result)
            
            print(f"  Run {run+1}/{n_runs}: Val Acc = {result['best_val_acc']:.2f}%")
        
        # Calculate statistics
        best_accs = [r['best_val_acc'] for r in runs_data]
        final_accs = [r['final_val_acc'] for r in runs_data]
        
        all_results[name] = {
            'best_val_acc_mean': np.mean(best_accs),
            'best_val_acc_std': np.std(best_accs),
            'final_val_acc_mean': np.mean(final_accs),
            'final_val_acc_std': np.std(final_accs),
            'runs': runs_data
        }
        
        print(f"  Result: {np.mean(best_accs):.2f} ± {np.std(best_accs):.2f}%")
    
    # Print summary table
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY (mean ± std)")
    print("="*80)
    print(f"{'Activation':<15} | {'Best Val Acc':<20} | {'Final Val Acc':<20}")
    print("-"*80)
    
    for name, stats in all_results.items():
        print(f"{name:<15} | {stats['best_val_acc_mean']:6.2f} ± {stats['best_val_acc_std']:.2f}%    "
              f"| {stats['final_val_acc_mean']:6.2f} ± {stats['final_val_acc_std']:.2f}%")
    
    print("="*80)
    
    # Statistical significance (paired t-test vs ReLU)
    print("\n📊 Statistical Significance (vs ReLU baseline):")
    from scipy import stats
    baseline_runs = np.array([r['best_val_acc'] for r in all_results['ReLU']['runs']])
    
    for name, data in all_results.items():
        if name == 'ReLU':
            continue
        test_runs = np.array([r['best_val_acc'] for r in data['runs']])
        t_stat, p_value = stats.ttest_ind(test_runs, baseline_runs)
        significant = "✓ Significant" if p_value < 0.05 else "✗ Not significant"
        diff = np.mean(test_runs) - np.mean(baseline_runs)
        print(f"  {name}: diff={diff:+.2f}%, p={p_value:.4f} {significant}")
    
    print("="*80)
    return all_results


if __name__ == '__main__':
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Parent directory is the main project folder
    project_dir = os.path.dirname(script_dir)
    # MNIST dataset path
    dataset_dir = os.path.join(project_dir, 'MNIST dataset')
    
    print(f"Project directory: {project_dir}")
    print(f"Dataset directory: {dataset_dir}")
    
    results = run_statistical_test(dataset_dir, n_runs=5, epochs=10)
    
    # Save results
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'statistical_results.npy')
    np.save(save_path, results, allow_pickle=True)
    print(f"\nResults saved to: {save_path}")
