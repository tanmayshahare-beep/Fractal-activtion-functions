"""
Ablation Study - Different Combining Operations

Compare max vs sum vs product vs attention-based combining
for the Fractal Tree Activation.
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


class FractalTreeActivation(layers.Layer):
    """FTA with configurable combining operation."""
    def __init__(self, num_units, depth=2, branch_factor=2, combine_op='max', 
                 share_weights=False, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.depth = depth
        self.branch_factor = branch_factor
        self.combine_op = combine_op
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
        
        # Learnable parameters for weighted operations
        if self.combine_op == 'weighted_sum':
            self.combine_weights = self.add_weight(
                shape=(self.num_leaves,),
                initializer='ones',
                trainable=True,
                name='combine_weights'
            )
        
        super().build(input_shape)

    def _combine_children(self, current, batch_size):
        """Combine children using specified operation."""
        for level in range(self.depth):
            num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
            current = tf.reshape(current, [num_nodes, self.branch_factor, batch_size, self.num_units])
            
            if self.combine_op == 'max':
                current = tf.reduce_max(current, axis=1)
            elif self.combine_op == 'sum':
                current = tf.reduce_sum(current, axis=1)
            elif self.combine_op == 'mean':
                current = tf.reduce_mean(current, axis=1)
            elif self.combine_op == 'product':
                # Product with clipping for numerical stability
                current = tf.reduce_prod(tf.clip_by_value(current, -10, 10), axis=1)
            elif self.combine_op == 'weighted_sum':
                # Apply learnable weights to each branch
                weights = tf.nn.softmax(self.combine_weights)
                weighted = current * tf.reshape(weights, [1, self.branch_factor, 1, 1])
                current = tf.reduce_sum(weighted, axis=1)
            elif self.combine_op == 'max_mean':
                # Hybrid: max + mean (like maxout with residual)
                max_val = tf.reduce_max(current, axis=1)
                mean_val = tf.reduce_mean(current, axis=1)
                current = 0.7 * max_val + 0.3 * mean_val
            else:
                current = tf.reduce_max(current, axis=1)
        
        return current

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
        
        batch_size = tf.shape(inputs)[0]
        current = self._combine_children(leaf_values, batch_size)
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


def create_model(combine_op='max', depth=2, branch_factor=2):
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, use_bias=False),
        layers.BatchNormalization(),
        FractalTreeActivation(num_units=128, depth=depth, branch_factor=branch_factor,
                             combine_op=combine_op),
        layers.Dense(10, activation='softmax')
    ])
    return model


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
        'final_train_acc': history.history['accuracy'][-1] * 100,
        'params': model.count_params(),
        'training_time': total_time
    }


def plot_ablation_results(results):
    """Plot ablation study results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # 1. Best validation accuracy comparison
    ax = axes[0, 0]
    best_accs = [r['best_val_acc'] for r in results.values()]
    bars = ax.bar(names, best_accs, color=colors)
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title('Combining Operation Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, best_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Training loss curves
    ax = axes[0, 1]
    for i, (name, result) in enumerate(results.items()):
        ax.plot(result['history'].history['loss'], color=colors[i], 
                label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Validation accuracy curves
    ax = axes[1, 0]
    for i, (name, result) in enumerate(results.items()):
        val_acc = [acc * 100 for acc in result['history'].history['val_accuracy']]
        ax.plot(val_acc, color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Training time vs accuracy scatter
    ax = axes[1, 1]
    times = [r['training_time'] for r in results.values()]
    ax.scatter(times, best_accs, s=150, c=colors)
    for i, name in enumerate(names):
        ax.annotate(name, (times[i], best_accs[i]), fontsize=8, 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Total Training Time (s)')
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title('Speed vs Accuracy Trade-off')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_study_results.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to 'ablation_study_results.png'")
    plt.show()


def run_ablation_study(dataset_dir, epochs=10):
    """Run ablation study on combining operations."""
    print("="*70)
    print("ABLATION STUDY - Combining Operations for FTA")
    print("="*70)
    
    (x_train, y_train), (x_test, y_test) = load_mnist_local(dataset_dir)
    
    print(f"\nDataset: MNIST")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    val_split = int(0.9 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train_sub, y_train_sub = x_train[:val_split], y_train[:val_split]
    
    # Combining operations to test
    operations = {
        'Max (baseline)': {'op': 'max', 'depth': 2, 'branch': 2},
        'Sum': {'op': 'sum', 'depth': 2, 'branch': 2},
        'Mean': {'op': 'mean', 'depth': 2, 'branch': 2},
        'Product': {'op': 'product', 'depth': 2, 'branch': 2},
        'Weighted Sum': {'op': 'weighted_sum', 'depth': 2, 'branch': 2},
        'Max+Mean (0.7/0.3)': {'op': 'max_mean', 'depth': 2, 'branch': 2},
        'Max (d=3)': {'op': 'max', 'depth': 3, 'branch': 2},
        'Max (d=2, k=3)': {'op': 'max', 'depth': 2, 'branch': 3},
    }
    
    results = {}
    
    for name, config in operations.items():
        try:
            model = create_model(combine_op=config['op'], 
                                depth=config['depth'], 
                                branch_factor=config['branch'])
            result = train_and_evaluate(model, x_train_sub, y_train_sub, x_val, y_val,
                                        x_test, y_test, epochs=epochs, model_name=name)
            results[name] = result
        except Exception as e:
            print(f"\n[ERROR] Training {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary table
    print("\n" + "="*90)
    print("ABLATION STUDY RESULTS")
    print("="*90)
    print(f"{'Operation':<20} | {'Best Val':<10} | {'Test Acc':<10} | {'Time (s)':<10} | {'Params':<10}")
    print("-"*90)
    
    for name, result in results.items():
        print(f"{name:<20} | {result['best_val_acc']:>8.2f}% | {result['test_acc']:>8.2f}% | "
              f"{result['training_time']:>8.1f} | {result['params']:>8,}")
    
    print("="*90)
    
    # Find best operation
    best_op = max(results, key=lambda x: results[x]['best_val_acc'])
    print(f"\n🏆 Best Operation: {best_op} with {results[best_op]['best_val_acc']:.2f}% accuracy")
    
    # Plot results
    plot_ablation_results(results)
    
    # Save results
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'history'} 
                   for k, v in results.items()}
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'ablation_results.npy')
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
    
    results = run_ablation_study(dataset_dir, epochs=10)
