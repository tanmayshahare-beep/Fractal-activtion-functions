"""
Regularization Study for FTA

Address overfitting by testing various regularization techniques:
- Dropout
- L2 weight decay
- Batch normalization adjustments
- Parameter-matched architectures
- Early stopping
- Data augmentation
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
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
    """Load MNIST from local idx files."""
    # Try both naming conventions (with dots or hyphens)
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


def create_image_generator(x_train, y_train, rotation_range=10, width_shift=0.1, 
                           height_shift=0.1, zoom_range=0.1, batch_size=64):
    """Create data augmentation generator."""
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        zoom_range=zoom_range,
        fill_mode='nearest'
    )
    datagen.fit(x_train.reshape(-1, 28, 28, 1))
    return datagen.flow(x_train.reshape(-1, 28, 28, 1), y_train, batch_size=batch_size)


def create_model(regularization_config, hidden_units=128, depth=2, branch_factor=2):
    """
    Create FTA model with specified regularization.
    
    regularization_config: dict with keys:
        - dropout: float (0 = no dropout)
        - l2: float (0 = no L2 regularization)
        - bn_after_fta: bool (batch norm after FTA)
        - early_stopping: bool
        - data_augmentation: dict or None
        - reduced_architecture: dict or None (smaller depth/branch_factor)
    """
    # Use reduced architecture if specified
    if regularization_config.get('reduced_architecture'):
        depth = regularization_config['reduced_architecture']['depth']
        branch_factor = regularization_config['reduced_architecture']['branch_factor']
    
    # L2 regularization
    kernel_regularizer = None
    if regularization_config.get('l2', 0) > 0:
        kernel_regularizer = tf.keras.regularizers.l2(regularization_config['l2'])
    
    # Build model
    inputs = layers.Input(shape=(784,))
    
    # First dense layer
    x = layers.Dense(hidden_units, use_bias=False, kernel_regularizer=kernel_regularizer)(inputs)
    x = layers.BatchNormalization()(x)
    
    # FTA layer
    fta = FractalTreeActivation(num_units=hidden_units, depth=depth, 
                                branch_factor=branch_factor)
    x = fta(x)
    
    # Batch norm after FTA if specified
    if regularization_config.get('bn_after_fta', False):
        x = layers.BatchNormalization()(x)
    
    # Dropout after FTA
    if regularization_config.get('dropout', 0) > 0:
        x = layers.Dropout(regularization_config['dropout'])(x)
    
    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model


def train_with_regularization(model, x_train, y_train, x_val, y_val, x_test, y_test,
                               regularization_config, epochs=50, model_name="Model"):
    """Train model with specified regularization including early stopping."""
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Regularization: {regularization_config}")
    
    # Callbacks
    callback_list = []
    
    # Early stopping
    if regularization_config.get('early_stopping', False):
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        print("  - Early stopping enabled (patience=5)")
    
    # Learning rate reduction
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    callback_list.append(reduce_lr)
    
    # Training
    if regularization_config.get('data_augmentation'):
        # Use data augmentation generator
        aug_config = regularization_config['data_augmentation']
        train_gen = create_image_generator(
            x_train, y_train,
            rotation_range=aug_config.get('rotation', 10),
            width_shift=aug_config.get('width_shift', 0.1),
            height_shift=aug_config.get('height_shift', 0.1),
            zoom_range=aug_config.get('zoom', 0.1),
            batch_size=64
        )
        print(f"  - Data augmentation: rotation={aug_config.get('rotation', 10)}°, "
              f"shift={aug_config.get('width_shift', 0.1)}")
        
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callback_list,
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=64,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callback_list,
            verbose=1
        )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Calculate generalization gap
    best_val_acc = max(history.history['val_accuracy']) * 100
    final_train_acc = history.history['accuracy'][-1] * 100
    gen_gap = final_train_acc - best_val_acc
    
    return {
        'name': model_name,
        'history': history,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc * 100,
        'final_train_acc': final_train_acc,
        'generalization_gap': gen_gap,
        'params': model.count_params(),
        'epochs_trained': len(history.history['loss'])
    }


def plot_regularization_results(results):
    """Plot regularization study results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # 1. Best validation accuracy
    ax = axes[0, 0]
    best_accs = [r['best_val_acc'] for r in results.values()]
    bars = ax.bar(range(len(names)), best_accs, color=colors)
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Comparison')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, best_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Generalization gap
    ax = axes[0, 1]
    gaps = [r['generalization_gap'] for r in results.values()]
    bars = ax.bar(range(len(names)), gaps, color=colors)
    ax.set_ylabel('Generalization Gap (Train - Val %)')
    ax.set_title('Overfitting Measure (Lower is Better)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{gap:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. Test accuracy
    ax = axes[0, 2]
    test_accs = [r['test_acc'] for r in results.values()]
    bars = ax.bar(range(len(names)), test_accs, color=colors)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Final Test Performance')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 4. Training curves comparison
    ax = axes[1, 0]
    for i, (name, result) in enumerate(results.items()):
        val_acc = [acc * 100 for acc in result['history'].history['val_accuracy']]
        ax.plot(val_acc, color=colors[i], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Curves')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 5. Epochs trained (early stopping effect)
    ax = axes[1, 1]
    epochs = [r['epochs_trained'] for r in results.values()]
    bars = ax.bar(range(len(names)), epochs, color=colors)
    ax.set_ylabel('Epochs Trained')
    ax.set_title('Training Duration (Early Stopping Effect)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Accuracy vs Parameters scatter
    ax = axes[1, 2]
    params = [r['params'] for r in results.values()]
    ax.scatter(params, best_accs, s=150, c=colors)
    for i, name in enumerate(names):
        ax.annotate(name, (params[i], best_accs[i]), fontsize=8, 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title('Parameter Efficiency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regularization_study_results.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to 'regularization_study_results.png'")
    plt.show()


def run_regularization_study(dataset_dir, epochs=50):
    """Run comprehensive regularization study."""
    print("="*70)
    print("FTA REGULARIZATION STUDY - Addressing Overfitting")
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
    
    # Regularization configurations to test
    configs = {
        'Baseline (no reg)': {
            'dropout': 0,
            'l2': 0,
            'bn_after_fta': False,
            'early_stopping': False,
            'data_augmentation': None,
            'reduced_architecture': None,
        },
        'Dropout 0.3': {
            'dropout': 0.3,
            'l2': 0,
            'bn_after_fta': False,
            'early_stopping': False,
            'data_augmentation': None,
            'reduced_architecture': None,
        },
        'Dropout 0.5': {
            'dropout': 0.5,
            'l2': 0,
            'bn_after_fta': False,
            'early_stopping': False,
            'data_augmentation': None,
            'reduced_architecture': None,
        },
        'L2 (1e-4)': {
            'dropout': 0,
            'l2': 1e-4,
            'bn_after_fta': False,
            'early_stopping': False,
            'data_augmentation': None,
            'reduced_architecture': None,
        },
        'L2 (5e-4)': {
            'dropout': 0,
            'l2': 5e-4,
            'bn_after_fta': False,
            'early_stopping': False,
            'data_augmentation': None,
            'reduced_architecture': None,
        },
        'Dropout + L2': {
            'dropout': 0.3,
            'l2': 1e-4,
            'bn_after_fta': False,
            'early_stopping': False,
            'data_augmentation': None,
            'reduced_architecture': None,
        },
        'BN after FTA': {
            'dropout': 0,
            'l2': 0,
            'bn_after_fta': True,
            'early_stopping': False,
            'data_augmentation': None,
            'reduced_architecture': None,
        },
        'Early Stopping': {
            'dropout': 0,
            'l2': 0,
            'bn_after_fta': False,
            'early_stopping': True,
            'data_augmentation': None,
            'reduced_architecture': None,
        },
        'Data Augmentation': {
            'dropout': 0,
            'l2': 0,
            'bn_after_fta': False,
            'early_stopping': False,
            'data_augmentation': {
                'rotation': 10,
                'width_shift': 0.1,
                'height_shift': 0.1,
                'zoom': 0.1
            },
            'reduced_architecture': None,
        },
        'Reduced Arch (d=2,k=2)': {
            'dropout': 0,
            'l2': 0,
            'bn_after_fta': False,
            'early_stopping': False,
            'data_augmentation': None,
            'reduced_architecture': {'depth': 2, 'branch_factor': 2},
        },
        'Full Regularization': {
            'dropout': 0.3,
            'l2': 1e-4,
            'bn_after_fta': True,
            'early_stopping': True,
            'data_augmentation': None,
            'reduced_architecture': None,
        },
        'Full + Data Aug': {
            'dropout': 0.3,
            'l2': 1e-4,
            'bn_after_fta': True,
            'early_stopping': True,
            'data_augmentation': {
                'rotation': 10,
                'width_shift': 0.1,
                'height_shift': 0.1,
                'zoom': 0.1
            },
            'reduced_architecture': None,
        },
    }
    
    results = {}
    
    for name, config in configs.items():
        try:
            model = create_model(config)
            result = train_with_regularization(
                model, x_train_sub, y_train_sub, x_val, y_val, x_test, y_test,
                config, epochs=epochs, model_name=name
            )
            results[name] = result
        except Exception as e:
            print(f"\n[ERROR] Training {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary table
    print("\n" + "="*110)
    print("REGULARIZATION STUDY RESULTS")
    print("="*110)
    print(f"{'Model':<20} | {'Best Val':<10} | {'Test':<8} | {'Gap':<8} | {'Epochs':<8} | {'Params':<10}")
    print("-"*110)
    
    for name, result in results.items():
        print(f"{name:<20} | {result['best_val_acc']:>8.2f}% | {result['test_acc']:>6.2f}% | "
              f"{result['generalization_gap']:>6.2f}% | {result['epochs_trained']:>6} | {result['params']:>8,}")
    
    print("="*110)
    
    # Find best configurations
    best_val = max(results, key=lambda x: results[x]['best_val_acc'])
    best_test = max(results, key=lambda x: results[x]['test_acc'])
    lowest_gap = min(results, key=lambda x: results[x]['generalization_gap'])
    
    print(f"\n🏆 Best Validation: {best_val} ({results[best_val]['best_val_acc']:.2f}%)")
    print(f"🏆 Best Test: {best_test} ({results[best_test]['test_acc']:.2f}%)")
    print(f"🏆 Lowest Overfitting: {lowest_gap} (gap={results[lowest_gap]['generalization_gap']:.2f}%)")
    
    # Plot results
    plot_regularization_results(results)
    
    # Save results
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'history'} 
                   for k, v in results.items()}
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'regularization_results.npy')
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
    
    results = run_regularization_study(dataset_dir, epochs=50)
