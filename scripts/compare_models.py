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
# Baseline Model (Sigmoid)
# ----------------------------
def create_baseline_model():
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(10, activation='softmax')
    ])
    return model


# ----------------------------
# Maxout Model
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


def create_maxout_model(k=4):
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, use_bias=False),
        layers.BatchNormalization(),
        MaxoutLayer(num_units=128, k=k),
        layers.Dense(10, activation='softmax')
    ])
    return model


# ----------------------------
# Training & Evaluation Functions
# ----------------------------
def count_parameters(model):
    return model.count_params()


def train_model(model, x_train, y_train, x_test, y_test, epochs=5, model_name="Model"):
    # Compile model with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Parameters: {count_parameters(model):,}")
    print()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epoch_times = []
    
    best_val_acc = 0
    best_epoch = 0
    
    # Split training data for validation (10%)
    val_split = int(0.9 * len(x_train))
    x_val = x_train[val_split:]
    y_val = y_train[val_split:]
    x_train_sub = x_train[:val_split]
    y_train_sub = y_train[:val_split]
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train for one epoch
        history = model.fit(
            x_train_sub, y_train_sub,
            batch_size=64,
            epochs=1,
            verbose=0,
            validation_data=(x_val, y_val)
        )
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0] * 100
        val_loss = history.history['val_loss'][0]
        val_acc = history.history['val_accuracy'][0] * 100
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        if np.isnan(train_loss):
            print("NaN detected! Stopping training.")
            break
    
    # Final evaluation on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    test_acc *= 100
    
    if test_acc > best_val_acc:
        best_val_acc = test_acc
        best_epoch = epochs
    
    avg_epoch_time = np.mean(epoch_times)
    
    print(f"\n{model_name} Summary:")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Avg Training Time: {avg_epoch_time:.1f}s per epoch")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'best_epoch': best_epoch,
        'avg_epoch_time': avg_epoch_time,
        'params': count_parameters(model)
    }


# ----------------------------
# Plot Comparison
# ----------------------------
def plot_comparison(baseline_results, maxout_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training Loss
    axes[0, 0].plot(baseline_results['train_losses'], 'b-o', label='Baseline (Sigmoid)', linewidth=2)
    axes[0, 0].plot(maxout_results['train_losses'], 'r-s', label='Maxout (k=4)', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation Loss
    axes[0, 1].plot(baseline_results['val_losses'], 'b-o', label='Baseline (Sigmoid)', linewidth=2)
    axes[0, 1].plot(maxout_results['val_losses'], 'r-s', label='Maxout (k=4)', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Accuracy
    axes[1, 0].plot(baseline_results['train_accs'], 'b-o', label='Baseline (Sigmoid)', linewidth=2)
    axes[1, 0].plot(maxout_results['train_accs'], 'r-s', label='Maxout (k=4)', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation Accuracy
    axes[1, 1].plot(baseline_results['val_accs'], 'b-o', label='Baseline (Sigmoid)', linewidth=2)
    axes[1, 1].plot(maxout_results['val_accs'], 'r-s', label='Maxout (k=4)', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=150)
    print("\nPlots saved to 'comparison_plot.png'")
    plt.show()


# ----------------------------
# Comparison Table
# ----------------------------
def print_comparison_table(baseline_results, maxout_results):
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<35} | {'Baseline':<15} | {'Maxout':<15}")
    print("-"*70)
    print(f"{'Parameters':<35} | {baseline_results['params']:>12,} | {maxout_results['params']:>12,}")
    print(f"{'Best Val Accuracy (%)':<35} | {baseline_results['best_val_acc']:>14.2f} | {maxout_results['best_val_acc']:>14.2f}")
    print(f"{'Test Accuracy (%)':<35} | {baseline_results['test_acc']:>14.2f} | {maxout_results['test_acc']:>14.2f}")
    print(f"{'Best Epoch':<35} | {baseline_results['best_epoch']:>14} | {maxout_results['best_epoch']:>14}")
    print(f"{'Avg Training Time (s/epoch)':<35} | {baseline_results['avg_epoch_time']:>14.1f} | {maxout_results['avg_epoch_time']:>14.1f}")
    print("="*70)
    
    # Calculate improvement
    acc_improvement = maxout_results['test_acc'] - baseline_results['test_acc']
    speed_ratio = baseline_results['avg_epoch_time'] / maxout_results['avg_epoch_time'] if maxout_results['avg_epoch_time'] > 0 else 0
    param_ratio = maxout_results['params'] / baseline_results['params'] if baseline_results['params'] > 0 else 0
    
    print(f"\nAnalysis:")
    print(f"  Accuracy {'improvement' if acc_improvement > 0 else 'decrease'}: {abs(acc_improvement):.2f}%")
    print(f"  Maxout has {param_ratio:.1f}x more parameters")
    if speed_ratio != 0:
        if speed_ratio > 1:
            print(f"  Maxout is faster by {abs(speed_ratio - 1):.0%}")
        else:
            print(f"  Maxout is slower by {abs(1 - speed_ratio):.0%}")
    else:
        print(f"  Training speed similar")
    print("="*70)


# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    print("="*60)
    print("MNIST Model Comparison: Baseline vs Maxout (TensorFlow)")
    print("="*60)
    
    # Load data
    dataset_dir = r'.\MNIST dataset'
    (x_train, y_train), (x_test, y_test) = load_mnist_local(dataset_dir)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Preprocess
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Train Baseline
    baseline_model = create_baseline_model()
    baseline_results = train_model(
        baseline_model, x_train, y_train, x_test, y_test,
        epochs=5, model_name="Baseline (Sigmoid)"
    )
    
    # Train Maxout
    maxout_model = create_maxout_model(k=4)
    maxout_results = train_model(
        maxout_model, x_train, y_train, x_test, y_test,
        epochs=5, model_name="Maxout (k=4)"
    )
    
    # Print comparison
    print_comparison_table(baseline_results, maxout_results)
    
    # Plot comparison
    try:
        plot_comparison(baseline_results, maxout_results)
    except Exception as e:
        print(f"\nCould not generate plots: {e}")
        print("Install matplotlib for visualization: pip install matplotlib")
