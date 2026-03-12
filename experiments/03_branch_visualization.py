"""
Branch Visualization - What do FTA leaves learn?

Feed digit images and trace which leaves activate most strongly.
Analyze if different branches specialize in different features.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import idx2numpy
import matplotlib.pyplot as plt
import seaborn as sns

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

    def call(self, inputs, return_leaf_values=False):
        if self.share_weights:
            leaf_values = [tf.matmul(inputs, self.leaf_weights[0]) + self.leaf_biases[0]
                          for _ in range(self.num_leaves)]
            leaf_values = tf.stack(leaf_values, axis=0)
        else:
            leaf_values = tf.stack([
                tf.matmul(inputs, self.leaf_weights[i]) + self.leaf_biases[i]
                for i in range(self.num_leaves)
            ], axis=0)
        
        if return_leaf_values:
            return leaf_values
        
        current = leaf_values
        for level in range(self.depth):
            batch_size = tf.shape(current)[1]
            num_nodes = self.num_leaves // (self.branch_factor ** (level + 1))
            current = tf.reshape(current, [num_nodes, self.branch_factor, batch_size, self.num_units])
            current = tf.reduce_max(current, axis=1)
        return tf.squeeze(current, axis=0)


class FTAWithLeafAccess(layers.Layer):
    """FTA wrapper that allows accessing leaf values."""
    def __init__(self, fta_layer, **kwargs):
        super().__init__(**kwargs)
        self.fta = fta_layer
    
    def get_leaf_values(self, inputs):
        """Get leaf values for analysis."""
        return self.fta.call(inputs, return_leaf_values=True)


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


def create_fta_model(depth=2, branch_factor=2):
    """Create FTA model with leaf access."""
    inputs = layers.Input(shape=(784,))
    x = layers.Dense(128, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    fta = FractalTreeActivation(num_units=128, depth=depth, branch_factor=branch_factor)
    x = fta(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, fta


def analyze_leaf_activation(fta_layer, images, labels, num_samples_per_digit=10):
    """
    Analyze which leaves activate most strongly for each digit class.
    """
    print("\nAnalyzing leaf activations...")
    
    # Get leaf values: (num_leaves, batch, num_units)
    leaf_values = fta_layer.call(tf.constant(images.astype('float32')), return_leaf_values=True)
    leaf_values = leaf_values.numpy()  # (num_leaves, batch, num_units)
    
    num_leaves = leaf_values.shape[0]
    num_digits = 10
    
    # For each digit, find which leaves activate most strongly
    digit_leaf_activations = {d: [] for d in range(num_digits)}
    
    unique_digits = np.unique(labels)
    for digit in unique_digits:
        digit_mask = labels == digit
        digit_indices = np.where(digit_mask)[0][:num_samples_per_digit]
        
        for idx in digit_indices:
            # Get leaf values for this sample: (num_leaves, num_units)
            sample_leaf_values = leaf_values[:, idx, :]
            # Mean activation across neurons for each leaf
            mean_leaf_activation = np.mean(sample_leaf_values, axis=1)
            digit_leaf_activations[digit].append(mean_leaf_activation)
        
        digit_leaf_activations[digit] = np.mean(digit_leaf_activations[digit], axis=0)
    
    return digit_leaf_activations, num_leaves


def plot_leaf_analysis(digit_leaf_activations, num_leaves, depth, branch_factor):
    """Visualize leaf activation patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    digits = list(digit_leaf_activations.keys())
    
    # 1. Heatmap: digits x leaves
    ax = axes[0, 0]
    heatmap_data = np.array([digit_leaf_activations[d] for d in digits])
    sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Mean Activation'})
    ax.set_xlabel('Leaf Index')
    ax.set_ylabel('Digit')
    ax.set_title(f'Leaf Activation Patterns by Digit\n(FTA depth={depth}, branch={branch_factor}, {num_leaves} leaves)')
    
    # 2. Top activating leaves per digit
    ax = axes[0, 1]
    top_k = 5
    for i, digit in enumerate(digits):
        activations = digit_leaf_activations[digit]
        top_indices = np.argsort(activations)[-top_k:][::-1]
        y_pos = np.arange(top_k)
        ax.scatter(activations[top_indices], y_pos + i * top_k, label=f'Digit {digit}', alpha=0.7, s=50)
    
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Top-K Rank')
    ax.set_title('Top Activating Leaves per Digit')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8)
    
    # 3. Leaf specialization score (variance across digits)
    ax = axes[1, 0]
    all_activations = np.array([digit_leaf_activations[d] for d in digits])
    specialization_score = np.std(all_activations, axis=0)  # High std = specialized leaf
    
    # Find most specialized leaves
    top_specialized = np.argsort(specialization_score)[-10:][::-1]
    ax.bar(range(len(top_specialized)), specialization_score[top_specialized])
    ax.set_xlabel('Specialized Leaf Rank')
    ax.set_ylabel('Specialization Score (std across digits)')
    ax.set_title('Most Specialized Leaves')
    ax.set_xticks(range(len(top_specialized)))
    ax.set_xticklabels([f'Leaf {i}' for i in top_specialized], rotation=45, ha='right')
    
    # 4. Dendrogram-like visualization of tree structure
    ax = axes[1, 1]
    
    # Show tree structure with activation intensity
    y_positions = {}
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_leaves))
    
    # Draw tree levels
    for level in range(depth + 1):
        nodes_at_level = branch_factor ** level
        y = depth - level
        x_positions = np.linspace(0, 1, nodes_at_level + 1)[1:]
        
        for i, x in enumerate(x_positions):
            if level == depth:
                # Leaf node - color by average activation
                leaf_idx = i
                avg_activation = np.mean([digit_leaf_activations[d][leaf_idx] for d in digits])
                color = plt.cm.Reds(0.3 + 0.7 * (avg_activation - np.min([digit_leaf_activations[d] for d in digits])) / 
                           (np.max([digit_leaf_activations[d] for d in digits]) - np.min([digit_leaf_activations[d] for d in digits]) + 1e-6))
                ax.plot(x, y, 'o', color=color, markersize=15, markeredgecolor='black')
            else:
                ax.plot(x, y, 'o', color='lightgray', markersize=10, markeredgecolor='black')
            
            # Draw connections to children
            if level < depth:
                children_start = i * branch_factor
                for c in range(branch_factor):
                    child_x = (children_start + c + 1) / (branch_factor ** (level + 1))
                    ax.plot([x, child_x], [y, y - 0.9], 'k-', alpha=0.3)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, depth + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Tree Structure with Leaf Activations\n(Darker = Higher Activation)')
    
    plt.tight_layout()
    plt.savefig('leaf_activation_analysis.png', dpi=150, bbox_inches='tight')
    print("Plots saved to 'leaf_activation_analysis.png'")
    plt.show()


def visualize_specialized_digits(fta_layer, test_images, test_labels, num_leaves, top_k=5):
    """Find and visualize images that maximally activate specific leaves."""
    print("\nFinding images that maximally activate specific leaves...")
    
    # Get all leaf values
    leaf_values = fta_layer.call(tf.constant(test_images.astype('float32')), return_leaf_values=True)
    leaf_values = leaf_values.numpy()  # (num_leaves, batch, num_units)
    
    # For each leaf, find top activating images
    fig, axes = plt.subplots(top_k, min(6, num_leaves), figsize=(15, 3 * top_k))
    
    for leaf_idx in range(min(6, num_leaves)):
        # Mean activation across neurons for this leaf
        leaf_activations = np.mean(leaf_values[leaf_idx, :, :], axis=1)
        top_indices = np.argsort(leaf_activations)[-top_k:][::-1]
        
        for i, img_idx in enumerate(top_indices):
            ax = axes[i, leaf_idx] if top_k > 1 else axes[leaf_idx]
            ax.imshow(test_images[img_idx].reshape(28, 28), cmap='gray')
            digit = test_labels[img_idx]
            activation = leaf_activations[img_idx]
            ax.set_title(f'Digit: {digit}\nActivation: {activation:.2f}', fontsize=9)
            ax.axis('off')
    
    plt.suptitle(f'Top Activating Images for Leaves 0-{min(5, num_leaves-1)}', y=1.02)
    plt.tight_layout()
    plt.savefig('specialized_leaf_images.png', dpi=150, bbox_inches='tight')
    print("Specialized images saved to 'specialized_leaf_images.png'")
    plt.show()


def run_branch_visualization(dataset_dir):
    """Run branch visualization analysis."""
    print("="*70)
    print("BRANCH VISUALIZATION - What do FTA leaves learn?")
    print("="*70)
    
    (x_train, y_train), (x_test, y_test) = load_mnist_local(dataset_dir)
    
    # Normalize
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Create and train model
    print("\nCreating and training FTA model...")
    model, fta_layer = create_fta_model(depth=2, branch_factor=2)
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    val_split = int(0.9 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train_sub, y_train_sub = x_train[:val_split], y_train[:val_split]
    
    print("Training for 10 epochs...")
    model.fit(x_train_sub, y_train_sub, batch_size=64, epochs=10,
              validation_data=(x_val, y_val), verbose=1)
    
    # Analyze leaf activations
    digit_activations, num_leaves = analyze_leaf_activation(
        fta_layer, x_test[:500], y_test[:500], num_samples_per_digit=50
    )
    
    print(f"\nNumber of leaves: {num_leaves}")
    print(f"Digits analyzed: {list(digit_activations.keys())}")
    
    # Plot analysis
    plot_leaf_analysis(digit_activations, num_leaves, depth=2, branch_factor=2)
    
    # Visualize specialized images
    visualize_specialized_digits(fta_layer, x_test, y_test, num_leaves, top_k=5)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey questions to consider:")
    print("1. Do different leaves specialize in different digit classes?")
    print("2. Are there leaves that activate strongly for similar digits (e.g., 4 and 9)?")
    print("3. What visual features might each leaf be detecting?")
    print("="*70)


if __name__ == '__main__':
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Parent directory is the main project folder
    project_dir = os.path.dirname(script_dir)
    # MNIST dataset path
    dataset_dir = os.path.join(project_dir, 'MNIST dataset')
    
    print(f"Project directory: {project_dir}")
    print(f"Dataset directory: {dataset_dir}")
    
    run_branch_visualization(dataset_dir)
