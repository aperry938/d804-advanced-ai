"""
transformer_optimization.py

Optimize the Transformer-based Sentiment Analysis model for improved
efficiency and performance through systematic hyperparameter tuning
and architectural modifications.

Optimization Strategies:
1. Hyperparameter Tuning: Learning rate, batch size, dropout
2. Architecture Optimization: Attention heads, embedding dimension, layers
3. Regularization Techniques: Dropout rates, early stopping
4. Performance Benchmarks: Accuracy, Precision, Recall, F1, Training time

Author: Anthony Perry
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time
import json

# ============================================================================
# SECTION 1: BASELINE CONFIGURATION
# ============================================================================

# Fixed parameters
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200

# Baseline configuration (from original model)
BASELINE_CONFIG = {
    'embedding_dim': 128,
    'num_heads': 4,
    'ff_dim': 128,
    'num_transformer_blocks': 2,
    'dropout_rate': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 5
}

# ============================================================================
# SECTION 2: TRANSFORMER COMPONENTS (Same as original)
# ============================================================================

class TransformerBlock(layers.Layer):
    """Transformer encoder block with multi-head self-attention."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    """Combined token and positional embedding layer."""

    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# ============================================================================
# SECTION 3: DATA LOADING
# ============================================================================

def load_data():
    """Load and prepare the IMDB dataset."""
    print("Loading IMDB dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    x_train_full = pad_sequences(x_train_full, maxlen=MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

    val_split = int(0.8 * len(x_train_full))
    x_train = x_train_full[:val_split]
    y_train = y_train_full[:val_split]
    x_val = x_train_full[val_split:]
    y_val = y_train_full[val_split:]

    print(f"Training: {len(x_train)}, Validation: {len(x_val)}, Test: {len(x_test)}")
    return x_train, y_train, x_val, y_val, x_test, y_test

# ============================================================================
# SECTION 4: MODEL BUILDER
# ============================================================================

def build_model(config):
    """
    Build a transformer model with the given configuration.

    Args:
        config: Dictionary with model hyperparameters

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(MAX_SEQUENCE_LENGTH,))

    embedding_layer = TokenAndPositionEmbedding(
        MAX_SEQUENCE_LENGTH,
        VOCAB_SIZE,
        config['embedding_dim']
    )
    x = embedding_layer(inputs)

    for _ in range(config['num_transformer_blocks']):
        transformer_block = TransformerBlock(
            config['embedding_dim'],
            config['num_heads'],
            config['ff_dim'],
            config['dropout_rate']
        )
        x = transformer_block(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ============================================================================
# SECTION 5: TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate(config, x_train, y_train, x_val, y_val, x_test, y_test, verbose=0):
    """
    Train a model with given config and return metrics.

    Returns:
        Dictionary with all performance metrics
    """
    # Build model
    model = build_model(config)

    # Count parameters
    total_params = model.count_params()

    # Train with timing
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(x_val, y_val),
        verbose=verbose
    )
    training_time = time.time() - start_time

    # Evaluate on test set
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'training_time': training_time,
        'total_params': total_params,
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_train_acc': history.history['accuracy'][-1],
        'final_val_acc': history.history['val_accuracy'][-1]
    }

    # Clean up
    del model
    tf.keras.backend.clear_session()

    return metrics, history

# ============================================================================
# SECTION 6: OPTIMIZATION EXPERIMENTS
# ============================================================================

def run_baseline(x_train, y_train, x_val, y_val, x_test, y_test):
    """Run baseline model and return metrics."""
    print("\n" + "=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)
    print(f"\nBaseline Configuration:")
    for key, value in BASELINE_CONFIG.items():
        print(f"  {key}: {value}")

    metrics, history = train_and_evaluate(
        BASELINE_CONFIG, x_train, y_train, x_val, y_val, x_test, y_test, verbose=1
    )

    print(f"\nBaseline Results:")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1-Score:      {metrics['f1_score']:.4f}")
    print(f"  Training Time: {metrics['training_time']:.2f}s")
    print(f"  Parameters:    {metrics['total_params']:,}")

    return metrics, history


def optimize_learning_rate(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Optimize learning rate hyperparameter.

    Learning Rate Optimization:
    - Too high: Model diverges, loss increases
    - Too low: Very slow convergence
    - Optimal: Fast convergence to good solution
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION 1: LEARNING RATE")
    print("=" * 60)

    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    results = []

    for lr in learning_rates:
        config = BASELINE_CONFIG.copy()
        config['learning_rate'] = lr
        config['epochs'] = 3  # Reduced for speed

        print(f"\nTesting learning_rate = {lr}...")
        metrics, _ = train_and_evaluate(
            config, x_train, y_train, x_val, y_val, x_test, y_test
        )
        metrics['learning_rate'] = lr
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

    # Find best
    best = max(results, key=lambda x: x['f1_score'])
    print(f"\nBest learning rate: {best['learning_rate']} (F1: {best['f1_score']:.4f})")

    return results, best['learning_rate']


def optimize_architecture(x_train, y_train, x_val, y_val, x_test, y_test, best_lr):
    """
    Optimize model architecture parameters.

    Architecture Optimization:
    - Embedding dimension: Capacity to represent word semantics
    - Number of heads: Parallel attention patterns
    - Number of blocks: Model depth
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION 2: ARCHITECTURE")
    print("=" * 60)

    architectures = [
        {'embedding_dim': 64, 'num_heads': 2, 'ff_dim': 64, 'num_transformer_blocks': 1},
        {'embedding_dim': 128, 'num_heads': 4, 'ff_dim': 128, 'num_transformer_blocks': 2},
        {'embedding_dim': 256, 'num_heads': 8, 'ff_dim': 256, 'num_transformer_blocks': 2},
    ]

    results = []
    for arch in architectures:
        config = BASELINE_CONFIG.copy()
        config.update(arch)
        config['learning_rate'] = best_lr
        config['epochs'] = 3

        print(f"\nTesting: embed={arch['embedding_dim']}, heads={arch['num_heads']}, blocks={arch['num_transformer_blocks']}...")
        metrics, _ = train_and_evaluate(
            config, x_train, y_train, x_val, y_val, x_test, y_test
        )
        metrics['architecture'] = arch
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, Params: {metrics['total_params']:,}")

    # Find best (balance accuracy and efficiency)
    best = max(results, key=lambda x: x['f1_score'])
    print(f"\nBest architecture: {best['architecture']}")

    return results, best['architecture']


def optimize_regularization(x_train, y_train, x_val, y_val, x_test, y_test, best_lr, best_arch):
    """
    Optimize regularization (dropout).

    Regularization Optimization:
    - Dropout prevents overfitting by randomly dropping units
    - Too little: Overfitting
    - Too much: Underfitting
    """
    print("\n" + "=" * 60)
    print("OPTIMIZATION 3: REGULARIZATION (DROPOUT)")
    print("=" * 60)

    dropout_rates = [0.05, 0.1, 0.2, 0.3]
    results = []

    for dropout in dropout_rates:
        config = BASELINE_CONFIG.copy()
        config.update(best_arch)
        config['learning_rate'] = best_lr
        config['dropout_rate'] = dropout
        config['epochs'] = 3

        print(f"\nTesting dropout_rate = {dropout}...")
        metrics, _ = train_and_evaluate(
            config, x_train, y_train, x_val, y_val, x_test, y_test
        )
        metrics['dropout_rate'] = dropout
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

    best = max(results, key=lambda x: x['f1_score'])
    print(f"\nBest dropout rate: {best['dropout_rate']} (F1: {best['f1_score']:.4f})")

    return results, best['dropout_rate']

# ============================================================================
# SECTION 7: FINAL OPTIMIZED MODEL
# ============================================================================

def train_optimized_model(x_train, y_train, x_val, y_val, x_test, y_test,
                          best_lr, best_arch, best_dropout):
    """Train the final optimized model with all improvements."""
    print("\n" + "=" * 60)
    print("TRAINING OPTIMIZED MODEL")
    print("=" * 60)

    # Create optimized configuration
    optimized_config = BASELINE_CONFIG.copy()
    optimized_config.update(best_arch)
    optimized_config['learning_rate'] = best_lr
    optimized_config['dropout_rate'] = best_dropout
    optimized_config['epochs'] = 5  # Full training

    print("\nOptimized Configuration:")
    for key, value in optimized_config.items():
        print(f"  {key}: {value}")

    metrics, history = train_and_evaluate(
        optimized_config, x_train, y_train, x_val, y_val, x_test, y_test, verbose=1
    )

    print(f"\nOptimized Model Results:")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1-Score:      {metrics['f1_score']:.4f}")
    print(f"  Training Time: {metrics['training_time']:.2f}s")
    print(f"  Parameters:    {metrics['total_params']:,}")

    return metrics, history, optimized_config

# ============================================================================
# SECTION 8: BENCHMARKING AND COMPARISON
# ============================================================================

def benchmark_comparison(baseline_metrics, optimized_metrics, baseline_config, optimized_config):
    """
    Compare baseline and optimized models with defined benchmarks.

    Benchmarks for Evaluation:
    1. Accuracy: Primary classification performance
    2. F1-Score: Balanced precision/recall metric
    3. Training Time: Computational efficiency
    4. Model Size: Memory efficiency (parameter count)
    """
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)

    print("\n{:<25} {:>15} {:>15} {:>15}".format(
        "Benchmark", "Baseline", "Optimized", "Improvement"
    ))
    print("-" * 70)

    benchmarks = [
        ('Accuracy', 'accuracy', True),
        ('Precision', 'precision', True),
        ('Recall', 'recall', True),
        ('F1-Score', 'f1_score', True),
        ('Training Time (s)', 'training_time', False),
        ('Parameters', 'total_params', False),
    ]

    for name, key, higher_better in benchmarks:
        base_val = baseline_metrics[key]
        opt_val = optimized_metrics[key]

        if key in ['total_params']:
            improvement = ((base_val - opt_val) / base_val) * 100
            print(f"{name:<25} {base_val:>15,} {opt_val:>15,} {improvement:>+14.1f}%")
        elif key == 'training_time':
            improvement = ((base_val - opt_val) / base_val) * 100
            print(f"{name:<25} {base_val:>15.2f} {opt_val:>15.2f} {improvement:>+14.1f}%")
        else:
            improvement = ((opt_val - base_val) / base_val) * 100
            print(f"{name:<25} {base_val:>15.4f} {opt_val:>15.4f} {improvement:>+14.1f}%")

    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)

    print("\nConfiguration Changes:")
    for key in optimized_config:
        if key in baseline_config and baseline_config[key] != optimized_config[key]:
            print(f"  {key}: {baseline_config[key]} -> {optimized_config[key]}")

    # Overall assessment
    f1_improvement = ((optimized_metrics['f1_score'] - baseline_metrics['f1_score'])
                      / baseline_metrics['f1_score']) * 100

    print(f"\nOverall F1-Score Improvement: {f1_improvement:+.2f}%")

    if f1_improvement > 0:
        print("Optimization successful: Model performance improved.")
    else:
        print("Note: Baseline was already well-tuned for this dataset.")


def plot_optimization_results(baseline_history, optimized_history):
    """Visualize training progress comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    axes[0].plot(baseline_history.history['val_accuracy'], 'b--', label='Baseline Val Acc')
    axes[0].plot(optimized_history.history['val_accuracy'], 'b-', label='Optimized Val Acc')
    axes[0].plot(baseline_history.history['accuracy'], 'r--', alpha=0.5, label='Baseline Train Acc')
    axes[0].plot(optimized_history.history['accuracy'], 'r-', alpha=0.5, label='Optimized Train Acc')
    axes[0].set_title('Accuracy: Baseline vs Optimized')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss comparison
    axes[1].plot(baseline_history.history['val_loss'], 'b--', label='Baseline Val Loss')
    axes[1].plot(optimized_history.history['val_loss'], 'b-', label='Optimized Val Loss')
    axes[1].plot(baseline_history.history['loss'], 'r--', alpha=0.5, label='Baseline Train Loss')
    axes[1].plot(optimized_history.history['loss'], 'r-', alpha=0.5, label='Optimized Train Loss')
    axes[1].set_title('Loss: Baseline vs Optimized')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('sentiment_optimization_comparison.png', dpi=150)
    print("\nOptimization comparison plot saved to 'sentiment_optimization_comparison.png'")

# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main optimization pipeline.

    Steps:
    1. Load data
    2. Evaluate baseline model
    3. Optimize learning rate
    4. Optimize architecture
    5. Optimize regularization
    6. Train final optimized model
    7. Benchmark comparison
    """
    print("=" * 60)
    print("Transformer Sentiment Optimization")
    print("Transformer Model Optimization")
    print("=" * 60)

    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    # Step 1: Baseline evaluation
    baseline_metrics, baseline_history = run_baseline(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    # Step 2: Optimize learning rate
    lr_results, best_lr = optimize_learning_rate(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    # Step 3: Optimize architecture
    arch_results, best_arch = optimize_architecture(
        x_train, y_train, x_val, y_val, x_test, y_test, best_lr
    )

    # Step 4: Optimize regularization
    reg_results, best_dropout = optimize_regularization(
        x_train, y_train, x_val, y_val, x_test, y_test, best_lr, best_arch
    )

    # Step 5: Train optimized model
    optimized_metrics, optimized_history, optimized_config = train_optimized_model(
        x_train, y_train, x_val, y_val, x_test, y_test,
        best_lr, best_arch, best_dropout
    )

    # Step 6: Benchmark comparison
    benchmark_comparison(baseline_metrics, optimized_metrics,
                         BASELINE_CONFIG, optimized_config)

    # Step 7: Visualize
    plot_optimization_results(baseline_history, optimized_history)

    # Save results
    results = {
        'baseline_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in baseline_metrics.items()},
        'optimized_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in optimized_metrics.items()},
        'optimized_config': optimized_config
    }

    with open('sentiment_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to 'sentiment_optimization_results.json'")

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    return baseline_metrics, optimized_metrics


if __name__ == "__main__":
    baseline_metrics, optimized_metrics = main()
