"""
transformer_sentiment.py

Sentiment Analysis using Transformer with Self-Attention Mechanism

Dataset: IMDB Movie Reviews (50,000 reviews - 25,000 training, 25,000 testing)
Method: Transformer architecture with multi-head self-attention

This model classifies movie reviews as positive (1) or negative (0) sentiment
using a custom Transformer encoder architecture that leverages self-attention
to capture long-range dependencies in text sequences.

Author: Anthony Perry
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ============================================================================
# SECTION 1: CONFIGURATION AND HYPERPARAMETERS
# ============================================================================

# Dataset parameters
VOCAB_SIZE = 10000          # Maximum number of words to consider in vocabulary
MAX_SEQUENCE_LENGTH = 200   # Maximum length of each review (truncate/pad to this)
EMBEDDING_DIM = 128         # Dimension of word embeddings

# Transformer architecture parameters
NUM_HEADS = 4               # Number of attention heads in multi-head attention
FF_DIM = 128                # Hidden layer size in feed-forward network
NUM_TRANSFORMER_BLOCKS = 2  # Number of transformer encoder blocks

# Training parameters
BATCH_SIZE = 32             # Number of samples per gradient update
EPOCHS = 5                  # Number of training epochs
DROPOUT_RATE = 0.1          # Dropout rate for regularization

# ============================================================================
# SECTION 2: DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """
    Load the IMDB dataset and prepare it for training.

    The IMDB dataset contains 50,000 movie reviews split evenly into
    25,000 training and 25,000 testing samples. Each review is labeled
    as positive (1) or negative (0).

    Data Preparation Steps:
    1. Load pre-tokenized data from Keras (words replaced with integer indices)
    2. Pad sequences to uniform length for batch processing
    3. Split training data into train and validation sets

    Returns:
        x_train: Padded training sequences
        y_train: Training labels
        x_val: Padded validation sequences
        y_val: Validation labels
        x_test: Padded test sequences
        y_test: Test labels
    """
    print("=" * 60)
    print("LOADING IMDB DATASET")
    print("=" * 60)

    # Load the IMDB dataset with vocabulary limited to top VOCAB_SIZE words
    # Words are represented as integers based on their frequency ranking
    (x_train_full, y_train_full), (x_test, y_test) = imdb.load_data(
        num_words=VOCAB_SIZE
    )

    print(f"Training samples: {len(x_train_full)}")
    print(f"Test samples: {len(x_test)}")

    # Pad sequences to ensure uniform length
    # Sequences shorter than MAX_SEQUENCE_LENGTH are padded with zeros at the beginning
    # Sequences longer than MAX_SEQUENCE_LENGTH are truncated from the beginning
    print(f"\nPadding sequences to length {MAX_SEQUENCE_LENGTH}...")
    x_train_full = pad_sequences(x_train_full, maxlen=MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

    # Split training data into training and validation sets (80/20 split)
    # Validation set is used to monitor model performance during training
    val_split = int(0.8 * len(x_train_full))
    x_train = x_train_full[:val_split]
    y_train = y_train_full[:val_split]
    x_val = x_train_full[val_split:]
    y_val = y_train_full[val_split:]

    print(f"\nFinal data splits:")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Validation samples: {len(x_val)}")
    print(f"  Test samples: {len(x_test)}")
    print(f"  Sequence length: {MAX_SEQUENCE_LENGTH}")
    print(f"  Vocabulary size: {VOCAB_SIZE}")

    return x_train, y_train, x_val, y_val, x_test, y_test

# ============================================================================
# SECTION 3: TRANSFORMER COMPONENTS
# ============================================================================

class TransformerBlock(layers.Layer):
    """
    Transformer Encoder Block with Multi-Head Self-Attention.

    This block implements the core transformer encoder architecture:
    1. Multi-Head Self-Attention: Allows the model to attend to different
       positions in the sequence simultaneously from multiple representation
       subspaces.
    2. Layer Normalization: Normalizes activations for stable training.
    3. Feed-Forward Network: Two dense layers with ReLU activation that
       process each position independently.
    4. Residual Connections: Skip connections that help gradient flow.

    The self-attention mechanism computes attention scores between all pairs
    of positions in the sequence, allowing the model to capture long-range
    dependencies regardless of their distance in the sequence.

    Args:
        embed_dim: Dimension of the embedding/hidden states
        num_heads: Number of attention heads
        ff_dim: Dimension of the feed-forward network hidden layer
        rate: Dropout rate for regularization
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()

        # Multi-Head Self-Attention layer
        # Splits the embedding into multiple heads, applies attention,
        # then concatenates and projects back to embedding dimension
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        # Feed-Forward Network: Two dense layers
        # First layer expands to ff_dim with ReLU, second projects back
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])

        # Layer Normalization layers for stable training
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers for regularization
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        """
        Forward pass through the transformer block.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, embed_dim)
            training: Boolean indicating training mode (affects dropout)

        Returns:
            Output tensor of same shape as input
        """
        # Self-attention: query, key, and value all come from the same input
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)

        # First residual connection and layer normalization
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Second residual connection and layer normalization
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    """
    Combined Token and Position Embedding Layer.

    This layer creates two types of embeddings that are summed together:
    1. Token Embeddings: Dense vector representations for each word in the
       vocabulary. These capture semantic meaning of words.
    2. Position Embeddings: Dense vectors for each position in the sequence.
       Since transformers don't have inherent notion of order (unlike RNNs),
       position embeddings provide information about word positions.

    The combination allows the model to consider both what words are present
    and where they appear in the sequence.

    Args:
        maxlen: Maximum sequence length
        vocab_size: Size of the vocabulary
        embed_dim: Dimension of the embeddings
    """

    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()

        # Token embedding: maps word indices to dense vectors
        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        # Position embedding: maps position indices to dense vectors
        self.pos_emb = layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim
        )

    def call(self, x):
        """
        Create combined token and position embeddings.

        Args:
            x: Input tensor of token indices (batch_size, seq_len)

        Returns:
            Combined embeddings (batch_size, seq_len, embed_dim)
        """
        maxlen = tf.shape(x)[-1]

        # Create position indices [0, 1, 2, ..., maxlen-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)

        # Look up position embeddings
        positions = self.pos_emb(positions)

        # Look up token embeddings
        x = self.token_emb(x)

        # Sum token and position embeddings
        return x + positions

# ============================================================================
# SECTION 4: MODEL BUILDING
# ============================================================================

def build_transformer_model():
    """
    Build the complete Transformer model for sentiment classification.

    Model Architecture:
    1. Input Layer: Accepts padded sequences of word indices
    2. Token + Position Embedding: Converts indices to dense representations
    3. Transformer Blocks: Self-attention and feed-forward processing
    4. Global Average Pooling: Aggregates sequence into fixed-size vector
    5. Dense Layers: Final classification layers with dropout
    6. Output Layer: Sigmoid activation for binary classification

    Returns:
        Compiled Keras model ready for training
    """
    print("\n" + "=" * 60)
    print("BUILDING TRANSFORMER MODEL")
    print("=" * 60)

    # Input layer accepts sequences of integers (word indices)
    inputs = layers.Input(shape=(MAX_SEQUENCE_LENGTH,))

    # Embedding layer: combines token and position embeddings
    embedding_layer = TokenAndPositionEmbedding(
        MAX_SEQUENCE_LENGTH,
        VOCAB_SIZE,
        EMBEDDING_DIM
    )
    x = embedding_layer(inputs)

    # Stack multiple transformer blocks for deeper representation
    for i in range(NUM_TRANSFORMER_BLOCKS):
        transformer_block = TransformerBlock(
            EMBEDDING_DIM,
            NUM_HEADS,
            FF_DIM,
            DROPOUT_RATE
        )
        x = transformer_block(x)

    # Global average pooling: aggregate sequence information
    # Converts (batch, seq_len, embed_dim) to (batch, embed_dim)
    x = layers.GlobalAveragePooling1D()(x)

    # Dropout for regularization
    x = layers.Dropout(0.1)(x)

    # Dense layer for final representation
    x = layers.Dense(20, activation="relu")(x)

    # Dropout before output
    x = layers.Dropout(0.1)(x)

    # Output layer: sigmoid for binary classification
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Create and compile the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile with Adam optimizer and binary cross-entropy loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    print(f"\nModel Configuration:")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Number of attention heads: {NUM_HEADS}")
    print(f"  Feed-forward dimension: {FF_DIM}")
    print(f"  Number of transformer blocks: {NUM_TRANSFORMER_BLOCKS}")
    print(f"  Dropout rate: {DROPOUT_RATE}")

    return model

# ============================================================================
# SECTION 5: MODEL TRAINING
# ============================================================================

def train_model(model, x_train, y_train, x_val, y_val):
    """
    Train the transformer model on the IMDB dataset.

    Training Process:
    1. Model learns to predict sentiment from review text
    2. Validation set monitors for overfitting
    3. Training history tracks loss and accuracy per epoch

    Args:
        model: Compiled Keras model
        x_train: Training sequences
        y_train: Training labels
        x_val: Validation sequences
        y_val: Validation labels

    Returns:
        Training history object containing metrics per epoch
    """
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    print(f"\nTraining for {EPOCHS} epochs with batch size {BATCH_SIZE}")

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        verbose=1
    )

    return history

# ============================================================================
# SECTION 6: MODEL EVALUATION AND VALIDATION
# ============================================================================

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model and compute performance metrics.

    Validation Metrics:
    1. Accuracy: Proportion of correct predictions
    2. Precision: True positives / (True positives + False positives)
    3. Recall: True positives / (True positives + False negatives)
    4. F1-Score: Harmonic mean of precision and recall
    5. Confusion Matrix: Breakdown of prediction outcomes

    These metrics validate the model's performance and ensure it
    generalizes well to unseen data.

    Args:
        model: Trained Keras model
        x_test: Test sequences
        y_test: Test labels

    Returns:
        Dictionary containing all computed metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODEL PERFORMANCE")
    print("=" * 60)

    # Get model predictions
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print results
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]:5d}  |  False Positives: {cm[0][1]:5d}")
    print(f"  False Negatives: {cm[1][0]:5d}  |  True Positives:  {cm[1][1]:5d}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_training_history(history):
    """
    Visualize training progress with loss and accuracy curves.

    Plots help identify:
    - Convergence: Is the model learning?
    - Overfitting: Gap between training and validation curves
    - Optimal stopping point: Where validation performance peaks

    Args:
        history: Training history from model.fit()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('sentiment_training_history.png', dpi=150)
    print("\nTraining history plot saved to 'sentiment_training_history.png'")

# ============================================================================
# SECTION 7: DEPLOYMENT CONSIDERATIONS
# ============================================================================

def demonstrate_deployment(model):
    """
    Demonstrate how the model could be deployed for end users.

    Deployment Options:
    1. REST API: Wrap model in Flask/FastAPI for HTTP predictions
    2. TensorFlow Serving: Scalable model serving infrastructure
    3. Cloud Deployment: AWS SageMaker, Google Cloud AI Platform
    4. Edge Deployment: TensorFlow Lite for mobile/embedded devices

    This function saves the model in formats suitable for deployment.
    """
    print("\n" + "=" * 60)
    print("DEPLOYMENT CONSIDERATIONS")
    print("=" * 60)

    # Save the model in native Keras format (recommended)
    model.save('sentiment_model_saved.keras')
    print("\nModel saved in native Keras format: 'sentiment_model_saved.keras'")

    # Also export for TensorFlow Serving
    model.export('sentiment_model_export')
    print("Model exported for TF Serving: 'sentiment_model_export/'")

    print("\nDeployment Options:")
    print("1. REST API Deployment:")
    print("   - Use Flask or FastAPI to create prediction endpoints")
    print("   - Load model with tf.keras.models.load_model()")
    print("   - Accept text input, preprocess, and return predictions")
    print("\n2. TensorFlow Serving:")
    print("   - Deploy SavedModel format to TensorFlow Serving")
    print("   - Supports batching and model versioning")
    print("   - Scalable for production workloads")
    print("\n3. Cloud Platform Deployment:")
    print("   - AWS SageMaker: Upload model, create endpoint")
    print("   - Google Cloud AI Platform: Deploy SavedModel")
    print("   - Azure ML: Container-based deployment")
    print("\n4. Edge Deployment:")
    print("   - Convert to TensorFlow Lite for mobile apps")
    print("   - Use TensorFlow.js for browser deployment")

# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function that orchestrates the complete pipeline.

    Pipeline Steps:
    1. Load and prepare the IMDB dataset
    2. Build the transformer model architecture
    3. Train the model on training data
    4. Evaluate performance on test data
    5. Visualize training progress
    6. Demonstrate deployment options
    """
    print("=" * 60)
    print("Transformer Sentiment Analyzer")
    print("Transformer-Based Sentiment Analysis")
    print("=" * 60)

    # Step 1: Load and prepare data
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_prepare_data()

    # Step 2: Build the model
    model = build_transformer_model()

    # Step 3: Train the model
    history = train_model(model, x_train, y_train, x_val, y_val)

    # Step 4: Evaluate on test set
    metrics = evaluate_model(model, x_test, y_test)

    # Step 5: Plot training history
    plot_training_history(history)

    # Step 6: Demonstrate deployment
    demonstrate_deployment(model)

    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)

    return model, metrics, history


if __name__ == "__main__":
    model, metrics, history = main()
