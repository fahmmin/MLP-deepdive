import numpy as np

# ===== Reproducibility =====
SEED = 1337
# rng is a random number generator object, seeded for reproducibility.
# This means every time you run the code, you'll get the same random numbers.
rng = np.random.default_rng(SEED)

# ===== Synthetic dataset (2D blobs) =====
def make_blobs(n_per_class=200, spread=0.9):
    """
    Generates a synthetic 2D dataset for binary classification.
    - n_per_class: number of samples per class
    - spread: standard deviation of the blobs
    Returns:
        X: features (shape: [n_samples, 2])
        y: labels (0 or 1)
    """
    # Generate samples for class 0 centered at (-2, -2)
    c0 = rng.normal(loc=[-2.0, -2.0], scale=spread, size=(n_per_class, 2))
    # Generate samples for class 1 centered at (2, 2)
    c1 = rng.normal(loc=[ 2.0,  2.0], scale=spread, size=(n_per_class, 2))
    # Stack both classes together
    X = np.vstack([c0, c1]).astype(np.float32)
    # Create labels: first n_per_class are 0, next n_per_class are 1
    y = np.array([0]*n_per_class + [1]*n_per_class, dtype=np.int64)
    # Shuffle the dataset so the classes are mixed
    # rng.permutation returns a shuffled array of indices from 0 to len(X)-1
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

# Create the dataset
X, y = make_blobs(n_per_class=400)

# Split into training and validation sets (80% train, 20% validation)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val,   y_val   = X[split:], y[split:]

# ===== utilities =====
def one_hot(y, num_classes):
    """
    Converts integer labels to one-hot encoded vectors.
    Example: y=[1,0], num_classes=2 -> [[0,1],[1,0]]
    """
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def relu(z):
    """
    ReLU activation function: sets negative values to zero.
    Used to introduce non-linearity.
    """
    return np.maximum(0, z)

def relu_grad(z):
    """
    Gradient of ReLU: 1 for positive values, 0 for negative.
    Used in backpropagation.
    """
    return (z > 0).astype(z.dtype)

def softmax(logits):
    """
    Softmax activation: converts logits to probabilities that sum to 1.
    Used for multi-class classification.
    """
    # Subtract max for numerical stability (avoids large exponents)
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy(probs, y_true_oh, eps=1e-12):
    """
    Cross-entropy loss: measures difference between predicted and true distributions.
    Lower loss means better predictions.
    """
    return -np.mean(np.sum(y_true_oh * np.log(probs + eps), axis=1))

def accuracy(probs, y_true):
    """
    Computes accuracy: fraction of correct predictions.
    """
    preds = probs.argmax(axis=1)
    return (preds == y_true).mean()

# ===== model params =====
in_dim = 2      # input dimension (2 features per sample)
hidden = 32     # number of hidden units in hidden layer
out_dim = 2     # output dimension (number of classes)

# Initialize weights and biases for both layers
W1 = rng.normal(0, 0.1, size=(in_dim, hidden)).astype(np.float32)  # input to hidden weights
b1 = np.zeros((hidden,), dtype=np.float32)                         # hidden layer bias
W2 = rng.normal(0, 0.1, size=(hidden, out_dim)).astype(np.float32) # hidden to output weights
b2 = np.zeros((out_dim,), dtype=np.float32)                        # output layer bias

# ===== forward pass =====
def forward(X):
    """
    Computes output of the network for input X.
    Returns:
        probs: predicted probabilities
        cache: intermediate values for backpropagation
    """
    z1 = X @ W1 + b1            # Linear transformation (input to hidden)
    h1 = relu(z1)               # Apply ReLU activation
    logits = h1 @ W2 + b2       # Linear transformation (hidden to output)
    probs = softmax(logits)     # Convert logits to probabilities
    cache = (X, z1, h1, logits, probs)  # Store for backprop
    return probs, cache

# ===== backward pass =====
def backward(cache, y_true):
    """
    Computes gradients for all parameters using backpropagation.
    Returns:
        Gradients for W1, b1, W2, b2
    """
    global W1, b1, W2, b2
    X, z1, h1, logits, probs = cache
    B = X.shape[0]  # batch size
    y_oh = one_hot(y_true, out_dim)  # one-hot encode labels

    # Gradient of loss w.r.t logits (softmax + cross-entropy)
    dlogits = (probs - y_oh) / B
    # Gradients for output layer weights and bias
    gW2 = h1.T @ dlogits
    gb2 = dlogits.sum(axis=0)

    # Backpropagate into hidden layer
    dh1 = dlogits @ W2.T
    dz1 = dh1 * relu_grad(z1)

    # Gradients for input layer weights and bias
    gW1 = X.T @ dz1
    gb1 = dz1.sum(axis=0)

    return gW1, gb1, gW2, gb2

# ===== training loop =====
def iterate_minibatches(X, y, batch_size=64, shuffle=True):
    """
    Yields batches of data for training.
    - yield is a Python keyword that turns the function into a generator.
      Instead of returning all batches at once, it produces one batch at a time.
      This is memory efficient and lets you loop over batches with 'for'.
    """
    N = len(X)
    idx = np.arange(N)
    if shuffle:
        rng.shuffle(idx)  # Randomly shuffle indices for each epoch
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]  # yield returns the batch to the caller

lr = 0.1
weight_decay = 1e-4   # L2 regularization strength (prevents overfitting)
epochs = 50
batch_size = 64

for epoch in range(1, epochs+1):
    # Training phase
    train_losses = []
    train_accs = []
    for xb, yb in iterate_minibatches(X_train, y_train, batch_size):
        probs, cache = forward(xb)  # Forward pass
        # Compute loss (cross-entropy + L2 regularization)
        ce = cross_entropy(probs, one_hot(yb, out_dim))
        l2 = 0.5 * weight_decay * (np.sum(W1*W1) + np.sum(W2*W2))  # Penalize large weights
        loss = ce + l2
        train_losses.append(loss)
        train_accs.append(accuracy(probs, yb))

        # Backward pass (compute gradients)
        gW1, gb1, gW2, gb2 = backward(cache, yb)
        # Add L2 gradients (derivative of regularization term)
        gW1 += weight_decay * W1
        gW2 += weight_decay * W2

        # SGD step: update weights and biases
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2

    # Validation phase (no training, just evaluation)
    probs_val, _ = forward(X_val)
    val_loss = cross_entropy(probs_val, one_hot(y_val, out_dim))
    val_acc = accuracy(probs_val, y_val)

    # Print progress every 5 epochs (and first epoch)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | "
              f"train_loss={np.mean(train_losses):.4f} "
              f"train_acc={np.mean(train_accs):.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

# Quick sanity check: final metrics
probs_tr, _ = forward(X_train)
probs_te, _ = forward(X_val)
print("Final Train Acc:", accuracy(probs_tr, y_train))
print("Final Val Acc  :", accuracy(probs_te, y_val))
