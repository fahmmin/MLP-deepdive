import numpy as np

# ===== Reproducibility =====
SEED = 1337
rng = np.random.default_rng(SEED)  # random number generator for reproducibility

# ===== Synthetic dataset (2D blobs) =====
def make_blobs(n_per_class=200, spread=0.9):
    # Generate two clusters ("blobs") for binary classification
    # class 0 centered at (-2, -2), class 1 at (2, 2)
    c0 = rng.normal(loc=[-2.0, -2.0], scale=spread, size=(n_per_class, 2))  # samples for class 0
    c1 = rng.normal(loc=[ 2.0,  2.0], scale=spread, size=(n_per_class, 2))  # samples for class 1
    X = np.vstack([c0, c1]).astype(np.float32)  # combine both classes into one array
    y = np.array([0]*n_per_class + [1]*n_per_class, dtype=np.int64)  # labels: 0 for c0, 1 for c1
    # shuffle dataset so classes are mixed
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

X, y = make_blobs(n_per_class=400)  # create dataset

# train/val split (80% train, 20% validation)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val,   y_val   = X[split:], y[split:]

# ===== utilities =====
def one_hot(y, num_classes):
    # Convert integer labels to one-hot encoded vectors
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def relu(z):
    # ReLU activation: sets negative values to zero
    return np.maximum(0, z)

def relu_grad(z):
    # Gradient of ReLU: 1 for positive, 0 for negative
    return (z > 0).astype(z.dtype)

def softmax(logits):
    # Softmax activation: converts logits to probabilities
    # logits: (B, C) where B=batch size, C=number of classes
    z = logits - logits.max(axis=1, keepdims=True)  # numerical stability
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy(probs, y_true_oh, eps=1e-12):
    # Cross-entropy loss: measures difference between predicted and true distributions
    # probs: predicted probabilities, y_true_oh: one-hot true labels
    # eps: small value to avoid log(0)
    return -np.mean(np.sum(y_true_oh * np.log(probs + eps), axis=1))

def accuracy(probs, y_true):
    # Compute accuracy: fraction of correct predictions
    preds = probs.argmax(axis=1)
    return (preds == y_true).mean()

# ===== model params =====
in_dim = 2      # input dimension (2 features per sample)
hidden = 32     # number of hidden units
out_dim = 2     # output dimension (number of classes)

# Initialize weights and biases for both layers
W1 = rng.normal(0, 0.1, size=(in_dim, hidden)).astype(np.float32)  # input to hidden weights
b1 = np.zeros((hidden,), dtype=np.float32)                         # hidden layer bias
W2 = rng.normal(0, 0.1, size=(hidden, out_dim)).astype(np.float32) # hidden to output weights
b2 = np.zeros((out_dim,), dtype=np.float32)                        # output layer bias

# ===== forward pass =====
def forward(X):
    # Compute output of the network for input X
    # layer 1: input to hidden
    z1 = X @ W1 + b1            # linear transformation
    h1 = relu(z1)               # apply ReLU activation
    # layer 2: hidden to output
    logits = h1 @ W2 + b2       # linear transformation
    probs = softmax(logits)     # convert logits to probabilities
    cache = (X, z1, h1, logits, probs)  # store intermediate values for backprop
    return probs, cache

# ===== backward pass =====
def backward(cache, y_true):
    # Compute gradients for all parameters using backpropagation
    global W1, b1, W2, b2
    X, z1, h1, logits, probs = cache
    B = X.shape[0]  # batch size
    y_oh = one_hot(y_true, out_dim)  # one-hot encode labels

    # Gradient of loss w.r.t logits (softmax + cross-entropy)
    dlogits = (probs - y_oh) / B                        # (B, out_dim)
    # Gradients for output layer weights and bias
    gW2 = h1.T @ dlogits                                # (hidden, out_dim)
    gb2 = dlogits.sum(axis=0)                           # (out_dim,)

    # Backpropagate into hidden layer
    dh1 = dlogits @ W2.T                                # (B, hidden)
    dz1 = dh1 * relu_grad(z1)                           # (B, hidden)

    # Gradients for input layer weights and bias
    gW1 = X.T @ dz1                                     # (in_dim, hidden)
    gb1 = dz1.sum(axis=0)                               # (hidden,)

    return gW1, gb1, gW2, gb2

# ===== training loop =====
def iterate_minibatches(X, y, batch_size=64, shuffle=True):
    # Yield batches of data for training
    N = len(X)
    idx = np.arange(N)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]

lr = 0.1
weight_decay = 1e-4   # L2 regularization strength
epochs = 50
batch_size = 64

for epoch in range(1, epochs+1):
    # training
    train_losses = []
    train_accs = []
    for xb, yb in iterate_minibatches(X_train, y_train, batch_size):
        probs, cache = forward(xb)  # forward pass
        # loss + L2 regularization
        ce = cross_entropy(probs, one_hot(yb, out_dim))
        l2 = 0.5 * weight_decay * (np.sum(W1*W1) + np.sum(W2*W2))  # penalize large weights
        loss = ce + l2
        train_losses.append(loss)
        train_accs.append(accuracy(probs, yb))

        # backward
        gW1, gb1, gW2, gb2 = backward(cache, yb)
        # add L2 gradients (derivative of regularization term)
        gW1 += weight_decay * W1
        gW2 += weight_decay * W2

        # SGD step: update weights and biases
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2

    # validation
    probs_val, _ = forward(X_val)
    val_loss = cross_entropy(probs_val, one_hot(y_val, out_dim))
    val_acc = accuracy(probs_val, y_val)

    # Print progress every 5 epochs (and first epoch)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | "
              f"train_loss={np.mean(train_losses):.4f} "
              f"train_acc={np.mean(train_accs):.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

# quick sanity: final metrics
probs_tr, _ = forward(X_train)
probs_te, _ = forward(X_val)
print("Final Train Acc:", accuracy(probs_tr, y_train))
print("Final Val Acc  :", accuracy(probs_te,

X, y = make_blobs(n_per_class=400)  # create dataset

# train/val split (80% train, 20% validation)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val,   y_val   = X[split:], y[split:]

# ===== utilities =====
def one_hot(y, num_classes):
    # Convert integer labels to one-hot encoded vectors
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def relu(z):
    # ReLU activation: sets negative values to zero
    return np.maximum(0, z)

def relu_grad(z):
    # Gradient of ReLU: 1 for positive, 0 for negative
    return (z > 0).astype(z.dtype)

def softmax(logits):
    # Softmax activation: converts logits to probabilities
    # logits: (B, C) where B=batch size, C=number of classes
    z = logits - logits.max(axis=1, keepdims=True)  # numerical stability
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy(probs, y_true_oh, eps=1e-12):
    # Cross-entropy loss: measures difference between predicted and true distributions
    # probs: predicted probabilities, y_true_oh: one-hot true labels
    # eps: small value to avoid log(0)
    return -np.mean(np.sum(y_true_oh * np.log(probs + eps), axis=1))

def accuracy(probs, y_true):
    # Compute accuracy: fraction of correct predictions
    preds = probs.argmax(axis=1)
    return (preds == y_true).mean()

# ===== model params =====
in_dim = 2      # input dimension (2 features per sample)
hidden = 32     # number of hidden units
out_dim = 2     # output dimension (number of classes)

# Initialize weights and biases for both layers
W1 = rng.normal(0, 0.1, size=(in_dim, hidden)).astype(np.float32)  # input to hidden weights
b1 = np.zeros((hidden,), dtype=np.float32)                         # hidden layer bias
W2 = rng.normal(0, 0.1, size=(hidden, out_dim)).astype(np.float32) # hidden to output weights
b2 = np.zeros((out_dim,), dtype=np.float32)                        # output layer bias

# ===== forward pass =====
def forward(X):
    # Compute output of the network for input X
    # layer 1: input to hidden
    z1 = X @ W1 + b1            # linear transformation
    h1 = relu(z1)               # apply ReLU activation
    # layer 2: hidden to output
    logits = h1 @ W2 + b2       # linear transformation
    probs = softmax(logits)     # convert logits to probabilities
    cache = (X, z1, h1, logits, probs)  # store intermediate values for backprop
    return probs, cache

# ===== backward pass =====
def backward(cache, y_true):
    # Compute gradients for all parameters using backpropagation
    global W1, b1, W2, b2
    X, z1, h1, logits, probs = cache
    B = X.shape[0]  # batch size
    y_oh = one_hot(y_true, out_dim)  # one-hot encode labels

    # Gradient of loss w.r.t logits (softmax + cross-entropy)
    dlogits = (probs - y_oh) / B                        # (B, out_dim)
    # Gradients for output layer weights and bias
    gW2 = h1.T @ dlogits                                # (hidden, out_dim)
    gb2 = dlogits.sum(axis=0)                           # (out_dim,)

    # Backpropagate into hidden layer
    dh1 = dlogits @ W2.T                                # (B, hidden)
    dz1 = dh1 * relu_grad(z1)                           # (B, hidden)

    # Gradients for input layer weights and bias
    gW1 = X.T @ dz1                                     # (in_dim, hidden)
    gb1 = dz1.sum(axis=0)                               # (hidden,)

    return gW1, gb1, gW2, gb2

# ===== training loop =====
def iterate_minibatches(X, y, batch_size=64, shuffle=True):
    # Yield batches of data for training
    N = len(X)
    idx = np.arange(N)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]

lr = 0.1
weight_decay = 1e-4   # L2 regularization
epochs = 50
batch_size = 64

for epoch in range(1, epochs+1):
    # training
    train_losses = []
    train_accs = []
    for xb, yb in iterate_minibatches(X_train, y_train, batch_size):
        probs, cache = forward(xb)
        # loss + L2
        ce = cross_entropy(probs, one_hot(yb, out_dim))
        l2 = 0.5 * weight_decay * (np.sum(W1*W1) + np.sum(W2*W2))
        loss = ce + l2
        train_losses.append(loss)
        train_accs.append(accuracy(probs, yb))

        # backward
        gW1, gb1, gW2, gb2 = backward(cache, yb)
        # add L2 grads
        gW1 += weight_decay * W1
        gW2 += weight_decay * W2

        # SGD step
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2

    # validation
    probs_val, _ = forward(X_val)
    val_loss = cross_entropy(probs_val, one_hot(y_val, out_dim))
    val_acc = accuracy(probs_val, y_val)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | "
              f"train_loss={np.mean(train_losses):.4f} "
              f"train_acc={np.mean(train_accs):.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

# quick sanity: final metrics
probs_tr, _ = forward(X_train)
probs_te, _ = forward(X_val)
print("Final Train Acc:", accuracy(probs_tr, y_train))
print("Final Val Acc  :", accuracy(probs_te, y_val))
