"""Multiclass softmax regression trained with the explicit gradient

    z_k = sum_i w_ki x_i + b_k
    p_k = exp(z_k) / sum_j exp(z_j)
    L   = -log p_target                         (+ optional L2 penalty)
    dL/dw_ki = (p_k - 1[k=target]) x_i
    w <- w - lr * dL/dw
"""
import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def onehot(y, n=10):
    Y = np.zeros((len(y), n))
    Y[np.arange(len(y)), y] = 1
    return Y


def loss_acc(z, y):
    p = softmax(z)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()), float((z.argmax(1) == y).mean())


def train_softmax(X, y, epochs=300, lr=0.5, batch=64, l2=1e-3, seed=0, Xval=None, yval=None):
    """Mini-batch SGD. Returns W (10,64), b (10,), history list of dicts."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = np.zeros((10, d)); b = np.zeros(10)
    Y = onehot(y)
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        for a in range(0, n, batch):
            j = idx[a:a + batch]
            p = softmax(X[j] @ W.T + b)
            G = p - Y[j]                              # (p_k - 1[k=target])
            gW = G.T @ X[j] / len(j) + l2 * W         # dL/dw_ki averaged over batch
            gb = G.mean(0)
            W -= lr * gW; b -= lr * gb
        l, a_ = loss_acc(X @ W.T + b, y)
        rec = dict(epoch=ep + 1, train_loss=l, train_acc=a_)
        if Xval is not None:
            lv, av = loss_acc(Xval @ W.T + b, yval)
            rec.update(val_loss=lv, val_acc=av)
        hist.append(rec)
    return W, b, hist
