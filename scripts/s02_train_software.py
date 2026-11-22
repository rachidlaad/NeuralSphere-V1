"""Software reference models (no circuit involved).

1. sklearn LogisticRegression(C=1) baseline on x = pixel/16.
2. Our own softmax regression trained with the explicit gradient (p_k - 1[k=t]) x_i.
   Hyper-parameters (lr=0.5, batch=64, 200 epochs, L2=1e-3) were chosen on a validation
   split carved out of the TRAINING set (see docs/DESIGN.md); the test set is not used here
   except for the final report line.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from _common import C, stamp
from ns.data import load_split
from ns.train import train_softmax, loss_acc
from ns.metrics import evaluate, save_json

Xtr, Xte, ytr, yte = load_split()
np.savez(C.DATA / "digits_split.npz", Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)

lr_model = LogisticRegression(C=1.0, max_iter=10000).fit(Xtr, ytr)
base = dict(train=evaluate(lr_model.predict(Xtr), ytr), test=evaluate(lr_model.predict(Xte), yte))
print("sklearn baseline test:", base["test"]["correct"], "/", base["test"]["total"])

W, b, hist = train_softmax(Xtr, ytr, epochs=200, lr=0.5, batch=64, l2=1e-3, seed=0)
zt = Xte @ W.T + b
ours = dict(train=evaluate((Xtr @ W.T + b).argmax(1), ytr), test=evaluate(zt.argmax(1), yte),
            test_loss=loss_acc(zt, yte)[0])
print("softmax-SGD test:", ours["test"]["correct"], "/", ours["test"]["total"])

np.savez(C.WEIGHTS / "software_weights.npz", W=W, b=b)
save_json(hist, C.RESULTS / "s02_training_history.json")
save_json(dict(stamp=stamp(), sklearn_logreg_C1=base, softmax_sgd=ours,
               hyperparameters=dict(lr=0.5, batch=64, epochs=200, l2=1e-3, seed=0),
               max_abs_w=float(np.abs(W).max()), max_abs_b=float(np.abs(b).max())),
          C.RESULTS / "s02_software_reference.json")
