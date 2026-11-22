import json
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def evaluate(pred, y):
    pred = np.asarray(pred); y = np.asarray(y)
    cm = confusion_matrix(y, pred, labels=range(10))
    per = {str(d): float(cm[d, d] / cm[d].sum()) for d in range(10)}
    return dict(correct=int((pred == y).sum()), total=int(len(y)),
                accuracy=float((pred == y).mean()),
                macro_f1=float(f1_score(y, pred, average="macro")),
                per_digit_accuracy=per, confusion_matrix=cm.tolist())


def save_json(obj, path):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
