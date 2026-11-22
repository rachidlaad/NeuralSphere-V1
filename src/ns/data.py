"""Dataset: sklearn digits, fixed split, pixel -> voltage encoding."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from . import config as C


def load_split():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # x_i = pixel_i / 16  (dimensionless, 0..1)
    return Xtr / 16.0, Xte / 16.0, ytr.astype(int), yte.astype(int)


def to_volts(x, vmax=C.VMAX):
    """V_i = Vmax * x_i   [V]"""
    return vmax * np.asarray(x)
