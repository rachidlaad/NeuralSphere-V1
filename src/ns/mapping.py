"""Map signed software weights to physical conductance pairs and back.

Augmented weight matrix  Wa = [W | b']  (10 x 65); the 65th input is the bias rail
VBIAS, so b' = b * VMAX / VBIAS makes  sum_i Wa_ki V_i = VMAX * z_k.

  G+ = Gmin + alpha * max(w, 0)          [S]
  G- = Gmin + alpha * max(-w, 0)         [S]
  alpha = (Gmax - Gmin) / max|Wa|         [S per unit weight]

TIA + difference amplifier output:
  S_k = Rf * (I+_k - I-_k) = Rf * alpha * VMAX * z_k     [V]
"""
import json
import numpy as np
from . import config as C

E24 = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
                3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1])


def e24_floor(x):
    dec = 10 ** np.floor(np.log10(x))
    m = x / dec
    return float(E24[E24 <= m + 1e-12].max() * dec)


def augment(W, b):
    return np.hstack([W, (b * C.VMAX / C.VBIAS)[:, None]])


def input_vector(x):
    """64 pixel voltages + bias rail -> 65 node voltages [V]."""
    x = np.atleast_2d(x)
    return np.hstack([C.VMAX * x, np.full((x.shape[0], 1), C.VBIAS)])


class Mapping:
    """Holds the physical programming of one network instance."""

    def __init__(self, W, b, w_clip=None, bits=None, Rf=None, Xcal=None, i_target=3.5):
        self.Wa = augment(np.asarray(W, float), np.asarray(b, float))
        self.wmax = float(w_clip if w_clip else np.abs(self.Wa).max())
        self.alpha = (C.G_MAX - C.G_MIN) / self.wmax
        w = np.clip(self.Wa, -self.wmax, self.wmax)
        self.Gp_target = C.G_MIN + self.alpha * np.maximum(w, 0)
        self.Gn_target = C.G_MIN + self.alpha * np.maximum(-w, 0)
        self.bits = bits
        self.Gp = quantize(self.Gp_target, bits)
        self.Gn = quantize(self.Gn_target, bits)
        if Rf is None:
            # choose the largest E24 feedback resistor that keeps every TIA output
            # magnitude below i_target volts on the calibration (training) inputs
            V = input_vector(Xcal)
            imax = max((V @ self.Gp.T).max(), (V @ self.Gn.T).max())
            Rf = e24_floor(i_target / imax)
        self.Rf = float(Rf)

    @property
    def volts_per_logit(self):
        return self.Rf * self.alpha * C.VMAX

    def ideal_scores(self, x, Gp=None, Gn=None):
        """Kirchhoff/Ohm prediction of S_k for ideal amplifiers [V]."""
        Gp = self.Gp if Gp is None else Gp
        Gn = self.Gn if Gn is None else Gn
        V = input_vector(x)
        return self.Rf * (V @ Gp.T - V @ Gn.T)

    def effective_weights(self, Gp=None, Gn=None):
        Gp = self.Gp if Gp is None else Gp
        Gn = self.Gn if Gn is None else Gn
        return (Gp - Gn) / self.alpha

    def to_json(self, path, extra=None):
        d = dict(R_MIN=C.R_MIN, R_MAX=C.R_MAX, G_MIN=C.G_MIN, G_MAX=C.G_MAX, wmax=self.wmax,
                 alpha=self.alpha, Rf=self.Rf, bits=self.bits, VMAX=C.VMAX, VBIAS=C.VBIAS,
                 volts_per_logit=self.volts_per_logit,
                 Rp_ohm=(1 / self.Gp).tolist(), Rn_ohm=(1 / self.Gn).tolist())
        if extra:
            d.update(extra)
        with open(path, "w") as f:
            json.dump(d, f, indent=1)


def quantize(G, bits):
    """Snap conductances to 2**bits evenly spaced levels between Gmin and Gmax."""
    if not bits:
        return np.array(G, float)
    n = 2 ** bits - 1
    step = (C.G_MAX - C.G_MIN) / n
    return C.G_MIN + np.round((np.asarray(G) - C.G_MIN) / step) * step


def device_errors(shape, tol, dist, rng):
    """Multiplicative conductance error per device.

    dist='uniform' : G_actual = G * (1 + U(-tol, +tol))   (resistor tolerance band)
    dist='gauss'   : G_actual = G * (1 + N(0, tol))       (sigma = tol)
    """
    if tol == 0:
        return np.ones(shape)
    if dist == "uniform":
        return 1 + rng.uniform(-tol, tol, shape)
    return np.clip(1 + rng.normal(0, tol, shape), 0.05, None)
