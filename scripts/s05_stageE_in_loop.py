"""Stage E: SPICE-in-the-loop fine-tuning ("chip-in-the-loop" training).

A fixed "chip instance" is created: every synapse conductance has a persistent
multiplicative device error, conductances are limited to 2**bits levels, and every
op-amp has a random input offset.  The software weights are first programmed into
this chip ("ex-situ"), then fine-tuned:

  repeat N iterations:
    1. SPICE solves the chip for every TRAINING image -> measured S_k [V]
    2. z_meas = S_k / (volts per logit);  p = softmax(z_meas)
    3. gradient  dL/dw_ki = mean over images of (p_k - 1[k=t]) x_i  (+ L2)
       -- computed from the MEASURED circuit outputs, so it sees the chip's errors
    4. update shadow weights, re-program conductances (quantize, same device errors)

The gradient arithmetic is done by the software controller (as in published
chip-in-the-loop work); the forward pass is the SPICE circuit.  The test set is
evaluated only before and after fine-tuning and is never used for any decision.
"""
import sys, json
import numpy as np
from _common import C, stamp
from ns import spice
from ns.mapping import Mapping, quantize, device_errors
from ns.train import softmax, onehot
from ns.metrics import evaluate, save_json

CHIPS = {
    "chipA_6bit_gauss5pct_vos5mV": dict(bits=6, tol=0.05, dist="gauss", vos_sigma=5e-3, seed=11),
    "chipB_4bit_gauss10pct_vos10mV": dict(bits=4, tol=0.10, dist="gauss", vos_sigma=10e-3, seed=22),
    "chipC_2bit_gauss20pct_vos20mV": dict(bits=2, tol=0.20, dist="gauss", vos_sigma=20e-3, seed=33),
}
if len(sys.argv) > 2:
    CHIPS = {k: v for k, v in CHIPS.items() if k in sys.argv[2].split(",")}
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
L2 = 1e-3

d = np.load(C.DATA / "digits_split.npz"); wts = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
W0, b0 = wts["W"], wts["b"]
Y = onehot(ytr)
out_json = C.RESULTS / "s05_stageE_in_loop.json"
summary = json.load(open(out_json)) if out_json.exists() else dict(chips={})
summary.update(stamp=stamp(), iterations=ITERS, lr=LR, l2=L2)

for name, cfg in CHIPS.items():
    rng = np.random.default_rng(cfg["seed"])
    base = Mapping(W0, b0, Xcal=Xtr)                       # fixes alpha, wmax, Rf
    Ep = device_errors(base.Gp.shape, cfg["tol"], cfg["dist"], rng)
    En = device_errors(base.Gn.shape, cfg["tol"], cfg["dist"], rng)
    vos = rng.normal(0, cfg["vos_sigma"], (10, 3))
    vpl = base.volts_per_logit

    def program(Wa):
        w = np.clip(Wa, -base.wmax, base.wmax)
        Gp = quantize(C.G_MIN + base.alpha * np.maximum(w, 0), cfg["bits"]) * Ep
        Gn = quantize(C.G_MIN + base.alpha * np.maximum(-w, 0), cfg["bits"]) * En
        return Gp, Gn

    def measure(Gp, Gn, X):
        return spice.run(Gp, Gn, base.Rf, C.VMAX * X, opamp="real", vos=vos)["s"]

    Wa = base.Wa.copy()                                    # shadow weights (bias col scaled by VMAX/VBIAS = 1)
    Gp, Gn = program(Wa)
    s_te0 = measure(Gp, Gn, Xte)
    hist = []
    best = (np.inf, None, None, None, -1)        # (measured TRAIN loss, Wa, Gp, Gn, iteration)
    for it in range(ITERS + 1):
        s_tr = measure(Gp, Gn, Xtr)
        z = s_tr / vpl
        p = softmax(z)
        loss = float(-np.log(p[np.arange(len(ytr)), ytr] + 1e-12).mean())
        acc = float((z.argmax(1) == ytr).mean())
        hist.append(dict(iteration=it, train_loss_measured=loss, train_acc_measured=acc))
        if loss < best[0]:
            best = (loss, Wa.copy(), Gp.copy(), Gn.copy(), it)
        print(name, it, round(loss, 4), round(acc, 4), flush=True)
        if it == ITERS:
            break
        G = p - Y
        Xa = np.hstack([Xtr, np.full((len(Xtr), 1), C.VBIAS / C.VMAX)])
        grad = G.T @ Xa / len(Xtr) + L2 * np.hstack([Wa[:, :64], np.zeros((10, 1))])
        Wa = Wa - LR * grad
        Gp, Gn = program(Wa)
    s_te1 = measure(Gp, Gn, Xte)                       # state after the last iteration
    s_te_best = measure(best[2], best[3], Xte)         # state with lowest MEASURED TRAINING loss
    np.savez(C.WEIGHTS / f"in_loop_{name}_lr{LR:g}.npz", Wa_final=Wa, Gp=Gp, Gn=Gn, Ep=Ep, En=En, vos=vos, Rf=base.Rf)
    np.savez(C.RESULTS / f"s05_{name}_lr{LR:g}_outputs.npz", s_te_before=s_te0, s_te_after=s_te1, s_te_best_train=s_te_best)
    key = name if LR == 2.0 else f"{name}_lr{LR:g}"
    summary["chips"][key] = dict(config=cfg, lr=LR, Rf=base.Rf, history=hist,
                                 test_before=evaluate(s_te0.argmax(1), yte),
                                 test_after=evaluate(s_te1.argmax(1), yte),
                                 best_train_loss_iteration=best[4],
                                 test_at_best_train_loss=evaluate(s_te_best.argmax(1), yte))
    print(key, "TEST before", summary["chips"][key]["test_before"]["correct"],
          "after", summary["chips"][key]["test_after"]["correct"],
          "at best-train-loss iter", best[4], summary["chips"][key]["test_at_best_train_loss"]["correct"], flush=True)
    save_json(summary, out_json)
