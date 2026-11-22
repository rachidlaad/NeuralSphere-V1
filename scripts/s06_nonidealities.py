"""Hardware non-ideality study. Every row below is a separate SPICE run on all 360 test
images with the realistic op-amp model (unless stated). Randomised conditions are
repeated with independent seeds.

Distributions
  device tolerance 'uniform t' : G_actual = G (1 + U(-t, +t))   per resistor, independent
  device tolerance 'gauss s'   : G_actual = G (1 + N(0, s))     per resistor, independent
  input noise s                : V_i -> V_i + N(0, s) per pixel per image (static sample)
  op-amp offset s              : VOS ~ N(0, s) for each of the 30 amplifiers
"""
import sys, csv
import numpy as np
from _common import C, stamp
from ns import spice
from ns.mapping import Mapping, quantize, device_errors
from ns.metrics import save_json

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
d = np.load(C.DATA / "digits_split.npz"); wts = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
base = Mapping(W := wts["W"], wts["b"], Xcal=Xtr)
Vte = C.VMAX * Xte
rows = []


def run(group, cond, trial, Gp=None, Gn=None, Rf=None, V=None, opamp="real", vos=None, vcc=C.VCC, vee=C.VEE):
    r = spice.run(base.Gp if Gp is None else Gp, base.Gn if Gn is None else Gn,
                  base.Rf if Rf is None else Rf, Vte if V is None else V,
                  opamp=opamp, vos=vos, vcc=vcc, vee=vee)
    acc = float((r["s"].argmax(1) == yte).mean())
    sat = float(np.mean(np.abs(np.concatenate([r["tp"], r["tn"], r["s"]], 1)) > (min(vcc, -vee) - 0.5)))
    P = spice.power(r)["p_total"].mean()
    rows.append(dict(group=group, condition=cond, trial=trial, accuracy=acc, correct=int(acc * 360 + 0.5),
                     frac_amp_outputs_near_rail=sat, mean_power_W=float(P)))
    print(group, cond, trial, acc, flush=True)


run("reference", "ideal op-amps, continuous G", 0, opamp="ideal")
run("reference", "real op-amps, continuous G", 0)
for bits in (8, 6, 5, 4, 3, 2):
    run("quantization", f"{bits}-bit", 0, Gp=quantize(base.Gp_target, bits), Gn=quantize(base.Gn_target, bits))
for dist in ("uniform", "gauss"):
    for tol in (0.01, 0.05, 0.10, 0.20):
        for t in range(TRIALS):
            rng = np.random.default_rng(1000 + t)
            run("device tolerance", f"{dist} {int(tol*100)}%", t,
                Gp=base.Gp * device_errors(base.Gp.shape, tol, dist, rng),
                Gn=base.Gn * device_errors(base.Gn.shape, tol, dist, rng))
for t in range(TRIALS):
    rng = np.random.default_rng(2000 + t)
    run("combined", "6-bit + gauss 5% + VOS 5 mV", t,
        Gp=quantize(base.Gp_target, 6) * device_errors(base.Gp.shape, 0.05, "gauss", rng),
        Gn=quantize(base.Gn_target, 6) * device_errors(base.Gn.shape, 0.05, "gauss", rng),
        vos=rng.normal(0, 5e-3, (10, 3)))
for sig in (0.005, 0.01, 0.02, 0.05, 0.1):
    for t in range(TRIALS):
        rng = np.random.default_rng(3000 + t)
        run("input noise", f"sigma {sig*1e3:.0f} mV", t, V=Vte + rng.normal(0, sig, Vte.shape))
for sig in (0.001, 0.005, 0.01, 0.03):
    for t in range(TRIALS):
        rng = np.random.default_rng(4000 + t)
        run("op-amp offset", f"sigma {sig*1e3:.0f} mV", t, vos=rng.normal(0, sig, (10, 3)))
for k in (2, 4, 8):
    run("saturation", f"Rf x{k} = {base.Rf*k:.0f} ohm", 0, Rf=base.Rf * k)
for vcc, vee in ((4.5, -4.5), (4.0, -4.0), (3.3, -3.3), (2.5, -2.5), (5.5, -5.5), (5.0, -4.0)):
    run("supply", f"+{vcc}/{vee} V", 0, vcc=vcc, vee=vee)

ref = rows[1]["accuracy"]
for r in rows:
    r["degradation_vs_real_reference_pct_points"] = 100 * (ref - r["accuracy"])
with open(C.RESULTS / "s06_nonidealities_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
summ = {}
for r in rows:
    key = (r["group"], r["condition"]); summ.setdefault(key, []).append(r["accuracy"])
table = [dict(group=g, condition=c, trials=len(a), mean_acc=float(np.mean(a)), std_acc=float(np.std(a)),
              min_acc=float(np.min(a)), degradation_pct_points=100 * (ref - float(np.mean(a))))
         for (g, c), a in summ.items()]
save_json(dict(stamp=stamp(), Rf=base.Rf, reference_accuracy_real=ref, table=table),
          C.RESULTS / "s06_nonidealities_summary.json")
