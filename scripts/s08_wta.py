"""Analog winner-take-all: 45 complementary comparators + diode-AND gates, simulated in
SPICE together with the realistic-op-amp crossbar on all 360 test images.
The predicted digit is the WTA output line w_k that is HIGH (> 2.5 V); a sample is
counted 'ambiguous' if not exactly one line is high.
"""
import numpy as np
from _common import C, stamp
from ns import spice
from ns.mapping import Mapping
from ns.metrics import evaluate, save_json

d = np.load(C.DATA / "digits_split.npz"); wts = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, yte = d["Xtr"], d["Xte"], d["yte"]
m = Mapping(wts["W"], wts["b"], Xcal=Xtr)
r = spice.run(m.Gp, m.Gn, m.Rf, C.VMAX * Xte, opamp="real", wta=True,
              keep_netlist=C.GEN / "full_64x10_real_with_WTA_testimage0.cir")
w = r["w"]; high = w > 2.5
n_high = high.sum(1)
pred_wta = np.where(n_high == 1, w.argmax(1), -1)
pred_argmax = r["s"].argmax(1)
gap = np.sort(r["s"], 1)[:, -1] - np.sort(r["s"], 1)[:, -2]
P = spice.power(r)
np.savez(C.RESULTS / "s08_wta_outputs.npz", w=w, s=r["s"], pred_wta=pred_wta)
res = dict(stamp=stamp(), n_comparators=45, n_diodes=90,
           wta_test=evaluate(pred_wta, yte), ambiguous=int((n_high != 1).sum()),
           wta_vs_argmax_disagreements=int((pred_wta != pred_argmax).sum()),
           winner_line_V=dict(min=float(w[high].min()), mean=float(w[high].mean())),
           loser_line_V=dict(max=float(w[~high].max()), mean=float(w[~high].mean())),
           min_top2_gap_V=float(gap.min()), median_top2_gap_V=float(np.median(gap)),
           mean_total_power_W=float(P["p_total"].mean()))
print({k: v for k, v in res.items() if k != "wta_test"}, res["wta_test"]["correct"])
save_json(res, C.RESULTS / "s08_wta.json")
