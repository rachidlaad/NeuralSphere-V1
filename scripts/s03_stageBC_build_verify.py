"""Stages B + C: build the full 64x10 differential crossbar netlist, run it with ideal
amplifiers on every train and test image, and check correspondence with
(1) Kirchhoff/Ohm hand-math on the SAME conductances and (2) the software logits.
Also performs a node-level KCL audit on one image.
"""
import numpy as np, re, subprocess
from _common import C, stamp
from ns import spice
from ns.mapping import Mapping, input_vector
from ns.metrics import save_json, evaluate

d = np.load(C.DATA / "digits_split.npz"); wts = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
W, b = wts["W"], wts["b"]

m = Mapping(W, b, Xcal=Xtr)          # unquantized ("analog-ideal") programming
m.to_json(C.WEIGHTS / "conductances_ideal.json")
print(f"alpha = {m.alpha*1e6:.4f} uS/unit, Rf = {m.Rf:.0f} ohm, volts/logit = {m.volts_per_logit:.4f} V")

# --- full netlist file for inspection (first test image applied)
res_te = spice.run(m.Gp, m.Gn, m.Rf, C.VMAX * Xte, opamp="ideal",
                   keep_netlist=C.GEN / "full_64x10_ideal_testimage0.cir")
res_tr = spice.run(m.Gp, m.Gn, m.Rf, C.VMAX * Xtr, opamp="ideal")
np.savez(C.RESULTS / "s03_spice_ideal_outputs.npz", s_te=res_te["s"], s_tr=res_tr["s"],
         tp_te=res_te["tp"], tn_te=res_te["tn"])

S_kcl = m.ideal_scores(Xte)                       # Rf * sum (G+ - G-) V
z_sw = Xte @ W.T + b                              # software logits
S_sw = m.volts_per_logit * z_sw
err_kcl = res_te["s"] - S_kcl
err_sw = res_te["s"] - S_sw
pred_spice = res_te["s"].argmax(1); pred_sw = z_sw.argmax(1)

# --- KCL audit on test image 0, output 0: currents into column node cp0
V = input_vector(Xte[:1])[0]
I_branch = V * m.Gp[0]                            # Ohm: I = G * (V_in - V_cp), V_cp ~ 0
audit = dict(sum_I_plus_branches_A=float(I_branch.sum()),
             I_through_Rf_A=float(-res_te["tp"][0, 0] / m.Rf),
             tp0_spice_V=float(res_te["tp"][0, 0]), tp0_hand_V=float(-m.Rf * I_branch.sum()))

# resistor values in the netlist equal 1/G
txt = (C.GEN / "full_64x10_ideal_testimage0.cir").read_text()
Rnet = {mm.group(1): float(mm.group(2)) for mm in re.finditer(r"^(R[PN]\d+_\d+) \S+ \S+ (\S+)", txt, re.M)}
rel = max(abs(Rnet[f"RP{k}_{i}"] * m.Gp[k, i] - 1) for k in range(10) for i in range(65))

out = dict(stamp=stamp(), Rf=m.Rf, alpha=m.alpha, volts_per_logit=m.volts_per_logit,
           n_resistors_synapse=len(Rnet), n_opamps=30, max_rel_err_R_vs_1_over_G=rel,
           max_abs_err_spice_vs_kcl_V=float(np.abs(err_kcl).max()),
           max_abs_err_spice_vs_software_V=float(np.abs(err_sw).max()),
           max_abs_err_in_logits=float(np.abs(err_sw).max() / m.volts_per_logit),
           prediction_disagreements_spice_vs_software=int((pred_spice != pred_sw).sum()),
           tia_output_range_V=[float(min(res_tr["tp"].min(), res_tr["tn"].min())),
                               float(max(res_tr["tp"].max(), res_tr["tn"].max()))],
           score_range_V=[float(res_tr["s"].min()), float(res_tr["s"].max())],
           kcl_audit=audit, test_ideal=evaluate(pred_spice, yte), train_ideal=evaluate(res_tr["s"].argmax(1), ytr))
print({k: v for k, v in out.items() if k not in ("test_ideal", "train_ideal")})
print("ideal-amp SPICE test acc:", out["test_ideal"]["correct"], "/ 360")
save_json(out, C.RESULTS / "s03_stageBC_verification.json")
