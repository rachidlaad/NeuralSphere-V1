"""Stage A: one neuron = 4 signed synapses + bias, differential TIAs, difference amp.

Weights (dimensionless) w = [+1.0, -0.5, +0.25, -2.0], bias b = +0.3
Inputs  x = [1.0, 0.5, 0.25, 1.0]  ->  V = 1 V * x
Expected z = 1.0*1 - 0.5*0.5 + 0.25*0.25 - 2.0*1 + 0.3 = -0.8875
"""
import re, subprocess
import numpy as np
from _common import C, stamp
from ns import spice
from ns.metrics import save_json

w = np.array([[1.0, -0.5, 0.25, -2.0, 0.3]])          # last = bias weight (VBIAS = VMAX = 1 V)
x = np.array([1.0, 0.5, 0.25, 1.0])
wmax = np.abs(w).max()
alpha = (C.G_MAX - C.G_MIN) / wmax                     # S per unit weight
Gp = C.G_MIN + alpha * np.maximum(w, 0)
Gn = C.G_MIN + alpha * np.maximum(-w, 0)
Rf = 10e3
V = np.append(C.VMAX * x, C.VBIAS)

rows = []
for i in range(5):
    rows.append(dict(input=i, V=V[i], w=w[0, i], Rp=1 / Gp[0, i], Rn=1 / Gn[0, i],
                     Ip_uA=V[i] * Gp[0, i] * 1e6, In_uA=V[i] * Gn[0, i] * 1e6))
Ip = (V * Gp[0]).sum(); In = (V * Gn[0]).sum()
z = float(w[0, :4] @ x + w[0, 4])
S_hand = Rf * (Ip - In)

results = {}
for model in ("ideal", "real"):
    body = spice.circuit_cards(Gp, Gn, Rf, opamp=model, n_in=4,
                               vos=np.zeros((1, 3)) if model == "real" else None)
    ctl = [".control", "op"] + [f"alter VIN{i} dc = {V[i]:.9g}" for i in range(4)] + [
        "op", "print v(s0) v(tp0) v(tn0) v(cp0) v(cn0) i(VCC) i(VEE)",
        "print " + " ".join(f"i(VIN{i})" for i in range(4)) + " i(VB)", "quit", ".endc", ".end"]
    net = C.GEN / f"stageA_single_neuron_{model}.cir"
    net.write_text("\n".join(body + ctl) + "\n")
    out = subprocess.run([C.ngspice_exe(), "-b", str(net)], capture_output=True, text=True).stdout
    vals = {m.group(1).lower(): float(m.group(2)) for m in
            re.finditer(r"^\s*([vi]\([^)]+\))\s*=\s*([-+0-9.eE]+)", out, re.M)}
    results[model] = vals

res = dict(stamp=stamp(), weights=w.tolist(), x=x.tolist(), alpha_S_per_unit=alpha, Rf=Rf,
           synapses=rows, I_plus_A=Ip, I_minus_A=In, z_math=z,
           volts_per_logit=Rf * alpha * C.VMAX, S_hand_V=S_hand, spice=results,
           S_spice_ideal=results["ideal"]["v(s0)"], S_spice_real=results["real"]["v(s0)"],
           tp_hand=-Rf * Ip, tn_hand=-Rf * In)
res["err_ideal_V"] = res["S_spice_ideal"] - S_hand
res["err_real_V"] = res["S_spice_real"] - S_hand
for r in rows:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print(f"I+ = {Ip*1e6:.4f} uA, I- = {In*1e6:.4f} uA, z = {z}, S_hand = {S_hand:.6f} V")
print("SPICE ideal:", results["ideal"])
print("SPICE real :", results["real"])
save_json(res, C.RESULTS / "s01_stageA_single_neuron.json")
