"""Transient simulation: 6 test images presented one after another (PWL sources,
20 us each, 100 ns edges) to the realistic-op-amp circuit. Measures
  - output score waveforms and settling time (latency)
  - instantaneous power P(t) = sum over sources of V * I
  - energy per inference E = integral P dt over one image period
"""
import re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from _common import C, stamp
from ns import spice
from ns.mapping import Mapping
from ns.metrics import save_json

# usage: python s07_transient_energy.py [real|opa196]   (default real = original behaviour/file names)
AMP = sys.argv[1] if len(sys.argv) > 1 else "real"
SUF = "" if AMP == "real" else f"_{AMP}"
d = np.load(C.DATA / "digits_split.npz"); wts = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, yte = d["Xtr"], d["Xte"], d["yte"]
m = Mapping(wts["W"], wts["b"], Xcal=Xtr)
IDX = [0, 1, 2, 3, 4, 5]
T, TR = 20e-6, 100e-9
V = C.VMAX * Xte[IDX]

body = spice.circuit_cards(m.Gp, m.Gn, m.Rf, opamp=AMP, vos=np.zeros((10, 3)))
lines = []
for l in body:
    if l.startswith("VIN"):
        name = l.split()[0]; i = int(name[3:])
        pts = ["0", f"{V[0, i]:.6g}"]
        for n in range(1, len(IDX)):
            t0 = n * T
            pts += [f"{t0:.9g}", f"{V[n-1, i]:.6g}", f"{t0+TR:.9g}", f"{V[n, i]:.6g}"]
        l = f"{name} in{i} 0 PWL(" + " ".join(pts) + ")"
    lines.append(l)
out_file = (C.RESULTS / f"s07_transient{SUF}.txt").as_posix()
vecs = [f"v(s{k})" for k in range(10)] + ["i(VCC)", "i(VEE)", "i(VB)"] + \
       [f"i(VIN{i})" for i in range(64)] + [f"v(in{i})" for i in range(64)]
ctl = [".control", "set wr_singlescale", "option numdgt=8",
       f"tran 20n {len(IDX)*T:.9g} 0 20n", f"wrdata {out_file} " + " ".join(vecs), "quit", ".endc", ".end"]
net = C.GEN / f"transient_6images_{AMP}.cir"
net.write_text("\n".join(lines + ctl) + "\n")
SPLIT = AMP in spice.VENDOR_MODELS


def neuron_slice(k):
    """Neuron k's exact sub-circuit: its 130 synapses, 2 TIAs and difference amplifier, plus the shared sources."""
    pat = (r"^(RP|RN)(\d+)_\d+ ", r"^(XTP|XTN|RFP|RFN|XD)(\d+) ", r"^(RD\d)_(\d+) ")
    out = []
    for l in lines:
        mm = next((re.match(p_, l) for p_ in pat if re.match(p_, l)), None)
        if mm is None or int(mm.group(2)) == k:
            out.append(l)
    return out


def run_neuron(k):
    wdk = Path(tempfile.mkdtemp(prefix=f"ns_tran{k}_")); spice.prepare_workdir(wdk, AMP)
    vk = [f"v(s{k})", "i(VCC)", "i(VEE)", "i(VB)"] + [f"i(VIN{i})" for i in range(64)] + [f"v(in{i})" for i in range(64)]
    ck = [".control", "set wr_singlescale", "option numdgt=8", "save " + " ".join(vk),
          f"tran 20n {len(IDX)*T:.9g} 0 20n", "linearize " + " ".join(vk),      # common uniform 20 ns grid
          f"wrdata {(wdk / 'o.txt').as_posix()} " + " ".join(vk), "quit", ".endc", ".end"]
    (wdk / "n.cir").write_text("\n".join(neuron_slice(k) + ck) + "\n")
    pk = subprocess.run([C.ngspice_exe(), "-b", "n.cir"], capture_output=True, text=True, cwd=wdk)
    if not (wdk / "o.txt").exists():
        raise RuntimeError(f"neuron {k}: " + pk.stdout[-1500:] + pk.stderr[-1500:])
    return np.loadtxt(wdk / "o.txt")


if SPLIT:
    # Vendor models: the full 30-amplifier transient is impractically slow in ngspice (a 2 us slice did not finish in
    # 28 min, while one neuron simulates 2 us in 0.7 s). Because the 64 inputs, the bias rail and the +/-5 V supplies are
    # ideal voltage sources, the 10 neurons do not interact, so the full circuit equals the 10 neuron sub-circuits
    # simulated separately; each source's current is the sum of its currents into the 10 sub-circuits.
    with ThreadPoolExecutor(10) as ex:
        parts = list(ex.map(run_neuron, range(10)))
    n = min(len(q) for q in parts); parts = [q[:n] for q in parts]
    t = parts[0][:, 0]
    assert all(np.allclose(q[:, 0], t) for q in parts), "neuron time grids differ"
    cur = sum(q[:, 2:4 + 1 + 64] for q in parts)                                # i(VCC), i(VEE), i(VB), i(VIN0..63)
    cols = np.column_stack([np.column_stack([q[:, 1] for q in parts]), cur, parts[0][:, 5 + 64:5 + 128]])
    np.savetxt(out_file, np.column_stack([t, cols]), fmt="%.8e")
else:
    wd = tempfile.mkdtemp(prefix="ns_tran_"); spice.prepare_workdir(wd, AMP)
    p = subprocess.run([C.ngspice_exe(), "-b", str(net)], capture_output=True, text=True, cwd=wd)
    A = np.loadtxt(out_file)
    t = A[:, 0]; cols = A[:, 1:]
s = cols[:, :10]; ivcc, ivee, ivb = cols[:, 10], cols[:, 11], cols[:, 12]
iin = cols[:, 13:77]; vin = cols[:, 77:141]
P = -C.VCC * ivcc - C.VEE * ivee - C.VBIAS * ivb - (vin * iin).sum(1)
P_amp = -C.VCC * ivcc - C.VEE * ivee
P_xbar = -C.VBIAS * ivb - (vin * iin).sum(1)

per = []
for n in range(len(IDX)):
    a, bnd = n * T, (n + 1) * T
    sel = (t >= a) & (t <= bnd)
    E = np.trapezoid(P[sel], t[sel]); Ex = np.trapezoid(P_xbar[sel], t[sel])
    final = s[sel][-1]
    pred = int(final.argmax())
    # settling: last time any output is farther than 1% of full-scale (2 V) from its final value
    dev = np.abs(s[sel] - final).max(1)
    tl = t[sel][dev > 0.02]
    settle = float(tl[-1] - a) if len(tl) else 0.0
    # decision time: earliest time after which argmax stays equal to final prediction
    am = s[sel].argmax(1); bad = t[sel][am != pred]
    dec = float(bad[-1] - a) if len(bad) else 0.0
    per.append(dict(test_index=IDX[n], true=int(yte[IDX[n]]), predicted=pred, energy_J=float(E),
                    crossbar_energy_J=float(Ex), settling_time_s=settle, decision_time_s=dec,
                    final_scores_V=final.tolist()))
np.savez(C.RESULTS / f"s07_transient{SUF}.npz", t=t, s=s, P=P, P_amp=P_amp, P_xbar=P_xbar, vin=vin, idx=IDX, T=T)
res = dict(stamp=stamp(), opamp=AMP, image_period_s=T, per_image=per,
           method=("10 exact neuron sub-circuits simulated separately (ideal shared sources), resampled to a 20 ns grid "
                   "with ngspice linearize, source currents summed" if SPLIT else "full circuit in one ngspice run"),
           mean_power_W=float(P.mean()), mean_amp_power_W=float(P_amp.mean()),
           mean_crossbar_power_W=float(P_xbar.mean()),
           notes=["Energy is integrated over the full 20 us image period, so it scales with the period chosen.",
                  ("Amplifier quiescent current (30 x 0.5 mA from +/-5 V = 150 mW) dominates; a real design would "
                   "power-gate the amplifiers between inferences or use micro-power amplifiers." if AMP == "real" else
                   "Amplifier quiescent power 30 x 140 uA x 10 V = 42 mW (TI OPA196 model) plus the load current "
                   "the TIAs sink from the -5 V rail."),
                  "Not included: DACs that generate the input voltages, programming circuitry, ADC/readout, "
                  "wiring capacitance, leakage, clock/control logic."])
for r in per:
    print(r["test_index"], r["true"], r["predicted"], f"E={r['energy_J']*1e6:.3f} uJ",
          f"settle={r['settling_time_s']*1e6:.2f} us", f"decide={r['decision_time_s']*1e6:.2f} us")
save_json(res, C.RESULTS / f"s07_transient_energy{SUF}.json")
