"""Stage D with a VENDOR op-amp model: Texas Instruments OPAx196 (PSpice, unmodified).

Part 1 - bench validation of the vendor model in ngspice against the datasheet
         (SBOS869, table 6.7, VS = +/-4..18 V, typical values):
           IQ per amplifier            140 uA
           AOL (RL = 10 k, +/-18 V)    140 dB
           GBW                         2.5 MHz
           en @100 Hz / 1 kHz          18 / 15 nV/rtHz
           VOS                         +/-25 uV typ, +/-100 uV max
         Measured: IQ from i(VCC) of a buffer; VOS = output of a grounded-input buffer;
         AOL with an L/C DC servo (L = C = 1 GH/1 GF: DC loop closed, AC loop open);
         GBW = AOL(1 kHz) * 1 kHz (single-pole region); en from .noise of a buffer.

Part 2 - full 64x10 crossbar, all 30 amplifiers = OPA196, continuous conductances,
         all 1437 train + 360 test images, same programming (same R_f = 5.6 k) as s04.
         Solved as an image-index DC sweep (see spice.sweep_cards) because repeated
         independent .op solves of this model need gmin stepping each time.

Power bookkeeping (formula used for the hand check):
  P_quiescent = N_amp * IQ * (VCC - VEE) = 30 * 140 uA * 10 V = 42 mW
  P_load     ~= sum over TIAs of |I_column| * |VEE|   (TIA outputs are negative, so each TIA
               sinks its column current through its output stage into the -5 V rail)
"""
import re, subprocess, tempfile
from pathlib import Path
import numpy as np
from _common import C, stamp
from ns import spice
from ns.mapping import Mapping, input_vector
from ns.metrics import evaluate, save_json

DATASHEET = dict(source="TI SBOS869 table 6.7 (VS = +/-4 V..+/-18 V, typ; AOL at +/-18 V, RL = 10 k)",
                 IQ_A=140e-6, AOL_dB=140.0, GBW_Hz=2.5e6, en_100Hz=18e-9, en_1kHz=15e-9,
                 VOS_typ_V=25e-6, VOS_max_V=100e-6, IB_typ_A=5e-12, SR_V_per_us="7.5 rise / 5.5 fall")

# ---------------------------------------------------------------- part 1: bench
if not C.OPA196_LIB.exists():
    raise SystemExit(f"{C.OPA196_LIB} missing: run  powershell -ExecutionPolicy Bypass -File .\\tools\\get_opa196_model.ps1")
wd =Path(tempfile.mkdtemp(prefix="ns_opa196_bench_")); spice.prepare_workdir(wd, "opa196")
inc = [f'.include "{C.OPA196_LIB.as_posix()}"', f'.include "{(C.MODELS / "opa196_wrapper.lib").as_posix()}"']
bench = ["* OPA196 bench (pins: inp inn out vcc vee)"] + inc + [
    "VCC vcc 0 DC 5", "VEE vee 0 DC -5",
    "X1 0 o1 o1 vcc vee OPA196",                                   # buffer, input grounded
    "VAC a 0 DC 0 AC 1", "X2 a n2 o2 vcc vee OPA196", "L2 o2 n2 1G", "C2 n2 0 1G",   # open loop
    ".control", "set wr_singlescale", "op",
    f"wrdata {(wd / 'op.txt').as_posix()} v(o1) i(VCC) i(VEE)",
    "ac dec 20 0.1 100meg", f"wrdata {(wd / 'ac.txt').as_posix()} vdb(o2)",
    "quit", ".endc", ".end"]
(wd / "bench.cir").write_text("\n".join(bench) + "\n")
nz = ["* OPA196 input voltage noise (buffer)"] + inc + [
    "VCC vcc 0 DC 5", "VEE vee 0 DC -5", "VN p 0 DC 0 AC 1", "X3 p o o vcc vee OPA196",
    ".control", "set wr_singlescale", "noise v(o) VN dec 20 0.1 1meg", "setplot noise1",
    f"wrdata {(wd / 'en.txt').as_posix()} inoise_spectrum", "quit", ".endc", ".end"]
(wd / "noise.cir").write_text("\n".join(nz) + "\n")
for f in ("bench.cir", "noise.cir"):
    subprocess.run([C.ngspice_exe(), "-b", f], cwd=wd, capture_output=True, text=True, timeout=300)
op = np.concatenate([[0.0], np.loadtxt(wd / "op.txt").ravel()[-3:]])   # -> op[1]=v(o1), op[2]=i(VCC), op[3]=i(VEE)
ac = np.loadtxt(wd / "ac.txt"); en = np.loadtxt(wd / "en.txt")
f_ac, aol = ac[:, 0], ac[:, 1]
at = lambda f, x, y: float(np.interp(np.log10(f), np.log10(x), y))
n_amp_bench = 2                                   # X1 and X2 draw IQ from VCC
bench_res = dict(
    IQ_per_amp_A=float(-op[2] / n_amp_bench), VOS_buffer_V=float(op[1]),
    AOL_dc_dB=float(aol[0]), AOL_1kHz_dB=at(1e3, f_ac, aol),
    GBW_Hz=10 ** (at(1e3, f_ac, aol) / 20) * 1e3,
    unity_gain_crossing_Hz=float(10 ** np.interp(0, -aol, np.log10(f_ac))),
    en_10Hz=at(10, en[:, 0], en[:, 1]), en_100Hz=at(100, en[:, 0], en[:, 1]), en_1kHz=at(1e3, en[:, 0], en[:, 1]))
bench_res["formulas"] = dict(
    IQ=f"IQ = -i(VCC)/2 = {-op[2]*1e6:.2f} uA / 2 = {bench_res['IQ_per_amp_A']*1e6:.2f} uA",
    GBW=f"GBW = 10^(AOL(1kHz)/20) * 1 kHz = 10^({bench_res['AOL_1kHz_dB']:.2f}/20) * 1 kHz = {bench_res['GBW_Hz']/1e6:.3f} MHz")
print("bench:", {k: v for k, v in bench_res.items() if k != "formulas"})

# ---------------------------------------------------------------- part 2: crossbar
d = np.load(C.DATA / "digits_split.npz"); wts = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
W, b = wts["W"], wts["b"]
m = Mapping(W, b, Xcal=Xtr)
z_sw = Xte @ W.T + b
rte = spice.run(m.Gp, m.Gn, m.Rf, C.VMAX * Xte, opamp="opa196", chunk=30,
                keep_netlist=C.GEN / "full_64x10_OPA196_testimage0.cir")
rtr = spice.run(m.Gp, m.Gn, m.Rf, C.VMAX * Xtr, opamp="opa196", chunk=120)
P = spice.power(rte); pred = rte["s"].argmax(1)
ref = np.load(C.RESULTS / "s04_D1_real_continuous_outputs.npz")
S_k = m.ideal_scores(Xte)
I_col = (input_vector(Xte) @ m.Gp.T).sum(1) + (input_vector(Xte) @ m.Gn.T).sum(1)    # sum of 20 column currents
P_q = 30 * DATASHEET["IQ_A"] * (C.VCC - C.VEE)
P_hand = P_q + I_col * abs(C.VEE) + spice.power(rte)["p_inputs"] + spice.power(rte)["p_bias"]
np.savez(C.RESULTS / "s12_opa196_outputs.npz", s_te=rte["s"], s_tr=rtr["s"], tp=rte["tp"], tn=rte["tn"],
         pred_te=pred, **{k: v for k, v in P.items()})
gap = np.sort(rte["s"], 1)
out = dict(stamp=stamp(), model="TI OPAx196 Final 1.5 (SBOMAF9E.ZIP), unmodified, ngspice ngbehavior=ps",
           datasheet=DATASHEET, bench=bench_res, Rf=m.Rf, volts_per_logit=m.volts_per_logit,
           label="ELECTRICAL simulation; weights trained in software (externally trained, circuit-executed inference)",
           test=evaluate(pred, yte), train=evaluate(rtr["s"].argmax(1), ytr),
           disagreements_vs_software=int((pred != z_sw.argmax(1)).sum()),
           disagreements_vs_OPA_REAL_s04=int((pred != ref["pred_te"]).sum()),
           max_abs_S_minus_kirchhoff_V=float(np.abs(rte["s"] - S_k).max()),
           mean_S_minus_kirchhoff_V=float((rte["s"] - S_k).mean()),
           max_abs_S_minus_OPA_REAL_V=float(np.abs(rte["s"] - ref["s_te"]).max()),
           max_abs_logit_err=float(np.abs(rte["s"] / m.volts_per_logit - z_sw).max()),
           min_top2_gap_V=float((gap[:, -1] - gap[:, -2]).min()),
           tia_output_range_V=[float(min(rte["tp"].min(), rte["tn"].min())), float(max(rte["tp"].max(), rte["tn"].max()))],
           power_W={k: dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max())) for k, v in P.items()},
           power_hand_check=dict(
               formula="P ~= 30*IQ*(VCC-VEE) + |VEE|*sum(I_column) + P_inputs + P_bias",
               P_quiescent_W=P_q, mean_sum_column_current_A=float(I_col.mean()),
               P_hand_mean_W=float(P_hand.mean()), P_spice_mean_W=float(P["p_total"].mean())),
           power_OPA_REAL_s04_mean_W=float(ref["p_total"].mean()))
save_json(out, C.RESULTS / "s12_opa196_stageD.json")
print("OPA196 test", out["test"]["correct"], "/ 360, train", out["train"]["correct"], "/ 1437, disagree vs software",
      out["disagreements_vs_software"], "| max|S-Kirchhoff|", out["max_abs_S_minus_kirchhoff_V"],
      "| P mean", out["power_W"]["p_total"]["mean"], "W (hand", out["power_hand_check"]["P_hand_mean_W"], ")")
