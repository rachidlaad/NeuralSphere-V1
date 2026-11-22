"""Stage D: externally (software-) trained weights, programmed as conductances,
inference EXECUTED by the SPICE circuit with the realistic op-amp macromodel.

Configurations (all on all 1437 train + 360 test images):
  D1  realistic op-amps, continuous (unquantized) conductances
  D2  realistic op-amps, 8-bit conductance levels
Power is computed from SPICE branch currents of every source.
"""
import numpy as np
from _common import C, stamp
from ns import spice
from ns.mapping import Mapping
from ns.metrics import evaluate, save_json

d = np.load(C.DATA / "digits_split.npz"); wts = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
W, b = wts["W"], wts["b"]
z_sw_te = Xte @ W.T + b

out = dict(stamp=stamp(), opamp_model="OPA_REAL AOL=1e5 GBW=1MHz HR=0.2V IQ=0.5mA ROUT=50ohm VOS=0",
           label="externally trained, circuit-executed inference")
for tag, bits in (("D1_real_continuous", None), ("D2_real_8bit", 8)):
    m = Mapping(W, b, bits=bits, Xcal=Xtr)
    m.to_json(C.WEIGHTS / f"conductances_{tag}.json")
    rte = spice.run(m.Gp, m.Gn, m.Rf, C.VMAX * Xte, opamp="real",
                    keep_netlist=C.GEN / f"full_64x10_{tag}_testimage0.cir")
    rtr = spice.run(m.Gp, m.Gn, m.Rf, C.VMAX * Xtr, opamp="real")
    P = spice.power(rte)
    pred = rte["s"].argmax(1)
    np.savez(C.RESULTS / f"s04_{tag}_outputs.npz", s_te=rte["s"], s_tr=rtr["s"], tp=rte["tp"], tn=rte["tn"],
             pred_te=pred, **{k: v for k, v in P.items()})
    out[tag] = dict(bits=bits, Rf=m.Rf, volts_per_logit=m.volts_per_logit,
                    train=evaluate(rtr["s"].argmax(1), ytr), test=evaluate(pred, yte),
                    disagreements_vs_software=int((pred != z_sw_te.argmax(1)).sum()),
                    max_abs_logit_err=float(np.abs(rte["s"] / m.volts_per_logit - z_sw_te).max()),
                    power_W={k: dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
                             for k, v in P.items()},
                    supply_current_A=dict(i_vcc_mean=float(-rte["i_vcc"].mean()),
                                          i_vee_mean=float(rte["i_vee"].mean())),
                    input_current_total_A_mean=float(-rte["i_in"].sum(1).mean()))
    print(tag, "test", out[tag]["test"]["correct"], "/ 360, train", out[tag]["train"]["correct"], "/ 1437,",
          "disagree", out[tag]["disagreements_vs_software"], "P_mean", out[tag]["power_W"]["p_total"]["mean"])
save_json(out, C.RESULTS / "s04_stageD_inference.json")
