"""Electrical noise analysis of the 64x10 crossbar with TI OPA196 amplifiers (ngspice .noise).

For each output S_k a small-signal noise analysis is run around the DC operating point of
test image 0 (0.1 Hz .. 100 MHz, 20 points/decade), in three variants:
  A  all noise sources on: OPA196 en/in (vendor model) + thermal noise of every resistor
  B  as A, but the 1300 synapse resistors noiseless (instance parameter noisy=0)
  C  as B, and also R_f and the 4 difference-amplifier resistors noiseless -> op-amp noise only
Contributions (uncorrelated sources add in power):
  synapse thermal  = A - B ,  R_f + diff-amp resistors = B - C ,  op-amps = C        [V^2]

Hand calculation in the flat (white) region, f = 1 kHz, T = 300.15 K (ngspice 27 C):
  column total conductance         G_sum = sum_i G_ki          (65 synapses; source ends are AC ground)
  TIA noise gain                   NG = 1 + R_f * G_sum
  TIA output density               e_t^2 = 4kT R_f + 4kT G_sum R_f^2 + e_n^2 NG^2 + i_n^2 R_f^2
  diff-amp (4 x R_d) own noise     e_d^2 = 4kT R_d (RD1) + 4kT R_d (RD2) + 4kT (R_d/2) * 2^2 (RD3||RD4) + e_n^2 * 2^2
                                         = 16 kT R_d + 4 e_n^2
  output density                   e_S^2 = e_tp^2 + e_tn^2 + e_d^2
Integrated (single-pole approximation, ENBW = pi/2 * f_-3dB):
  sigma_S^2 ~= (e_tp^2 + e_tn^2) * pi/2 * GBW/NG  +  e_d^2 * pi/2 * GBW/2
SPICE rms noise is obtained by integrating the ngspice density spectrum ourselves,
  sigma^2(band) = integral e_S(f)^2 |H(f)|^2 df   (trapezoid on the log grid),
for explicit readout bandwidths: brick-wall 0.1 Hz..f_max, and a first-order readout filter
|H|^2 = 1/(1+(f/f_c)^2). The answer depends on the readout bandwidth because the vendor model's
output noise does not roll off above ~3 MHz (flat ~80 nV/rtHz plateau, op-amp-only variant).
(ngspice's own onoise_total vector was not usable here: wrdata returned 1.4e9 for every run.)

Classification risk (Gaussian noise, independent outputs, union bound) for every test image:
  P_err <= sum_{j != winner} Q( (S_w - S_j) / sqrt(sigma_w^2 + sigma_j^2) ),  Q(x) = erfc(x/sqrt2)/2
using the OPA196 test outputs S from s12 and sigma_k from variant A.
"""
import json, re, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from math import erfc, sqrt, pi
from pathlib import Path
import numpy as np
from _common import C, stamp
from ns import spice
from ns.mapping import Mapping, input_vector
from ns.metrics import save_json

IMG = 0
k_B, T = 1.380649e-23, 300.15
d = np.load(C.DATA / "digits_split.npz"); wts = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, yte = d["Xtr"], d["Xte"], d["yte"]
m = Mapping(wts["W"], wts["b"], Xcal=Xtr)
V = C.VMAX * Xte[IMG]
s12 = json.load(open(C.RESULTS / "s12_opa196_stageD.json"))
en_model = s12["bench"]["en_1kHz"]; GBW_model = s12["bench"]["GBW_Hz"]
in_ds = 1.5e-15                                                   # A/rtHz, datasheet

base = spice.circuit_cards(m.Gp, m.Gn, m.Rf, opamp="opa196", vos=np.zeros((10, 3)))
def set_input(l):                                                 # apply test image IMG as DC values
    mm = re.match(r"^VIN(\d+) in\d+ 0 DC 0$", l)
    if l.startswith("VB "):                                       # .noise needs an AC reference source; it only
        return l + " AC 1"                                        # scales input-referred noise (unused), not onoise
    return f"VIN{mm.group(1)} in{mm.group(1)} 0 DC {V[int(mm.group(1))]:.9g}" if mm else l


base = [set_input(l) for l in base]


def variant(lines, v):
    out = []
    for l in lines:
        quiet = (v in "BC" and re.match(r"^R[PN]\d+_\d+ ", l)) or (v == "C" and re.match(r"^R(F[PN]\d+|D\d_\d+) ", l))
        out.append(l + " noisy=0" if quiet else l)
    return out


wd = Path(tempfile.mkdtemp(prefix="ns_noise_"))


def job(arg):
    v, k = arg
    sub = wd / f"{v}{k}"; sub.mkdir(); spice.prepare_workdir(sub, "opa196")
    ctl = [".control", "set wr_singlescale", "option numdgt=8",
           f"noise v(s{k}) VB dec 20 0.1 100meg", "setplot noise1",
           f"wrdata {(sub / 'spec.txt').as_posix()} onoise_spectrum", "quit", ".endc", ".end"]
    (sub / "n.cir").write_text("\n".join(variant(base, v) + ctl) + "\n")
    p = subprocess.run([C.ngspice_exe(), "-b", "n.cir"], cwd=sub, capture_output=True, text=True, timeout=3600)
    if not (sub / "spec.txt").exists():
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    return v, k, np.loadtxt(sub / "spec.txt")


with ThreadPoolExecutor(12) as ex:
    runs = list(ex.map(job, [(v, k) for v in "ABC" for k in range(10)]))
R = {(v, k): spec for v, k, spec in runs}
f = R[("A", 0)][:, 0]
dens = {v: np.array([R[(v, k)][:, 1] for k in range(10)]) for v in "ABC"}           # V/rtHz


def rms(e, fmax=None, fc=None):                                   # integrate e(f)^2 |H|^2 df
    H2 = np.ones_like(f) if fc is None else 1 / (1 + (f / fc) ** 2)
    sel = f <= (fmax or f[-1])
    return np.sqrt(np.trapezoid((e ** 2 * H2)[..., sel], f[sel], axis=-1))


BANDS = {"brickwall 0.1 Hz-10 kHz": dict(fmax=1e4), "brickwall 0.1 Hz-100 kHz": dict(fmax=1e5),
         "brickwall 0.1 Hz-1 MHz": dict(fmax=1e6), "brickwall 0.1 Hz-10 MHz": dict(fmax=1e7),
         "brickwall 0.1 Hz-100 MHz (worst case, whole simulated band)": dict(fmax=1e8),
         "1st-order readout filter f_c = 100 kHz": dict(fc=1e5), "1st-order readout filter f_c = 1 MHz": dict(fc=1e6)}
PRIMARY = "1st-order readout filter f_c = 1 MHz"                  # settles in 5 tau = 0.8 us << 20 us image period
WORST = "brickwall 0.1 Hz-100 MHz (worst case, whole simulated band)"
sigma = {b: {v: rms(dens[v], **kw) for v in "ABC"} for b, kw in BANDS.items()}
tot = sigma[PRIMARY]                                              # V rms per output, per variant
at = lambda y, f0: float(10 ** np.interp(np.log10(f0), np.log10(f), np.log10(y)))

# ---- hand calculation per output
fourkT = 4 * k_B * T
hand = []
for k in range(10):
    gp, gn = m.Gp[k].sum(), m.Gn[k].sum()
    et = lambda g, en: fourkT * m.Rf + fourkT * g * m.Rf ** 2 + (en * (1 + m.Rf * g)) ** 2 + (in_ds * m.Rf) ** 2
    ed2 = 16 * k_B * T * C.R_DIFF + 4 * en_model ** 2
    eS = sqrt(et(gp, en_model) + et(gn, en_model) + ed2)
    sig = sqrt(et(gp, en_model) * pi / 2 * GBW_model / (1 + m.Rf * gp) + et(gn, en_model) * pi / 2 * GBW_model / (1 + m.Rf * gn)
               + ed2 * pi / 2 * GBW_model / 2)
    hand.append(dict(k=k, G_sum_plus_S=gp, G_sum_minus_S=gn, noise_gain_plus=1 + m.Rf * gp, noise_gain_minus=1 + m.Rf * gn,
                     e_S_1kHz_hand=eS, e_S_1kHz_spice=at(dens["A"][k], 1e3),
                     sigma_S_hand_singlepole=sig, sigma_S_spice_brickwall_1MHz=float(sigma["brickwall 0.1 Hz-1 MHz"]["A"][k])))
k0 = hand[0]; gp0 = k0["G_sum_plus_S"]
worked = {
    "4kT": f"4 k T = 4 * 1.380649e-23 J/K * 300.15 K = {fourkT:.4e} J",
    "R_f thermal": f"sqrt(4kT R_f) = sqrt({fourkT:.4e} * {m.Rf:.0f}) = {sqrt(fourkT*m.Rf)*1e9:.2f} nV/rtHz",
    "synapse thermal (k=0, + column)": f"R_f sqrt(4kT G_sum) = {m.Rf:.0f} * sqrt({fourkT:.4e} * {gp0*1e3:.4f} mS) = {m.Rf*sqrt(fourkT*gp0)*1e9:.2f} nV/rtHz",
    "op-amp e_n x noise gain (k=0, +)": f"e_n (1 + R_f G_sum) = {en_model*1e9:.2f} nV * (1 + {m.Rf:.0f} * {gp0*1e3:.4f} mS) = {en_model*1e9:.2f} * {1+m.Rf*gp0:.3f} = {en_model*(1+m.Rf*gp0)*1e9:.2f} nV/rtHz",
    "diff-amp own noise": f"sqrt(16 kT R_d + 4 e_n^2) = sqrt({16*k_B*T*C.R_DIFF:.4e} + 4*({en_model*1e9:.2f} nV)^2) = {sqrt(16*k_B*T*C.R_DIFF + 4*en_model**2)*1e9:.2f} nV/rtHz",
    "e_S(1 kHz), k=0": f"hand {k0['e_S_1kHz_hand']*1e9:.1f} nV/rtHz vs ngspice {k0['e_S_1kHz_spice']*1e9:.1f} nV/rtHz",
    "sigma_S, k=0": f"hand (single-pole, white noise only) {k0['sigma_S_hand_singlepole']*1e6:.1f} uV rms vs ngspice spectrum integrated 0.1 Hz-1 MHz {k0['sigma_S_spice_brickwall_1MHz']*1e6:.1f} uV rms (includes 1/f region and the peaking near 1 MHz)"}

# ---- contributions (power split)
contrib = dict(synapse_thermal_V2=(tot["A"] ** 2 - tot["B"] ** 2).tolist(),
               Rf_and_diffamp_resistors_V2=(tot["B"] ** 2 - tot["C"] ** 2).tolist(),
               opamps_V2=(tot["C"] ** 2).tolist())
share = {kk: float(np.mean(np.array(v) / tot["A"] ** 2)) for kk, v in contrib.items()}

# ---- classification risk on the 360 OPA196 test outputs, for every readout bandwidth
S = np.load(C.RESULTS / "s12_opa196_outputs.npz")["s_te"]
Q = lambda x: 0.5 * erfc(x / sqrt(2))


def risk_for(sig):
    perr, zmin, kr = [], [], []
    for s in S:
        w = int(s.argmax()); others = [j for j in range(10) if j != w]
        zz = [(s[w] - s[j]) / sqrt(sig[w] ** 2 + sig[j] ** 2) for j in others]
        zmin.append(min(zz)); perr.append(sum(Q(z) for z in zz)); kr.append((w, others[int(np.argmin(zz))]))
    perr = np.array(perr); zmin = np.array(zmin); iw = int(zmin.argmin()); w, r = kr[iw]
    dS, sd = S[iw, w] - S[iw, r], sqrt(sig[w] ** 2 + sig[r] ** 2)
    return dict(worst_image=iw, winner=w, runner_up=r, worst_margin_V=float(dS), worst_margin_in_sigma=float(zmin[iw]),
                worst_P_err_union_bound=float(perr[iw]), expected_noise_flips_over_360=float(perr.sum()),
                formula=f"z = (S_{w} - S_{r}) / sqrt(sig_{w}^2 + sig_{r}^2) = {dS*1e3:.3f} mV / sqrt(({sig[w]*1e6:.1f} uV)^2 + "
                        f"({sig[r]*1e6:.1f} uV)^2) = {dS*1e3:.3f} mV / {sd*1e6:.1f} uV = {zmin[iw]:.2f};  Q(z) = {Q(zmin[iw]):.3g}")


risk = {b: risk_for(sigma[b]["A"]) for b in BANDS}
sig = tot["A"]
np.savez(C.RESULTS / "s13_noise_spectra.npz", f=f, A=dens["A"], B=dens["B"], C=dens["C"],
         **{f"sigma_{i}_{v}": sigma[b][v] for i, b in enumerate(BANDS) for v in "ABC"}, band_names=np.array(list(BANDS)))
res = dict(stamp=stamp(), label="ELECTRICAL small-signal noise simulation (ngspice .noise), TI OPA196 vendor model",
           operating_point=f"test image {IMG}", band_Hz=[0.1, 100e6], temperature_K=T,
           en_model_1kHz=en_model, en_datasheet_1kHz=15e-9, GBW_model_Hz=GBW_model,
           primary_band=PRIMARY, sigma_S_V=tot["A"].tolist(), sigma_S_logit_units=(tot["A"] / m.volts_per_logit).tolist(),
           sigma_S_V_by_band={b: sigma[b]["A"].tolist() for b in BANDS},
           sigma_S0_variants_by_band_V={b: {v: float(sigma[b][v][0]) for v in "ABC"} for b in BANDS},
           density_S0_V_per_rtHz={f"{x:g} Hz": at(dens["A"][0], x) for x in (0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)},
           density_1kHz_V_per_rtHz=[at(dens["A"][k], 1e3) for k in range(10)],
           density_10Hz_V_per_rtHz=[at(dens["A"][k], 10) for k in range(10)],
           contributions=contrib, mean_power_share=share, hand=hand, worked_example=worked, classification_risk=risk,
           limits=["Small-signal (linearised) noise around one DC operating point; resistor 1/f (excess) noise not modelled.",
                   "Input voltage sources (DACs) are noiseless here; their noise would add like the s06 input-noise study.",
                   "Integration down to 0.1 Hz treats slow 1/f drift as noise; it is really a slowly varying offset.",
                   "Above ~3 MHz the vendor model's output noise stays flat (~80 nV/rtHz, op-amp-only variant C equals A), so "
                   "rms noise depends on the readout bandwidth; results are given per band, with 0.1 Hz-100 MHz as the worst case.",
                   "Outputs treated as independent: they share only the noiseless input/bias sources and the supply rails "
                   "(ideal sources here)."])
save_json(res, C.RESULTS / "s13_noise.json")

# ---- figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for v, col, lab in (("A", "#c8102e", "all sources"), ("B", "#e8833a", "synapse resistors noiseless"), ("C", "#1f5fbf", "op-amps only")):
    ax[0].loglog(f, dens[v][0] * 1e9, color=col, label=lab)
ax[0].set(xlabel="frequency (Hz)", ylabel="output noise density of S_0 (nV/rtHz)")
ax[0].set_title(f"ELECTRICAL: ngspice .noise of S_0, OPA196 model, test image {IMG}\n"
                f"rms: {tot['A'][0]*1e6:.0f} uV (1 MHz 1st-order readout),  {sigma[WORST]['A'][0]*1e6:.0f} uV (0.1 Hz-100 MHz)",
                fontsize=9)
ax[0].legend(fontsize=7); ax[0].grid(which="both", lw=.3, alpha=.5)
gaps = np.sort(S, 1); gaps = gaps[:, -1] - gaps[:, -2]
ax[1].hist(gaps * 1e3, bins=np.logspace(np.log10(gaps.min() * 1e3 / 1.5), np.log10(gaps.max() * 1e3 * 1.2), 40), color="#1f5fbf")
ax[1].axvline(6 * sqrt(2) * sig.mean() * 1e3, color="#c8102e", ls="--", label="6 sigma of (S_w - S_r), 1 MHz readout")
ax[1].axvline(6 * sqrt(2) * sigma[WORST]["A"].mean() * 1e3, color="#e8833a", ls=":", label="6 sigma, 0.1 Hz-100 MHz")
ax[1].set(xscale="log", xlabel="decision margin S_winner - S_runner-up (mV), 360 OPA196 test outputs", ylabel="images")
ax[1].set_title(f"smallest margin {risk[PRIMARY]['worst_margin_V']*1e3:.2f} mV = {risk[PRIMARY]['worst_margin_in_sigma']:.1f} sigma "
                f"(1 MHz readout), {risk[WORST]['worst_margin_in_sigma']:.1f} sigma (100 MHz)", fontsize=9)
ax[1].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(C.PLOTS / "fig11_noise.png", dpi=130)
print(json.dumps(dict(sigma_uV_primary=np.round(tot["A"] * 1e6, 2).tolist(),
                      sigma_S0_uV_by_band={b: round(float(sigma[b]["A"][0]) * 1e6, 1) for b in BANDS},
                      share=share, worked=worked, risk=risk), indent=1))
