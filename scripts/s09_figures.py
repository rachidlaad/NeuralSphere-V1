"""Make every figure from SAVED result files (no numbers are typed in by hand)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import C
from ns.mapping import Mapping

R, P = C.RESULTS, C.PLOTS
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GRAY, RED, GREEN = "#2a6fdb", "#e8833a", "#8a8a8a", "#d64545", "#2e9e6a"
d = np.load(C.DATA / "digits_split.npz"); wts = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]
m = Mapping(wts["W"], wts["b"], Xcal=Xtr)

# 1 training history
h = json.load(open(R / "s02_training_history.json"))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], color=BLUE)
ax[0].set(xlabel="epoch", ylabel="cross-entropy loss (train)", title="Software training (softmax SGD)")
ax[1].plot([r["epoch"] for r in h], [100 * r["train_acc"] for r in h], color=BLUE)
ax[1].set(xlabel="epoch", ylabel="train accuracy (%)", title="Train accuracy")
fig.tight_layout(); fig.savefig(P / "fig1_training_history.png"); plt.close(fig)

# 2 SPICE vs software correspondence
s3 = np.load(R / "s03_spice_ideal_outputs.npz"); s4 = np.load(R / "s04_D1_real_continuous_outputs.npz")
z = Xte @ wts["W"].T + wts["b"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a, s, t in ((ax[0], s3["s_te"], "ideal op-amps"), (ax[1], s4["s_te"], "realistic op-amp macromodel")):
    a.scatter(m.volts_per_logit * z.ravel(), s.ravel(), s=3, color=BLUE, alpha=.5)
    lim = [s.min() - .1, s.max() + .1]; a.plot(lim, lim, color=GRAY, lw=.8)
    err = np.abs(s - m.volts_per_logit * z).max()
    a.set(xlabel="software prediction  k·z (V)", ylabel="SPICE output S (V)",
          title=f"{t}: max |error| = {err*1e3:.3f} mV")
fig.tight_layout(); fig.savefig(P / "fig2_spice_vs_software.png"); plt.close(fig)

# 3 confusion matrix (stage D real, test)
sd = json.load(open(R / "s04_stageD_inference.json"))["D1_real_continuous"]["test"]
cm = np.array(sd["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.imshow(cm, cmap="Blues")
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 20 else "black")
ax.set(xticks=range(10), yticks=range(10), xlabel="predicted (SPICE argmax)", ylabel="true digit",
       title=f"SPICE circuit, 360 test images: {sd['correct']}/360")
fig.tight_layout(); fig.savefig(P / "fig3_confusion_matrix.png"); plt.close(fig)

# 4 per digit
fig, ax = plt.subplots(figsize=(6, 2.8))
pd_ = sd["per_digit_accuracy"]
ax.bar(range(10), [100 * pd_[str(k)] for k in range(10)], color=BLUE)
ax.set(xticks=range(10), ylim=(80, 101), xlabel="digit", ylabel="accuracy (%)", title="Per-digit test accuracy (SPICE, realistic op-amps)")
fig.tight_layout(); fig.savefig(P / "fig4_per_digit.png"); plt.close(fig)

# 5 conductance maps + resistor histogram
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw=dict(width_ratios=[1, 1, .8]))
for a, G, t in ((ax[0], m.Gp, "G+ (uS)"), (ax[1], m.Gn, "G- (uS)")):
    im = a.imshow(G * 1e6, aspect="auto", cmap="viridis"); plt.colorbar(im, ax=a)
    a.set(title=t, xlabel="input (0-63 pixels, 64 = bias)", ylabel="output neuron")
R_all = np.concatenate([1 / m.Gp.ravel(), 1 / m.Gn.ravel()])
ax[2].hist(np.log10(R_all), bins=40, color=BLUE)
ax[2].set(xlabel="log10(R / ohm)", ylabel="count", title=f"{R_all.size} synapse resistors")
fig.tight_layout(); fig.savefig(P / "fig5_conductance_maps.png"); plt.close(fig)

# 6 robustness
if (R / "s06_nonidealities_summary.json").exists():
  s6 = json.load(open(R / "s06_nonidealities_summary.json"))
  tab = s6["table"]
  fig, ax = plt.subplots(figsize=(9, 6.2))
  labels = [f"{t['group']}: {t['condition']}" for t in tab]
  mean = [100 * t["mean_acc"] for t in tab]; sd_ = [100 * t["std_acc"] for t in tab]
  cols = [GREEN if t["group"] == "reference" else BLUE for t in tab]
  y = np.arange(len(tab))[::-1]
  ax.barh(y, mean, xerr=sd_, color=cols, height=.7, error_kw=dict(lw=.7))
  ax.axvline(100 * s6["reference_accuracy_real"], color=GRAY, lw=.8, ls="--")
  ax.set(yticks=y, yticklabels=labels, xlim=(0, 100), xlabel="test accuracy (%)  (mean ± std over trials)",
         title="SPICE accuracy under hardware non-idealities (360 test images per run)")
  ax.tick_params(axis="y", labelsize=7)
  fig.tight_layout(); fig.savefig(P / "fig6_robustness.png"); plt.close(fig)

# 7 in-loop
if (R / "s05_stageE_in_loop.json").exists():
    s5 = json.load(open(R / "s05_stageE_in_loop.json"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for (name, c), col in zip(s5["chips"].items(), (BLUE, ORANGE, RED, GREEN, GRAY)):
        it = [r["iteration"] for r in c["history"]]
        lab = f"{name}{'' if 'lr' in name else ' (lr 2)'}"
        ax[0].plot(it, [r["train_loss_measured"] for r in c["history"]], "o-", ms=3, color=col, label=lab)
        t = f"test {c['test_before']['correct']}->{c['test_after']['correct']}"
        if "test_at_best_train_loss" in c:
            t += f" (best-train-loss state: {c['test_at_best_train_loss']['correct']})"
            ax[0].plot(c["best_train_loss_iteration"], c["history"][c["best_train_loss_iteration"]]["train_loss_measured"],
                       "*", ms=12, color=col)
        ax[1].plot(it, [100 * r["train_acc_measured"] for r in c["history"]], "o-", ms=3, color=col, label=f"{lab}: {t}")
    ax[0].set(xlabel="in-loop iteration", ylabel="loss from MEASURED SPICE outputs", title="SPICE-in-the-loop fine-tuning (train set)")
    ax[1].set(xlabel="in-loop iteration", ylabel="train accuracy measured on chip (%)")
    ax[0].legend(fontsize=6); ax[1].legend(fontsize=6); fig.tight_layout(); fig.savefig(P / "fig7_in_loop.png"); plt.close(fig)

# 8 transient
tr = np.load(R / "s07_transient.npz"); t = tr["t"] * 1e6
fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for i in (19, 27, 36, 44):
    ax[0].plot(t, tr["vin"][:, i], lw=1, label=f"V_in{i}")
ax[0].set(ylabel="input voltage (V)", title="Transient: 6 test images presented every 20 us"); ax[0].legend(fontsize=7, ncol=4)
for k in range(10):
    ax[1].plot(t, tr["s"][:, k], lw=1, label=str(k))
ax[1].set(ylabel="output score S_k (V)"); ax[1].legend(fontsize=7, ncol=10, title="digit k", title_fontsize=7)
ax[2].plot(t, tr["P"] * 1e3, color=RED, lw=1, label="total"); ax[2].plot(t, tr["P_xbar"] * 1e3, color=BLUE, lw=1, label="crossbar + bias rail")
ax[2].set(ylabel="power (mW)", xlabel="time (us)"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(P / "fig8_transient.png"); plt.close(fig)

# 9 example classification panels (from saved SPICE test outputs)
pred = s4["s_te"].argmax(1)
wrong = np.where(pred != yte)[0]
show = [0, 1, 2, int(wrong[0])] if len(wrong) else [0, 1, 2, 3]
fig, ax = plt.subplots(len(show), 3, figsize=(10, 2.4 * len(show)), gridspec_kw=dict(width_ratios=[.6, 1.4, 1]))
for r, idx in enumerate(show):
    ax[r, 0].imshow(Xte[idx].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
    ax[r, 0].set_title(f"test #{idx}  true = {yte[idx]}")
    ax[r, 1].bar(range(64), C.VMAX * Xte[idx], color=GRAY); ax[r, 1].set(ylabel="V_in (V)", ylim=(0, 1.05))
    s = s4["s_te"][idx]
    ax[r, 2].bar(range(10), s, color=[GREEN if k == pred[idx] else BLUE for k in range(10)])
    ax[r, 2].set(xticks=range(10), ylabel="S_k (V)", title=f"SPICE prediction = {pred[idx]}" + ("" if pred[idx] == yte[idx] else "  (wrong)"))
ax[-1, 1].set_xlabel("pixel index"); ax[-1, 2].set_xlabel("output neuron k")
fig.tight_layout(); fig.savefig(P / "fig9_example_classifications.png"); plt.close(fig)
print("figures written to", P)
