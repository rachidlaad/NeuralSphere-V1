"""NeuralSphere-V1 visual demonstration (live ngspice).

Every number drawn in the ELECTRICAL panels is read from the ngspice output file of
the run that was just executed for the selected image:

  8x8 digit  ->  64 DC voltage sources  ->  1300-resistor differential crossbar
             ->  20 transimpedance amplifiers  ->  10 difference amplifiers  ->  WTA

Per-synapse currents are obtained with Ohm's law from the SPICE-solved node voltages,
I_ki = G_ki * (V_i - v(col_k)), and checked by KCL against the current through R_f,
(v(col_k) - v(t_k)) / R_f, which is also taken from the SPICE solution.

SOFTWARE quantities (trained weights, software logits) are shown only as labelled
reference markers. The weights were trained in software and programmed into the
netlist; this is NOT on-chip or autonomous physical learning.

Run:   .venv\\Scripts\\python.exe demo\\neuralsphere_visual.py
       .venv\\Scripts\\python.exe demo\\neuralsphere_visual.py --snapshot 5     (headless PNG)
GUI:   choose a test image, or draw your own digit on the 8x8 pad, then "Run SPICE".
"""
import sys, time, subprocess
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ns import config as C, spice           # noqa: E402
from ns.mapping import Mapping, input_vector  # noqa: E402

d = np.load(C.DATA / "digits_split.npz"); wts = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, yte = d["Xtr"], d["Xte"], d["yte"]
W_SW, B_SW = wts["W"], wts["b"]
MAPS = {"continuous G": Mapping(W_SW, B_SW, Xcal=Xtr),
        "8-bit G": Mapping(W_SW, B_SW, bits=8, Xcal=Xtr),
        "4-bit G": Mapping(W_SW, B_SW, bits=4, Xcal=Xtr)}
WORK = ROOT / "demo" / "last_run_visual"
EXTRA = [f"v(cp{k})" for k in range(10)] + [f"v(cn{k})" for k in range(10)]
RED, BLUE, GREEN, GRAY, INK = "#c8102e", "#1f5fbf", "#1a8f5a", "#9a9a9a", "#222222"
_NGV = None


def ngspice_version():
    global _NGV
    if _NGV is None:
        try:
            out = subprocess.run([C.ngspice_exe(), "-v"], capture_output=True, text=True, timeout=20).stdout
            _NGV = next((l.strip("* ").split(":")[0].strip() for l in out.splitlines() if "ngspice-" in l), "ngspice")
        except Exception:
            _NGV = "ngspice"
    return _NGV


def solve(x, mapping_name="continuous G", amp="real", wta=True):
    """Run ngspice for one 64-pixel input x (0..1) and derive all displayed quantities."""
    m = MAPS[mapping_name]
    WORK.mkdir(parents=True, exist_ok=True)
    out_file = WORK / "out_0.txt"
    t0 = time.time()
    r = spice.run(m.Gp, m.Gn, m.Rf, C.VMAX * np.atleast_2d(x), opamp=amp, wta=wta, workdir=WORK,
                  keep_netlist=WORK / "netlist_for_this_image.cir", extra=EXTRA)
    dt = time.time() - t0
    fresh = out_file.exists() and out_file.stat().st_mtime >= t0 - 1
    V = input_vector(np.atleast_2d(x))[0]                       # 64 pixel sources + bias rail [V]
    vcp = np.array([r["extra"][f"v(cp{k})"][0] for k in range(10)])
    vcn = np.array([r["extra"][f"v(cn{k})"][0] for k in range(10)])
    tp, tn, s = r["tp"][0], r["tn"][0], r["s"][0]
    Ip_syn = m.Gp * (V[None, :] - vcp[:, None])                # Ohm, per synapse [A]
    In_syn = m.Gn * (V[None, :] - vcn[:, None])
    Ip_rf = (vcp - tp) / m.Rf                                   # current through R_f (SPICE voltages)
    In_rf = (vcn - tn) / m.Rf
    w = r["w"][0] if "w" in r else None
    high = (w > 2.5) if w is not None else None
    if w is not None and high.sum() == 1:
        pred, how = int(np.argmax(w)), "analog WTA line"
    else:
        pred, how = int(np.argmax(s)), "argmax of S_k (measurement program)"
    z_sw = W_SW @ np.asarray(x) + B_SW                          # SOFTWARE reference logits
    return dict(m=m, x=np.asarray(x), V=V, vcp=vcp, vcn=vcn, tp=tp, tn=tn, s=s, w=w, high=high,
                Ip_syn=Ip_syn, In_syn=In_syn, Ip_rf=Ip_rf, In_rf=In_rf, pred=pred, how=how,
                z_sw=z_sw, P=float(spice.power(r)["p_total"][0]), dt=dt, fresh=fresh,
                out_file=out_file, mtime=time.strftime("%H:%M:%S", time.localtime(out_file.stat().st_mtime)),
                amp=amp, mapping=mapping_name)


def draw(fig, R, true_label=None, title_extra=""):
    m = R["m"]; fig.clf(); fig.patch.set_facecolor("white")
    # ---- flow strip -----------------------------------------------------------------
    stages = ["8x8 digit", "64 DC sources\nV_i = 1 V * pixel/16", "1300 resistors\n(65 x 10 x 2)",
              "20 TIAs\nt = -R_f I", "10 diff-amps\nS = t- - t+", "WTA\n45 comparators", "digit"]
    for j, st in enumerate(stages):
        xx = 0.035 + j * 0.138
        fig.text(xx, 0.955, st, ha="center", va="center", fontsize=8, color=INK,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#eef3fb" if 0 < j < 6 else "#f4f4f4", ec="#9db4d8"))
        if j < len(stages) - 1:
            fig.text(xx + 0.069, 0.955, "->", ha="center", va="center", fontsize=11, color="#5b7fb5")
    fig.text(0.995, 0.995, "ELECTRICAL = ngspice solution   |   SOFTWARE = trained weights / reference logits",
             ha="right", va="top", fontsize=7, color="#555")

    # ---- digit ----------------------------------------------------------------------
    a = fig.add_axes([0.015, 0.50, 0.135, 0.36])
    a.imshow(R["x"].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    for p in range(64):
        v = int(round(16 * R["x"][p]))
        if v:
            a.text(p % 8, p // 8, v, ha="center", va="center", fontsize=6, color="white" if v > 8 else "black")
    a.set_xticks([]); a.set_yticks([])
    a.set_title("input digit (pixel 0..16)" + (f"\ntrue label = {true_label}" if true_label is not None else "\n(hand-drawn)"),
                fontsize=8)

    # ---- 64 input voltages ------------------------------------------------------------
    a = fig.add_axes([0.165, 0.50, 0.135, 0.36])
    a.imshow(R["V"][:64].reshape(8, 8), cmap="YlOrRd", vmin=0, vmax=C.VMAX)
    for p in range(64):
        a.text(p % 8, p // 8, f"{R['V'][p]:.2f}"[1:] if R["V"][p] < 1 else "1.0", ha="center", va="center", fontsize=5.5)
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"ELECTRICAL: 64 source voltages (V)\n+ bias rail {C.VBIAS:.1f} V", fontsize=8)
    ip = int(np.argmax(R["x"]))
    fig.text(0.015, 0.455, f"Encoding:  V_i = V_max * pixel_i / 16\n"
                           f"e.g. pixel {ip} = {int(round(16*R['x'][ip]))}/16  ->  "
                           f"V_{ip} = 1 V * {16*R['x'][ip]:.0f}/16 = {R['V'][ip]:.4f} V\n"
                           f"Total current drawn from the 64 sources + bias:\n"
                           f"  sum_k (I+_k + I-_k) = {1e3*(R['Ip_syn'].sum()+R['In_syn'].sum()):.3f} mA",
             fontsize=7.2, family="monospace", va="top", color=INK)

    # ---- crossbar: per-synapse currents ----------------------------------------------
    ax = fig.add_axes([0.345, 0.43, 0.31, 0.46])
    M = np.zeros((65, 20))
    M[:, 0::2] = (R["Ip_syn"] * 1e6).T                          # columns cp_k   (positive)
    M[:, 1::2] = -(R["In_syn"] * 1e6).T                         # columns cn_k   (shown negative)
    lim = max(np.abs(M).max(), 1e-3)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_xticks(range(20)); ax.set_xticklabels([f"{'+' if c % 2 == 0 else '-'}{c//2}" for c in range(20)], fontsize=6)
    ax.set_yticks([0, 8, 16, 24, 32, 40, 48, 56, 64]); ax.set_yticklabels([f"px{q}" for q in range(0, 64, 8)] + ["bias"], fontsize=6)
    for c in range(1, 10):
        ax.axvline(2 * c - 0.5, color="white", lw=1.2)
    ax.set_title("ELECTRICAL: current in each of the 1300 synapse resistors (uA)\n"
                 "I_ki = G_ki * (V_i - v(column_k)),  column voltages from ngspice;  red = into + column, blue = into - column",
                 fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.ax.tick_params(labelsize=6)
    ax2 = fig.add_axes([0.345, 0.27, 0.31 * 0.97, 0.12], sharex=ax)
    col = np.zeros(20); col[0::2] = R["Ip_rf"] * 1e6; col[1::2] = -R["In_rf"] * 1e6
    ax2.bar(range(20), col, color=[RED if c % 2 == 0 else BLUE for c in range(20)])
    ax2.axhline(0, color=GRAY, lw=.6); ax2.tick_params(labelsize=6)
    ax2.set_ylabel("column\ncurrent (uA)", fontsize=7)
    ax2.set_title("ELECTRICAL: column current through R_f = (v(col) - v(TIA out)) / R_f  (KCL sum of the synapses above)", fontsize=7)

    # ---- neuron outputs S_k ----------------------------------------------------------
    a = fig.add_axes([0.70, 0.43, 0.13, 0.46])
    s = R["s"]; k = np.arange(10)
    a.barh(k, s, color=[GREEN if q == R["pred"] else BLUE for q in k], height=.65, label="ngspice v(s_k)")
    a.plot(m.volts_per_logit * R["z_sw"], k, "k|", ms=11, mew=1.6, label="SOFTWARE ref: 0.2374 V * z_k")
    for q in k:
        a.text(s[q] + (0.05 if s[q] >= 0 else -0.05), q, f"{s[q]:+.3f} V", va="center",
               ha="left" if s[q] >= 0 else "right", fontsize=6.5)
    a.set_yticks(k); a.set_yticklabels([f"S{q}" for q in k], fontsize=7); a.invert_yaxis()
    xl = max(np.abs(s).max(), 0.5) * 1.6; a.set_xlim(-xl, xl); a.axvline(0, color=GRAY, lw=.6)
    a.tick_params(labelsize=6); a.set_xlabel("volts", fontsize=7)
    a.set_title("ELECTRICAL: 10 neuron outputs\nS_k = v(tn_k) - v(tp_k)", fontsize=8)
    a.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False)

    # ---- WTA + prediction ------------------------------------------------------------
    a = fig.add_axes([0.85, 0.43, 0.14, 0.46]); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(10.6, -1.2)
    a.set_title("ELECTRICAL: winner-take-all lines", fontsize=8)
    for q in range(10):
        if R["w"] is not None:
            on = bool(R["high"][q])
            a.scatter([0.15], [q], s=170, c=RED if on else "#dddddd", edgecolors="#555", zorder=3)
            a.text(0.32, q, f"w{q} = {R['w'][q]:.2f} V", va="center", fontsize=7, color=INK if on else "#777")
        else:
            a.text(0.05, q, f"w{q}: WTA not simulated", va="center", fontsize=6.5, color="#888")
    fig.text(0.92, 0.33, str(R["pred"]), ha="center", va="center", fontsize=54, weight="bold",
             color=GREEN if (true_label is None or R["pred"] == true_label) else RED)
    verdict = "" if true_label is None else ("correct" if R["pred"] == true_label else f"WRONG (true {true_label})")
    fig.text(0.92, 0.265, f"predicted digit {verdict}\nfrom {R['how']}", ha="center", va="top", fontsize=7)

    # ---- formulas with substituted values ---------------------------------------------
    order = np.argsort(R["s"])[::-1]; kw, kr = int(order[0]), int(order[1])
    vpl = m.volts_per_logit

    def block(q, tag):
        Ip, In = R["Ip_syn"][q].sum(), R["In_syn"][q].sum()
        return (f"Neuron {q} ({tag})  [ELECTRICAL, ngspice .op]\n"
                f"  I+_{q} = sum_i G+_{q},i (V_i - v(cp{q})) = {Ip*1e6:9.3f} uA   KCL check, current in R_f = {R['Ip_rf'][q]*1e6:9.3f} uA   (v(cp{q}) = {R['vcp'][q]*1e6:+.1f} uV)\n"
                f"  I-_{q} = sum_i G-_{q},i (V_i - v(cn{q})) = {In*1e6:9.3f} uA   KCL check, current in R_f = {R['In_rf'][q]*1e6:9.3f} uA   (v(cn{q}) = {R['vcn'][q]*1e6:+.1f} uV)\n"
                f"  S_{q} = R_f (I+ - I-) = {m.Rf:.0f} ohm * ({Ip*1e6:.3f} - {In*1e6:.3f}) uA = {m.Rf*(Ip-In):+.5f} V      ngspice v(s{q}) = {R['s'][q]:+.5f} V\n"
                f"  z_{q} = S_{q} / (R_f alpha V_max) = {R['s'][q]:+.5f} V / ({m.Rf:.0f} ohm * {m.alpha*1e6:.3f} uS * {C.VMAX:.0f} V = {vpl:.5f} V)"
                f" = {R['s'][q]/vpl:+.4f}     [SOFTWARE ref z_{q} = sum_i w x + b = {R['z_sw'][q]:+.4f}]")
    txt = (block(kw, "winner") + "\n" + block(kr, "runner-up") + "\n"
           f"Decision margin  S_{kw} - S_{kr} = {R['s'][kw]:+.5f} - ({R['s'][kr]:+.5f}) = {1e3*(R['s'][kw]-R['s'][kr]):.2f} mV"
           f"      |  power P = sum(-V i(V)) over all sources = {R['P']*1e3:.1f} mW"
           f"      |  synapses: {R['mapping']}, op-amps: {R['amp']}")
    fig.text(0.015, 0.215, txt, fontsize=6.6, family="monospace", va="top", color=INK)
    fig.text(0.015, 0.012,
             f"{ngspice_version()} solved this circuit in {R['dt']:.2f} s | output file {R['out_file']} written {R['mtime']}"
             f" ({'fresh for this run' if R['fresh'] else 'NOT FRESH'}) | Weights were trained in software and programmed "
             f"as resistances; this is a circuit simulation, not on-chip learning. {title_extra}",
             fontsize=6, color="#555")


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.root = root; self.tk = tk
        root.title("NeuralSphere-V1 - live ngspice visual demonstration")
        bar = ttk.Frame(root, padding=4); bar.pack(fill="x")
        ttk.Label(bar, text="test image #").pack(side="left")
        self.idx = tk.IntVar(value=5)
        ttk.Spinbox(bar, from_=0, to=len(Xte) - 1, textvariable=self.idx, width=5).pack(side="left")
        ttk.Button(bar, text="<", width=2, command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(bar, text=">", width=2, command=lambda: self.step(1)).pack(side="left")
        ttk.Button(bar, text="random", command=self.rand).pack(side="left", padx=3)
        ttk.Label(bar, text="op-amp").pack(side="left", padx=(10, 2))
        self.amp = tk.StringVar(value="real")
        # "opa196" = Texas Instruments vendor macromodel (slower: ~5-15 s per image)
        ttk.Combobox(bar, textvariable=self.amp, values=["real", "ideal", "opa196"], width=7, state="readonly").pack(side="left")
        ttk.Label(bar, text="synapses").pack(side="left", padx=(10, 2))
        self.mp = tk.StringVar(value="continuous G")
        ttk.Combobox(bar, textvariable=self.mp, values=list(MAPS), width=12, state="readonly").pack(side="left")
        self.wta = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="analog WTA", variable=self.wta).pack(side="left", padx=6)
        ttk.Button(bar, text="Run SPICE on test image", command=self.run_test).pack(side="left", padx=6)
        self.status = ttk.Label(root, text=f"ngspice: {C.ngspice_exe()}", padding=3); self.status.pack(fill="x")
        body = ttk.Frame(root); body.pack(fill="both", expand=True)
        # ---- 8x8 drawing pad
        pad = ttk.Frame(body, padding=6); pad.pack(side="left", fill="y")
        ttk.Label(pad, text="Draw a digit\n(left = paint, right = erase)").pack()
        self.cell = 26
        self.cv = tk.Canvas(pad, width=8 * self.cell, height=8 * self.cell, bg="white", highlightthickness=1)
        self.cv.pack(pady=4)
        self.pix = np.zeros(64)
        for ev in ("<Button-1>", "<B1-Motion>"):
            self.cv.bind(ev, lambda e: self.paint(e, +1))
        for ev in ("<Button-3>", "<B3-Motion>"):
            self.cv.bind(ev, lambda e: self.paint(e, -1))
        ttk.Button(pad, text="Run SPICE on drawing", command=self.run_drawing).pack(fill="x")
        ttk.Button(pad, text="copy test image to pad", command=self.copy_to_pad).pack(fill="x", pady=2)
        ttk.Button(pad, text="clear", command=self.clear_pad).pack(fill="x")
        try:
            root.state("zoomed")                      # maximise on Windows; the figure scales with the window
        except Exception:
            pass
        self.fig = Figure(figsize=(10, 6), dpi=90)
        self.canvas = FigureCanvasTkAgg(self.fig, body); self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.redraw_pad(); self.run_test()

    # drawing pad -------------------------------------------------------------------
    def paint(self, e, sign):
        c, r = e.x // self.cell, e.y // self.cell
        if not (0 <= c < 8 and 0 <= r < 8):
            return
        if sign > 0:
            self.pix[r * 8 + c] = 1.0
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):             # soft brush like the blurred dataset
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    self.pix[rr * 8 + cc] = min(1.0, self.pix[rr * 8 + cc] + 0.15)
        else:
            self.pix[r * 8 + c] = 0.0
        self.redraw_pad()

    def redraw_pad(self):
        self.cv.delete("all")
        for p in range(64):
            g = int(255 * (1 - self.pix[p])); col = f"#{g:02x}{g:02x}{g:02x}"
            r, c = divmod(p, 8)
            self.cv.create_rectangle(c * self.cell, r * self.cell, (c + 1) * self.cell, (r + 1) * self.cell,
                                     fill=col, outline="#dddddd")

    def clear_pad(self):
        self.pix[:] = 0; self.redraw_pad()

    def copy_to_pad(self):
        self.pix = np.round(Xte[int(self.idx.get())] * 16) / 16; self.redraw_pad()

    # runs ----------------------------------------------------------------------------
    def step(self, dlt):
        self.idx.set((int(self.idx.get()) + dlt) % len(Xte)); self.run_test()

    def rand(self):
        self.idx.set(int(np.random.randint(len(Xte)))); self.run_test()

    def _run(self, x, true_label, label):
        self.status.config(text="running ngspice ..."); self.root.update()
        R = solve(x, self.mp.get(), self.amp.get(), self.wta.get())
        draw(self.fig, R, true_label, label)
        self.canvas.draw()
        self.status.config(text=f"{label}: ngspice solved in {R['dt']:.2f} s | predicted {R['pred']} ({R['how']}) | "
                                f"P = {R['P']*1e3:.1f} mW | raw output {R['out_file']} @ {R['mtime']}")
        self.last = R

    def run_test(self):
        i = int(self.idx.get()); self._run(Xte[i], int(yte[i]), f"test image #{i}")

    def run_drawing(self):
        if self.pix.sum() == 0:
            self.status.config(text="draw something on the pad first"); return
        self._run(self.pix.copy(), None, "hand-drawn digit")


if __name__ == "__main__":
    if "--snapshot" in sys.argv:
        i = int(sys.argv[sys.argv.index("--snapshot") + 1])
        amp = sys.argv[sys.argv.index("--amp") + 1] if "--amp" in sys.argv else "real"
        fig = Figure(figsize=(16, 9.2), dpi=110)
        R = solve(Xte[i], "continuous G", amp, True)
        draw(fig, R, int(yte[i]), f"(test image #{i})")
        out = ROOT / "plots" / f"visual_demo_test{i}.png"; fig.savefig(out, facecolor="white")
        print("saved", out, "| pred", R["pred"], "| true", yte[i], "| S =", np.round(R["s"], 5).tolist())
    else:
        import tkinter as tk
        root = tk.Tk(); App(root); root.mainloop()
