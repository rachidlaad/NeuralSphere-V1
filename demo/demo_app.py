"""NeuralSphere-V1 interactive demo.

Pick any held-out test image, press "Run SPICE": the program writes a netlist of the
full 64x10 differential resistive crossbar with the selected image applied as 64
voltage sources, launches ngspice, parses the solved node voltages and shows them.
Nothing is precomputed or hardcoded: every bar comes from the ngspice output file
of that run (path shown in the window).

Run:  .venv\\Scripts\\python.exe demo\\demo_app.py      (Windows)
      python3 demo/demo_app.py                        (Linux/macOS)
"""
import sys, time
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ns import config as C, spice          # noqa: E402
from ns.mapping import Mapping              # noqa: E402

d = np.load(C.DATA / "digits_split.npz"); w = np.load(C.WEIGHTS / "software_weights.npz")
Xtr, Xte, yte = d["Xtr"], d["Xte"], d["yte"]
MAPS = {"continuous G": Mapping(w["W"], w["b"], Xcal=Xtr), "8-bit G": Mapping(w["W"], w["b"], bits=8, Xcal=Xtr),
        "4-bit G": Mapping(w["W"], w["b"], bits=4, Xcal=Xtr)}
WORK = ROOT / "demo" / "last_run"




def simulate_and_draw(fig, i, mapping_name="continuous G", amp="real", wta=True, workdir=WORK):
    """Run ngspice for test image i and draw the result into a matplotlib Figure."""
    m = MAPS[mapping_name]
    t0 = time.time()
    r = spice.run(m.Gp, m.Gn, m.Rf, C.VMAX * Xte[i:i + 1], opamp=amp, wta=wta,
                  workdir=workdir, keep_netlist=Path(workdir) / "netlist_for_this_image.cir")
    dt = time.time() - t0
    s = r["s"][0]; pred = int(s.argmax()); P = spice.power(r)["p_total"][0]
    fig.clf()
    g = fig.add_gridspec(2, 3, width_ratios=[.7, 1.6, 1.2])
    a0 = fig.add_subplot(g[:, 0]); a0.imshow(Xte[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    a0.set_title(f"test #{i}   true label = {yte[i]}"); a0.set_xticks([]); a0.set_yticks([])
    a1 = fig.add_subplot(g[0, 1]); a1.bar(range(64), r["vin"][0], color="#8a8a8a")
    a1.set(title="64 input voltage sources V_i = 1 V x pixel/16", ylabel="V", ylim=(0, 1.05))
    a2 = fig.add_subplot(g[1, 1])
    a2.bar(np.arange(10) - .2, r["tp"][0], .4, label="TIA+ out v(tp_k)", color="#2a6fdb")
    a2.bar(np.arange(10) + .2, r["tn"][0], .4, label="TIA- out v(tn_k)", color="#e8833a")
    a2.set(xticks=range(10), ylabel="V", title="transimpedance amplifier outputs (= -Rf*I)"); a2.legend(fontsize=7)
    a3 = fig.add_subplot(g[0, 2])
    ok = pred == yte[i]
    a3.bar(range(10), s, color=["#2e9e6a" if k == pred else "#2a6fdb" for k in range(10)])
    a3.set(xticks=range(10), ylabel="S_k (V)",
           title=f"10 analog output scores -> predicted {pred} " + ("(correct)" if ok else "(WRONG)"))
    a4 = fig.add_subplot(g[1, 2])
    if "w" in r:
        a4.bar(range(10), r["w"][0], color=["#d64545" if v > 2.5 else "#bbbbbb" for v in r["w"][0]])
        a4.set(xticks=range(10), ylabel="V", ylim=(0, 5.5), title="analog winner-take-all lines w_k")
    else:
        a4.axis("off")
    fig.suptitle(f"ngspice output ({amp} op-amps, {mapping_name}) | solved in {dt:.2f} s | "
                 f"power {P*1e3:.1f} mW | file {Path(workdir) / 'out_0.txt'}", fontsize=8)
    fig.tight_layout()
    return r, pred, dt, P


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.tk = tk
        self.root = root
        root.title("NeuralSphere-V1 - SPICE-simulated analog neural network")
        bar = ttk.Frame(root, padding=6); bar.pack(fill="x")
        ttk.Label(bar, text="test image #").pack(side="left")
        self.idx = tk.IntVar(value=0)
        ttk.Spinbox(bar, from_=0, to=len(Xte) - 1, textvariable=self.idx, width=6).pack(side="left")
        ttk.Button(bar, text="random", command=self.rand).pack(side="left", padx=4)
        ttk.Label(bar, text="op-amp").pack(side="left", padx=(12, 2))
        self.amp = tk.StringVar(value="real")
        ttk.Combobox(bar, textvariable=self.amp, values=["real", "ideal"], width=6, state="readonly").pack(side="left")
        ttk.Label(bar, text="synapses").pack(side="left", padx=(12, 2))
        self.mp = tk.StringVar(value="continuous G")
        ttk.Combobox(bar, textvariable=self.mp, values=list(MAPS), width=12, state="readonly").pack(side="left")
        self.wta = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="analog WTA", variable=self.wta).pack(side="left", padx=8)
        ttk.Button(bar, text="Run SPICE", command=self.run).pack(side="left", padx=8)
        self.status = ttk.Label(root, text=f"ngspice: {C.ngspice_exe()}", padding=4); self.status.pack(fill="x")
        self.fig = Figure(figsize=(12, 6.2), dpi=95)
        self.canvas = FigureCanvasTkAgg(self.fig, root); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.run()

    def rand(self):
        self.idx.set(int(np.random.randint(len(Xte)))); self.run()

    def run(self):
        self.status.config(text="running ngspice ..."); self.root.update()
        r, pred, dt, P = simulate_and_draw(self.fig, int(self.idx.get()), self.mp.get(), self.amp.get(), self.wta.get())
        self.canvas.draw()
        self.status.config(text=f"ngspice solved the circuit in {dt:.2f} s | predicted {pred} | power {P*1e3:.1f} mW"
                                f" | raw output: {WORK / 'out_0.txt'}")


if __name__ == "__main__":
    if "--snapshot" in sys.argv:          # headless: python demo_app.py --snapshot 17
        i = int(sys.argv[sys.argv.index("--snapshot") + 1])
        fig = Figure(figsize=(12, 6.2), dpi=110)
        simulate_and_draw(fig, i)
        out = ROOT / "plots" / f"demo_snapshot_test{i}.png"; fig.savefig(out); print("saved", out)
    else:
        import tkinter as tk
        root = tk.Tk(); App(root); root.mainloop()
