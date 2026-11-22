"""Drive the Tkinter demos exactly as a user would and prove that every selection
launches ngspice and yields ten electrical output voltages.

For each demo (demo/demo_app.py and demo/neuralsphere_visual.py):
  - open the real Tk window,
  - select several test images (setting the spinbox variable, then pressing the same
    callback the "Run SPICE" button calls),
  - wrap subprocess.run to record every ngspice process that is started,
  - check the ngspice output file is (re)written after the click,
  - parse the 10 S_k voltages from that raw file independently of the demo,
  - grab a screenshot of the window.
Writes results/demo_gui_verification.json and plots/demo_gui_<name>.png.
"""
import sys, time, json, importlib.util, subprocess
from pathlib import Path
import numpy as np
from _common import C, ROOT, stamp
from ns import spice

calls = []
_orig = subprocess.run


def spy(args, *a, **k):
    if args and "ngspice" in str(args[0]).lower():
        calls.append(dict(t=time.time(), exe=str(args[0]), netlist=str(args[-1])))
    return _orig(args, *a, **k)


subprocess.run = spy
spice.subprocess.run = spy


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def check(mod, name, run_attr, work, images=(0, 17, 123, 359)):
    import tkinter as tk
    root = tk.Tk(); root.geometry("1500x900+20+20")
    app = mod.App(root); root.update()
    rows = []
    for i in images:
        n0 = len(calls); t0 = time.time()
        app.idx.set(i); getattr(app, run_attr)(); root.update()
        out = Path(work) / "out_0.txt"
        A = np.array([[float(v) for v in l.split()] for l in out.read_text().splitlines() if l.strip()])
        S = A[0, 1:11]                                   # wr_singlescale: col 0 = scale, then v(s0..s9)
        rows.append(dict(image=i, true=int(mod.yte[i]), ngspice_processes_started=len(calls) - n0,
                         output_file=str(out), output_written_after_click=out.stat().st_mtime >= t0 - 1,
                         S_volts_from_raw_file=S.round(6).tolist(), predicted_argmax=int(S.argmax()),
                         n_output_voltages=int(S.size)))
    time.sleep(0.3); root.update()
    shot = C.PLOTS / f"demo_gui_{name}.png"
    try:
        from PIL import ImageGrab
        x, y, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
        full = ImageGrab.grab()                       # physical pixels; Tk reports logical (DPI-scaled) pixels
        f = full.width / root.winfo_screenwidth()
        full.crop(tuple(int(round(v * f)) for v in (x, y, x + w, y + h))).save(shot)
    except Exception as e:  # pragma: no cover
        shot = f"screenshot failed: {e}"
    root.destroy()
    ok = all(r["ngspice_processes_started"] >= 1 and r["output_written_after_click"] and r["n_output_voltages"] == 10
             for r in rows) and len({tuple(r["S_volts_from_raw_file"]) for r in rows}) == len(rows)
    return dict(demo=name, ok=ok, screenshot=str(shot), runs=rows)


res = dict(stamp=stamp(), demos=[])
res["demos"].append(check(load(ROOT / "demo" / "demo_app.py", "demo_app"), "demo_app", "run", ROOT / "demo" / "last_run"))
res["demos"].append(check(load(ROOT / "demo" / "neuralsphere_visual.py", "visual"), "neuralsphere_visual", "run_test",
                          ROOT / "demo" / "last_run_visual"))
res["ngspice_exe_used"] = sorted({c["exe"] for c in calls})
res["ok"] = all(d["ok"] for d in res["demos"])
from ns.metrics import save_json  # noqa: E402
save_json(res, C.RESULTS / "demo_gui_verification.json")
for dmo in res["demos"]:
    print(dmo["demo"], "OK" if dmo["ok"] else "FAILED", dmo["screenshot"])
    for r in dmo["runs"]:
        print("  image", r["image"], "true", r["true"], "pred", r["predicted_argmax"], "ngspice runs", r["ngspice_processes_started"],
              "fresh", r["output_written_after_click"], "S =", r["S_volts_from_raw_file"])
assert res["ok"]
