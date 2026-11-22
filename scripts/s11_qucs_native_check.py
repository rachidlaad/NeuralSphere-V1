"""Electrical check of the Qucs-S schematic using Qucs-S ITSELF (not our own parser).

s10 checks the drawing by re-deriving connectivity from wire geometry in Python.
This script removes that assumption: it asks the Qucs-S program to
  1. netlist the schematic for ngspice      (qucs-s -n --ngspice -i sch -o cir)
  2. netlist + run ngspice, return dataset   (qucs-s -n --ngspice --run ...)
  3. render the schematic to PNG             (qucs-s -p ...)
and compares the solved node voltages with the Stage A hand calculation:

  I+ = sum_i V_i G+_i ,  I- = sum_i V_i G-_i              (Ohm + KCL, column at virtual ground)
  tp = -Rf I+ ,  tn = -Rf I- ,  S = tn - tp = Rf (I+ - I-)

The op-amp pin assignment is read from the Qucs-S netlist: the behavioural source
`V = A*V(a,b)` means a = non-inverting input, b = inverting input.
"""
import json, re, subprocess, shutil, tempfile
from pathlib import Path
from _common import C, ROOT, stamp

SCH = C.CIRCUITS / "qucs" / "neuron_stageA.sch"
QUCS = next((ROOT / "tools" / "qucs-s").rglob("qucs-s.exe"), None) or shutil.which("qucs-s")
st = json.load(open(C.RESULTS / "s01_stageA_single_neuron.json"))
tmp = Path(tempfile.mkdtemp(prefix="ns_qucs_"))
sch = tmp / SCH.name
shutil.copy(SCH, sch)


def q(*args):
    return subprocess.run([str(QUCS), *args], capture_output=True, text=True, cwd=tmp, timeout=120)


q("-n", "--ngspice", "-i", str(sch), "-o", str(tmp / "net.cir"))
q("-n", "--ngspice", "--run", "-i", str(sch), "-o", str(tmp / "run.dat"))
# Qucs-S 26.1.1 CLI PNG export leaves the background uninitialised; SVG export is clean.
q("-p", "-i", str(sch), "-o", str(tmp / "sch.svg"))

net = (tmp / "net.cir").read_text()
dat = (tmp / "run.dat").read_text()
vals = {m.group(1): float(m.group(2)) for m in re.finditer(r"<indep (v\([^)]+\)) 1>\s*([-+]?[0-9.]+(?:[eE][-+]?\d+)?)", dat)}
amps = {m.group(1): (m.group(2), m.group(3), m.group(4)) for m in
        re.finditer(r"^B_(\w+) (\S+) 0 V = [0-9.Ee+]+\*V\(([^,]+),([^)]+)\)", net, re.M)}
# (output, non-inverting, inverting) per op-amp
amp_pins = {k: dict(out=o, non_inverting=a, inverting=b) for k, (o, a, b) in amps.items()}
Rf = st["Rf"]; Ip = st["I_plus_A"]; In = st["I_minus_A"]
hand = {"v(tp)": -Rf * Ip, "v(tn)": -Rf * In, "v(s)": Rf * (Ip - In)}
cmp = {k: dict(qucs_ngspice=vals.get(k), hand=h, abs_err=abs(vals.get(k, float("nan")) - h)) for k, h in hand.items()}
topology_ok = (amp_pins.get("OPP", {}).get("non_inverting") == "0" and amp_pins.get("OPN", {}).get("non_inverting") == "0"
               and amp_pins.get("OPD", {}).get("out", "").lower() == "s")
n_res = len(re.findall(r"^R\w+ ", net, re.M)); n_src = len(re.findall(r"^V\w+ ", net, re.M))
res = dict(stamp=stamp(), qucs_s=str(QUCS), qucs_version=net.splitlines()[0],
           components_in_qucs_netlist=dict(resistors=n_res, voltage_sources=n_src, opamps=len(amps)),
           opamp_pins_from_qucs_netlist=amp_pins, inverting_TIAs_and_diffamp_output_ok=topology_ok,
           node_voltages=cmp,
           formulas=dict(tp=f"-Rf*I+ = -{Rf:g} ohm * {Ip*1e6:.4f} uA = {hand['v(tp)']:.6f} V",
                         tn=f"-Rf*I- = -{Rf:g} ohm * {In*1e6:.4f} uA = {hand['v(tn)']:.6f} V",
                         S=f"Rf*(I+ - I-) = {Rf:g} ohm * ({Ip*1e6:.4f} - {In*1e6:.4f}) uA = {hand['v(s)']:.6f} V"),
           note="Resistor values in the .sch are written with 6 significant digits, so ~1e-6 V deviations are expected.",
           ok=topology_ok and all(c["abs_err"] < 1e-4 for c in cmp.values()))
shutil.copy(tmp / "net.cir", C.CIRCUITS / "qucs" / "neuron_stageA_qucs_ngspice.cir")
shutil.copy(tmp / "sch.svg", C.PLOTS / "fig10_qucs_schematic.svg")
edge = next((p for p in (Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
                         Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")) if p.exists()), None)
if edge:                                # rasterise the SVG (as drawn by Qucs-S) on a white page
    (tmp / "v.html").write_text("<html><body style='margin:0;background:white'>"
                                "<img src='sch.svg' style='width:1860px'></body></html>")
    png = C.PLOTS / "fig10_qucs_schematic.png"
    subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--screenshot={png}",
                    "--window-size=1860,720", (tmp / "v.html").as_uri()], timeout=60)
    res["render_png"] = str(png)
else:
    res["render_png"] = "Edge not found; open plots/fig10_qucs_schematic.svg in a browser"
from ns.metrics import save_json  # noqa: E402
save_json(res, C.RESULTS / "s11_qucs_native_check.json")
print(json.dumps({k: v for k, v in res.items() if k != "stamp"}, indent=1))
assert res["ok"], "Qucs-S netlist does not reproduce the Stage A neuron"
