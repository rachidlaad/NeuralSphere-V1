"""Generate a Qucs-S schematic of one representative analog neuron (Stage A neuron:
4 pixel inputs + bias rail, differential conductance pairs, two transimpedance
amplifiers and a difference amplifier) and an equivalent SPICE netlist derived
from the SAME geometric description, so the drawing's connectivity can be checked
in ngspice.

Qucs pin geometry used (component placed at X,Y):
  R      rot 0: (X-30,Y) (X+30,Y)      rot 1: (X,Y-30) (X,Y+30)
  Vdc    rot 1: + at (X,Y-30), - at (X,Y+30)
  OpAmp  non-inverting (X-30,Y-20), inverting (X-30,Y+20), output (X+40,Y)
  GND    (X,Y)
"""
import json, subprocess, re
import numpy as np
from _common import C, ROOT

st = json.load(open(C.RESULTS / "s01_stageA_single_neuron.json"))
V = st["x"] + [1.0]                      # 4 pixel voltages + bias rail (all VMAX = 1 V)
Rp = [r["Rp"] for r in st["synapses"]]
Rn = [r["Rn"] for r in st["synapses"]]
Rf = st["Rf"]; Rd = C.R_DIFF

comps, wires, texts = [], [], []
pins = {}                                # name -> list of pin points
labels = []


def R(name, x, y, val, rot):
    comps.append(f'<R {name} 1 {x} {y} 15 -26 0 {rot} "{val:.6g} Ohm" 1 "26.85" 0 "0.0" 0 "0.0" 0 "26.85" 0 "european" 0>')
    pins[name] = [(x - 30, y), (x + 30, y)] if rot == 0 else [(x, y - 30), (x, y + 30)]


def Vdc(name, x, y, val):
    comps.append(f'<Vdc {name} 1 {x} {y} 18 -26 0 1 "{val:g} V" 1>')
    pins[name] = [(x, y - 30), (x, y + 30)]


def OP(name, x, y):
    comps.append(f'<OpAmp {name} 1 {x} {y} -26 42 0 0 "1e6" 1 "5 V" 0>')
    pins[name] = [(x - 30, y - 20), (x - 30, y + 20), (x + 40, y)]


gnd_pts = []


def GND(x, y):
    comps.append(f"<GND * 5 {x} {y} 0 0 0 0>")
    gnd_pts.append((x, y))


def Wr(a, b, label=""):
    if label:
        wires.append(f'<{a[0]} {a[1]} {b[0]} {b[1]} "{label}" {a[0]+10} {a[1]-30} 0 "">')
    else:
        wires.append(f'<{a[0]} {a[1]} {b[0]} {b[1]} "" 0 0 0 "">')


YP, YN = 400, 520                         # horizontal column (bit-line) wires
X0, DX = 100, 140
xs = [X0 + DX * i for i in range(5)]
for i, x in enumerate(xs):
    nm = f"VIN{i}" if i < 4 else "VBIAS"
    Vdc(nm, x - 60, 230, V[i]); GND(x - 60, 280); Wr((x - 60, 260), (x - 60, 280))
    Wr((x - 60, 200), (x - 60, 180)); Wr((x - 60, 180), (x, 180), label=f"in{i}" if i < 4 else "bias")
    Wr((x, 180), (x + 40, 180))
    R(f"RP{i}", x, 370, Rp[i], 1)         # pins (x,340) (x,400)
    Wr((x, 180), (x, 340))
    R(f"RN{i}", x + 40, 490, Rn[i], 1)     # pins (x+40,460) (x+40,520)
    Wr((x + 40, 180), (x + 40, 460))
# column wires as segments joining successive resistor pins
xe = xs[-1] + 120
for a, b in zip(xs, xs[1:] + [xe]):
    Wr((a, YP), (b, YP))
for a, b in zip([x + 40 for x in xs], [x + 40 for x in xs[1:]] + [xe]):
    Wr((a, YN), (b, YN))
labels += [((xe, YP), "cp"), ((xe, YN), "cn")]
# TIA+ : inverting input on YP
XT = xe + 80
OP("OPP", XT, YP - 20)                    # +(XT-30,YP-40) -(XT-30,YP) out(XT+40,YP-20)
GND(XT - 60, YP - 40); Wr((XT - 60, YP - 40), (XT - 30, YP - 40))
R("RFP", XT + 5, YP - 90, Rf, 0)          # pins (XT-25, YP-90) (XT+35, YP-90)
Wr((xe, YP), (XT - 30, YP)); Wr((xe, YP), (xe, YP - 90)); Wr((xe, YP - 90), (XT - 25, YP - 90))
Wr((XT + 35, YP - 90), (XT + 60, YP - 90)); Wr((XT + 60, YP - 90), (XT + 60, YP - 20))
Wr((XT + 40, YP - 20), (XT + 60, YP - 20), label="tp")
# TIA- : inverting input on YN
OP("OPN", XT, YN - 20)
GND(XT - 60, YN - 40); Wr((XT - 60, YN - 40), (XT - 30, YN - 40))
R("RFN", XT + 5, YN + 60, Rf, 0)          # pins (XT-25, YN+60) (XT+35, YN+60)
Wr((xe, YN), (XT - 30, YN)); Wr((xe, YN), (xe, YN + 60)); Wr((xe, YN + 60), (XT - 25, YN + 60))
Wr((XT + 35, YN + 60), (XT + 60, YN + 60)); Wr((XT + 60, YN + 60), (XT + 60, YN - 20))
Wr((XT + 40, YN - 20), (XT + 60, YN - 20), label="tn")
# difference amplifier: S = v(tn) - v(tp)
XD = XT + 260; YD = 460
OP("OPD", XD, YD)                         # +(XD-30,YD-20) -(XD-30,YD+20) out(XD+40,YD)
R("RD3", XT + 120, YN - 20, Rd, 0)        # tn -> dp   pins (XT+90,..) (XT+150,..)
Wr((XT + 60, YN - 20), (XT + 90, YN - 20))
Wr((XT + 150, YN - 20), (XD - 60, YN - 20)); Wr((XD - 60, YN - 20), (XD - 60, YD - 20)); Wr((XD - 60, YD - 20), (XD - 30, YD - 20))
R("RD4", XD - 60, YD - 70, Rd, 1)         # dp -> gnd  pins (XD-60,YD-100) (XD-60,YD-40)
Wr((XD - 60, YD - 40), (XD - 60, YD - 20)); GND(XD - 60, YD - 120); Wr((XD - 60, YD - 100), (XD - 60, YD - 120))
R("RD1", XT + 120, YP - 20, Rd, 0)        # tp -> dm
Wr((XT + 60, YP - 20), (XT + 90, YP - 20))
Wr((XT + 150, YP - 20), (XD - 100, YP - 20)); Wr((XD - 100, YP - 20), (XD - 100, YD + 20)); Wr((XD - 100, YD + 20), (XD - 30, YD + 20))
R("RD2", XD + 5, YD + 80, Rd, 0)          # dm -> S
Wr((XD - 100, YD + 20), (XD - 100, YD + 80)); Wr((XD - 100, YD + 80), (XD - 25, YD + 80))
Wr((XD + 35, YD + 80), (XD + 70, YD + 80)); Wr((XD + 70, YD + 80), (XD + 70, YD))
Wr((XD + 40, YD), (XD + 70, YD)); Wr((XD + 70, YD), (XD + 110, YD), label="S")
comps.append('<.DC DC1 1 100 640 0 40 0 0 "26.85" 0 "0.001" 0 "1 pA" 0 "1 uV" 0 "no" 0 "150" 0 "no" 0 "none" 0 "CroutLU" 0>')
texts += [f'<Text 60 40 16 #000000 0 "NeuralSphere-V1: one analog neuron (Stage A)">',
          f'<Text 60 70 11 #000000 0 "4 pixel inputs + bias rail; each signed weight = conductance pair (RP to column cp, RN to column cn)\\nOPP/OPN: transimpedance amplifiers hold cp, cn at virtual ground: v(tp) = -Rf*I+, v(tn) = -Rf*I-\\nOPD: difference amplifier, S = v(tn) - v(tp) = Rf*(I+ - I-).  Expected S = {st["S_hand_V"]:.4f} V (Stage A hand calculation)">',
          f'<Text {XD-40} {YD+130} 11 #000000 0 "analog inference hardware: inputs -> S\\nmeasurement interface: DC simulation / probe on S">']

sch = ["<Qucs Schematic 1.0.0>", "<Properties>", "  <View=0,0,1700,800,1,0,0>", "  <Grid=10,10,1>",
       "  <DataSet=neuron_stageA.dat>", "  <DataDisplay=neuron_stageA.dpl>", "  <OpenDisplay=0>",
       "  <Script=neuron_stageA.m>", "  <RunScript=0>", "  <showFrame=0>", "</Properties>",
       "<Symbol>", "</Symbol>", "<Components>"] + ["  " + c for c in comps] + ["</Components>", "<Wires>"] + \
      ["  " + w for w in wires] + ["</Wires>", "<Diagrams>", "</Diagrams>", "<Paintings>"] + \
      ["  " + t for t in texts] + ["</Paintings>"]
out = C.CIRCUITS / "qucs" / "neuron_stageA.sch"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(sch) + "\n")

# ---- connectivity extraction from the drawing (union-find on points) ----------
par = {}


def find(p):
    par.setdefault(p, p)
    while par[p] != p:
        par[p] = par[par[p]]; p = par[p]
    return p


def union(a, b):
    par[find(a)] = find(b)


segs = []
for w in wires:
    a = list(map(int, re.match(r"<(-?\d+) (-?\d+) (-?\d+) (-?\d+)", w).groups()))
    segs.append(((a[0], a[1]), (a[2], a[3]))); union((a[0], a[1]), (a[2], a[3]))
allpts = {p for s in segs for p in s} | {p for v in pins.values() for p in v} | set(gnd_pts)
for p in allpts:                                    # endpoints lying on another wire's interior join it
    for a, b in segs:
        if a[0] == b[0] == p[0] and min(a[1], b[1]) < p[1] < max(a[1], b[1]) or \
           a[1] == b[1] == p[1] and min(a[0], b[0]) < p[0] < max(a[0], b[0]):
            union(p, a)
g0 = find(gnd_pts[0])
for p in gnd_pts:
    union(p, g0)
names = {}


def net(p):
    r = find(p)
    if r == find(gnd_pts[0]):
        return "0"
    return names.setdefault(r, f"n{len(names)+1}")


for p, lab in labels:
    names[find(p)] = lab
for w in wires:
    m = re.match(r'<(-?\d+) (-?\d+) \S+ \S+ "([^"]+)"', w)
    if m:
        names[find((int(m.group(1)), int(m.group(2))))] = m.group(3)
cards = ["* netlist extracted from neuron_stageA.sch geometry",
         "* OpAmp = Qucs ideal op-amp (gain 1e6, output limited to +/-5 V)",
         ".subckt QOPAMP inp inn out", "B1 out 0 V = 5*tanh(1e6*(v(inp)-v(inn))/5)", ".ends"]
vals = {**{f"VIN{i}": V[i] for i in range(4)}, "VBIAS": V[4]}
for c in comps:
    t = c.split()
    kind, name = t[0][1:], t[1]
    if kind == "R":
        val = float(re.search(r'"([0-9.e+-]+) Ohm"', c).group(1))
        a, b = pins[name]; cards.append(f"{name} {net(a)} {net(b)} {val:.9g}")
    elif kind == "Vdc":
        a, b = pins[name]; cards.append(f"{name} {net(a)} {net(b)} DC {vals[name]}")
    elif kind == "OpAmp":
        a, b, o = pins[name]; cards.append(f"X{name} {net(a)} {net(b)} {net(o)} QOPAMP")
cards += [".control", "op", "print v(S) v(tp) v(tn) v(cp) v(cn)", "quit", ".endc", ".end"]
cir = C.CIRCUITS / "qucs" / "neuron_stageA_extracted.cir"
cir.write_text("\n".join(cards) + "\n")
o = subprocess.run([C.ngspice_exe(), "-b", str(cir)], capture_output=True, text=True).stdout
vS = float(re.search(r"v\(s\)\s*=\s*([-+0-9.eE]+)", o).group(1))
print("\n".join(cards[:12])); print(o[-400:])
print(f"S from schematic geometry = {vS:.6f} V ; Stage A hand value = {st['S_hand_V']:.6f} V")
json.dump(dict(S_extracted_V=vS, S_hand_V=st["S_hand_V"], ok=abs(vS - st["S_hand_V"]) < 1e-4),
          open(C.RESULTS / "s10_qucs_schematic_check.json", "w"), indent=1)
