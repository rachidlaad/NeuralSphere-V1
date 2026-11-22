"""Generate physical ngspice netlists of the 64x10 differential crossbar and run them.

Nothing in this module computes the classification: it only writes resistor,
source and amplifier cards, launches ngspice, and parses the node voltages and
branch currents that ngspice solved with modified nodal analysis.
"""
import os, re, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from . import config as C

N_AMPS = 3 * C.N_OUT  # 2 TIAs + 1 difference amplifier per output


def _fmt(x):
    return f"{x:.9g}"


def circuit_cards(Gp, Gn, Rf, opamp="ideal", vos=None, vcc=C.VCC, vee=C.VEE,
                  opa_params="", wta=False, n_in=C.N_IN):
    """Return the netlist body (no analysis / control) as a list of lines.

    Gp, Gn : (n_out, n_in+1) actual conductances [S]; last column = bias synapse.
    vos    : (n_out, 3) input offset voltages [V] for TIA+, TIA-, diff-amp (real model)
    """
    n_out = Gp.shape[0]
    model = {"ideal": "OPA_IDEAL", "real": "OPA_REAL", "opa196": "OPA196"}[opamp]
    L = [f"* NeuralSphere-V1 differential resistive crossbar: {n_in} inputs + bias -> {n_out} outputs",
         f"* amplifier model: {model}",
         f".include \"{(C.MODELS / 'neuralsphere_models.lib').as_posix()}\""]
    if opamp == "opa196":
        if not C.OPA196_LIB.exists():
            raise FileNotFoundError(f"{C.OPA196_LIB} missing. The TI model is not redistributed; fetch it with "
                                    "'powershell -ExecutionPolicy Bypass -File .\\tools\\get_opa196_model.ps1'")
        L += [f".include \"{C.OPA196_LIB.as_posix()}\"",
              f".include \"{(C.MODELS / 'opa196_wrapper.lib').as_posix()}\""]
    L += [
         "* ---- supplies and reference rail",
         f"VCC vcc 0 DC {_fmt(vcc)}",
         f"VEE vee 0 DC {_fmt(vee)}",
         f"VB bias 0 DC {_fmt(C.VBIAS)}",
         "* ---- input voltage sources (one per pixel), values set per image by alter"]
    for i in range(n_in):
        L.append(f"VIN{i} in{i} 0 DC 0")
    inode = [f"in{i}" for i in range(n_in)] + ["bias"]
    L.append("* ---- synapses: one resistor per conductance, R = 1/G")
    for k in range(n_out):
        for i, nd in enumerate(inode):
            L.append(f"RP{k}_{i} {nd} cp{k} {_fmt(1 / Gp[k, i])}")
            L.append(f"RN{k}_{i} {nd} cn{k} {_fmt(1 / Gn[k, i])}")
    L.append("* ---- neurons: two transimpedance amplifiers + one difference amplifier")
    for k in range(n_out):
        ext = ""
        if opamp != "ideal":
            v = (0, 0, 0) if vos is None else vos[k]
            ext = [f" params: VOS={_fmt(v[j])} {opa_params}" for j in range(3)]
        else:
            ext = ["", "", ""]
        L += [f"XTP{k} 0 cp{k} tp{k} vcc vee {model}{ext[0]}",
              f"RFP{k} cp{k} tp{k} {_fmt(Rf)}",
              f"XTN{k} 0 cn{k} tn{k} vcc vee {model}{ext[1]}",
              f"RFN{k} cn{k} tn{k} {_fmt(Rf)}",
              f"RD1_{k} tp{k} dm{k} {_fmt(C.R_DIFF)}",
              f"RD2_{k} dm{k} s{k} {_fmt(C.R_DIFF)}",
              f"RD3_{k} tn{k} dp{k} {_fmt(C.R_DIFF)}",
              f"RD4_{k} dp{k} 0 {_fmt(C.R_DIFF)}",
              f"XD{k} dp{k} dm{k} s{k} vcc vee {model}{ext[2]}"]
    if wta:
        L += wta_cards(n_out)
    return L


def wta_cards(n_out):
    """Pairwise-comparator winner-take-all with diode-AND gates.

    For every pair (k<j) one COMP2 compares s_k with s_j and provides
    q_kj = [s_k > s_j] and q_jk = [s_j > s_k].  Output w_k is pulled up to VCC by
    100 kohm and pulled down through a diode by any q_kj that is LOW, so
    w_k is HIGH only when s_k beats all nine competitors.
    """
    L = ["* ---- analog winner-take-all (45 comparators, 90 diodes, 10 pull-ups)"]
    for k in range(n_out):
        for j in range(k + 1, n_out):
            L.append(f"XC{k}_{j} s{k} s{j} q{k}_{j} q{j}_{k} vcc vee COMP2")
    for k in range(n_out):
        L.append(f"RPU{k} vcc w{k} 100k")
        for j in range(n_out):
            if j != k:
                L.append(f"DW{k}_{j} w{k} q{k}_{j} DSW")
    return L


def out_vectors(n_out=C.N_OUT, wta=False, n_in=C.N_IN):
    v = [f"v(s{k})" for k in range(n_out)]
    v += [f"v(tp{k})" for k in range(n_out)] + [f"v(tn{k})" for k in range(n_out)]
    v += ["i(VCC)", "i(VEE)", "i(VB)"] + [f"i(VIN{i})" for i in range(n_in)]
    if wta:
        v += [f"v(w{k})" for k in range(n_out)]
    return v


def control_block(Vin, vecs, out_file):
    """ngspice control script: for each image set the 64 sources, solve .op, append a row."""
    L = [".control", "set noaskquit", "set wr_singlescale", "set appendwrite",
         "option numdgt=10", f"shell rm -f {out_file}" if os.name != "nt" else ""]
    prev = None
    for r, row in enumerate(Vin):
        for i, v in enumerate(row):
            if prev is None or v != prev[i]:
                L.append(f"alter VIN{i} dc = {_fmt(v)}")
        prev = row
        L.append("op")
        L.append(f"wrdata {out_file} " + " ".join(vecs))
    L += ["quit", ".endc"]
    return L


def sweep_cards(body, Vchunk):
    """Image-index DC sweep (used for slow-converging vendor models).

    Each pixel source VIN_i (kept as a 0 V ammeter so i(VIN_i) is still its current) is
    driven by a behavioural source whose value is a piecewise-linear function of the
    sweep voltage v(sel):  V_i(sel = r) = Vchunk[r, i] exactly at integer r.
    A `dc VSEL 0 n-1 1` sweep then solves image after image, each Newton solve starting
    from the previous image's operating point instead of from zero."""
    out = []
    for l in body:
        mm = re.match(r"^VIN(\d+) in\d+ 0 DC 0$", l)
        if mm:
            i = int(mm.group(1)); col = Vchunk[:, i]
            if len(col) == 1:
                pts = f"{_fmt(col[0])}"
            else:
                pts = "pwl(v(sel), " + ", ".join(f"{r}, {_fmt(v)}" for r, v in enumerate(col)) + ")"
            out += [f"VIN{i} in{i} xin{i} DC 0", f"BIN{i} xin{i} 0 V = {pts}"]
        else:
            out.append(l)
    return out + ["VSEL sel 0 DC 0", "RSEL sel 0 1meg"]


def sweep_control(n, vecs, out_file):
    return [".control", "set noaskquit", "set wr_singlescale", "option numdgt=10",
            f"dc VSEL 0 {n - 1} 1" if n > 1 else "op",
            f"wrdata {out_file} " + " ".join(vecs), "quit", ".endc"]


def run(Gp, Gn, Rf, Vin, opamp="ideal", vos=None, vcc=C.VCC, vee=C.VEE, opa_params="",
        wta=False, workers=None, chunk=200, keep_netlist=None, workdir=None, extra=(), sweep=None):
    """Simulate a batch of images. Vin: (n_img, 64) input voltages [V].

    opamp : "ideal" | "real" (own macromodels) | "opa196" (TI vendor model, PSpice mode)
    extra : additional ngspice vectors to record, e.g. ("v(cp0)", "v(cn0)"); returned in
            res["extra"][name] as (n,) arrays. Default () keeps the original behaviour.
    sweep : None -> automatic (True for vendor models), else force on/off. Same circuit and
            same equations; only the order/initial guess of ngspice's Newton solves differs.
    Returns dict of arrays: s (n,10), tp, tn, i_vcc, i_vee, i_vb, i_in (n,64), [w], [extra].
    """
    sweep = (opamp in VENDOR_MODELS) if sweep is None else sweep
    Vin = np.atleast_2d(np.asarray(Vin, float))
    body = circuit_cards(Gp, Gn, Rf, opamp, vos, vcc, vee, opa_params, wta)
    vecs = out_vectors(Gp.shape[0], wta) + list(extra)
    workdir = Path(workdir or tempfile.mkdtemp(prefix="ns_spice_"))
    workdir.mkdir(parents=True, exist_ok=True)
    prepare_workdir(workdir, opamp)
    chunks = [Vin[a:a + chunk] for a in range(0, len(Vin), chunk)]

    def job(ci):
        net = workdir / f"net_{ci}.cir"
        out = workdir / f"out_{ci}.txt"
        if out.exists():
            out.unlink()
        if sweep:
            # row 0 = blank image (all pixel sources 0 V): a defined power-up state from which
            # every sweep starts; its solution is discarded below.
            Vs = np.vstack([np.zeros((1, chunks[ci].shape[1])), chunks[ci]])
            lines = sweep_cards(body, Vs) + sweep_control(len(Vs), vecs, out.as_posix()) + [".end"]
        else:
            lines = body + control_block(chunks[ci], vecs, out.as_posix()) + [".end"]
        net.write_text("\n".join(l for l in lines if l is not None) + "\n")
        p = subprocess.run([C.ngspice_exe(), "-b", str(net)], capture_output=True, text=True, cwd=workdir)
        if not out.exists():
            raise RuntimeError(f"ngspice failed (chunk {ci}):\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
        if sweep:
            return parse_wrdata(out, len(vecs), len(chunks[ci]) + 1)[1:]
        return parse_wrdata(out, len(vecs), len(chunks[ci]))

    workers = workers or min(len(chunks), os.cpu_count() or 2)
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(job, range(len(chunks))))
    if keep_netlist:
        lines = body + control_block(Vin[:1], vecs, "single_image_output.txt") + [".end"]
        Path(keep_netlist).write_text("\n".join(lines) + "\n")
    A = np.vstack(parts)
    n = C.N_OUT
    res = dict(s=A[:, 0:n], tp=A[:, n:2 * n], tn=A[:, 2 * n:3 * n],
               i_vcc=A[:, 3 * n], i_vee=A[:, 3 * n + 1], i_vb=A[:, 3 * n + 2],
               i_in=A[:, 3 * n + 3:3 * n + 3 + C.N_IN])
    base = 3 * n + 3 + C.N_IN
    if wta:
        res["w"] = A[:, base:base + n]
        base += n
    if extra:
        res["extra"] = {name: A[:, base + j] for j, name in enumerate(extra)}
    res["vin"] = Vin
    res["vcc"], res["vee"] = vcc, vee
    return res


def prepare_workdir(workdir, opamp):
    """Vendor (PSpice) models need ngspice's PSpice compatibility mode. ngspice reads
    .spiceinit from its working directory before parsing the netlist, so we put one
    there for vendor models only (our own models keep the default ngspice behaviour)."""
    init = Path(workdir) / ".spiceinit"
    if opamp in VENDOR_MODELS:
        init.write_text("set ngbehavior=ps\n")
    elif init.exists():
        init.unlink()


VENDOR_MODELS = ("opa196",)


def parse_wrdata(path, nvec, nrows):
    """wrdata with wr_singlescale writes: scale value, then one column per vector."""
    rows = []
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if not t or not re.match(r"^[-+0-9.]", t[0]):
            continue
        vals = [float(x) for x in t]
        rows.append(vals[-nvec:])
    A = np.array(rows)
    if A.shape != (nrows, nvec):
        raise RuntimeError(f"unexpected wrdata shape {A.shape}, expected {(nrows, nvec)}")
    return A


def power(res):
    """Electrical power delivered by every source, P = V * I  [W].

    ngspice reports i(V) as current flowing INTO the + terminal, so the power
    delivered by a source is  -V * i(V).
    """
    p_vcc = -res["vcc"] * res["i_vcc"]
    p_vee = -res["vee"] * res["i_vee"]
    p_vb = -C.VBIAS * res["i_vb"]
    p_in = -(res["vin"] * res["i_in"]).sum(1)
    return dict(p_vcc=p_vcc, p_vee=p_vee, p_bias=p_vb, p_inputs=p_in,
                p_total=p_vcc + p_vee + p_vb + p_in)
