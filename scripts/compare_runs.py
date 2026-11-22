"""Compare a fresh run of the pipeline against an archived reference run.

Usage:  python compare_runs.py <reference_dir> [<report.md>]
        reference_dir = e.g. ../archive/linux_ngspice47_2026-11-21

For every results/*.json present in both runs, every numeric leaf is compared:
    abs_diff = |new - ref|,   rel_diff = abs_diff / max(|ref|, 1e-30)
For every results/*.npz and weights/*.npz, every common array is compared (max abs diff).
Then the pass/fail checks that the HANDOFF defines are evaluated on the NEW run.
Nothing here re-simulates anything: it only reads saved files.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "archive" / "linux_ngspice47_2026-11-21"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "logs" / "compare_vs_reference.md"
SKIP_KEYS = {"stamp", "time", "host", "platform", "python", "ngspice", "ngspice_version"}


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in SKIP_KEYS:
                continue
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, o


def cmp_json(a, b):
    A, B = dict(leaves(a)), dict(leaves(b))
    rows, missing = [], sorted(set(A) ^ set(B))
    for k in sorted(set(A) & set(B)):
        x, y = A[k], B[k]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            if x != y:
                rows.append((k, x, y, None, None))
            continue
        d = abs(y - x)
        if d > 0:
            rows.append((k, x, y, d, d / max(abs(x), 1e-30)))
    return rows, missing


def main():
    lines = [f"# Comparison: new run vs reference `{REF.name}`", ""]
    worst = []
    lines += ["## JSON result files", "",
              "| file | numeric leaves differing | max abs diff | max rel diff | non-numeric changes | keys only in one run |",
              "|---|---|---|---|---|---|"]
    details = []
    for f in sorted((ROOT / "results").glob("*.json")):
        rf = REF / "results" / f.name
        if not rf.exists():
            lines.append(f"| {f.name} | (no reference) | | | | |"); continue
        rows, missing = cmp_json(json.load(open(rf)), json.load(open(f)))
        num = [r for r in rows if r[3] is not None]; other = [r for r in rows if r[3] is None]
        ma = max((r[3] for r in num), default=0.0); mr = max((r[4] for r in num), default=0.0)
        lines.append(f"| {f.name} | {len(num)} | {ma:.3g} | {mr:.3g} | {len(other)} | {len(missing)} |")
        top = sorted(num, key=lambda r: -r[4])[:8]
        if top or other or missing:
            details += [f"### {f.name}", "", "| key | reference | new | abs diff | rel diff |", "|---|---|---|---|---|"]
            details += [f"| `{k}` | {x:.10g} | {y:.10g} | {d:.3g} | {r:.3g} |" for k, x, y, d, r in top]
            details += [f"| `{k}` | {x} | {y} | changed | |" for k, x, y, _, _ in other[:8]]
            details += [f"| `{k}` | key present in only one run | | | |" for k in missing[:8]]
            details.append("")
    lines += ["", "## NPZ arrays (max |new - ref| over all elements)", "", "| file | array | shape | max abs diff |", "|---|---|---|---|"]
    for sub in ("results", "weights"):
        for f in sorted((ROOT / sub).glob("*.npz")):
            rf = REF / sub / f.name
            if not rf.exists():
                continue
            a, b = np.load(rf), np.load(f)
            for k in sorted(set(a.files) & set(b.files)):
                x, y = a[k], b[k]
                if x.shape != y.shape:
                    lines.append(f"| {sub}/{f.name} | {k} | {x.shape} vs {y.shape} | SHAPE CHANGED |"); continue
                if x.dtype.kind in "fiub" and x.size:
                    d = float(np.abs(y.astype(float) - x.astype(float)).max())
                    lines.append(f"| {sub}/{f.name} | {k} | {x.shape} | {d:.3g} |")

    # ---- checks on the NEW run ----------------------------------------------------
    R = ROOT / "results"
    J = lambda n: json.load(open(R / n))
    checks = []
    def chk(name, ok, val):
        checks.append((name, bool(ok), val))
    s0 = J("s00_validate_ngspice.json"); chk("s00 divider: |V_spice - V_hand| < 1 uV", s0["passed"], f"{s0['abs_err_V']:.3g} V")
    s1 = J("s01_stageA_single_neuron.json")
    chk("s01 Stage A ideal: |S_spice - S_hand| < 10 uV", abs(s1["err_ideal_V"]) < 1e-5, f"{s1['err_ideal_V']:.3g} V")
    chk("s01 Stage A real: |S_spice - S_hand| < 100 uV", abs(s1["err_real_V"]) < 1e-4, f"{s1['err_real_V']:.3g} V")
    s2 = J("s02_software_reference.json")
    chk("s02 software softmax test = 348/360", s2["softmax_sgd"]["test"]["correct"] == 348, s2["softmax_sgd"]["test"]["correct"])
    chk("s02 sklearn baseline test = 344/360", s2["sklearn_logreg_C1"]["test"]["correct"] == 344, s2["sklearn_logreg_C1"]["test"]["correct"])
    s3 = J("s03_stageBC_verification.json")
    chk("s03 SPICE ideal test = 348/360", s3["test_ideal"]["correct"] == 348, s3["test_ideal"]["correct"])
    chk("s03 SPICE vs software disagreements = 0", s3["prediction_disagreements_spice_vs_software"] == 0, s3["prediction_disagreements_spice_vs_software"])
    chk("s03 max |S_spice - S_kcl| < 100 uV", s3["max_abs_err_spice_vs_kcl_V"] < 1e-4, f"{s3['max_abs_err_spice_vs_kcl_V']:.3g} V")
    s4 = J("s04_stageD_inference.json")
    for t in ("D1_real_continuous", "D2_real_8bit"):
        chk(f"s04 {t} test = 348/360", s4[t]["test"]["correct"] == 348, s4[t]["test"]["correct"])
        chk(f"s04 {t} disagreements vs software = 0", s4[t]["disagreements_vs_software"] == 0, s4[t]["disagreements_vs_software"])
    s7 = J("s07_transient_energy.json")
    dc_pred = np.load(R / "s04_D1_real_continuous_outputs.npz")["pred_te"]
    chk("s07 transient: final prediction equals DC (.op) prediction for all 6 images",
        all(p["predicted"] == int(dc_pred[p["test_index"]]) for p in s7["per_image"]),
        [(p["true"], p["predicted"]) for p in s7["per_image"]])
    s8 = J("s08_wta.json")
    chk("s08 WTA test = 348/360", s8["wta_test"]["correct"] == 348, s8["wta_test"]["correct"])
    chk("s08 WTA ambiguous = 0", s8["ambiguous"] == 0, s8["ambiguous"])
    chk("s08 WTA vs argmax disagreements = 0", s8["wta_vs_argmax_disagreements"] == 0, s8["wta_vs_argmax_disagreements"])
    s10 = J("s10_qucs_schematic_check.json"); chk("s10 schematic-extracted S matches hand value", s10["ok"], s10["S_extracted_V"])
    head = ["## Pass/fail checks on the new run", "", "| check | result | value |", "|---|---|---|"]
    head += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {v} |" for n, ok, v in checks]
    text = "\n".join(lines[:2] + head + [""] + lines[2:] + ["", "## Largest differences per file", ""] + details) + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print("FAILED CHECKS:", sum(not ok for _, ok, _ in checks))


if __name__ == "__main__":
    main()
