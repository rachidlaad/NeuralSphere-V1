# NeuralSphere-V1: a trainable analog neural processor (SPICE simulation)

A 64-input, 10-output **differential resistive crossbar** with op-amp transimpedance neurons. It classifies the 8×8 handwritten digits from `sklearn.datasets.load_digits`. Every inference result reported here is the voltage that **ngspice** solved for the physical netlist: 1300 synapse resistors and 30 op-amps.

> This is a circuit **simulation**, not a fabricated or breadboarded device. See `HANDOFF.md` §1 and §8.

## Headline results (360 held-out test images)

| | Test accuracy | Source file |
|---|---|---|
| sklearn LogisticRegression C=1 (baseline) | 344/360 = 95.56 % | `results/s02_software_reference.json` |
| Software softmax (our training) | 348/360 = 96.67 % | same |
| **SPICE, ideal op-amps** | **348/360**, 0 disagreements with software | `results/s03_stageBC_verification.json` |
| **SPICE, realistic op-amp macromodel** | **348/360**, macro-F1 0.966 | `results/s04_stageD_inference.json` |
| SPICE, realistic op-amps + 8-bit conductances | 348/360 | same |
| SPICE + analog winner-take-all (45 comparators) | 348/360, 0 ambiguous | `results/s08_wta.json` |
| Harsh chip (2-bit, 20 % mismatch, 20 mV offsets), after SPICE-in-the-loop fine-tuning | 342 → 346 /360 | `results/s05_stageE_in_loop.json` |

The robustness sweep, power, latency and energy figures are in `HANDOFF.md` §4 and `plots/`.

## Quick start (Windows)

```powershell
cd C:\Users\intel\Documents\NeuralSphere-V1
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Quick
.\.venv\Scripts\python.exe demo\demo_app.py
```

- `run_all.ps1` creates `.venv` if it is missing and finds ngspice under `tools\`.
- Without `-Quick` it also runs the in-loop training and the full non-ideality sweep (1–2 h).
- It overwrites `results\` and `plots\`.

**Install from scratch on another machine:**
1. Qucs-S portable from https://ra3xdh.github.io/. Unpack it to `tools\qucs-s`; it bundles `ngspice_con.exe`.
2. Or install ngspice from https://ngspice.sourceforge.io/. Unpack it to `tools\ngspice`, or set the `NGSPICE` environment variable.
3. Python 3.10+ and `pip install -r requirements.txt`.

**Linux/macOS:** install ngspice, then run `cd scripts && python3 s00_validate_ngspice.py` and the rest in numeric order (s09 last).

## Pipeline

| Script | Stage |
|---|---|
| `s00_validate_ngspice.py` | voltage divider vs hand calculation |
| `s01_stageA_single_neuron.py` | Stage A: one 4-input neuron, hand calculation vs SPICE |
| `s02_train_software.py` | baseline + softmax training (explicit gradient) |
| `s03_stageBC_build_verify.py` | Stages B/C: full netlist, Kirchhoff and software correspondence, KCL audit |
| `s04_stageD_inference.py` | Stage D: externally trained, circuit-executed inference (realistic op-amps) |
| `s05_stageE_in_loop.py` | Stage E: SPICE-in-the-loop fine-tuning on simulated defective chips |
| `s06_nonidealities.py` | quantization, tolerance, noise, offset, saturation, supply |
| `s07_transient_energy.py` | transient latency, P(t), energy per inference |
| `s08_wta.py` | analog winner-take-all |
| `s10_qucs_schematic.py` | Qucs-S schematic of one neuron + connectivity check |
| `s09_figures.py` | all figures from saved results |

See `docs/DESIGN.md` for the physics, formulas and worked numbers, and `HANDOFF.md` for the full status, problems and next steps.
