# NeuralSphere-V1: Handoff

**Project folder (Windows):** `C:\Users\intel\Documents\NeuralSphere-V1`
**Status date:** 2026-11-21
**State:** V1 milestone reached in simulation. A 64-input, 10-output differential resistive crossbar with op-amp neurons recognises handwritten digits: **348/360 = 96.67 %** on the held-out test set, measured from ngspice circuit solutions.
**What this is:** SPICE circuit simulation. **What this is not:** a fabricated or breadboarded neuromorphic processor.

---

## 1. Honest summary of what was done where

| Item | Where it ran | Status |
|---|---|---|
| Windows inspection | Your PC (Windows 11 build 22631, Python 3.12.10 at `C:\Python312`, no SPICE before) | done |
| Qucs-S 26.1.1 portable (official GitHub release) | Your PC: `tools\qucs-s\bin\qucs-s.exe` | **downloaded and unpacked; not opened yet** (see §8) |
| ngspice on Windows | `tools\qucs-s\bin\ngspice_con.exe`, bundled with Qucs-S (its binary identifies as **ngspice-46**) | present; **not yet run on Windows** |
| Official ngspice-47 Windows build | `tools\setup_ngspice.ps1` would download it | **not run** (see §8) |
| Python venv (numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, pandas 3.1.1, scikit-learn 1.9.1) | Your PC: `.venv\` | done (`tools\setup_log.txt`) |
| ngspice-47 source tarball (SHA-256 `894e6496…675f`) | Downloaded on your PC from SourceForge, compiled from source in Claude's Linux workspace | done |
| **All simulations, training and figures in `results\` and `plots\`** | Claude's Linux workspace, **ngspice-47** (KLU solver), Python 3.13 | done |

Every number below comes from a saved file in `results\`. Each JSON has a `stamp` field (host, ngspice version, time).

**Not reproduced on Windows yet.** Running `run_all.ps1` (§7) on your PC is the first next step.

---

## 2. Architecture

```
x_i = pixel_i/16  ->  V_i = 1 V * x_i  (64 DC sources)  +  1 V bias rail
          |
   1300 synapse resistors: for each output k and input i (65 inputs incl. bias)
      R+_ki from input i to column cp_k ;  R-_ki from input i to column cn_k
          |
   per output k:  TIA+ (op-amp, Rf from cp_k to tp_k, + input grounded) -> tp_k = -Rf * I+_k
                  TIA- (same on cn_k)                                   -> tn_k = -Rf * I-_k
                  difference amp (4 x 100 kohm)                         -> S_k  = tn_k - tp_k
          |
   S_0..S_9 -> (a) measurement program takes argmax       OR
               (b) analog WTA: 45 complementary comparators + 90 diodes + 10 pull-ups -> w_k HIGH for winner
```

- **Analog inference hardware:** input sources, synapse resistors, TIAs, difference amplifiers, WTA.
- **Programming interface (software):** `src/ns/mapping.py` turns weights into resistor values; `src/ns/spice.py` writes them into the netlist.
- **Measurement interface:** ngspice `wrdata` output, parsed by `spice.parse_wrdata`.

**Circuit size of the full network** (`circuits/generated/full_64x10_*.cir`):

| Part | Count |
|---|---|
| Synapse resistors | 1300 |
| Feedback resistors | 20 |
| Difference-amplifier resistors | 40 |
| Op-amps | 30 |
| Voltage sources (64 inputs + bias + VCC + VEE) | 67 |
| WTA (added): comparators / diodes / pull-ups | 45 / 90 / 10 |

### Design values

| Quantity | Value | Where set |
|---|---|---|
| Input encoding | V_i = 1 V × pixel/16 | `config.VMAX` |
| Bias rail | 1 V | `config.VBIAS` |
| Programmable R range | 10 kΩ … 1 MΩ (G 1 µS … 100 µS) | `config.R_MIN/R_MAX` |
| Weight→conductance scale α | 42.394 µS per unit weight (max\|w\| = 2.3352) | `mapping.Mapping` |
| Feedback R_f | 5.6 kΩ (E24 value keeping TIA output ≤ 3.5 V on train set; max column current 570.4 µA) | `mapping.Mapping` |
| Score scale | 0.2374 V per logit unit | R_f·α·V_max |
| Supplies | ±5 V | `config.VCC/VEE` |
| Difference-amp resistors | 100 kΩ | `config.R_DIFF` |

---

## 3. Formulas

- **Ohm (one synapse):** I⁺_ki = G⁺_ki · V_i, because the column node is held at virtual ground (~µV).
- **KCL (column sum):** I⁺_k = Σ_i G⁺_ki V_i. The TIA gives tp_k = −R_f I⁺_k (likewise tn_k).
- **Neuron output:** S_k = tn_k − tp_k = R_f (I⁺_k − I⁻_k).
- **Signed weights:**
  - G⁺ = G_min + α·max(w,0) and G⁻ = G_min + α·max(−w,0)
  - So G⁺ − G⁻ = α·w and S_k = R_f·α·V_max·z_k
  - α = (G_max − G_min)/max|w|
- **Software model:** z_k = Σ w_ki x_i + b_k; p = softmax(z); L = −log p_target; ∂L/∂w_ki = (p_k − 1[k=t]) x_i; w ← w − η∇.
- **Quantization (n bits):** G snapped to G_min + j·(G_max − G_min)/(2ⁿ − 1), j = 0 … 2ⁿ − 1.
- **Device tolerance:** G_actual = G·(1 + U(−t, t)) (uniform) or G·(1 + N(0, t)) (Gaussian), independent per resistor.
- **Power:** P = Σ_sources −V·i(V). ngspice's i(V) is the current into the + terminal.
- **Energy:** E = ∫P dt over one image period (numerical trapezoid integration).

A worked numeric example of every step is in `docs/DESIGN.md` §3–5.

---

## 4. Experiments and results

### Stage 0: simulator validation (`results/s00_validate_ngspice.json`)
5 V across a 10 kΩ / 4.7 kΩ divider. ngspice gives 1.598639 V; the hand calculation gives 5·4.7/14.7 = 1.598639 V. **Pass.**

### Stage A: one neuron (`results/s01_stageA_single_neuron.json`)
Setup: w = [1, −0.5, 0.25, −2], b = 0.3, x = [1, 0.5, 0.25, 1].

| Quantity | Value |
|---|---|
| I⁺ | 71.194 µA |
| I⁻ | 115.125 µA |
| S (hand calculation) | −0.439312 V |
| S (SPICE, ideal op-amp) | −0.43931 V |
| S (SPICE, realistic macromodel) | −0.43929 V |
| Column node voltage | 0.7 µV (virtual ground confirmed) |

### Software reference (`results/s02_software_reference.json`)

| Model | Train | Test | Macro-F1 |
|---|---|---|---|
| sklearn LogisticRegression C=1, x/16 (your baseline) | – | **344/360 = 95.56 %** (reproduced exactly) | 0.9549 |
| Own softmax SGD (lr 0.5, batch 64, 200 epochs, L2 1e-3; chosen on a validation split from the train set) | 1410/1437 | **348/360 = 96.67 %** | 0.9659 |

### Stages B and C: full netlist, software vs SPICE (`results/s03_stageBC_verification.json`)
Ideal op-amps, all 1797 images.

| Check | Result |
|---|---|
| Max \|S_SPICE − S_Kirchhoff\| | 16.4 µV |
| Max error in logit units | 6.9·10⁻⁵ |
| Prediction disagreements with software | **0** |
| Resistor values in netlist vs 1/G | agree to 5·10⁻⁹ relative |
| KCL audit (sum of branch currents vs current through R_f) | 197.376 µA vs 197.375 µA |
| Accuracy | 348/360 test, 1410/1437 train |

### Stage D: externally trained, circuit-executed inference (`results/s04_stageD_inference.json`)
Realistic op-amp macromodel: A_OL 10⁵, GBW 1 MHz, rail headroom 0.2 V, I_q 0.5 mA, R_out 50 Ω.

| Configuration | Test | Train | Macro-F1 | Disagree vs software | Max logit error |
|---|---|---|---|---|---|
| D1: continuous conductances | **348/360 (96.67 %)** | 1410/1437 | 0.9659 | 0 | 8.3·10⁻⁴ |
| D2: 8-bit conductance levels | **348/360** | 1410/1437 | 0.9659 | 0 | 0.023 |

- **Per-digit test accuracy (D1):**

  | Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Accuracy | 97.2 % | **80.6 %** | 100 % | 100 % | 100 % | 100 % | 97.2 % | 100 % | 94.3 % | 97.2 % |

- Digit 1 is the weak class. The confusion matrix is in `plots/fig3_confusion_matrix.png`.
- **Mean power (D1, from SPICE currents): 183.0 mW total.**
  - VCC 75.4 mW and VEE 103.3 mW: amplifier quiescent current 30 × 0.5 mA × 10 V = 150 mW, plus load current.
  - Input sources 4.05 mW and bias rail 0.28 mW.
  - Mean total input current 5.22 mA.

### Stage E: SPICE-in-the-loop fine-tuning (`results/s05_stageE_in_loop.json`)

**Method.** Each chip instance has fixed device errors, quantization and op-amp offsets. Each iteration:
1. SPICE solves the chip for all 1437 training images.
2. The measured S_k gives z = S/0.2374 V, then softmax, then the gradient.
3. The software controller updates the shadow weights and re-programs the conductances.

The test set is used only before and after. Final state = last iteration. For the lr 0.5 run, the state with the lowest *measured training* loss is also recorded; it turned out to be the last iteration.

| Chip | lr | Train loss (measured) | Train acc (measured) | **Test before → after** |
|---|---|---|---|---|
| A: 6-bit, Gauss 5 %, V_OS σ 5 mV | 2.0 | 0.141 → 0.137 | 98.26 → 98.33 % | 347 → 347 |
| B: 4-bit, Gauss 10 %, V_OS σ 10 mV | 2.0 | 0.158 → 0.140 | 97.56 → 98.12 % | 348 → 346 |
| C: 2-bit, Gauss 20 %, V_OS σ 20 mV | 2.0 | 0.226 → 0.233 (oscillated) | 95.06 → 93.32 % | **342 → 336** (worse) |
| C: same chip | 0.5 | 0.226 → 0.168 | 95.06 → 97.43 % | **342 → 346** (+4 images, +1.1 pp) |

**Interpretation.**
- In-loop training reliably lowers the loss measured on the chip.
- It improves held-out accuracy only when the chip is badly degraded (chip C), and only with a small enough step.
- With 2-bit conductances, lr = 2 oscillates, because each level step is large compared with the update.
- On mild chips (A, B) the circuit was already at the software baseline, so the ±1–2 test-image changes are within noise.

### Non-idealities (`results/s06_nonidealities_summary.json`, per-run rows in `s06_nonidealities_runs.csv`)
Realistic op-amps, all 360 test images per run, 5 seeds for random conditions. Reference = 96.67 %.

| Condition | Mean test acc | Degradation |
|---|---|---|
| Quantization 8 / 6 / 5 / 4 / 3 / 2 bit | 96.67 / 96.94 / 96.67 / 96.11 / 95.56 / 94.44 % | 0 / −0.3 / 0 / 0.6 / 1.1 / 2.2 pp |
| Uniform tolerance 1 / 5 / 10 / 20 % | 96.61 / 96.61 / 96.50 / 96.11 % | ≤ 0.6 pp |
| Gaussian tolerance 1 / 5 / 10 / 20 % | 96.56 / 96.33 / 96.33 / 94.89 % (min 93.61) | up to 1.8 pp |
| 6-bit + Gauss 5 % + V_OS 5 mV | 96.67 ± 0.35 % | 0 |
| Input noise σ 5 / 10 / 20 / 50 / 100 mV | 96.67 / 96.67 / 96.44 / 96.17 / 95.28 % | up to 1.4 pp |
| Op-amp offset σ 1 / 5 / 10 / 30 mV | 96.56 / 96.56 / 96.61 / 94.28 % (min 92.78) | up to 2.4 pp |
| **Saturation:** R_f ×2 / ×4 / ×8 | 96.67 / **78.61** / **32.78** % | 0 / 18 / 64 pp |
| Supply ±4.5 / ±4 / ±3.3 / ±2.5 / ±5.5 / +5/−4 V | 96.67 / 96.67 / 96.67 / 96.11 / 96.67 / 96.67 % | ≤ 0.6 pp |

**Main finding:** amplifier saturation is the dominant failure mode, not device variation. That makes choosing R_f (signal range management) the critical design parameter.

Note: the same seeds were used at each tolerance level (common random numbers). That is why uniform 1 % and 5 % give identical accuracies: the same draws are scaled.

### Transient, latency and energy (`results/s07_transient_energy.json`)
Six test images were presented every 20 µs.

- **Settling** to within 20 mV: 4.1–5.1 µs.
- **Decision time** (argmax stable): 0.65–4.0 µs.
- **Energy per 20 µs window:**
  - 3.55–3.72 µJ total, of which 66–98 nJ is crossbar plus bias rail.
  - Mean power: 181.8 mW total, 177.6 mW amplifiers, 4.19 mW crossbar.
- **Not included:** DACs, programming circuits, ADC/readout, wiring capacitance, leakage, control logic.
- Ideal-amplifier runs draw zero supply power by construction.
- **These numbers say nothing about ASIC efficiency.**

### Analog winner-take-all (`results/s08_wta.json`)

| Quantity | Value |
|---|---|
| Test accuracy | **348/360**; agrees with argmax on all 360 images |
| Ambiguous outputs | 0 |
| Winner line | ≈ 5.0 V |
| Loser lines | ≤ 0.45 V |
| Smallest top-2 score gap that was resolved | 4.6 mV |
| Mean power with WTA | 275 mW |

### Qucs-S schematic (`circuits/qucs/neuron_stageA.sch`, check in `results/s10_qucs_schematic_check.json`)
The Stage A neuron, drawn with Qucs built-in components. A script extracts the connectivity from the drawing's geometry and simulates it: S = −0.439311 V, which matches.
**Not yet visually confirmed in the Qucs-S GUI** (see §8).

---

## 5. File locations

```
NeuralSphere-V1\
  HANDOFF.md                this file
  README.md                 quick start and reproduction
  run_all.ps1               runs every stage on Windows
  requirements.txt
  src\ns\                   library: config.py (constants, ngspice lookup), data.py, mapping.py
                            (weights<->conductances, quantization, device errors), spice.py (netlist
                            writer, ngspice runner, wrdata parser, power), train.py (softmax SGD), metrics.py
  scripts\                  s00_validate_ngspice … s10_qucs_schematic (one script per stage; _common.py)
  circuits\models\neuralsphere_models.lib   OPA_IDEAL, OPA_REAL, COMP2, diode model
  circuits\generated\       full 64x10 netlists (ideal, real, 8-bit, with WTA), Stage A netlists, transient netlist
  circuits\qucs\            neuron_stageA.sch (Qucs-S), neuron_stageA_extracted.cir
  data\digits_split.npz     the exact train/test split used
  weights\                  software_weights.npz, conductances_*.json (every resistor in ohms), in_loop_*.npz
  results\                  every metric (JSON/CSV/NPZ) + run logs
  plots\                    fig1…fig9 + demo_snapshot_test5.png
  demo\demo_app.py          interactive SPICE demo (Tkinter)
  docs\DESIGN.md            teaching document: vocabulary, physics, worked numbers, model limits
  tools\                    qucs-s\ (portable Qucs-S 26.1.1 + bundled ngspice), ngspice-47.tar.gz (source),
                            setup.ps1 / setup_log.txt, setup_ngspice.ps1, inspect.ps1 / inspect_log.txt, chunks\
  .venv\                    Python virtual environment
```

---

## 6. Dependencies

- **Python 3.12** with numpy, scipy, matplotlib, pandas, scikit-learn (`requirements.txt`). Tkinter is needed for the demo; it comes with the python.org Windows installer.
- **ngspice batch executable.**
  - The code searches `tools\ngspice\**\ngspice_con.exe`, then `tools\qucs-s\**\ngspice_con.exe`.
  - Override with the `NGSPICE` environment variable.
  - Results so far were produced with ngspice-47. The Qucs-S copy is ngspice-46: expect last-digit differences, not different predictions. Verify this (§9, step 1).
- **Qucs-S 26.1.1:** only needed to view or edit the schematic.

---

## 7. How to reproduce (Windows)

```powershell
cd C:\Users\intel\Documents\NeuralSphere-V1
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Quick   # s00-s04, s07, s08, s10, s09 (~10 min)
powershell -ExecutionPolicy Bypass -File .\run_all.ps1          # adds s05 in-loop + s06 sweep (~1-2 h)
.\.venv\Scripts\python.exe demo\demo_app.py                     # interactive demo
```

- `run_all.ps1` **overwrites** the files in `results\`, `plots\`, `weights\`, `circuits\generated\`.
- **Copy the folder first** if you want to keep the Linux/ngspice-47 reference results for comparison.
- The Stage E script's chip C lr 0.5 run needs its arguments: `python s05_stageE_in_loop.py 15 chipC_2bit_gauss20pct_vos20mV 0.5`.

---

## 8. Unresolved problems and blockers

1. *(Resolved 2026-11-21, see §8a.)* **Not yet executed on Windows.** Partway through the session, the automatic safety check in Claude's tooling blocked Claude from launching further PowerShell scripts on your PC through the Run dialog; an earlier attempt to start a background job-runner had been blocked the same way. `tools\setup_ngspice.ps1` (download of the official ngspice-47 Windows build) was therefore never run, and `run_all.ps1` has not been run on Windows. You need to start these yourself.
2. *(Resolved, see §8a.)* **Qucs-S schematic not visually checked.** Computer use (screen control) was not turned on in the Claude desktop app, so Claude could not open Qucs-S. The pin geometry assumed for the Qucs `OpAmp` symbol is "+ input on top, − input on the bottom". It is consistent with the Qucs source as recalled, but unverified. If the opened schematic gives S ≈ ±5 V instead of −0.439 V, the op-amp inputs are swapped: mirror the three op-amps.
3. **Digit "1" accuracy is 80.6 %** (7 of 36 misclassified). This is a limitation of the linear model, not the circuit: software has the same errors. Fixing it needs a hidden layer (V0.3/V0.4).
4. *(Addressed with the TI OPA196 vendor model, see §8b.)* **Op-amp macromodel is generic.** It omits slew rate, noise, input bias current, CMRR/PSRR and temperature, and it is not a specific part. The amplifier choice dominates power (150 mW of 183 mW).
5. **Comparator model is behavioural** (tanh), not a transistor-level or vendor model.
6. **Energy numbers exclude DACs, ADC, programming circuitry and parasitics.** No ASIC-level efficiency claim can be made.
7. **Stage E learning is chip-in-the-loop, not on-device learning.** Gradients are computed digitally from measured outputs, and conductances are "re-programmed" by rewriting the netlist (an ideal write with no write noise or endurance limits).
8. *(`.noise` analysis added, see §8b.)* **Static, noise-free resistors.** Input noise was injected as a static per-image sample. There is no thermal/1/f noise analysis (`.noise`) yet.
9. The `chunks\` folder (split source tarball) and `ngspice-47.tar.gz` are transfer leftovers. They are safe to delete once you no longer need the ngspice source.

---

## 8a. Windows validation (2026-11-21, second session)

- **Backup:** `archive\linux_ngspice47_2026-11-21\` holds the Linux/ngspice-47 `results`, `plots`, `weights`, `data`, `circuits\generated`, `circuits\qucs` (55 files, `MANIFEST.sha256` verified, files set read-only).
- **Git:** repository initialised in this folder only (`main`); commit `4176f66` is the as-handed-over baseline. `.venv`, `tools\qucs-s`, transfer archives and demo scratch output are ignored.
- **`run_all.ps1 -Quick` on Windows** (ngspice-46 from Qucs-S, Python 3.12.10): 7 min 20 s, all 17 checks in `scripts\compare_runs.py` pass. Report: `logs\compare_quick_windows_vs_linux.md`.
  - Software weights differ from Linux by ≤ 1.8·10⁻¹⁵ (floating-point rounding); every SPICE voltage by ≤ 2.6·10⁻⁹ V (print resolution of `numdgt=10`); all predictions identical.
  - 348/360 in s03 (ideal), s04 D1 and D2 (realistic), s08 (WTA); s00 passes.
- **Demo:** `scripts\verify_demo_gui.py` drives both Tk demos; each image selection starts exactly one ngspice process, rewrites `out_0.txt`, and yields 10 distinct S_k voltages (`results\demo_gui_verification.json`, screenshots `plots\demo_gui_*.png`).
- **New visual demo** `demo\neuralsphere_visual.py`: digit → 64 source voltages → per-synapse currents of all 1300 resistors (Ohm's law on ngspice node voltages, KCL-checked against the R_f current) → column currents → S_k → WTA lines → predicted digit, with substituted formulas; includes an 8×8 drawing pad. Software logits are shown only as labelled reference markers.
- **Qucs-S schematic (§8.2 resolved):** `scripts\s11_qucs_native_check.py` uses Qucs-S's own netlister and `--run` (same path as F2). Qucs-S assigns + input = ground for both TIAs; v(s) = −0.439311 V vs hand −0.4393125 V. Rendered drawing: `plots\fig10_qucs_schematic.png/.svg`. Qucs-S 26.1.1's CLI PNG export is broken (uninitialised background), so SVG is rasterised with Edge.
- **`run_all.ps1` fix:** the full run now repeats the archived Stage E exactly (chips A,B: 10 iterations lr 2; chip C: 15 iterations at lr 2 and at lr 0.5). Before, it ran all chips for 10 iterations at lr 2 only, and skipped lr 0.5.
- `spice.run` gained an optional `extra=` list of vectors (default unchanged).

## 8b. Vendor op-amp, noise and publication (2026-11-21/22)

- **TI OPA196 macromodel** (`scripts/s12_opa196_stageD.py`, `results/s12_opa196_stageD.json`, DESIGN §9).
  - Bench check vs datasheet: I_Q 140.00 µA (140), A_OL 141.6 dB (140), GBW 2.87 MHz (2.5), e_n 18.1 nV/√Hz at 1 kHz (15), V_OS 16.8 µV.
  - Full inference: 348/360 test and 1410/1437 train; 0 disagreements with software or with OPA_REAL; mean power 75.0 mW vs 183.0 mW.
  - Batches run as an image-index DC sweep (`spice.sweep_cards`), because a fresh `.op` per image is slow and once landed in a different DC state (hysteretic switches in the model).
  - The TI library is copyrighted and is **not** in the git repository. `tools/get_opa196_model.ps1` downloads it and verifies its SHA-256.
- **Noise** (`scripts/s13_noise.py`, `results/s13_noise.json`, `plots/fig11_noise.png`, DESIGN §10).
  - ngspice `.noise` at the operating point of test image 0, 0.1 Hz – 100 MHz, in 3 variants to split the contributions.
  - Hand density = SPICE within 0.01 % (139.8 nV/√Hz at 1 kHz for S_0).
  - σ(S_k) = 291–302 µV with a 1 MHz first-order readout filter; 861–866 µV over the whole 0.1 Hz – 100 MHz band, because the vendor model's output noise plateaus above ~3 MHz.
  - Worst decision margin 4.56 mV (image 18, S_9 − S_4) = 10.8 σ, so P ≈ 2·10⁻²⁷ (3.7 σ, P ≈ 1·10⁻⁴ without any band limit). Noise causes none of the 12 errors.
  - Shares: op-amps 80 %, R_f + difference-amplifier resistors 19 %, synapses 1 %.
  - ngspice's `onoise_total` was unusable (returned 1.4·10⁹), so the rms values are integrals of the density.
- **DESIGN §6 correction:** the TIA noise gain is 4.0–5.3 (ΣG = 0.537–0.760 mS per column), not ≈ 37.
- **Open issue: OPA196 transient.**
  - The full 30-amplifier transient did not progress usefully in ngspice: a 2 µs slice was still unfinished after 28 min, and the full run was stopped after 3.4 h at 1.9 GB.
  - `s07_transient_energy.py opa196` therefore simulates the 10 neurons as separate exact sub-circuits (the shared inputs and supplies are ideal sources) and sums their source currents.
  - Neurons 1–9 finish in 1–14 min. **Neuron 0 does not finish** (stopped after 17 h at 3.7 GB): probable timestep collapse at a hysteretic switch in the model.
  - This step is excluded from `run_all.ps1`. Ideas: `.options method=gear`, raising `abstol`/`chgtol`, smoothing the PWL edges, or starting from the DC-sweep operating point.
- **Published** as a public GitHub repository (MIT license). Commits use the GitHub noreply address.

## 9. Exact next steps (in order)

Done since the first handoff: Windows run, Qucs-S schematic check, demo verification, vendor op-amp model, `.noise` analysis (old steps 1–5).

1. **Fix the OPA196 transient for neuron 0** (see §8b). Then report settling time and energy per inference with the vendor model, next to the generic model's 3.55–3.72 µJ.
2. **Repeat s06 non-idealities with the OPA196** (offset and tolerance with the vendor part; pass `vos` through the wrapper's `VOS` parameter).
3. **Power reduction study.** Raise R_min/R_max (for example 100 kΩ–10 MΩ), use micro-power amplifiers and power-gate between inferences. Re-measure E per inference with s07.
4. **V0.3/V0.4:** add a hidden layer (64 → 16 → 10) with an analog nonlinearity (diode/ReLU stage) between crossbars. Train ex situ, then in-loop. Target: fix digit "1" and test signal restoration between layers.
5. **Stage E improvements:** stochastic rounding for low-bit programming, write-noise and endurance models, a per-column offset calibration step. Repeat with ≥ 3 seeds per chip so the improvement has error bars.
6. **Toward V0.6 (hardware):** pick a digital potentiometer or DAC-plus-multiplying approach for programmable synapses. Build one 8-input differential column on a breadboard, then compare measurements with `s01`-style SPICE predictions.
