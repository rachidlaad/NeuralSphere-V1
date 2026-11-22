# NeuralSphere-V1: circuit design and the physics behind it

This document explains every design decision, with the formula, the actual numbers substituted, and units.
All numbers come from the saved result files in `results/` and `weights/`.

---

## 1. Vocabulary (short definitions)

| Term | Meaning |
|---|---|
| **Resistance R** (ohm, Ω) | How strongly a component opposes current. Ohm's law: V = I·R. |
| **Conductance G** (siemens, S) | The inverse of resistance, G = 1/R. A larger G passes more current. **In this circuit G is the synaptic weight.** 1 µS = 10⁻⁶ S. |
| **Node** | A point in the circuit where component terminals meet. Every point on it has one voltage. |
| **KCL** (Kirchhoff's current law) | The currents flowing into a node add up to zero. This law makes the circuit **sum**. |
| **Op-amp** (operational amplifier) | A very-high-gain differential amplifier: v_out = A·(v₊ − v₋), with A ≈ 10⁵…10⁶. |
| **Negative feedback** | Connecting the output back to v₋. The op-amp then drives its output to whatever value makes v₋ ≈ v₊. |
| **Virtual ground** | A node held at ≈ 0 V by negative feedback without being wired to ground. |
| **TIA** (transimpedance amplifier) | An op-amp plus a feedback resistor R_f. It converts an input current I into an output voltage v = −R_f·I. |
| **Difference amplifier** | An op-amp with 4 equal resistors. Its output is v₂ − v₁. |
| **Rail** | A supply voltage (+5 V, −5 V). An op-amp output cannot go beyond its rails. |
| **Saturation** | The output is stuck near a rail, so the circuit is no longer linear. |
| **Offset voltage V_OS** | A small built-in error: the op-amp behaves as if V_OS were added to its input. |
| **Quantization** | A programmable device can only take N discrete conductance levels (N = 2^bits). |
| **Tolerance / mismatch** | A manufactured part differs from its nominal value by a random amount. |
| **SPICE** | A circuit simulator. It writes KCL at every node, plus each device's equation, and solves the resulting system (modified nodal analysis). |
| **.op** | SPICE "operating point" analysis: the DC steady state. **.tran** is the time-domain analysis. |

---

## 2. Architecture

```
  pixel_i/16 ──► V_i = 1 V·x_i ─┬─ R+_k,i ─┐                          ┌─ R_f ─┐
  (64 sources)                   │          ├─ column cp_k (virtual gnd)┤─ OPP_k ┴─► tp_k = −R_f·I+_k ─┐
  bias rail 1 V ─────────────────┤          │                                                          ├─ diff-amp ─► S_k
                                 └─ R−_k,i ─┴─ column cn_k (virtual gnd)── OPN_k ──► tn_k = −R_f·I−_k ─┘
                                                                                     S_k = tn_k − tp_k = R_f(I+_k − I−_k)
  S_0 … S_9  ──► [measurement program: argmax]   or   [analog WTA: 45 comparators + 90 diodes]
```

**Analog inference hardware:** input sources → 1300 synapse resistors → 20 TIAs → 10 difference amplifiers → (optional) WTA.
**Programming interface (software controller):** chooses the 1300 resistance values and writes them into the netlist.
**Measurement interface:** reads the node voltages S_k (or w_k) that ngspice solved.

Circuit size, counted in the generated netlist `circuits/generated/full_64x10_*.cir`:
- 1300 synapse resistors: 10 outputs × 65 inputs (64 pixels + bias) × 2 (G⁺ and G⁻)
- 20 feedback resistors and 40 difference-amplifier resistors
- 20 difference-amplifier compensation capacitors, 4.7 pF (added 2026-11-22, §11.2; open at DC)
- 30 op-amps
- 67 voltage sources: 64 inputs, the bias rail, VCC and VEE
- Adding the WTA: 45 comparators, 90 diodes and 10 pull-up resistors

---

## 3. Why currents add: Ohm + Kirchhoff

The TIA holds each column node at about 0 V, so the current through synapse resistor R⁺_k,i is, by Ohm's law:

  I⁺_k,i = (V_i − 0) / R⁺_k,i = G⁺_k,i · V_i

KCL at the column node says that all those currents must leave through R_f:

  I⁺_k = Σ_i G⁺_k,i · V_i   →   tp_k = −R_f · I⁺_k

The same holds for the negative column. The difference amplifier then computes S_k = tn_k − tp_k = R_f·(I⁺_k − I⁻_k).

**Why the virtual ground matters (this fixes the V0.1 breadboard weakness):** in the V0.1 passive divider, the summing node voltage moved with the inputs. Each weight's effect then depended on all the other weights (Millman's theorem). Here the column voltage stays at about 0 V (measured: 0.7 µV in Stage A). Each current depends only on its own G and V, so the weights are independent.

### Worked example (Stage A, `results/s01_stageA_single_neuron.json`)

The example neuron is w = [+1.0, −0.5, +0.25, −2.0], bias +0.3, inputs x = [1, 0.5, 0.25, 1].

- Largest |w| = 2.0, so α = (G_max − G_min)/2.0 = (100 µS − 1 µS)/2.0 = **49.5 µS per unit weight**.
- Synapse 0 (w = +1): G⁺ = 1 µS + 49.5 µS × 1 = 50.5 µS, so R⁺ = 1/50.5 µS = **19.80 kΩ**. G⁻ = G_min = 1 µS, so R⁻ = **1 MΩ**.
- Current: I⁺ = 1 V × 50.5 µS = **50.5 µA**.
- Summing all five rows: I⁺ = 71.194 µA and I⁻ = 115.125 µA.
- Output: S = R_f(I⁺ − I⁻) = 10 kΩ × (71.194 − 115.125) µA = **−0.43931 V**.
- Check: z = 1·1 − 0.5·0.5 + 0.25·0.25 − 2·1 + 0.3 = −0.8875, and S = R_f·α·V_max·z = 10 kΩ × 49.5 µS × 1 V × (−0.8875) = −0.43931 V ✓
- ngspice results: **−0.43931 V** with the ideal op-amp and **−0.43929 V** with the realistic macromodel.

---

## 4. Signed weights with differential pairs

A resistor cannot have negative conductance, so each weight uses two:

  G⁺ = G_min + α·max(w, 0)  G⁻ = G_min + α·max(−w, 0)  ⇒  G⁺ − G⁻ = α·w

The G_min terms cancel in the subtraction, but they are physically necessary: a "zero" resistance level is a real resistor of R_max = 1 MΩ, not an open circuit.

For the trained network (`weights/conductances_ideal.json`):
- The largest |weight| is w_max = 2.3352, so α = (100 µS − 1 µS)/2.3352 = **42.394 µS per unit weight**.
- Example: neuron 3, pixel 36 has w = +0.8037. Then G⁺ = 1 µS + 42.394 µS × 0.8037 = 35.07 µS, so **R⁺ = 28.51 kΩ**, and **R⁻ = 1 MΩ**.
- 52.7 % of the 1300 resistors sit at R_max = 1 MΩ: every weight uses only one side of its pair.

**Bias** is a 65th input row driven by a constant 1 V rail. That makes the bias just another conductance pair.

---

## 5. Choosing R_f, supply and output range

- Supplies: ±5 V.
- The realistic op-amp model cannot swing closer than 0.2 V to a rail (soft tanh limit).
- Largest column current over the 1437 training images: I_max = 570.4 µA.
- R_f target: the largest E24 value with R_f·I_max ≤ 3.5 V. 3.5 V / 570.4 µA = 6.14 kΩ, so **R_f = 5.6 kΩ**. The worst-case TIA output is then 5.6 kΩ × 570.4 µA = 3.19 V (measured range −3.19 … −0.17 V).
- Score scale: S_k = R_f·α·V_max·z_k = 5600 Ω × 42.394 µS × 1 V × z = **0.2374 V per logit unit**.
- Measured score range: −1.94 … +2.26 V, comfortably inside ±4.8 V.
- Difference-amplifier resistors are 100 kΩ, so they barely load the TIAs.

The training rule Σ_k (p_k − 1[k=t]) = 0 keeps Σ_k w_ki = 0 when training starts from zero. That is why the mean of I⁺ equals the mean of I⁻ (both 278.8 µA).

---

## 6. Amplifier models (`circuits/models/neuralsphere_models.lib`)

| Model | What it is | Used for |
|---|---|---|
| `OPA_IDEAL` | Linear VCVS, gain 10⁶; no rails, no offset, no bandwidth limit, no supply current | Mathematical verification (Stage C) |
| `OPA_REAL` | Single-pole macromodel: A_OL = 10⁵ (100 dB), GBW = 1 MHz, input offset V_OS, soft output saturation at rail − 0.2 V that **follows the actual supply voltage**, 50 Ω output resistance, 0.5 mA quiescent current. Load current is drawn from the rail that sources it. | All "realistic" results |
| `COMP2` | Behavioural comparator with complementary 0/5 V outputs | WTA |
| `DSW` | 1N4148-like diode | WTA diode-AND gates |

`OPA_REAL` does **not** model slew rate, noise, input bias current, CMRR/PSRR or temperature. These are listed as limitations.

**Finite-gain error.** The TIA's noise gain is NG = 1 + R_f·ΣG, where ΣG is the total conductance of the 65 synapses on that column. The actual column sums are ΣG = 0.537 … 0.760 mS, so NG = 1 + 5.6 kΩ × (0.537 … 0.760) mS = **4.00 … 5.26**. With A_OL = 10⁵, the relative error is at most 5.26/10⁵ ≈ 0.005 %. The measured maximum error versus software is given in `results/s04_stageD_inference.json`.
*(Corrected 2026-11-21. The earlier text used ΣG ≈ 6.5 mS = 65 × G_max, giving NG ≈ 37. That is only an upper bound: 52.7 % of the resistors sit at G_min.)*

---

## 7. Training: what is software and what is circuit

| Stage | Forward pass (inference) | Gradient computation | Label |
|---|---|---|---|
| D | SPICE circuit | software, done earlier on software logits | **externally trained, circuit-executed inference** |
| E | **SPICE circuit (measured S_k)** | software controller, using measured S_k | **chip-in-the-loop fine-tuning**: the circuit's own errors enter the gradient, but the update arithmetic is digital |

Nothing in this project is "fully physical learning". The conductances are rewritten by a program, not by local physics in the devices.

Stage E update used: z_meas = S_k / 0.2374 V; p = softmax(z_meas); ΔW = −lr·[(p − y)ᵀX/N + λW] with lr = 2.0 and λ = 10⁻³. The new conductances are re-quantized and keep the chip's fixed device errors. The test set is evaluated only before and after.

Hyper-parameters for software training (lr 0.5, batch 64, 200 epochs, L2 10⁻³) were chosen on a validation split taken from the **training** set:

| L2 | train acc | val acc | max\|w\| |
|---|---|---|---|
| 0 | 99.5 % | 95.8 % | 3.96 |
| 10⁻³ | 98.5 % | 95.5 % | 2.48 |
| 10⁻² | 95.2 % | 92.4 % | 1.14 |

10⁻³ was chosen because it keeps nearly all of the validation accuracy while shrinking the largest weight. A smaller |w|_max means a larger α, and so more current signal above the G_min floor.

---

## 8. Power: what is and is not counted

P = V·I for every source, using ngspice's branch currents (`ns.spice.power`).

- **Counted:** input sources (64 × V_i·I_i), the bias rail, ±5 V amplifier supplies (quiescent current plus load current) and, in the WTA run, the comparators and diode pull-ups.
- **Not counted:** DACs that would generate the 64 input voltages, the programming circuitry for the conductances, ADC/readout, wiring capacitance and leakage, clocking and control.
- Ideal-op-amp runs draw **zero** supply current by construction. They say nothing about energy.

Discrete-amplifier power (30 × 0.5 mA × 10 V = 150 mW) dominates. It describes a breadboard built from general-purpose parts, **not** what an integrated ASIC would consume. ASIC efficiency claims would need transistor-level design in a real process (PDK), layout parasitics and the peripheral circuits listed above.

---

## 9. Vendor amplifier: Texas Instruments OPA196 (`scripts/s12_opa196_stageD.py`)

**Why this part.** The crossbar runs from ±5 V and the TIA outputs swing to −3.24 V (test set; slightly beyond the 3.19 V training-set maximum used to choose R_f, still 1.7 V from the rail), so the amplifier must accept a 10 V supply and swing close to its rails. The OPA196 is rated 4.5–36 V total supply with rail-to-rail output, its quiescent current is 140 µA (vs 0.5 mA assumed in `OPA_REAL`), and its GBW (2.5 MHz) is in the same range as the generic model, so results remain comparable. TI publishes an unencrypted PSpice macromodel that models noise, offset, Iq, slew rate, output swing, CMRR/PSRR.

**Model provenance.** `SBOMAF9E.ZIP` (27 388 bytes, SHA-256 `9c264813…ccdf5`) from ti.com, file `OPAx196.LIB`, "Final 1.5", 29 Aug 2022. It is used **unmodified** (`tools/vendor_models/opa196_extracted/`); `circuits/models/opa196_wrapper.lib` only maps pin order (TI: IN+ IN− VCC VEE OUT → ours: inp inn out vcc vee) and adds an optional extra offset `VOS`. ngspice needs PSpice compatibility mode (`set ngbehavior=ps`), which `spice.prepare_workdir` writes to a `.spiceinit` in ngspice's working directory for vendor models only.

**Bench check of the model against the datasheet** (SBOS869 table 6.7, typical; our supplies ±5 V):

| Quantity | How measured in ngspice | Datasheet | Model |
|---|---|---|---|
| I_Q per amplifier | −i(VCC)/2 for two idle amplifiers: 280.00 µA / 2 | 140 µA | **140.00 µA** |
| A_OL (DC) | L/C DC servo (L = 1 GH, C = 1 GF), vdb(out) at 0.1 Hz | 140 dB (±18 V, 10 kΩ) | **141.6 dB** |
| GBW | 10^(A_OL(1 kHz)/20) × 1 kHz = 10^(69.17/20) × 1 kHz | 2.5 MHz | **2.87 MHz** |
| e_n at 100 Hz / 1 kHz | `.noise` of a unity buffer, input-referred | 18 / 15 nV/√Hz | **20.3 / 18.1 nV/√Hz** |
| V_OS | output of a grounded-input buffer | ±25 µV typ, ±100 µV max | **16.8 µV** |

The model is ~15 % faster and ~20 % noisier than the datasheet *typical* values; we keep the vendor model as delivered, so the noise results below are slightly pessimistic.

**Results (all 1437 train + 360 test images, same programming as Stage D, R_f = 5.6 kΩ):**

| | OPA_REAL (generic) | OPA196 (TI model) |
|---|---|---|
| Test / train correct | 348/360 / 1410/1437 | **348/360 / 1410/1437** |
| Disagreements with software / with OPA_REAL | 0 / – | **0 / 0** |
| max \|S − S_Kirchhoff\| | – | 42.3 µV (mean +33.2 µV) |
| TIA output range (360 test images) | −3.244 … −0.185 V | −3.244 … −0.185 V |
| Mean total power | 183.0 mW | **75.0 mW** |

*Where the 33 µV offset comes from:* each TIA amplifies its input offset by its noise gain, and the difference amplifier by 2. For test image 0 the measured TIA errors were 0.07–0.09 mV ≈ 16.9 µV × (1 + R_f G_Σ) with 1 + R_f G_Σ ≈ 4–5, and the difference amplifier added 2 × 16.7 µV = 33 µV; the TIA errors largely cancel in tn − tp because both columns see similar offsets. In logits: 42 µV / 0.2374 V = 1.8·10⁻⁴.

*Power hand check:* P ≈ 30·I_Q·(V_CC − V_EE) + |V_EE|·ΣI_col + P_inputs + P_bias = 30 × 140 µA × 10 V + 5 V × 5.504 mA + 4.05 mW + 0.28 mW = 42.0 + 27.5 + 4.3 = **73.9 mW** (SPICE: 75.0 mW). The TIA outputs are negative, so every TIA sinks its column current through its output stage into the −5 V rail; that is why P_VEE (49.3 mW) > P_VCC (21.4 mW).

**Numerical method note (important for reproducing):** with this model a fresh `.op` per image needs dynamic gmin stepping (≈11–16 s per image) and, for 1 of 20 images tested, converged to a *different valid DC state* (S off by 3.9 mV). The model contains voltage-controlled switches with hysteresis (overload switch VON 0.9 V / VOFF 0.8 V; clamp switches 10 mV / 0 V), so the DC solution can depend on history. We therefore solve vendor-model batches as an image-index DC sweep starting from a blank image (`spice.sweep_cards`): each solve starts from the previous image's operating point, which is also how a physical chip sees consecutive inputs. With our own models the sweep and the per-image `.op` agree to 5·10⁻⁸ V; with the OPA196 all images then sit at 0.041–0.042 mV from Kirchhoff, and the speed is ~1 s per image.

---

## 10. Electrical noise with the OPA196 (`scripts/s13_noise.py`)

**This is ELECTRICAL simulation:** ngspice `.noise` small-signal analysis around the DC operating point of test image 0, 0.1 Hz – 100 MHz, 20 points/decade, one run per output S_k. Noise sources: the vendor model's own voltage and current noise (including 1/f), and the thermal noise of every resistor at T = 300.15 K (ngspice's default 27 °C). The input DACs are ideal and noiseless (see the s06 input-noise study for that). Three variants separate the contributions:

- **A**: everything noisy.
- **B**: the 1300 synapse resistors noiseless (`noisy=0`).
- **C**: R_f and the difference-amplifier resistors also noiseless, leaving the op-amps only.

### 10.1 Noise density: hand calculation vs ngspice at 1 kHz (white region)

Constants: 4kT = 4 × 1.380649·10⁻²³ J/K × 300.15 K = **1.6576·10⁻²⁰ J**. Model e_n(1 kHz) = 18.07 nV/√Hz (bench, §9). Datasheet i_n = 1.5 fA/√Hz, so i_n·R_f = 1.5 fA × 5.6 kΩ = 8.4 pV/√Hz, which is negligible.

Worked example, output S_0, positive column. ΣG⁺ = 0.5365 mS, so NG = 1 + R_f·ΣG = 1 + 5600 × 0.5365·10⁻³ = 4.004.

| Source (referred to the TIA output) | Formula | Value |
|---|---|---|
| R_f thermal | √(4kT·R_f) = √(1.6576·10⁻²⁰ × 5600) | 9.63 nV/√Hz |
| 65 synapse resistors | R_f·√(4kT·ΣG) = 5600 × √(1.6576·10⁻²⁰ × 0.5365·10⁻³) | 16.70 nV/√Hz |
| op-amp e_n × noise gain | e_n·NG = 18.07 × 4.004 | 72.36 nV/√Hz |
| TIA total | √(9.63² + 16.70² + 72.36²) | 74.9 nV/√Hz |

The negative column (ΣG⁻ = 0.5610 mS, NG = 4.142) gives √(9.63² + 17.08² + 74.84²) = 77.4 nV/√Hz.

The difference amplifier's own noise:

- Its four 100 kΩ resistors give 4kT·R_d (RD1) + 4kT·R_d (RD2) + 4kT·(R_d/2)·2² (RD3‖RD4) = 16kT·R_d.
- Its own e_n appears with noise gain 2.
- So e_d = √(16kT·R_d + 4e_n²) = √(6.630·10⁻¹⁵ + 4 × (18.07 nV)²) = **89.09 nV/√Hz**.

The output is e_S = √(e_tp² + e_tn² + e_d²) = √(74.9² + 77.4² + 89.09²) = **139.8 nV/√Hz**. ngspice gives **139.8 nV/√Hz**. For all 10 outputs (139.8 … 158.6 nV/√Hz) the hand formula and ngspice agree within **0.01 %**, which validates the noise-gain picture.

### 10.2 Spectrum and why "the" rms noise depends on readout bandwidth

The circuit includes the difference-amplifier compensation C_d = 4.7 pF (§11, adopted 2026-11-22). The values from before it are kept for comparison.

ngspice noise density of S_0 (`results/s13_noise.json`, `plots/fig11_noise.png`):

| f | 0.1 Hz | 1 Hz | 10 Hz | 100 Hz | 1 kHz | 10 kHz | 100 kHz | 1 MHz | 10 MHz | 100 MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| e_S (nV/√Hz), with C_d | 1873 | 607 | 233 | 151 | 140 | 139 | 135 | 96 | 89 | 41 |
| without C_d | 1873 | 607 | 233 | 151 | 140 | 139 | 142 | 217 (peak 570 at 0.71 MHz) | 84 | 77 |

- **Below ~100 Hz:** 1/f noise of the vendor model. It contributes only 2.1 µV rms from 0.1 to 100 Hz.
- **100 Hz – 100 kHz:** flat. Check: σ = e_S·√B = 139.8 nV/√Hz × √(10⁴ Hz) = 14.0 µV, and ngspice, integrated 0.1 Hz – 10 kHz, also gives **14.0 µV**.
- **Without C_d** there was a resonance peak at 0.71 MHz: the difference-amplifier ringing of §11. **With C_d** the peak is gone. Every output component passes through the difference amplifier, whose compensation pole is f_d = 1/(2π·R_d·C_d) = 1/(2π × 100 kΩ × 4.7 pF) = **338.6 kHz**.
- **Above ~3 MHz** the vendor model leaves a tail of op-amp output noise (~90 nV/√Hz, falling to 41 nV/√Hz at 100 MHz). A real readout is band-limited, so rms noise is quoted **per readout bandwidth**. ngspice's own `onoise_total` vector was unusable here (wrdata returned 1.4·10⁹ for every run), so the rms values are integrals of the density, σ² = ∫ e_S²·|H(f)|² df.

**Hand check of the rms noise with C_d:** σ ≈ e_S·√(π/2·f_d) = 139.8 nV/√Hz × √(π/2 × 338.6 kHz) = **101.9 µV**. ngspice integrated 0.1 Hz – 1 MHz gives **110.7 µV**; the difference is the 1/f region and the tail above f_d. For the 10 outputs: hand 102–116 µV vs ngspice 111–118 µV.

| Readout bandwidth | σ(S_k), with C_d | worst-image margin / σ_diff | P_err (union bound), worst image | σ(S_k) without C_d |
|---|---|---|---|---|
| brick-wall 0.1 Hz – 10 kHz | 14.0 – 15.9 µV | 210.6 | < 10⁻³⁰⁰ | 14.0 – 15.9 µV |
| brick-wall 0.1 Hz – 100 kHz | 43.5 – 49.2 µV | 68.0 | < 10⁻³⁰⁰ | 44.2 – 50.0 µV |
| 1st-order filter f_c = 100 kHz | 51.0 – 56.8 µV | 58.6 | < 10⁻³⁰⁰ | 72.1 – 78.0 µV |
| **1st-order filter f_c = 1 MHz** (primary) | **131.5 – 137.5 µV** | **23.8** | **3·10⁻¹²⁵** | 291 – 302 µV |
| brick-wall 0.1 Hz – 1 MHz | 110.7 – 118.2 µV | 27.8 | 2·10⁻¹⁷⁰ | 325 – 337 µV |
| brick-wall 0.1 Hz – 10 MHz | 305.6 – 308.4 µV | 10.5 | 4.9·10⁻²⁶ | 429 – 438 µV |
| brick-wall 0.1 Hz – 100 MHz (worst case) | 627.1 – 628.5 µV | 5.14 | 1.4·10⁻⁷ | 861 – 866 µV |

The primary choice is a 1st-order readout filter at f_c = 1 MHz. Its time constant is τ = 1/(2π·1 MHz) = 0.16 µs, so it settles in 5τ = 0.8 µs, well inside the 20 µs image period. In logits, 131.5–137.5 µV / 0.23741 V = **5.5·10⁻⁴–5.8·10⁻⁴ logit units**. The compensation capacitor lowered this σ by a factor of 2.2.

**Where the noise comes from** (power shares, 1 MHz readout, mean of the 10 outputs; A − B, B − C, C):

| Source | Share of σ² |
|---|---|
| op-amps (e_n × noise gain, model tail) | 83.7 % |
| R_f and difference-amplifier resistors | 15.1 % |
| 1300 synapse resistors | 1.2 % |

The synapse resistors matter least because their thermal-noise current, √(4kT·ΣG) = √(1.6576·10⁻²⁰ × 0.54 mS) = 3.0 pA/√Hz, is small next to the op-amp noise multiplied by NG = 4–5.

### 10.3 Classification risk from noise

The 360 OPA196 test outputs come from §9. Noise on different outputs is assumed independent: they share only ideal sources. Winner w, competitor j:

  z_j = (S_w − S_j) / √(σ_w² + σ_j²),  P_err ≤ Σ_j Q(z_j),  Q(x) = ½·erfc(x/√2)

The smallest margin is test image 18 (true 9, predicted 9), S_9 − S_4 = 0.87580 − 0.87124 V = **4.562 mV**:

- With the 1 MHz readout: z = 4.562 mV / √((133.9 µV)² + (137.5 µV)²) = 4.562 mV / 191.9 µV = **23.8**, so Q(23.8) = **3·10⁻¹²⁵**. Without C_d it was z = 10.8 and Q = 1.8·10⁻²⁷.
- With the full 0.1 Hz – 100 MHz band (worst case): z = 4.562 mV / √(627.7² + 628.5²) µV = 4.562 mV / 888.2 µV = **5.14**, so Q = **1.4·10⁻⁷**. Without C_d it was 3.73 and 9.5·10⁻⁵.

**Conclusion (electrical simulation).** With any readout bandwidth up to 10 MHz, amplifier and resistor noise cannot change a single test decision; P ≤ 5·10⁻²⁶ per image. Even without a band limit, P is at most 1.4·10⁻⁷. The 12 test errors (348/360) are therefore classification errors of the **software-trained** linear model, not noise. For comparison, the static non-idealities in s06 do flip decisions: uniform ±1 % resistor tolerance lowers the worst of 5 trials to 96.39 %, and 2-bit quantization lowers accuracy to 94.44 %. Static device errors, not noise, are the accuracy limit of this circuit.

**Limits:**

- Small-signal analysis around one operating point (test image 0). σ varies with ΣG, which is fixed by the weights, not by the image; the image only changes the operating point, and the noise is computed linearised.
- No resistor excess (1/f) noise.
- Ideal, noiseless input DACs and supplies.
- Integration down to 0.1 Hz treats slow 1/f drift as noise, although it really behaves as a slowly varying offset.

---

## 11. Transient response and difference-amplifier compensation (`scripts/s07_transient_energy.py`)

**This is ELECTRICAL simulation:** ngspice `.tran` of the full circuit. Six test images are presented for 20 µs each, with 100 ns input edges and a 20 ns maximum step. Outputs are `results/s07_transient_energy*.json` and `plots/fig12_transient_opa196.png`.

**Numerical method.** With ngspice's default trapezoidal integration the OPA196 run did not finish: it ran 3.4 h and reached 1.9 GB of memory. One neuron simulated alone ran 17 h without finishing.

The cause is a real circuit effect, ringing, which keeps the model's internal hysteretic switches toggling and forces bursts of tiny time steps. For vendor models s07 therefore uses **Gear integration** (`.options method=gear`), and the full run takes 40–60 s. Three checks support the result:
- Where both methods finish, they agree within 3 mV.
- Neuron 0 simulated alone agrees with the full circuit within 0.15 mV, as it should, because the neurons share only ideal sources.
- The end-of-image outputs agree with the DC-sweep results of §9 within **1.4 µV**.

### 11.1 The ringing (design as first built, without compensation)

With the OPA196 the difference amplifiers ringed after every image change, while the TIA outputs settled smoothly. The generic single-pole model cannot show this. Measured on output S_2 after the image change at 20 µs, successive positive peaks were +0.684, +0.276, +0.100 V:

- Ringing period T_d = 1.47 µs, so f_d = 1/T_d = **0.679 MHz**. This is the same resonance as the 0.71 MHz noise peak, which comes from the AC analysis and therefore does not depend on the integration method.
- Logarithmic decrement δ = ln(0.684/0.276) = 0.907, so ζ = δ/√(4π² + δ²) = **0.143**. That is a phase margin of only about 15°, and the outputs overshot by 38–50 % of each step.
- Decay time constant τ = 1/(ζ·ω_n) = 1/(0.143 × 2π × 0.686 MHz) = 1.62 µs.
- Predicted settling time: τ·ln(0.684 V / 20 mV) = 1.62 µs × 3.53 = 5.7 µs, the same as the 5.7 µs measured.

Likely mechanism: the difference amplifier's inverting input sees R_d‖R_d = 50 kΩ together with the op-amp's input capacitance, which adds a pole near the loop crossover (noise gain 2 → crossover ≈ GBW/2 ≈ 1.4 MHz) and erodes the phase margin.

### 11.2 Compensation: C_d = 4.7 pF across RD2 and RD4 (adopted 2026-11-22)

`config.C_DIFF = 4.7e-12`. `spice.circuit_cards` adds `CD2_k` across RD2 (feedback) and a matching `CD4_k` across RD4, so the gain stays balanced. This adds 20 capacitors in total. Their time constant is R_d·C_d = 100 kΩ × 4.7 pF = 0.47 µs.

A capacitor is an open circuit at DC, so **no DC result changes**. Re-running the pipeline confirmed it: s03, s04, s08 and s12 outputs are identical to within print resolution (≤ 1·10⁻¹⁰ V; WTA lines ≤ 3·10⁻⁹ V), and all 17 checks against the archived Linux run pass.

Candidate values, full-circuit OPA196 runs before adoption:

| C_d | Overshoot | Settling (1 % FS) | Change in settled S_k |
|---|---|---|---|
| none | 38 – 50 % | 5.0 – 6.3 µs | – |
| **4.7 pF** | **0** | **2.4 – 2.8 µs** | ≤ 0.4 µV |
| 10 pF | 0 | 4.6 – 5.3 µs | ≤ 0.4 µV |
| 22 pF | 0 | 9.3 – 10.9 µs | ≤ 0.34 mV (τ = 2.2 µs, too slow for 20 µs) |

**Results with C_d = 4.7 pF** (6 images, true labels 5, 2, 8, 1, 7, 2):

| | Generic OPA_REAL | TI OPA196 | OPA196 without C_d (before) |
|---|---|---|---|
| Predictions | 5, 2, 8, 6, 7, 2 | 5, 2, 8, 6, 7, 2 (= DC result) | same |
| Settling to 1 % of 2 V full scale | 5.0 – 5.9 µs (was 4.1 – 5.1 µs) | **2.4 – 2.8 µs** | 5.0 – 6.3 µs |
| Overshoot | none | **none** | 38 – 50 % |
| Mean power | 181.8 mW | **73.8 mW** | 73.8 mW |
| Energy per 20 µs image period | 3.55 – 3.72 µJ | **1.39 – 1.57 µJ** | 1.39 – 1.57 µJ |
| Noise σ(S_k), 1 MHz readout (§10) | – | **132 – 138 µV** | 291 – 302 µV |

The trade-off is visible in the generic model: it never rang, so C_d only adds its 0.47 µs pole there, and settling grows by about 0.9 µs. With the real part, C_d **halves** settling, removes the overshoot and lowers the noise by a factor of 2.2.

Energy hand check: E = ∫P dt ≈ P̄·T = 73.84 mW × 20 µs = 1.477 µJ. The mean of the six SPICE values is also 1.477 µJ. The crossbar itself (inputs + bias rail) uses only 0.066–0.099 µJ per image; the amplifiers' quiescent and load power dominate. C_d does not change the power.

*Not updated:* the Stage A Qucs-S schematic (`circuits/qucs/neuron_stageA.sch`, s01/s10/s11) is a DC check and does not include C_d. At DC the result is identical.
