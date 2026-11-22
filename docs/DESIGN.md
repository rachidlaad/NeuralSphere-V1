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

**Finite-gain error.** The TIA's noise gain is 1 + R_f·ΣG ≈ 1 + 5.6 kΩ × 6.5 mS ≈ 37. With A_OL = 10⁵, the relative error is about 37/10⁵ ≈ 0.04 %. The measured maximum error versus software is given in `results/s04_stageD_inference.json`.

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
