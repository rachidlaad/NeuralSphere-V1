# Comparison: new run vs reference `linux_ngspice47_2026-11-21`

## Pass/fail checks on the new run

| check | result | value |
|---|---|---|
| s00 divider: |V_spice - V_hand| < 1 uV | PASS | 4.56e-07 V |
| s01 Stage A ideal: |S_spice - S_hand| < 10 uV | PASS | 2.5e-06 V |
| s01 Stage A real: |S_spice - S_hand| < 100 uV | PASS | 2.25e-05 V |
| s02 software softmax test = 348/360 | PASS | 348 |
| s02 sklearn baseline test = 344/360 | PASS | 344 |
| s03 SPICE ideal test = 348/360 | PASS | 348 |
| s03 SPICE vs software disagreements = 0 | PASS | 0 |
| s03 max |S_spice - S_kcl| < 100 uV | PASS | 1.64e-05 V |
| s04 D1_real_continuous test = 348/360 | PASS | 348 |
| s04 D1_real_continuous disagreements vs software = 0 | PASS | 0 |
| s04 D2_real_8bit test = 348/360 | PASS | 348 |
| s04 D2_real_8bit disagreements vs software = 0 | PASS | 0 |
| s07 transient: final prediction equals DC (.op) prediction for all 6 images | PASS | [(5, 5), (2, 2), (8, 8), (1, 6), (7, 7), (2, 2)] |
| s08 WTA test = 348/360 | PASS | 348 |
| s08 WTA ambiguous = 0 | PASS | 0 |
| s08 WTA vs argmax disagreements = 0 | PASS | 0 |
| s10 schematic-extracted S matches hand value | PASS | -0.439311 |

## JSON result files

| file | numeric leaves differing | max abs diff | max rel diff | non-numeric changes | keys only in one run |
|---|---|---|---|---|---|
| demo_gui_verification.json | (no reference) | | | | |
| s00_validate_ngspice.json | 0 | 0 | 0 | 0 | 0 |
| s01_stageA_single_neuron.json | 0 | 0 | 0 | 0 | 0 |
| s02_software_reference.json | 1 | 1.11e-15 | 6.89e-16 | 0 | 0 |
| s02_training_history.json | 70 | 2.22e-16 | 4.87e-16 | 0 | 0 |
| s03_stageBC_verification.json | 1 | 2.89e-15 | 5.87e-07 | 0 | 0 |
| s04_stageD_inference.json | 12 | 8.42e-11 | 3.64e-09 | 0 | 0 |
| s05_stageE_in_loop.json | 0 | 0 | 0 | 0 | 0 |
| s06_nonidealities_summary.json | 0 | 0 | 0 | 0 | 0 |
| s07_transient_energy.json | 37 | 9.8e-07 | 0.554 | 0 | 2 |
| s07_transient_energy_opa196.json | (no reference) | | | | |
| s08_wta.json | 6 | 1.4e-09 | 2.19e-09 | 0 | 0 |
| s10_qucs_schematic_check.json | 0 | 0 | 0 | 0 | 0 |
| s11_qucs_native_check.json | (no reference) | | | | |
| s12_opa196_stageD.json | (no reference) | | | | |
| s13_noise.json | (no reference) | | | | |

## NPZ arrays (max |new - ref| over all elements)

| file | array | shape | max abs diff |
|---|---|---|---|
| results/s03_spice_ideal_outputs.npz | s_te | (360, 10) | 0 |
| results/s03_spice_ideal_outputs.npz | s_tr | (1437, 10) | 0 |
| results/s03_spice_ideal_outputs.npz | tn_te | (360, 10) | 0 |
| results/s03_spice_ideal_outputs.npz | tp_te | (360, 10) | 0 |
| results/s04_D1_real_continuous_outputs.npz | p_bias | (360,) | 0 |
| results/s04_D1_real_continuous_outputs.npz | p_inputs | (360,) | 0 |
| results/s04_D1_real_continuous_outputs.npz | p_total | (360,) | 1e-11 |
| results/s04_D1_real_continuous_outputs.npz | p_vcc | (360,) | 5e-12 |
| results/s04_D1_real_continuous_outputs.npz | p_vee | (360,) | 5e-12 |
| results/s04_D1_real_continuous_outputs.npz | pred_te | (360,) | 0 |
| results/s04_D1_real_continuous_outputs.npz | s_te | (360, 10) | 1e-10 |
| results/s04_D1_real_continuous_outputs.npz | s_tr | (1437, 10) | 1e-10 |
| results/s04_D1_real_continuous_outputs.npz | tn | (360, 10) | 0 |
| results/s04_D1_real_continuous_outputs.npz | tp | (360, 10) | 0 |
| results/s04_D2_real_8bit_outputs.npz | p_bias | (360,) | 0 |
| results/s04_D2_real_8bit_outputs.npz | p_inputs | (360,) | 0 |
| results/s04_D2_real_8bit_outputs.npz | p_total | (360,) | 5e-12 |
| results/s04_D2_real_8bit_outputs.npz | p_vcc | (360,) | 5e-12 |
| results/s04_D2_real_8bit_outputs.npz | p_vee | (360,) | 5e-12 |
| results/s04_D2_real_8bit_outputs.npz | pred_te | (360,) | 0 |
| results/s04_D2_real_8bit_outputs.npz | s_te | (360, 10) | 1e-10 |
| results/s04_D2_real_8bit_outputs.npz | s_tr | (1437, 10) | 1e-10 |
| results/s04_D2_real_8bit_outputs.npz | tn | (360, 10) | 0 |
| results/s04_D2_real_8bit_outputs.npz | tp | (360, 10) | 0 |
| results/s05_chipA_6bit_gauss5pct_vos5mV_outputs.npz | s_te_after | (360, 10) | 0 |
| results/s05_chipA_6bit_gauss5pct_vos5mV_outputs.npz | s_te_before | (360, 10) | 0 |
| results/s05_chipB_4bit_gauss10pct_vos10mV_outputs.npz | s_te_after | (360, 10) | 0 |
| results/s05_chipB_4bit_gauss10pct_vos10mV_outputs.npz | s_te_before | (360, 10) | 0 |
| results/s05_chipC_2bit_gauss20pct_vos20mV_lr0.5_outputs.npz | s_te_after | (360, 10) | 0 |
| results/s05_chipC_2bit_gauss20pct_vos20mV_lr0.5_outputs.npz | s_te_before | (360, 10) | 0 |
| results/s05_chipC_2bit_gauss20pct_vos20mV_lr0.5_outputs.npz | s_te_best_train | (360, 10) | 0 |
| results/s05_chipC_2bit_gauss20pct_vos20mV_outputs.npz | s_te_after | (360, 10) | 0 |
| results/s05_chipC_2bit_gauss20pct_vos20mV_outputs.npz | s_te_before | (360, 10) | 0 |
| results/s07_transient.npz | P | (6038,) | 5.27e-05 |
| results/s07_transient.npz | P_amp | (6038,) | 5.27e-05 |
| results/s07_transient.npz | P_xbar | (6038,) | 1.95e-08 |
| results/s07_transient.npz | T | () | 0 |
| results/s07_transient.npz | idx | (6,) | 0 |
| results/s07_transient.npz | s | (6038, 10) | 0.435 |
| results/s07_transient.npz | t | (6038,) | 0 |
| results/s07_transient.npz | vin | (6038, 64) | 0 |
| results/s08_wta_outputs.npz | pred_wta | (360,) | 0 |
| results/s08_wta_outputs.npz | s | (360, 10) | 1e-10 |
| results/s08_wta_outputs.npz | w | (360, 10) | 3e-09 |
| weights/in_loop_chipA_6bit_gauss5pct_vos5mV.npz | En | (10, 65) | 0 |
| weights/in_loop_chipA_6bit_gauss5pct_vos5mV.npz | Ep | (10, 65) | 0 |
| weights/in_loop_chipA_6bit_gauss5pct_vos5mV.npz | Gn | (10, 65) | 0 |
| weights/in_loop_chipA_6bit_gauss5pct_vos5mV.npz | Gp | (10, 65) | 0 |
| weights/in_loop_chipA_6bit_gauss5pct_vos5mV.npz | Rf | () | 0 |
| weights/in_loop_chipA_6bit_gauss5pct_vos5mV.npz | Wa_final | (10, 65) | 0 |
| weights/in_loop_chipA_6bit_gauss5pct_vos5mV.npz | vos | (10, 3) | 0 |
| weights/in_loop_chipB_4bit_gauss10pct_vos10mV.npz | En | (10, 65) | 0 |
| weights/in_loop_chipB_4bit_gauss10pct_vos10mV.npz | Ep | (10, 65) | 0 |
| weights/in_loop_chipB_4bit_gauss10pct_vos10mV.npz | Gn | (10, 65) | 0 |
| weights/in_loop_chipB_4bit_gauss10pct_vos10mV.npz | Gp | (10, 65) | 0 |
| weights/in_loop_chipB_4bit_gauss10pct_vos10mV.npz | Rf | () | 0 |
| weights/in_loop_chipB_4bit_gauss10pct_vos10mV.npz | Wa_final | (10, 65) | 0 |
| weights/in_loop_chipB_4bit_gauss10pct_vos10mV.npz | vos | (10, 3) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV.npz | En | (10, 65) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV.npz | Ep | (10, 65) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV.npz | Gn | (10, 65) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV.npz | Gp | (10, 65) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV.npz | Rf | () | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV.npz | Wa_final | (10, 65) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV.npz | vos | (10, 3) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV_lr0.5.npz | En | (10, 65) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV_lr0.5.npz | Ep | (10, 65) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV_lr0.5.npz | Gn | (10, 65) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV_lr0.5.npz | Gp | (10, 65) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV_lr0.5.npz | Rf | () | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV_lr0.5.npz | Wa_final | (10, 65) | 0 |
| weights/in_loop_chipC_2bit_gauss20pct_vos20mV_lr0.5.npz | vos | (10, 3) | 0 |
| weights/software_weights.npz | W | (10, 64) | 1.11e-15 |
| weights/software_weights.npz | b | (10,) | 1.78e-15 |

## Largest differences per file

### s02_software_reference.json

| key | reference | new | abs diff | rel diff |
|---|---|---|---|---|
| `max_abs_b` | 1.611233085 | 1.611233085 | 1.11e-15 | 6.89e-16 |

### s02_training_history.json

| key | reference | new | abs diff | rel diff |
|---|---|---|---|---|
| `[35].train_loss` | 0.1708751401 | 0.1708751401 | 8.33e-17 | 4.87e-16 |
| `[2].train_loss` | 0.531493498 | 0.531493498 | 2.22e-16 | 4.18e-16 |
| `[135].train_loss` | 0.1396570923 | 0.1396570923 | 5.55e-17 | 3.97e-16 |
| `[139].train_loss` | 0.1397304082 | 0.1397304082 | 5.55e-17 | 3.97e-16 |
| `[80].train_loss` | 0.1461038455 | 0.1461038455 | 5.55e-17 | 3.8e-16 |
| `[77].train_loss` | 0.1469469037 | 0.1469469037 | 5.55e-17 | 3.78e-16 |
| `[71].train_loss` | 0.1500789395 | 0.1500789395 | 5.55e-17 | 3.7e-16 |
| `[75].train_loss` | 0.1506267281 | 0.1506267281 | 5.55e-17 | 3.69e-16 |

### s03_stageBC_verification.json

| key | reference | new | abs diff | rel diff |
|---|---|---|---|---|
| `max_rel_err_R_vs_1_over_G` | 4.919934904e-09 | 4.919932017e-09 | 2.89e-15 | 5.87e-07 |

### s04_stageD_inference.json

| key | reference | new | abs diff | rel diff |
|---|---|---|---|---|
| `D2_real_8bit.max_abs_logit_err` | 0.02312937946 | 0.02312937938 | 8.42e-11 | 3.64e-09 |
| `D1_real_continuous.max_abs_logit_err` | 0.0008262500177 | 0.0008262500177 | 3.55e-15 | 4.3e-12 |
| `D1_real_continuous.supply_current_A.i_vcc_mean` | 0.01507785498 | 0.01507785498 | 3.33e-14 | 2.21e-12 |
| `D1_real_continuous.power_W.p_vcc.mean` | 0.07538927489 | 0.07538927489 | 1.67e-13 | 2.21e-12 |
| `D2_real_8bit.power_W.p_vcc.mean` | 0.07538944312 | 0.07538944312 | 1.67e-13 | 2.21e-12 |
| `D2_real_8bit.supply_current_A.i_vcc_mean` | 0.01507788862 | 0.01507788862 | 3.33e-14 | 2.21e-12 |
| `D1_real_continuous.power_W.p_total.mean` | 0.1830143367 | 0.1830143367 | 1.39e-13 | 7.59e-13 |
| `D2_real_8bit.power_W.p_total.mean` | 0.183018978 | 0.183018978 | 1.25e-13 | 6.83e-13 |

### s07_transient_energy.json

| key | reference | new | abs diff | rel diff |
|---|---|---|---|---|
| `per_image[4].decision_time_s` | 6.5e-07 | 1.01e-06 | 3.6e-07 | 0.554 |
| `per_image[2].decision_time_s` | 9.3e-07 | 1.35e-06 | 4.2e-07 | 0.452 |
| `per_image[5].decision_time_s` | 1.35e-06 | 1.89e-06 | 5.4e-07 | 0.4 |
| `per_image[1].decision_time_s` | 1.49e-06 | 2.07e-06 | 5.8e-07 | 0.389 |
| `per_image[3].decision_time_s` | 4.01e-06 | 4.99e-06 | 9.8e-07 | 0.244 |
| `per_image[4].settling_time_s` | 4.25e-06 | 5.17e-06 | 9.2e-07 | 0.216 |
| `per_image[3].settling_time_s` | 4.13e-06 | 4.99e-06 | 8.6e-07 | 0.208 |
| `per_image[2].settling_time_s` | 4.33e-06 | 5.23e-06 | 9e-07 | 0.208 |
| `method` | key present in only one run | | | |
| `opamp` | key present in only one run | | | |

### s08_wta.json

| key | reference | new | abs diff | rel diff |
|---|---|---|---|---|
| `min_top2_gap_V` | 0.00456765069 | 0.0045676507 | 1e-11 | 2.19e-09 |
| `winner_line_V.min` | 4.999997131 | 4.99999713 | 1.4e-09 | 2.8e-10 |
| `winner_line_V.mean` | 4.99999999 | 4.999999991 | 2.78e-10 | 5.56e-11 |
| `loser_line_V.max` | 0.449292207 | 0.449292207 | 2e-11 | 4.45e-11 |
| `loser_line_V.mean` | 0.3819317793 | 0.3819317793 | 1.3e-13 | 3.39e-13 |
| `mean_total_power_W` | 0.2750924674 | 0.2750924674 | 4.16e-14 | 1.51e-13 |

