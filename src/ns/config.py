"""Physical design constants for NeuralSphere-V1 (all SI units).

Every number here is a *design choice*, documented in docs/DESIGN.md.
"""
from pathlib import Path
import os, shutil, sys

ROOT = Path(__file__).resolve().parents[2]
CIRCUITS = ROOT / "circuits"
GEN = CIRCUITS / "generated"
MODELS = CIRCUITS / "models"
WEIGHTS = ROOT / "weights"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
DATA = ROOT / "data"

# ---- Inputs ---------------------------------------------------------------
VMAX = 1.0          # V   : pixel value 16/16 -> 1 V
VBIAS = 1.0         # V   : constant reference rail that drives the bias synapses
N_IN = 64
N_OUT = 10

# ---- Programmable synapse (conductance) range ------------------------------
R_MIN = 10e3        # ohm : lowest programmable resistance  -> G_MAX = 100 uS
R_MAX = 1e6         # ohm : highest programmable resistance -> G_MIN = 1 uS
G_MAX = 1.0 / R_MIN
G_MIN = 1.0 / R_MAX

# ---- Amplifiers -------------------------------------------------------------
VCC = 5.0           # V   : positive rail
VEE = -5.0          # V   : negative rail
R_DIFF = 100e3      # ohm : the four equal resistors of each difference amplifier
C_DIFF = 4.7e-12    # F   : compensation capacitor across RD2 (feedback) and RD4 (matching leg) of every
                    #       difference amplifier. Open at DC, so no DC result depends on it. With the TI OPA196
                    #       model it removes the 0.68 MHz ringing (zeta 0.14 -> no overshoot) and halves settling
                    #       (DESIGN.md section 11). Time constant R_DIFF * C_DIFF = 0.47 us. Set 0 to omit.
# R_F (transimpedance feedback) is chosen from the trained weights, see mapping.py

# ---- Vendor amplifier model (Texas Instruments OPAx196, PSpice, unmodified) ---
OPA196_LIB = ROOT / "tools" / "vendor_models" / "opa196_extracted" / "OPAx196.LIB"

# ---- ngspice executable ------------------------------------------------------
def ngspice_exe():
    env = os.environ.get("NGSPICE")
    if env:
        return env
    if sys.platform.startswith("win"):
        # official ngspice-47 Windows build (tools/ngspice) first, then the copy bundled with Qucs-S
        for sub in ("ngspice", "qucs-s"):
            cands = list((ROOT / "tools" / sub).rglob("ngspice_con.exe"))
            if cands:
                return str(cands[0])
    exe = shutil.which("ngspice_con") or shutil.which("ngspice")
    if exe:
        return exe
    raise FileNotFoundError("ngspice not found: set NGSPICE env var to ngspice(_con) executable")
