import sys, time, json, platform, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ns import config as C  # noqa: E402


def stamp():
    exe = C.ngspice_exe()
    try:
        v = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        v = next((l for l in v if "ngspice" in l.lower()), v[0] if v else "?")
    except Exception as e:  # pragma: no cover
        v = f"? ({e})"
    return dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                platform=platform.platform(), python=platform.python_version(),
                ngspice=exe, ngspice_version=v.strip())


for d in (C.GEN, C.WEIGHTS, C.RESULTS, C.PLOTS):
    d.mkdir(parents=True, exist_ok=True)
