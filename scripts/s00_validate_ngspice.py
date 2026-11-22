"""Step 0: prove that ngspice solves a physical circuit we can check by hand.

Circuit: 5 V source -> 10 kOhm -> node mid -> 4.7 kOhm -> ground.
Hand calculation:  V_mid = 5 V * 4.7k / (10k + 4.7k) = 1.598639 V
                   I     = 5 V / 14.7 kOhm           = 340.136 uA
"""
import subprocess, re
from _common import C, stamp, ROOT
from ns.metrics import save_json

net = C.GEN / "s00_divider.cir"
net.write_text("""* validation divider
V1 in 0 DC 5
R1 in mid 10k
R2 mid 0 4.7k
.control
op
print v(mid) i(V1)
quit
.endc
.end
""")
out = subprocess.run([C.ngspice_exe(), "-b", str(net)], capture_output=True, text=True).stdout
vmid = float(re.search(r"v\(mid\)\s*=\s*([-+0-9.eE]+)", out).group(1))
i1 = float(re.search(r"i\(v1\)\s*=\s*([-+0-9.eE]+)", out, re.I).group(1))
v_hand = 5 * 4.7e3 / 14.7e3
i_hand = 5 / 14.7e3
res = dict(stamp=stamp(), v_mid_spice=vmid, v_mid_hand=v_hand, i_source_spice=i1,
           i_hand=i_hand, abs_err_V=abs(vmid - v_hand),
           note="ngspice reports i(V1) flowing into the + terminal, hence the negative sign",
           passed=abs(vmid - v_hand) < 1e-6 and abs(-i1 - i_hand) < 1e-9)
print(res)
save_json(res, C.RESULTS / "s00_validate_ngspice.json")
assert res["passed"]
