"""Membership values of the compromise in three scenarios (note to the fixed-normalisation table of the Supplementary Material).
Usage: python membership_check.py <model_eu.py arguments>, e.g. YEAR=2030 EFLOW=1 DELTA=0.05 CAPBOUND=1 DESIGN=2
CO2CAP=1 BASIN=1 CO2CAP_POWER=0 PRDRY=0 [FIXNORM=2 PEN_SCALE=10]. Memberships are measured against the endpoints saved
by a FIXNORM=1 central run (results/norm_endpoints)."""
import sys, json
from pathlib import Path
sys.argv = ['x'] + sys.argv[1:]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_eu as M

for i, sc in enumerate(M.scenarios()):
    if i not in (0, 5, 11):                                  # P05-none, P10-desal, P20-both
        continue
    r = M.run(sc)
    e = json.load(open(M.OUT/'norm_endpoints'/f"{sc['scenario']}_{sc['transfer']}_{int(sc['unconventional'])}.json"))
    mu = {k: (c@r['z']-e['worst'][k])/(e['best'][k]-e['worst'][k]) for k, (c, _) in M.OBJ.items()}
    print(sc['scenario'], sc['transfer'], sc['unconventional'], 'lambda=%.3f' % r['z'][M.iL],
          ' '.join(f'mu_{k}={v:.3f}' for k, v in mu.items()), flush=True)
