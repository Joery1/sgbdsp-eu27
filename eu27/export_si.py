"""Model internals for the Supplementary Material (central 2030 configuration):
basin layer by country, problem size, and the implied direct cost of each technology option."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = ['x', 'YEAR=2030', 'EFLOW=1', 'DELTA=0.05', 'CAPBOUND=1', 'DESIGN=2', 'CO2CAP=1', 'BASIN=1',
            'CO2CAP_POWER=0', 'PRDRY=0']
import model_eu as m

OUT = HERE/'results/y2030'
EU = m.EU

# basin layer by country
rows = []
for c in EU:
    i = EU.index(c)
    f = np.where(m.fc == i)[0]
    rows.append(dict(country=c, fragments=len(f), coastal=int(m.is_coast[f].sum()),
                     summer_compliant=int((m.WEIb[f, 2] <= m.THETA).sum()) if len(f) else 0,
                     WEI_national=float(m.WEI_nat[i]), penalty_EUR_m3=float(m.pen[i]*1e3),
                     withdrawal_2030_bn_m3=float(m.W0[i]/1e3)))
pd.DataFrame(rows).to_csv(OUT/'S_basins_by_country.csv', index=False)

# problem size (first scenario of the central run)
sc = next(iter(m.scenarios()))
A, b, bnd = m.base_constraints(sc)
size = dict(n_variables=int(m.NV), n_constraints=int(A.shape[0]), n_options=int(m.K),
            reuse_bn_m3={k: float(v) for k, v in zip(EU, sc['reuse']/1e3)} if 'reuse' in sc else {})
# reuse and desalination volumes of the 'both' option
for s_ in m.scenarios():
    if s_['transfer'] == 'both':
        size['reuse_total_bn_m3'] = float(s_['reuse'].sum()/1e3)
        size['desal_total_bn_m3'] = float(s_['desal'].sum()/1e3)
        break
json.dump(size, open(OUT/'S_problem_size.json', 'w'), indent=1)

# implied direct cost per m3 saved and input increase relative to output (EU, weighted by reference output)
t = m.tech
xref = (m.LEE@m.y0+m.cst)[m.tnode]
rows = []
for k in m.OPTIONS:
    sel = t['name'] == k
    w, v, xr = -t['dw'][sel], -t['dv'][sel], xref[sel]
    rows.append(dict(option=k, n_options=int(sel.sum()), cost_EUR_per_m3=float((v*xr).sum()/(w*xr).sum()*1e3),
                     input_increase_pct_output=float(100*(v*xr).sum()/xr.sum()),
                     potential_bn_m3=float((w*xr*t['cap'][sel]).sum()/1e3)))
pd.DataFrame(rows).to_csv(OUT/'S_technology_cost.csv', index=False)
print(json.dumps(size, indent=1)[:400])
print(pd.DataFrame(rows))
