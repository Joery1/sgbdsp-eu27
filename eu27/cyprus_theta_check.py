"""Headline scale numbers (national vs basin exceedance, share missed) when Cyprus and Malta use the EEA WEI+ (and
Eurostat-scaled withdrawal) instead of EXIOBASE withdrawal / Eurostat resources; plans fixed, recomputed ex post.
Usage: python cyprus_theta_check.py <theta>   (needs results/eea2023_country.csv from compare_eea_2023.py)"""
import sys
theta = sys.argv[1]
sys.argv = ['x', 'YEAR=2030', 'EFLOW=1', 'DELTA=0.05', 'CAPBOUND=1', 'DESIGN=2', 'CO2CAP=1', 'BASIN=1', 'CO2CAP_POWER=0', 'PRDRY=0', f'THETA={theta}']
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import model_eu as M
R = M.OUT; ct = pd.read_csv(R/'eea2023_country.csv').set_index('country').apply(pd.to_numeric, errors='coerce'); va = pd.read_csv(R/'validation_water_exio_vs_eurostat.csv').set_index('c')
P = 'YEAR2030_EFLOW1_DELTA0.05_CAPBOUND1_DESIGN2_CO2CAP1_'
tag = '' if theta == '0.4' else f'_THETA{theta}'
files = {'basin': f'province_results_{P}BASIN1_CO2CAP_POWER0_PRDRY0{tag}.csv', 'national': f'province_results_{P}BASIN0_CO2CAP_POWER0_PRDRY0{tag}.csv'}
sfiles = {'basin': f'scenario_results_{P}BASIN1_CO2CAP_POWER0_PRDRY0{tag}.csv', 'national': f'scenario_results_{P}BASIN0_CO2CAP_POWER0_PRDRY0{tag}.csv'}
scs = {(s['scenario'], s['transfer'], s['unconventional']): s for s in M.scenarios()}
out = {}
for lab in ('basin', 'national'):
    prov = pd.read_csv(R/files[lab]); sr = pd.read_csv(R/sfiles[lab]); tot = sr[sr.model == 'SGBDSP'].exp_shortage.mean()
    res = {}
    for v in ('model', 'EEA WEI+', 'EEA WEI+ & Eurostat'):
        delta = 0.0
        for c in ('CY', 'MT'):
            i = M.EU.index(c); p = prov[prov.province == c]
            wei, k = {'model': (M.WEI_nat[i], 1.0), 'EEA WEI+': (ct.loc[c, 'new_WEI_annual_0214'], 1.0),
                      'EEA WEI+ & Eurostat': (ct.loc[c, 'new_WEI_annual_0214'], va.loc[c, 'euro_tot']/va.loc[c, 'exio_tot'])}[v]
            ex = []
            for _, r in p.iterrows():
                sc = scs[(r.scenario, r.transfer, r.unconventional)]
                cap = np.array([min(k*M.W17[i]*M.THETA*(M.flow[h, i]/M.Rbar[i])/wei, k*M.W0[i]*(1+M.INFRA)) for h in range(M.H)])+sc['alt'][i]
                ex.append(float(M.PR@np.maximum(k*r.alloc-cap, 0)))
            delta += np.mean(ex)-p.exp_shortage.mean()
        res[v] = (tot+delta)/1e3
    out[lab] = res
for v in out['basin']:
    n, b = out['national'][v], out['basin'][v]
    print(f'theta={theta} | {v:22s} national {n:.3f}  basin {b:.3f}  missed {100*(1-n/b):.0f}%')
