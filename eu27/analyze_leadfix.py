"""Decoupling test for Section 3.4. Under fixed normalisation the drier regime drives lambda below zero, and the leader's
band (1-lambda) t widens with it. This table compares, for standard and fixed normalisation, the coupled band (central
model), a band whose width is fixed at the central run's (1-lambda_ref) t (LEADFIX=1) and a band whose width and centre
are both fixed at the central run (LEADFIX=2). Bi-level means over 12 scenarios; changes against the central run with
the same normalisation and band. Writes results/table_fixednorm_leadfix.csv."""
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent/'results'
P = 'scenario_results_YEAR2030_EFLOW1_DELTA0.05_CAPBOUND1_DESIGN2_CO2CAP1_BASIN1_CO2CAP_POWER0_'
VAR = {'central': 'PRDRY0', 'pen x0.1': 'PRDRY0_PEN_SCALE0.1', 'pen x10': 'PRDRY0_PEN_SCALE10', 'drier': 'PRDRY1'}
BAND = {0: 'coupled', 1: 'fixed width', 2: 'fixed width and centre'}
lam_ref = {f.stem: float(np.load(f)['lam']) for f in (RES/'leader_ref').glob('*.npz')}

def suffix(var, norm, lf):
    s = VAR[var]
    if norm == 'fixed':
        s = s.replace('PRDRY0_PEN', 'PRDRY0_FIXNORM2_PEN') if 'PEN' in s else s+'_FIXNORM2'
    return s+(f'_LEADFIX{lf}' if lf else '')

rows = []
for norm in ('standard', 'fixed'):
    for lf, band in BAND.items():
        for var in VAR:
            d = pd.read_csv(RES/f'{P}{suffix(var, norm, lf)}.csv'); b = d[(d.model == 'SGBDSP') & d.feasible]
            key = b.scenario+'_'+b.transfer+'_'+b.unconventional.astype(int).astype(str)
            width = 1-(b['lambda'] if lf == 0 else key.map(lam_ref))
            rows.append(dict(normalisation=norm, band=band, variant=var, feasible=len(b), lam=b['lambda'].mean(),
                             band_width_x_t=width.mean(), VA_tn=b.GDP.mean()/1e3, CO2_Mt=b.CO2.mean(), food=b.food.mean(),
                             exceedance_bn=b.exp_shortage.mean()/1e3, irrigation_bn=b.saved_IRR.mean()/1e3,
                             dry_cooling_bn=b.saved_AIR.mean()/1e3))
t = pd.DataFrame(rows)
for c in ['VA_tn', 'CO2_Mt', 'food', 'exceedance_bn', 'irrigation_bn', 'dry_cooling_bn']:
    base = t[t.variant == 'central'].set_index(['normalisation', 'band'])[c]
    t[c+'_chg%'] = 100*(t[c]/pd.MultiIndex.from_frame(t[['normalisation', 'band']]).map(base)-1)
t.to_csv(RES/'table_fixednorm_leadfix.csv', index=False)
pd.set_option('display.width', 250)
print(t[['normalisation', 'band', 'variant', 'feasible', 'lam', 'band_width_x_t', 'VA_tn_chg%', 'food_chg%', 'exceedance_bn_chg%',
         'irrigation_bn', 'irrigation_bn_chg%', 'dry_cooling_bn']].round(3).to_string(index=False))
