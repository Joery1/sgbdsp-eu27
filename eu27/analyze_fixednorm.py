"""Standard versus fixed normalisation of the memberships (Section 3.4): central run, penalty x0.1 and x10, drier regime,
bi-level means over 12 scenarios and changes against the central run with the same normalisation.
Writes results/table_fixed_normalisation.csv (Supplementary Table S22)."""
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parent/'results'
P = 'scenario_results_YEAR2030_EFLOW1_DELTA0.05_CAPBOUND1_DESIGN2_CO2CAP1_BASIN1_CO2CAP_POWER0_'
F = {('standard', 'central'): 'PRDRY0', ('standard', 'pen x0.1'): 'PRDRY0_PEN_SCALE0.1', ('standard', 'pen x10'): 'PRDRY0_PEN_SCALE10',
     ('standard', 'drier'): 'PRDRY1', ('fixed', 'central'): 'PRDRY0_FIXNORM2', ('fixed', 'pen x0.1'): 'PRDRY0_FIXNORM2_PEN_SCALE0.1',
     ('fixed', 'pen x10'): 'PRDRY0_FIXNORM2_PEN_SCALE10', ('fixed', 'drier'): 'PRDRY1_FIXNORM2'}
rows = []
for (norm, var), suf in F.items():
    d = pd.read_csv(RES/f'{P}{suf}.csv'); b = d[(d.model == 'SGBDSP') & d.feasible]
    rows.append(dict(normalisation=norm, variant=var, feasible=len(b), lam=b['lambda'].mean(), VA_tn=b.GDP.mean()/1e3, CO2_Mt=b.CO2.mean(),
                     food=b.food.mean(), exceedance_bn=b.exp_shortage.mean()/1e3, irrigation_bn=b.saved_IRR.mean()/1e3,
                     dry_cooling_bn=b.saved_AIR.mean()/1e3))
t = pd.DataFrame(rows)
for c in ['VA_tn', 'CO2_Mt', 'food', 'exceedance_bn', 'irrigation_bn', 'dry_cooling_bn']:
    base = t[t.variant == 'central'].set_index('normalisation')[c]
    t[c+'_chg%'] = 100*(t[c]/t.normalisation.map(base)-1)
t.to_csv(RES/'table_fixed_normalisation.csv', index=False)
pd.set_option('display.width', 250)
print(t.round(3).to_string(index=False))
