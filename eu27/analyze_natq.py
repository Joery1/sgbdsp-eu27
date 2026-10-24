"""Seasonal and spatial parts of the scale contrast (Section 3.1). Three resolutions: national-annual (NA),
national-quarterly (NQ) and sub-basin-quarterly (BQ). Detection: each model's expected exceedance under its own rule
(central and at theta = 0.2, 0.6); the share of the BQ exceedance that NA misses splits into a seasonal part (NQ - NA)
and a spatial part (BQ - NQ). Decision: the plans of the three models evaluated under all three rules (results/crosseval);
under the BQ rule, the extra exceedance of the NA plan splits into what quarterly national planning recovers (NA - NQ)
and what only basin resolution recovers (NQ - BQ). Writes results/table_natq_decomposition.csv and table_natq_crosseval.csv."""
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parent/'results'
P = 'YEAR2030_EFLOW1_DELTA0.05_CAPBOUND1_DESIGN2_CO2CAP1_'
PLAN = {'NA': 'BASIN0_CO2CAP_POWER{p}_PRDRY0', 'NQ': 'BASIN0_CO2CAP_POWER{p}_PRDRY0_NATQ1', 'BQ': 'BASIN1_CO2CAP_POWER{p}_PRDRY0'}
RULE = {'NA': 'eval_national', 'NQ': 'eval_natq', 'BQ': 'eval_basin'}

def bi(tag):
    d = pd.read_csv(RES/f'scenario_results_{P}{tag}.csv'); return d[(d.model == 'SGBDSP') & d.feasible]

rows = []
for th, ex in ((0.2, '_THETA0.2'), (0.4, ''), (0.6, '_THETA0.6')):
    na = bi('BASIN0_CO2CAP_POWER0_PRDRY0'+ex).exp_shortage.mean()/1e3
    nq = bi('BASIN0_CO2CAP_POWER0_PRDRY0_NATQ1'+ex).exp_shortage.mean()/1e3
    bq = bi('BASIN1_CO2CAP_POWER0_PRDRY0'+ex).exp_shortage.mean()/1e3
    rows.append(dict(theta=th, NA=na, NQ=nq, BQ=bq, missed_NA_pct=100*(1-na/bq), missed_NQ_pct=100*(1-nq/bq),
                     seasonal_pct=100*(nq-na)/bq, spatial_pct=100*(bq-nq)/bq))
det = pd.DataFrame(rows)
det.to_csv(RES/'table_natq_decomposition.csv', index=False)
cx = []
for pol, pname in (('0', 'Ref'), ('0.8138', 'Power CO2 cap')):
    for plan, tag in PLAN.items():
        d = pd.read_csv(RES/'crosseval'/f'crosseval_{P}{tag.format(p=pol)}.csv')
        for m in ('bi-level', 'single'):
            g = d[d.model == m].mean(numeric_only=True)
            cx.append(dict(policy=pname, solution=m, plan=plan, **{f'eval_{r}': g[c]/1e3 for r, c in RULE.items()},
                           ES_BQ=g.basin_ES/1e3, ES_NQ=g.natq_ES/1e3, irrigation=g.irrigation/1e3, VA=g.GDP/1e3,
                           own_check=float((d[d.model == m].reported-d[d.model == m][RULE[plan]]).abs().max())))
X = pd.DataFrame(cx)
X.to_csv(RES/'table_natq_crosseval.csv', index=False)
pd.set_option('display.width', 220)
print('--- detection (own rule, Ref, historical; bn m3)'); print(det.round(3).to_string(index=False))
print('--- plans x rules (bn m3)'); print(X.round(3).to_string(index=False))
for pol in ('Ref', 'Power CO2 cap'):
    for m in ('bi-level', 'single'):
        x = X[(X.policy == pol) & (X.solution == m)].set_index('plan')
        na, nq, bq = x.loc['NA', 'eval_BQ'], x.loc['NQ', 'eval_BQ'], x.loc['BQ', 'eval_BQ']
        print(f'{pol:14s} {m:8s} BQ-rule: NA plan {na:.2f}  NQ plan {nq:.2f}  BQ plan {bq:.2f} | total gain {na-bq:.2f}'
              f' = quarterly national {na-nq:.2f} ({100*(na-nq)/(na-bq):.0f}%) + basin {nq-bq:.2f} ({100*(nq-bq)/(na-bq):.0f}%)')
