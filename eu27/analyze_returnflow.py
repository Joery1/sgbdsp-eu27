"""Return-flow sensitivity: does each main conclusion hold when only a share RHO_IRR of the withdrawal saved by
efficient irrigation reduces net consumption? Reads the cross-scale evaluation (results/crosseval) and the
leader-tolerance runs for RHO_IRR = 1 (central), 0.5 and 0.25, Ref policy and power-sector CO2 cap.
Exceedance is in consumption-equivalent withdrawal units when RHO_IRR < 1. At RHO_IRR = 0.25 the basin model is
infeasible without water reuse, so every RHO is also compared on the six reuse scenarios (subset 'reuse').
Writes results/table_return_flow.csv."""
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parent/'results'
P = 'YEAR2030_EFLOW1_DELTA0.05_CAPBOUND1_DESIGN2_CO2CAP1_'
KEY = ['scenario', 'transfer', 'unconventional']

def rho_tag(rho):
    return '' if rho == 1 else f'_RHO_IRR{rho:g}'

SUBSET = 'all'

def sub(d):
    return d[d.unconventional] if SUBSET == 'reuse' else d

def crosseval(b, pol, rho):
    return sub(pd.read_csv(RES/'crosseval'/f'crosseval_{P}BASIN{b}_CO2CAP_POWER{pol}_PRDRY0{rho_tag(rho)}.csv'))

def tol_run(pol, rho, tol):
    if rho == 1:
        name = (f'{P}BASIN1_CO2CAP_POWER{pol}_PRDRY0_LEADTOL0' if tol == 'none' else f'{P}BASIN1_PRDRY0_CO2CAP_POWER{pol}_TOL{tol}')
    else:
        name = f'{P}BASIN1_CO2CAP_POWER{pol}_PRDRY0{rho_tag(rho)}_' + ('LEADTOL0' if tol == 'none' else f'TOL{tol}')
    d = pd.read_csv(RES/f'scenario_results_{name}.csv')
    return sub(d[d.feasible])

def build():
  rows = []
  for rho in (1, 0.5, 0.25):
      for pol, pname in (('0', 'Ref'), ('0.8138', 'Power CO2 cap')):
          r = dict(rho=rho, policy=pname)
          nat, bas = crosseval(0, pol, rho), crosseval(1, pol, rho)
          r['feasible_national'], r['feasible_basin'] = len(nat)//2, len(bas)//2
          for m, tag in (('bi-level', 'bi'), ('single', 'single')):
              n, b = nat[nat.model == m], bas[bas.model == m]
              r[f'{tag}_natplan_at_basin'] = n.eval_basin.mean()/1e3
              r[f'{tag}_basinplan_at_basin'] = b.eval_basin.mean()/1e3
              r[f'{tag}_natplan_at_national'] = n.eval_national.mean()/1e3
              r[f'{tag}_basinplan_at_national'] = b.eval_national.mean()/1e3
              r[f'{tag}_planning_gain_bn'] = r[f'{tag}_natplan_at_basin']-r[f'{tag}_basinplan_at_basin']
              r[f'{tag}_planning_gain_pct'] = 100*(r[f'{tag}_natplan_at_basin']/r[f'{tag}_basinplan_at_basin']-1)
              r[f'{tag}_VA_diff_pct'] = 100*(n.GDP.mean()/b.GDP.mean()-1)
              r[f'{tag}_irr_natplan'] = n.irrigation.mean()/1e3
          bi3, s3 = bas[bas.model == 'bi-level'], bas[bas.model == 'single']
          r['irr_single'] = s3.irrigation.mean()/1e3
          r['irr_bi_tol3'] = bi3.irrigation.mean()/1e3
          r['exc_bi_tol3'] = bi3.eval_basin.mean()/1e3
          for tol, lab in (('0', 'tol0'), ('0.06', 'tol6'), ('none', 'none')):
              d = tol_run(pol, rho, tol)
              b = d[d.model == 'SGBDSP']
              r[f'feasible_{lab}'] = len(b)
              r[f'irr_bi_{lab}'] = b.saved_IRR.mean()/1e3
              r[f'exc_bi_{lab}'] = b.exp_shortage.mean()/1e3
          for lab in ('tol0', 'tol3', 'tol6', 'none'):
              r[f'gap_{lab}_pct'] = 100*(r[f'irr_bi_{lab}']/r['irr_single']-1)
          r['net_consumption_saving_bi_tol3'] = rho*r['irr_bi_tol3']
          rows.append(r)
  t = pd.DataFrame(rows)
  # effect of the power-sector cap on the bi-level plan, at the central tolerance and without the tolerance constraint
  for lab in ('tol0', 'tol3', 'tol6', 'none'):
      for rho in (1, 0.5, 0.25):
          ref, cap = t[(t.rho == rho) & (t.policy == 'Ref')].iloc[0], t[(t.rho == rho) & (t.policy != 'Ref')].index[0]
          t.loc[cap, f'cap_irr_effect_{lab}_pct'] = 100*(t.loc[cap, f'irr_bi_{lab}']/ref[f'irr_bi_{lab}']-1)
          t.loc[cap, f'cap_exc_effect_{lab}_pct'] = 100*(t.loc[cap, f'exc_bi_{lab}']/ref[f'exc_bi_{lab}']-1)
  t.insert(0, 'subset', SUBSET)
  return t

parts = []
for SUBSET in ('all', 'reuse'):
    parts.append(build())
T = pd.concat(parts, ignore_index=True)
T.to_csv(RES/'table_return_flow.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 80)
show = ['subset', 'rho', 'policy', 'bi_natplan_at_national', 'bi_basinplan_at_national', 'bi_natplan_at_basin', 'bi_basinplan_at_basin',
        'bi_planning_gain_bn', 'bi_planning_gain_pct', 'single_planning_gain_pct', 'bi_VA_diff_pct', 'bi_irr_natplan']
print('--- scale contrast (bn m3; bi-level unless noted)'); print(T[show].round(3).to_string(index=False))
show = ['subset', 'rho', 'policy', 'irr_single', 'irr_bi_tol0', 'irr_bi_tol3', 'irr_bi_tol6', 'irr_bi_none',
        'gap_tol0_pct', 'gap_tol3_pct', 'gap_tol6_pct', 'gap_none_pct', 'feasible_tol0', 'feasible_tol6', 'feasible_none']
print('--- value-added priority (water saved by efficient irrigation, bn m3)'); print(T[show].round(2).to_string(index=False))
show = ['subset', 'rho'] + [f'cap_{v}_effect_{lab}_pct' for lab in ('tol0', 'tol3', 'tol6', 'none') for v in ('irr', 'exc')]
print('--- power cap vs Ref, bi-level (%)'); print(T[T.policy != 'Ref'][show].round(1).to_string(index=False))
