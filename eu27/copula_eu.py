"""EU27 marginals / FSP flow levels and a 3-D vine for Spain, France, Italy.
Reuses the copula machinery of ../copula.py. Dependence is fitted on internal
flow (longest common record); marginal quantiles use each country's series."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import copula as cp

OUT = HERE/'results'
JOINT = ['ES', 'FR', 'IT']
IMPUTE = {'DK': ['DE', 'SE'], 'GR': ['IT', 'CY', 'BG']}   # no Eurostat series

meta = json.load(open(OUT/'meta.json'))
EU = meta['EU']
ser = pd.read_csv(OUT/'water_resources_series.csv', index_col=0)
rows, fits = [], {}
for c in EU:
    if c not in ser:
        continue
    v = ser[c].dropna().to_numpy()
    name, par, aic = cp.fit_marginal(v)
    dist = getattr(stats, name)
    fits[c] = (name, par)
    rows.append(dict(country=c, dist=name, n_years=len(v), mean_obs=v.mean(),
                     L=cp.band_mean(dist, par, 0, .2), M=cp.band_mean(dist, par, .2, .8),
                     H=cp.band_mean(dist, par, .8, 1), ks_p=stats.kstest(v, name, par).pvalue, imputed=''))
mg = pd.DataFrame(rows).set_index('country')
for c, nb in IMPUTE.items():                                 # relative levels from neighbours
    rel = (mg.loc[nb, ['L', 'M', 'H']].div(mg.loc[nb, 'mean_obs'], axis=0)).mean()
    mg.loc[c] = dict(dist='imputed', n_years=0, mean_obs=1.0, L=rel.L, M=rel.M, H=rel.H,
                     ks_p=np.nan, imputed='+'.join(nb))
mg = mg.loc[EU]
mg.to_csv(OUT/'marginals_flow_levels.csv', encoding='utf-8-sig')
json.dump({c: [n, list(p)] for c, (n, p) in fits.items()}, open(OUT/'marginal_params.json', 'w'))

res = pd.read_csv(HERE/'data/env_wat_res_long.csv')
fl = res[(res.wat_proc == 'INTFL') & res.geo.isin(JOINT) & (res.value > 0)] \
       .pivot(index='time', columns='geo', values='value')[JOINT].dropna()
X = fl.to_numpy()
U = stats.rankdata(X, axis=0)/(len(X)+1)
vine = cp.fit_vine(U)
for pc in (vine['p_ac'], vine['p_bc'], vine['p_ab_c']):
    ww, vv = cp.RNG.random(1000), cp.RNG.random(1000)
    assert np.max(np.abs(pc.h(pc.hinv(ww, vv), vv)-ww)) < 1e-6, pc.label()
Sim = cp.simulate(vine, 1_000_000)
design = {.05: .004, .10: .03, .15: .05, .20: .004}
out = []
for tag, k in zip('XIN', [2, 1, 0]):                         # X: Italy unknown, I: France, N: Spain
    for m, (p, kn) in enumerate(design.items(), 1):
        q = cp.solve_unknown(Sim, p, [kn, kn], k)
        lv = [kn, kn]; lv.insert(k, q)
        out.append(dict(scenario=f'{tag}{m}', p=p, **{f'p_{c}': l for c, l in zip(JOINT, lv)},
                        p_indep=1-np.prod(1-np.array(lv)), check=cp.joint_violation(Sim, lv)))
sc = pd.DataFrame(out)
sc.to_csv(OUT/'joint_violation_scenarios.csv', index=False)
# symmetric design (2030 experiment): one common individual level q with joint violation p
sym = []
for p in [.05, .10, .20]:
    q = cp.optimize.brentq(lambda v: cp.joint_violation(Sim, [v]*3)-p, 1e-6, p)
    sym.append(dict(scenario=f'P{int(p*100):02d}', p=p, **{f'p_{c}': q for c in JOINT},
                    p_indep=1-(1-q)**3, check=cp.joint_violation(Sim, [q]*3)))
pd.DataFrame(sym).to_csv(OUT/'joint_violation_symmetric.csv', index=False)
print(pd.DataFrame(sym).round(4).to_string(index=False))
nm = lambda i: JOINT[i]
info = dict(years=[int(fl.index.min()), int(fl.index.max())], n_years=len(fl), root=nm(vine['root']),
            pairs={f"{nm(vine['a'])}-{nm(vine['root'])}": [vine['p_ac'].label(), vine['p_ac'].th],
                   f"{nm(vine['b'])}-{nm(vine['root'])}": [vine['p_bc'].label(), vine['p_bc'].th],
                   f"{nm(vine['a'])}-{nm(vine['b'])}|{nm(vine['root'])}": [vine['p_ab_c'].label(), vine['p_ab_c'].th]},
            tau_obs=[stats.kendalltau(X[:, i], X[:, j])[0] for i, j in [(0, 1), (0, 2), (1, 2)]],
            tau_sim=[stats.kendalltau(Sim[:20000, i], Sim[:20000, j])[0] for i, j in [(0, 1), (0, 2), (1, 2)]])
json.dump(info, open(OUT/'vine_fit.json', 'w'), indent=1, default=float)
print(json.dumps(info, indent=1, default=float))
print(mg.round(3).to_string())
print(sc.round(4).to_string(index=False))
