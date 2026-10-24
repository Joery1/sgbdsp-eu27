"""Per-fragment detail of the reliability check (central run, scenario P05 without alternative water): violation probability of each
compliant ES/FR/IT fragment under the national marginal and under the fragment's own Q3 hydrology (lognormal fit, 2002-2014),
and historical summers with at least one violation, by fragment group. Writes results/reliability_fragments_P05.csv."""
import numpy as np, pandas as pd
import sys
from pathlib import Path
sys.argv = ['x', 'YEAR=2030', 'EFLOW=1', 'DELTA=0.05', 'CAPBOUND=1', 'DESIGN=2', 'CO2CAP=1', 'BASIN=1', 'CO2CAP_POWER=0', 'PRDRY=0']
from scipy import stats
import reliability_check as R
M = R.M
sc = next(M.scenarios())                      # P05, none
pl = R.plan(sc); rs = R.rstar(pl); grp = R.groups(pl['levels'])
d = pd.read_csv(M.HERE/'data/wei_subbasin_2002_2014.csv'); d = d[d.groupby('SB').Year_.transform('size') == 52]
d['R'] = np.where(d.WEI_2 > 0, d[M.GRP].sum(1)/(d.WEI_2/100), np.nan)
q3 = d[d.Quarter == 'Q3'].pivot_table(index='Year_', columns='SB', values='R')
rows = []
for c in R.JOINT:
    for g in ('free', 'grandfathered'):
        for f in grp[c][g]:
            sb = M.frags[f].split('|')[0]
            Rf = q3[sb].dropna().to_numpy() if sb in q3 else np.array([])
            pb = np.nan
            if len(Rf) >= 8 and Rf.min() > 0 and Rf.std() > 0:
                s, _, scl = stats.lognorm.fit(Rf, floc=0); pb = stats.lognorm.cdf(rs[f]*Rf.mean(), s, 0, scl)
            name, par = M.mpar[c]; dist = getattr(stats, name)
            pn = dist.cdf(rs[f]*dist.mean(*par), *par)
            rows.append(dict(c=c, g=g, frag=M.frags[f], WEI=M.WEIb[f, 2], lhs_over_2017=(pl['lhs'][f]-pl['alt'][f])/M.base17_fq[f, 2],
                             rstar=rs[f], p_national=pn, p_basin=pb, n_years=len(Rf), cv=Rf.std()/Rf.mean() if len(Rf) else np.nan))
t = pd.DataFrame(rows); pd.set_option('display.width', 200)
print(t[t.g == 'grandfathered'].round(3).to_string(index=False))
print(t.groupby(['c', 'g'])[['p_national', 'p_basin']].agg(['mean', 'max', 'count']).round(3))
q = pl['levels']['ES']
print('nominal q', round(q, 4))
print('free fragments with p_basin > q:', int((t[t.g == 'free'].p_basin > q).sum()), 'of', int(t[t.g == 'free'].p_basin.notna().sum()),
      '| with p_national > q:', int((t[t.g == 'free'].p_national > q).sum()))
print(t[(t.g == 'free') & (t.p_basin > q)][['c', 'frag', 'WEI', 'lhs_over_2017', 'rstar', 'p_national', 'p_basin', 'cv']].round(3).to_string(index=False))
for g in ('free', 'grandfathered'):
    yrs = np.zeros(len(q3), bool)
    for _, r in t[t.g == g].iterrows():
        sb = r.frag.split('|')[0]
        if sb in q3:
            Rf = q3[sb]; yrs |= (Rf/Rf.mean() < r.rstar).fillna(False).to_numpy()
    print(g, 'historical summers with >=1 violation:', int(yrs.sum()), 'of', len(yrs), list(q3.index[yrs]))
t.to_csv(M.OUT/'reliability_fragments_P05.csv', index=False)
