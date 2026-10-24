"""EXIOBASE 2017 blue-water withdrawal vs Eurostat freshwater abstraction (env_wat_abs, year closest to 2017) by country
and for agriculture, electricity cooling and public supply -> results/validation_water_exio_vs_eurostat.csv
(Table S7; also read by compare_eea_2023.py and cyprus_theta_check.py). Needs cache/inputs.npz from prepare_eu.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
w = pd.read_csv(HERE/'data/env_wat_abs_long.csv'); w['geo'] = w.geo.replace({'EL': 'GR'})
EU = json.load(open(HERE/'results/meta.json'))['EU']
d = np.load(HERE/'cache/inputs.npz'); iE = d['iE']; S = 163
ex = (d['omega'][iE]*d['x'][iE]).reshape(27, S)
grp = np.full(S, 'IND'); grp[:19] = 'AGR'; grp[95:111] = 'ELC'; grp[111] = 'PWS'; grp[114:] = 'SER'; grp[[112, 113]] = 'CON'
exio = pd.DataFrame({g: ex[:, grp == g].sum(1) for g in ['AGR', 'ELC', 'IND', 'PWS', 'SER', 'CON']}, index=EU)
exio['HH'] = d['dom']; exio['TOT'] = exio.sum(1)

def near(g, proc, yr=2017):
    s = w[(w.wat_src == 'FRW') & (w.wat_proc == proc) & (w.geo == g) & w.value.notna()]
    if s.empty:
        return np.nan, None
    r = s.assign(dist=(s.time-yr).abs()).sort_values('dist').iloc[0]
    return r.value, int(r.time)

rows = []
for c in EU:
    v, y = near(c, 'ABST'); a, _ = near(c, 'ABS_AGR'); e, _ = near(c, 'ABS_ELC_CL'); p, _ = near(c, 'ABS_PWS')
    rows.append(dict(c=c, euro_tot=v, yr=y, exio_tot=exio.loc[c, 'TOT'], ratio=exio.loc[c, 'TOT']/v if v else np.nan,
                     euro_agr=a, exio_agr=exio.loc[c, 'AGR'], euro_elc=e, exio_elc=exio.loc[c, 'ELC'], euro_pws=p,
                     exio_hh_pws=exio.loc[c, 'HH']+exio.loc[c, 'PWS']))
t = pd.DataFrame(rows).set_index('c')
t.to_csv(HERE/'results/validation_water_exio_vs_eurostat.csv')
ok = t.dropna(subset=['euro_tot'])
print('EU sum (countries with Eurostat total):', round(ok.exio_tot.sum()), 'vs', round(ok.euro_tot.sum()))
