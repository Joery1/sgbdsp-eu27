"""Basin x quarter layer for the EU27 model from EEA WEI+ sub-basin data (2002-2014).

1. Sub-basins are split into basin-country fragments by area (point sampling in
   lon/lat, matplotlib Path; no GIS stack needed).
2. Fragment water use by EEA sector group and quarter, as a share of the
   country's annual use of that group, maps EXIOBASE country water into
   fragment-quarter water: use_fq = sum_g a[f,q,g] * W[c,g].
3. Implied renewable resources R = use / WEI give basin-quarter flow levels
   L/M/H (bands 0-20/20-80/80-100 of a lognormal fit) and stress WEI_bq.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE/'data', HERE/'results/basins'
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = ['AgricForestFish', 'ElecGasSteam', 'MiningConstruct', 'Service', 'WaterCollectTreat']
Q = ['Q1', 'Q2', 'Q3', 'Q4']
NAME2CODE = {'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Cyprus': 'CY', 'Czech Republic': 'CZ',
             'Germany': 'DE', 'Denmark': 'DK', 'Estonia': 'EE', 'Spain': 'ES', 'Finland': 'FI', 'France': 'FR',
             'Greece': 'GR', 'Croatia': 'HR', 'Hungary': 'HU', 'Ireland': 'IE', 'Italy': 'IT', 'Lithuania': 'LT',
             'Luxembourg': 'LU', 'Latvia': 'LV', 'Malta': 'MT', 'Netherlands': 'NL', 'Poland': 'PL',
             'Portugal': 'PT', 'Romania': 'RO', 'Sweden': 'SE', 'Slovenia': 'SI', 'Slovakia': 'SK'}
N_PTS = 800

def inside(rings, pts):
    """Even-odd rule over all rings (outer rings and holes)."""
    hit = np.zeros(len(pts), bool)
    for ring in rings:
        r = np.asarray(ring)
        lo, hi = r.min(0), r.max(0)
        cand = np.all((pts >= lo) & (pts <= hi), axis=1)
        if cand.any():
            hit[cand] ^= MPath(r).contains_points(pts[cand])
    return hit

def sample(rings, rng):
    allpts = np.vstack([np.asarray(r) for r in rings])
    lo, hi = allpts.min(0), allpts.max(0)
    for _ in range(6):
        p = lo+(hi-lo)*rng.random((N_PTS*4, 2))
        p = p[inside(rings, p)]
        if len(p) >= 50:
            return p[:N_PTS]
    return p

if __name__ == '__main__':
    rng = np.random.default_rng(7)
    sb_rings = json.load(open(DATA/'wei_subbasin_rings.json'))
    c_rings = {NAME2CODE[k]: v for k, v in json.load(open(DATA/'wei_country_rings.json')).items() if k in NAME2CODE}
    c_box = {c: (np.vstack([np.asarray(r) for r in v]).min(0), np.vstack([np.asarray(r) for r in v]).max(0))
             for c, v in c_rings.items()}
    share = []
    for sb, rings in sb_rings.items():
        p = sample(rings, rng)
        lab = np.full(len(p), '', dtype=object)
        for c, rr in c_rings.items():
            lo, hi = c_box[c]
            m = np.all((p >= lo) & (p <= hi), axis=1) & (lab == '')
            if m.any():
                idx = np.where(m)[0]
                lab[idx[inside(rr, p[idx])]] = c
        for c, k in pd.Series(lab[lab != '']).value_counts().items():
            share.append(dict(SB=sb, country=c, area_share=k/len(p)))
        cen = p.mean(0) if len(p) else np.full(2, np.nan)
        share.append(dict(SB=sb, country='_centroid', area_share=np.nan, lon=cen[0], lat=cen[1]))
    sh = pd.DataFrame(share)
    cent = sh[sh.country == '_centroid'][['SB', 'lon', 'lat']]
    sh = sh[(sh.country != '_centroid') & (sh.area_share >= 0.02)].drop(columns=['lon', 'lat'])

    d = pd.read_csv(DATA/'wei_subbasin_2002_2014.csv')
    d = d[d.groupby('SB').Year_.transform('size') == 52]                 # complete 2002-2014 records only
    d['use'] = d[GROUPS].sum(1)
    d['R'] = np.where(d.WEI_2 > 0, d.use/(d.WEI_2/100), np.nan)

    # basin-quarter hydrology: stress and L/M/H relative flow levels
    rows = []
    for (sb, q), g in d.groupby(['SB', 'Quarter']):
        R = g.R.dropna().to_numpy()
        wei = (g.use.sum()/R.sum()) if len(R) and R.sum() > 0 else 0.0
        if len(R) >= 8 and R.min() > 0 and R.std() > 0:
            s, loc, sc = stats.lognorm.fit(R, floc=0)
            qs = lambda a, b: stats.lognorm.ppf(np.linspace(a, b, 402)[1:-1], s, 0, sc).mean()
            lv = np.array([qs(0, .2), qs(.2, .8), qs(.8, 1)])/R.mean()
        else:
            lv = np.ones(3)
        rows.append(dict(SB=sb, Quarter=q, WEI=wei, rL=lv[0], rM=lv[1], rH=lv[2], n=len(R),
                         R_cv=R.std()/R.mean() if len(R) else np.nan))
    hyd = pd.DataFrame(rows)

    # fragment x quarter x group use (annual mean over years)
    mean_q = d.groupby(['SB', 'Quarter'])[GROUPS].mean().reset_index()
    fr = sh.merge(mean_q, on='SB')
    for g in GROUPS:
        fr[g] = fr[g]*fr.area_share
    tot = fr.groupby('country')[GROUPS].sum()                           # annual use per country & group
    a = fr.copy()
    for g in GROUPS:
        a['U_'+g] = a[g]                                                 # raw area-weighted EEA use (per quarter)
        a[g] = a[g]/a.country.map(tot[g]).replace(0, np.nan)
    a = a.fillna(0).merge(hyd, on=['SB', 'Quarter'], how='left')
    a['frag'] = a.SB+'|'+a.country
    a = a.merge(d.groupby('SB').SUB_BASIN.first().reset_index(), on='SB').merge(cent, on='SB')
    a.to_csv(OUT/'fragment_quarter_coefficients.csv', index=False)
    hyd.to_csv(OUT/'basin_quarter_hydrology.csv', index=False)

    cov = sh.groupby('country').SB.nunique().reindex(list(NAME2CODE.values())).fillna(0).astype(int)
    print('fragments', a.frag.nunique(), 'rows', len(a))
    print('sub-basins per country:', cov.to_dict())
    q3 = hyd[hyd.Quarter == 'Q3']
    print('Q3 stress: WEI>0.2:', (q3.WEI > .2).sum(), ' WEI>0.4:', (q3.WEI > .4).sum(), ' of', len(q3))
    print('Q3 dry-level ratio rL quantiles', q3.rL.quantile([.1, .5, .9]).round(2).to_dict())
    chk = a.groupby('country')[GROUPS].sum().round(3)
    print('coefficient sums per country (should be 1 or 0):', sorted(set(chk.to_numpy().ravel().round(2))))
