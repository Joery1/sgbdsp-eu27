"""2030 reference baseline for the EU27 model (EU Reference Scenario 2020).

The IO structure (A, value-added coefficients) stays at 2017. What changes:
1. Final demand of EU consumers grows with their real GDP (Ref2020, EUR2015), differentiated by
   product group with the Ref2020 GVA shares: g_{c,G} = GDP_2030/GDP_2017 x share_{G,2030}/share_{G,2015},
   G = agriculture (EXIOBASE 0-18), industry (19-113), services (114-162); GDP_2017 = Ref GDP_2015 x
   Eurostat CLV15 growth 2015-17. Rest-of-world final demand is held at 2017.
2. CO2 intensities are rescaled per country so that, at 2030 baseline output,
   power/district-heat CO2 and non-power CO2 follow the Ref2020 ratios 2030/2017.
   Ref 2017 is anchored on Ref 2015 x observed Eurostat change 2015-17 (CRF1A1A for
   power; TOTX4_MEMO - CRF1A1A - CRF1A4B for non-power, non-residential), avoiding the
   COVID-affected 2020 point.
3. Cooling-water intensity of power nodes is rescaled so power-sector water follows
   thermal generation (nuclear, solid, gas, oil, biomass); 2017 thermal generation is
   interpolated linearly between Ref 2015 and 2025.
4. Household water scales with population (Ref2020, 2017 interpolated 2015-2020).
Other intensities (agricultural irrigation, industry, nutrients) stay at 2017 values.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from paths import EXIO, REF2020 as REF
CACHE, OUT, DATA = HERE/'cache', HERE/'results', HERE/'data'
S = 163
POWER = list(range(95, 107)) + [110]                       # generation technologies + steam/hot water
THERMAL = ['Nuclear energy', 'Solid fossil fuels', 'Natural and manufactured gases', 'Petroleum products',
           'Biomass-waste (including biogas and waste gas)']
NONPOWER_REF = ['Energy Branch', 'Industry', 'Services (and agriculture)', 'Transport (9)',
                'Other CO2 Emissions (non land-use related) (MtCO2)']

def eurostat(path):
    j = json.load(open(path)); ids, size = j['id'], j['size']
    keys = [[k for k, _ in sorted(j['dimension'][d]['category']['index'].items(), key=lambda kv: kv[1])] for d in ids]
    rows = []
    for flat, v in j['value'].items():
        f, idx = int(flat), []
        for s in reversed(size):
            idx.append(f % s); f //= s
        idx = idx[::-1]
        rows.append({d: keys[i][idx[i]] for i, d in enumerate(ids)} | {'value': v})
    df = pd.DataFrame(rows)
    df['geo'] = df.geo.replace({'EL': 'GR'})
    return df

meta = json.load(open(OUT/'meta.json'))
EU = meta['EU']
d = dict(np.load(CACHE/'inputs.npz'))
iE = d['iE']; nE = len(iE)
LE = np.load(CACHE/'LE.npy')
lab = pd.read_csv(CACHE/'labels.csv')
ref = pd.read_parquet(REF)
def refv(item, yr, block):
    s = ref[(ref.block == block) & (ref.item == item) & (ref.year == yr)].set_index('country').value
    assert s.index.is_unique, (block, item, yr)
    return s.reindex(EU)

# ---- 1. GDP growth by consuming country
gdp = eurostat(DATA/'nama_gdp_clv15.json').pivot(index='geo', columns='time', values='value').reindex(EU)
gdp17 = refv('GDP (in 000 M€15)', 2015, 'macro')*gdp['2017']/gdp['2015']
g = refv('GDP (in 000 M€15)', 2030, 'macro')/gdp17
shares = {G: refv(f'Share of Gross Value-Added: {G} (%)', 2030, 'macro')/refv(f'Share of Gross Value-Added: {G} (%)', 2015, 'macro')
          for G in ['Agriculture', 'Industry', 'Services']}
Y = pd.read_csv(EXIO/'Y.txt', sep='\t', header=[0, 1], index_col=[0, 1])
reg = Y.columns.get_level_values(0)
rsec = np.arange(len(Y)) % S
rgrp = np.where(rsec <= 18, 'Agriculture', np.where(rsec <= 113, 'Industry', 'Services'))
F = np.ones((len(Y), len(reg)))
for j, r in enumerate(reg):
    if r in EU:
        for G in shares:
            F[rgrp == G, j] = g[r]*shares[G][r]
y17 = Y.to_numpy().sum(1)/1e3
y30 = (Y.to_numpy()*F).sum(1)/1e3                         # bn EUR
assert np.allclose(y17, d['y'])
iR = np.setdiff1d(np.arange(len(y17)), iE)
x17E = LE@y17
x30E = LE@y30
cst30 = LE[:, iR]@y30[iR]

# ---- 2. CO2 intensities
co2 = eurostat(DATA/'env_air_gge_crf.json').pivot_table(index='geo', columns=['src_crf', 'time'], values='value')
co2 = co2.reindex(EU)
pw_ratio17 = co2[('CRF1A1A', '2017')]/co2[('CRF1A1A', '2015')]
np17 = co2[('TOTX4_MEMO', '2017')]-co2[('CRF1A1A', '2017')]-co2[('CRF1A4B', '2017')]
np15 = co2[('TOTX4_MEMO', '2015')]-co2[('CRF1A1A', '2015')]-co2[('CRF1A4B', '2015')]
ref_pw17 = refv('Power generation/District heating', 2015, 'co2')*pw_ratio17
ref_np = lambda yr: sum(refv(i, yr, 'co2') for i in NONPOWER_REF)
ref_np17 = ref_np(2015)*np17/np15
R_pow = refv('Power generation/District heating', 2030, 'co2')/ref_pw17
R_np = ref_np(2030)/ref_np17

gE = d['gamma'][iE].copy(); oE = d['omega'][iE].copy()
cE = np.arange(nE)//S; sec = np.arange(nE) % S
isP = np.isin(sec, POWER)
rows = []
for i, c in enumerate(EU):
    for grp_, m, R in [('power', (cE == i) & isP, R_pow[c]), ('nonpower', (cE == i) & ~isP, R_np[c])]:
        e17, e30 = gE[m]@x17E[m], gE[m]@x30E[m]
        R = R if np.isfinite(R) and R > 0 else np.nan
        k = (e17*R)/e30 if (e30 > 0 and np.isfinite(R)) else 1.0
        flag = '' if np.isfinite(R) else 'no Ref ratio: kept 2017 intensity'
        if k > 2:
            flag, k = f'k={k:.2f} clipped to 2', 2.0
        gE[m] *= k
        rows.append(dict(country=c, group=grp_, co2_2017=e17, co2_2030_frozen=e30, ref_ratio=R, k=k,
                         co2_2030=gE[m]@x30E[m], note=flag))
co2cal = pd.DataFrame(rows)

# ---- 3. cooling water follows thermal generation
gen = lambda yr: sum(refv(i, yr, 'gen').fillna(0) for i in THERMAL)
th17 = gen(2015)+(gen(2025)-gen(2015))*0.2
T = (gen(2030)/th17).where(th17 > 0)
wrows = []
for i, c in enumerate(EU):
    m = (cE == i) & isP
    w17, w30 = oE[m]@x17E[m], oE[m]@x30E[m]
    k = (w17*T[c])/w30 if (w30 > 0 and np.isfinite(T[c])) else 1.0
    oE[m] *= k
    wrows.append(dict(country=c, power_water_2017=w17, frozen_2030=w30, thermal_ratio=T[c], k=k, power_water_2030=oE[m]@x30E[m]))
wcal = pd.DataFrame(wrows)

# ---- 4. households
pop = lambda yr: refv('Population (in million)', yr, 'macro')
dom30 = d['dom']*(pop(2030)/(pop(2015)+(pop(2020)-pop(2015))*0.4)).to_numpy()

gamma30, omega30 = d['gamma'].copy(), d['omega'].copy()
gamma30[iE], omega30[iE] = gE, oE
# 'x' stays the 2017 world output (defines inactive nodes); 2030 EU output is in 'x30E'
np.savez(CACHE/'inputs_2030.npz', **{**d, 'y': y30, 'x30E': x30E, 'gamma': gamma30, 'omega': omega30,
         'const_E': cst30, 'dom': dom30})
co2cal.to_csv(OUT/'calib2030_co2.csv', index=False); wcal.to_csv(OUT/'calib2030_power_water.csv', index=False)
pd.DataFrame({'gdp_growth_2017_2030': g, **{f'g_{G}': g*shares[G] for G in shares},
              'R_pow': R_pow, 'R_np': R_np, 'thermal_ratio': T}).to_csv(OUT/'calib2030_ratios.csv')

# ---- checks
nu = d['nu'][iE]
va17, va30 = nu@x17E, nu@x30E
e17 = d['gamma'][iE]@x17E; e30f = d['gamma'][iE]@x30E; e30 = gE@x30E
ref_total17 = (ref_pw17+ref_np17).sum(); ref_total30 = (refv('Power generation/District heating', 2030, 'co2')+ref_np(2030)).sum()
print('EU GDP growth 2017-2030 (Ref, consumption-weighted): value added x%.3f' % (va30/va17))
print('final demand EU consumers x%.3f; EU-produced final demand x%.3f' % (y30.sum()/y17.sum(), y30[iE].sum()/y17[iE].sum()))
print('EU industry CO2 2017 %.0f Mt | 2030 frozen intensity %.0f | 2030 calibrated %.0f (x%.3f)' % (e17, e30f, e30, e30/e17))
print('Ref2020 covered-scope ratio 2030/2017: %.3f' % (ref_total30/ref_total17))
print('power water 2017 %.0f -> 2030 %.0f Mm3; households %.0f -> %.0f' % (wcal.power_water_2017.sum(), wcal.power_water_2030.sum(), d['dom'].sum(), dom30.sum()))
print(co2cal[co2cal.note != ''].to_string(index=False))
print(pd.DataFrame({'g': g, **{f'g_{G[:4]}': g*shares[G] for G in shares}, 'R_pow': R_pow, 'R_np': R_np, 'T': T}).round(3).T.to_string())
agri = np.isin(np.arange(nE) % S, range(19))
print('EU agricultural output 2017->2030 x%.3f; agricultural water x%.3f' % (x30E[agri].sum()/x17E[agri].sum(), (oE*x30E)[agri].sum()/(d['omega'][iE]*x17E)[agri].sum()))
