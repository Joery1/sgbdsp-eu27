"""EU27 inputs from EXIOBASE 3.11.2 (2017, ixi, 49 regions x 163 sectors) and
Eurostat env_wat_res (annual renewable freshwater, internal flow).
Monetary unit: bn EUR. Water: Mm3. CO2: Mt. Eutrophication: kt PO4-eq."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from paths import EXIO
CACHE, OUT = HERE/'cache', HERE/'results'
CACHE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

EU = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE',
      'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
EP_N, EP_P = 0.42, 3.06
CO2_ROWS = ['CO2 - combustion - air', 'CO2 - non combustion - Cement production - air',
            'CO2 - non combustion - Lime production - air', 'CO2 - waste - fossil - air']
S = 163
FOOD = list(range(0, 15))                                  # crops, livestock, raw milk, wool
ENERGY = [19, 20, 21, 22, 55, 56, 57] + list(range(95, 111))

def read_sat(name, part='F'):
    return pd.read_csv(EXIO/name/f'{part}.txt', sep='\t', header=[0, 1], index_col=0)

if not (CACHE/'Z.npy').exists():
    Z = pd.read_csv(EXIO/'Z.txt', sep='\t', header=[0, 1], index_col=[0, 1])
    np.save(CACHE/'Z.npy', Z.to_numpy())
    pd.DataFrame(Z.index.tolist(), columns=['region', 'sector']).to_csv(CACHE/'labels.csv', index=False)
Z = np.load(CACHE/'Z.npy')
lab = pd.read_csv(CACHE/'labels.csv')
regions = list(dict.fromkeys(lab.region))
n = len(lab)
x = pd.read_csv(EXIO/'x.txt', sep='\t', index_col=[0, 1]).iloc[:, 0].to_numpy()
Y = pd.read_csv(EXIO/'Y.txt', sep='\t', header=[0, 1], index_col=[0, 1])
y = Y.to_numpy().sum(1)
inv_x = np.divide(1, x, out=np.zeros(n), where=x > 1e-9)
A = Z*inv_x
L = np.linalg.inv(np.eye(n)-A)
print('closure max rel err', np.max(np.abs(L@y-x)/np.maximum(x, 1)))

eu = np.isin(lab.region.to_numpy(), EU)
iE, iR = np.where(eu)[0], np.where(~eu)[0]
mask = x < 1e-3                                           # < 1000 EUR output: inactive

def coef(F_row):
    return np.where(mask, 0, F_row*inv_x)

# satellites (per M EUR; converted to per bn EUR below)
W = read_sat('water')
wd = W.index.str.startswith('Water Withdrawal Blue')
excl = W.index.str.contains('throughflow|hydropower|Domestic')
omega = coef(W[wd & ~excl].to_numpy().sum(0))             # Mm3 / M EUR
WY = read_sat('water', 'F_Y')
dom = WY.loc['Water Withdrawal Blue - Domestic - domestic'].groupby(level=0).sum().reindex(EU).to_numpy()
air = read_sat('air_emissions')
gamma = coef(air.loc[CO2_ROWS].to_numpy().sum(0)/1e9)     # Mt / M EUR
nu = coef(read_sat('factor_inputs').to_numpy().sum(0))    # M EUR / M EUR
nut = read_sat('nutrients')
ep = EP_N*(nut.loc['N - agriculture - water']+nut.loc['N - waste - water']) \
   + EP_P*(nut.loc['P - agriculture - water']+nut.loc['P - waste - water'])
eta = coef(ep.to_numpy()/1e6)                             # kt / M EUR

# switch to bn EUR
k = 1e3
omega, gamma, eta = omega*k, gamma*k, eta*k
y_bn, x_bn = y/k, x/k
LE = L[iE]                                               # EU rows, all columns
const_E = LE[:, iR]@y_bn[iR]                              # EU output driven by fixed RoW final demand
r_co2 = gamma[iR]@L[iR]                                  # RoW CO2 per unit of any final demand
r_w = omega[iR]@L[iR]
np.save(CACHE/'LE.npy', LE)
np.save(CACHE/'A_Ecols.npy', A[:, iE].astype(np.float32))
np.savez(CACHE/'inputs.npz', y=y_bn, x=x_bn, omega=omega, gamma=gamma, eta=eta, nu=nu, mask=mask,
         iE=iE, const_E=const_E, r_co2=r_co2, r_w=r_w, dom=dom)

# --- Eurostat water resources (EL in Eurostat = GR in EXIOBASE)
res = pd.read_csv(HERE/'data/env_wat_res_long.csv')
res['geo'] = res.geo.replace({'EL': 'GR'})
piv = {wp: res[(res.wat_proc == wp) & (res.value > 0)].pivot(index='time', columns='geo', values='value')
       for wp in ['RFW_RES', 'INTFL']}
src = {}
for c in EU:
    for wp, minyrs in [('RFW_RES', 20), ('INTFL', 12)]:
        if c in piv[wp] and piv[wp][c].notna().sum() >= minyrs:
            src[c] = wp; break
    else:
        src[c] = None
series = pd.DataFrame({c: piv[src[c]][c] for c in EU if src[c]}).sort_index()
series.to_csv(OUT/'water_resources_series.csv')
xE = x_bn[iE]
w0_prod = (omega[iE]*xE).reshape(len(EU), S).sum(1)
meta = dict(EU=EU, S=S, food=FOOD, energy=ENERGY, source=src,
            gdp0_bn=float(nu[iE]@xE), co2_0=float(gamma[iE]@xE), ep0=float(eta[iE]@xE),
            prodwater0=float(w0_prod.sum()), domestic=float(dom.sum()),
            prodwater_by_country=dict(zip(EU, w0_prod.round(1).tolist())))
json.dump(meta, open(OUT/'meta.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k not in ('EU', 'food', 'energy')}, indent=1))
