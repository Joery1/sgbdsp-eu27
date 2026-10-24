"""SGBDSP for EU27 on EXIOBASE 3.11.2 (2017), same structure as ../model.py.

Decision space: final demand for EU27 production nodes (27 x 163); rest-of-world
final demand fixed but RoW CO2/water responses are tracked as leakage.
Leader (EU level): max EU value added - expected shortage penalty; controls
non-food, non-energy final demand and industrial water recycling.
L1 food (max agri output; irrigation tech), L2 energy (min EU CO2; dry cooling,
renewable substitution), L3 ecology (max environmental-flow reserve share minus
eutrophication share).
Uncertainty: FSP L/M/H per country (Eurostat 1970-2024); joint chance
constraint for Spain/France/Italy (vine copula).
EU analogues of the China case: desalination expansion replaces inter-basin
transfer, water reuse (Reg. 2020/741) replaces unconventional water; there are
no binding national abstraction red lines, so the ecological follower decides a
non-negative e-flow reserve instead of a floor on ecological water.
"""
from pathlib import Path
import itertools, json, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack as svstack, hstack as shstack

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import technology
from paths import EXIO, REF2020

CACHE, OUT = HERE/'cache', HERE/'results'
meta = json.load(open(OUT/'meta.json'))
EU, S = meta['EU'], meta['S']
C, H = len(EU), 3
PR = np.array([.2, .6, .2])
JOINT = ['ES', 'FR', 'IT']

DELTA, PHI, TOL = 0.20, 0.30, 0.03
ECO_MAX = 0.10        # e-flow reserve up to 10% of 2017 withdrawal
FOOD_FLOOR = 1.0
# extra desalination follows the EU capacity ranking (Spain ~60%, then Italy, Cyprus, Greece, Malta; EU Blue Economy Observatory)
DESAL_SHARE = {'ES': 60, 'IT': 15, 'CY': 10, 'GR': 8, 'MT': 5, 'PT': 2}
DESAL_LEVELS = {'zero': 0, 'low': 1000, 'medium': 2000, 'high': 3000}     # extra Mm3/yr
REUSE_RATE = 0.05
PHI_IMPUTED = {'GR': 1.0, 'DK': 0.25}  # no Eurostat series: stressed / unstressed assumption
TECH = 1
CAPBOUND = 0          # 1: adoption caps as bounds on baseline output (valid for a narrow demand band)
BASIN = 0             # 1: basin x quarter FSP (EEA WEI+ sub-basins) replaces national annual FSP
YEAR = 2017           # 2030: EU Reference 2020 baseline (calib2030.py)
EFLOW = 0             # 1: e-flow limit WEI+ <= THETA replaces the phi supply rule (needs BASIN=1)
THETA = 0.4           # maximum admissible WEI+ (EEA severe-stress threshold); 0.2 = presumptive standard
INFRA = 0.2           # withdrawal capacity above the reference-year baseline
CHANCE_ND = 1         # 1: grandfather 2017 abstraction in the summer reliability rows (non-deterioration)
CO2CAP = 0            # EU-wide cap on industry CO2 as a share of the baseline (0 = none; 1 = Ref2020 level)
CO2CAP_POWER = 0      # cap on power/district-heat CO2 as a share of its baseline (0.814 = Fit-for-55 tightening)
DESIGN = 1            # 1: 96-scenario design of the source paper; 2: 2030 design (3 reliability x 4 water options)
PRDRY = 0             # 1: drier regime, dry/normal/wet probabilities 0.4/0.5/0.1 instead of 0.2/0.6/0.2
CHANCE = 1            # 0: drop the joint reliability rows (used to isolate the technology effect)
PEN_MODE = 1          # shortage penalty: 1 = agricultural VA per m3 of agricultural withdrawal (central);
                      # 0 = country value added per m3 of total withdrawal (sensitivity, ~30x higher)
PEN_SCALE = 1.0       # multiplier on the shortage penalty (sensitivity)
LEADTOL = 1           # 0: drop the leader-tolerance rows (control: same objectives and feasible set, no hierarchy term)
FIXNORM = 0           # 1: save this run's membership endpoints per scenario; 2: use the saved (central) endpoints and the
                      # historical-regime baseline exceedance D0, so memberships are not re-normalised (test of Section 3.4)
LEADFIX = 0           # leader tolerance (FIXNORM=1 saves the reference): 0 = |y_U - y_U*| <= (1-lambda) t (central);
                      # 1 = width fixed at (1-lambda_ref) t of the central run, centre at this run's leader optimum;
                      # 2 = width and centre (the central run's leader optimum) both fixed, so lambda no longer moves the band
RHO_IRR = 1.0         # share of the withdrawal saved by efficient irrigation that reduces net consumption (return-flow
                      # sensitivity); applies to the water-stress and reliability rows, not to reported withdrawal
NATWEI_EEA = 1        # countries without a basin layer (Cyprus, Malta): 1 = EEA WEI+ (consumption-based, 2002-2014 mean);
                      # 0 = EXIOBASE 2017 withdrawal / Eurostat renewable resources (withdrawal-based, results before 2026-10-23)
NATQ = 0              # 1 (with BASIN=0): national-quarterly rule for countries with a basin layer, the comparator between the
                      # national-annual and the sub-basin-quarterly resolution; Cyprus and Malta keep the national annual rule
JOBS = 4
SCALE = {}
for a in sys.argv[1:]:
    k, v = a.split('=')
    if k.endswith(('_COST', '_CAP', '_SAVE')): SCALE[k] = float(v)
    else: globals()[k] = float(v)
if PRDRY:
    PR = np.array([.4, .5, .1])

# EXIOBASE sector positions; costs in EUR/m3 x 1e-3 because omega is Mm3 per bn EUR
MACH, ELMACH, ELDIST, WWT = 85, 87, 108, 151
FOSSIL_IN = [19, 20, 21, 22, 55, 56, 109]
# parameter sources (manuscript Table S2):
#  IRR saving 0.26 = withdrawal cut when surface/sprinkler water (Eurostat 2010: 19.3/16.5 bn m3) moves to drip at FAO
#      field application efficiencies 60/75/90%; cost 0.4 EUR/m3 saved brackets the Spanish Plan de Choque capital cost
#      (2,409 MEUR for 1,162 hm3/yr, ~0.13 EUR/m3 annualised at target savings) plus pressurisation energy and lower realised savings
#  AIR saving 0.93, cost ~+12% of output (Zhai et al. 2022 retrofits: -93% water, LCOE +12% coal / +18% gas); heat-rate penalty 4%
#      within 1-15% (Qin et al. 2023)
#  REN saving 0.99 (PV ~1 and wind 0 vs >=255 gal/MWh withdrawal for gas CC with towers; Macknick et al. 2012); direct CO2 removed
#      with the fuel inputs (life-cycle emissions enter through the equipment inputs); cost 1.1x avoided fuel bill
#  REC saving 0.30 = low end of the 30-50% potential in steel, pulp and paper, chemicals (EEA 2025); cost 0.8 EUR/m3 above the
#      >0.5 EUR/m3 typical total cost of reclaimed water (Pistocchi et al. 2018)
OPTIONS = {
    'IRR': dict(sectors=list(range(8)), saving=0.26, cap=0.30, cost_per_m3=0.4e-3,
                inputs={MACH: 0.7, ELDIST: 0.3}, owner='L1'),
    'AIR': dict(sectors=[95, 96, 97, 101], saving=0.93, cap=0.30, fuel_up=0.04, capex=0.10,
                fuel_sectors=FOSSIL_IN+[57, 17], inputs={MACH: 1.0}, owner='L2'),
    'REN': dict(sectors=[95, 96, 100], saving=0.99, cap=0.20, co2_cut=1.0, premium=1.10,
                fuel_sectors=FOSSIL_IN, inputs={ELMACH: 1.0}, owner='L2'),
    'REC': dict(sectors=[51, 53, 58, 60, 61, 62, 71, 75, 79], saving=0.30, cap=0.30, cost_per_m3=0.8e-3,
                inputs={WWT: 0.5, MACH: 0.5}, owner='U'),
}

d = dict(np.load(CACHE/('inputs_2030.npz' if YEAR == 2030 else 'inputs.npz')))
d17 = dict(np.load(CACHE/'inputs.npz'))
iE = d['iE']; nE = len(iE); N = len(d['x'])
LE = np.load(CACHE/'LE.npy')
LEE = LE[:, iE]
omega, gamma, nu, eta = d['omega'], d['gamma'], d['nu'], d['eta']
mask = d['mask']
y0 = d['y'][iE]
x0 = d['x30E'] if YEAR == 2030 else d['x'][iE]
cst = d['const_E']                                   # EU output driven by fixed RoW final demand

class _ACols:                                        # A[:, EU columns] seen through world indices
    def __init__(self, AE, iE, n):
        self.AE, self.shape = AE, (n, n)
        self.pos = np.full(n, -1); self.pos[iE] = np.arange(len(iE))
    def __getitem__(self, key):
        rows, i = key
        return self.AE[rows, self.pos[i]].astype(float)

regions = sorted({int(i)//S for i in iE})
tech = technology.build(_ACols(np.load(CACHE/'A_Ecols.npy'), iE, N), omega, gamma, mask, SCALE,
                        options=OPTIONS, regions=regions, S=S)
if not TECH:
    tech = {k: v[:0] if k != 'D' else v[:, :0] for k, v in tech.items()}
K = len(tech['node'])
pos = np.full(N, -1); pos[iE] = np.arange(nE)
tnode = pos[tech['node']]                            # technology node in EU indexing
G = LE@tech['D']
cE = np.arange(nE)//S                               # country of each EU node
agg = np.zeros((C, nE)); agg[cE, np.arange(nE)] = 1
onehot = np.zeros((C, K)); onehot[cE[tnode], np.arange(K)] = 1
oE, gE, vE, eE = omega[iE], gamma[iE], nu[iE], eta[iE]
sec = np.arange(nE) % S
food_nodes = np.isin(sec, meta['food'])
energy_nodes = np.isin(sec, meta['energy'])
mE = mask[iE]
lead_nodes = ~(food_nodes | energy_nodes) & ~mE

Mw, Mw_z, Mw_c = (agg*oE)@LEE, (agg*oE)@G+onehot*tech['dw'], (agg*oE)@cst
dw_c = tech['dw']*np.where(tech['name'] == 'IRR', RHO_IRR, 1.0)            # consumption-equivalent saving in the limits
Mw_zc = (agg*oE)@G+onehot*dw_c
g_gdp, g_gdp_z = vE@LEE, vE@G+tech['dv']
g_food, g_food_z = LEE[food_nodes].sum(0), G[food_nodes].sum(0)
g_co2, g_co2_z = gE@LEE, gE@G+tech['dg']
isP = np.isin(sec, list(range(95, 107))+[110])                 # power generation + district heat
gP = gE*isP
g_pow, g_pow_z = gP@LEE, gP@G+tech['dg']*isP[tnode]
g_ep, g_ep_z = eE@LEE, eE@G
r_co2, r_co2_z = d['r_co2'][iE], d['r_co2']@tech['D']       # RoW CO2 response (leakage)
dom = d['dom']
W0 = Mw@y0+Mw_c+dom
gdp_c = (agg*vE)@(LEE@y0+cst)
if PEN_MODE == 1:
    _xb = LEE@y0+cst
    pen = ((agg*(vE*food_nodes))@_xb)/np.maximum((agg*(oE*food_nodes))@_xb, 1e-9)   # agricultural VA per m3
else:
    pen = gdp_c/W0                                           # bn EUR per Mm3
pen = pen*PEN_SCALE
EP0, Emax = g_ep@y0+eE@cst, ECO_MAX*W0

mg = pd.read_csv(OUT/'marginals_flow_levels.csv', index_col=0).loc[EU]
mpar = json.load(open(OUT/'marginal_params.json'))
Rbar = mg.mean_obs.to_numpy()
flow = mg[['L', 'M', 'H']].to_numpy().T
WEI = np.array([W0[i]/Rbar[i] if c not in PHI_IMPUTED else np.nan for i, c in enumerate(EU)])
phi_c = PHI*np.array([PHI_IMPUTED[c] if c in PHI_IMPUTED else min(1, WEI[i]/0.4) for i, c in enumerate(EU)])
head = np.array([0 if c in PHI_IMPUTED else DELTA*max(0, 1-WEI[i]/0.4) for i, c in enumerate(EU)])
Cap = W0*(1+head)
def supply(Rv, alt, r=slice(None)):
    return Cap[r]*(1-phi_c[r]+phi_c[r]*Rv/Rbar[r])+alt

# ---- basin x quarter layer (basins_eu.py): country water by EEA sector group is
# spread over basin-country fragments and quarters with observed 2002-2014 shares
GRP = ['AgricForestFish', 'ElecGasSteam', 'MiningConstruct', 'Service', 'WaterCollectTreat']
NG, NQ = len(GRP), 4
grp = np.full(nE, 2)                                          # mining, manufacturing, construction
grp[sec <= 18] = 0
grp[(sec >= 95) & (sec <= 110)] = 1
grp[((sec >= 114) & (sec <= 137)) | (sec >= 158)] = 3
grp[(sec == 111) | ((sec >= 138) & (sec <= 157))] = 4         # water supply, sewerage, waste
AG = np.zeros((C*NG, nE)); AG[cE*NG+grp, np.arange(nE)] = 1
onehotG = np.zeros((C*NG, K)); onehotG[cE[tnode]*NG+grp[tnode], np.arange(K)] = 1
MwG, MwG_z, MwG_c = (AG*oE)@LEE, (AG*oE)@G+onehotG*tech['dw'], (AG*oE)@cst
MwG_zc = (AG*oE)@G+onehotG*dw_c
MwG_c[np.arange(C)*NG+4] += dom                               # households in public water supply
W0G = MwG@y0+MwG_c
fq = pd.read_csv(HERE/'results/basins/fragment_quarter_coefficients.csv')
fq = fq[fq.country.isin(EU)].sort_values(['frag', 'Quarter']).reset_index(drop=True)
frags = fq.frag.unique().tolist()
NF = len(frags)
fc = np.array([EU.index(f.split('|')[1]) for f in frags])
acoef = fq[GRP].to_numpy().reshape(NF, NQ, NG)
tot_use = fq[['U_'+g for g in GRP]].to_numpy().sum(1).reshape(NF, NQ)   # raw EEA use, for missing groups
FALLBACK = []
for c in set(fc):                                             # groups absent in EEA data: use total-use shares
    m = fc == c
    for g in range(NG):
        if acoef[m, :, g].sum() < 1e-9:
            acoef[m, :, g] = tot_use[m]/tot_use[m].sum()
            FALLBACK.append((EU[c], GRP[g]))
basin_c = np.zeros(C, bool); basin_c[fc] = True
hyd = fq[['WEI', 'rL', 'rM', 'rH']].to_numpy().reshape(NF, NQ, 4)
WEIb, rlev = np.nan_to_num(hyd[..., 0]), np.nan_to_num(hyd[..., 1:], nan=1.0)
phi_b = PHI*np.minimum(1, WEIb/0.4)
head_b = DELTA*np.maximum(0, 1-WEIb/0.4)
base_fq = np.einsum('fqg,fg->fq', acoef, W0G.reshape(C, NG)[fc])
share_fq = base_fq/np.bincount(fc, base_fq.sum(1), minlength=C)[fc][:, None]
# e-flow limit: with a constant consumptive fraction per basin-quarter, WEI+ <= THETA becomes
# withdrawal <= W2017 * THETA / WEI+_h, where WEI+_h = WEI+ / (R_h / mean R); capacity <= base*(1+INFRA)
o17 = d17['omega'][iE]
W17G = (AG*o17)@(LEE@d17['y'][iE]+d17['const_E'])
W17G[np.arange(C)*NG+4] += d17['dom']
base17_fq = np.einsum('fqg,fg->fq', acoef, W17G.reshape(C, NG)[fc])
with np.errstate(divide='ignore', invalid='ignore'):
    eco_lim = np.where(WEIb[..., None] > 0, base17_fq[..., None]*THETA*rlev/WEIb[..., None], np.inf)
cap_e = np.minimum(eco_lim, (base_fq*(1+INFRA))[..., None])            # NF x NQ x H
W17 = (agg*o17)@(LEE@d17['y'][iE]+d17['const_E'])+d17['dom']
coastal = pd.read_csv(HERE/'data/wei_subbasin_outlets.csv').set_index('SB').OUTLET.eq('#SEA#')
# national annual WEI+ = EEA consumption summed over the country's fragments (additive) / long-term
# renewable resources from Eurostat; renewable water is NOT summed over sub-basins (downstream R contains
# upstream inflow). Greece and Denmark (no Eurostat series): R summed over sea-outlet fragments only.
_U = fq[['U_'+g for g in GRP]].to_numpy().sum(1); _W = fq['WEI'].to_numpy()
_R = np.divide(_U, _W, out=np.zeros_like(_U), where=_W > 0)
_fc_rows = np.array([EU.index(c) for c in fq.country])
_sea = np.array([coastal.get(s, False) for s in fq.SB])
U_nat = np.bincount(_fc_rows, _U, minlength=C)
# long-term renewable resources incl. inflows (Eurostat RFW_RES) where >= 10 years, else the series used
# for the flow levels (internal flow); this matters for Portugal (RFW 26.2 vs internal flow 9.5 bn m3)
_rfw = pd.read_csv(HERE/'data/env_wat_res_long.csv').query("wat_proc == 'RFW_RES' and value > 0")
_rfw = _rfw.assign(geo=_rfw.geo.replace({'EL': 'GR'})).groupby('geo').value.agg(['mean', 'count'])
R_nat = np.array([np.bincount(_fc_rows, _R*_sea, minlength=C)[i] if c in PHI_IMPUTED else
                  (_rfw.loc[c, 'mean'] if c in _rfw.index and _rfw.loc[c, 'count'] >= 10 else Rbar[i])
                  for i, c in enumerate(EU)])
WEI_nat = np.where(basin_c, np.divide(U_nat, R_nat, out=np.zeros(C), where=R_nat > 0), W17/Rbar)
if NATWEI_EEA:
    # EEA WEI+ at country scale (release of January 2026): annual net consumption / annual renewable resources, mean 2002-2014
    _cn = pd.read_excel(HERE/'data/eea_wei_2000_2023/WEI_2000_2023_SDI_CountryScale.xlsx', sheet_name='seasonal wei country scale')
    _cn = _cn[_cn.phenomenonTimePeriod_year.between(2002, 2014)].assign(c=_cn.countryCode.replace({'EL': 'GR'}))
    _a = _cn.groupby(['c', 'phenomenonTimePeriod_year'])[['netWaterConsumption', 'renewableWaterResources']].sum()
    _wei_eea = (_a.netWaterConsumption/_a.renewableWaterResources).groupby('c').mean()
    for i, c in enumerate(EU):
        if not basin_c[i]:
            WEI_nat[i] = _wei_eea[c]
# alternative water by geography: desalination only reaches coastal sub-basins (EEA OUTLET = #SEA#),
# reuse follows urban and industrial use (groups 2-4); both supplied evenly over the four quarters
is_coast = np.array([coastal.get(f.split('|')[0], False) for f in frags])
def _spread(annual):
    # country total split over fragments by `annual`, a quarter of it in each quarter
    tot = np.bincount(fc, annual, minlength=C)[fc]
    return np.repeat((np.divide(annual, tot, out=np.zeros(NF), where=tot > 0)/NQ)[:, None], NQ, 1)
nonag_fq = np.einsum('fqg,fg->fq', acoef[..., 2:], W0G.reshape(C, NG)[fc][:, 2:])
w_desal = _spread(np.where(is_coast, base_fq.sum(1), 0.0))
w_reuse = _spread(nonag_fq.sum(1))

# ---- national-quarterly rule (always built, so that any plan can be evaluated with it): national water by sector group
# split over quarters with the observed shares of the basin layer; consumption summed over the country's fragments;
# Eurostat renewable resources (as in the national-annual rule) split over quarters with the seasonal pattern of the
# sea-outlet sub-basins (all sub-basins in landlocked countries), so that the quarterly WEI+ average back to the annual
# one; dry/normal/wet levels from a lognormal fit of that national quarterly renewable water over 2002-2014
A_cq = np.zeros((C, NQ, NG)); np.add.at(A_cq, fc, acoef)                      # share of national group water in quarter q
U_cq = np.zeros((C, NQ)); np.add.at(U_cq, fc, tot_use)
_d = pd.read_csv(HERE/'data/wei_subbasin_2002_2014.csv')
_d = _d[_d.groupby('SB').Year_.transform('size') == 52]
_d = _d.assign(R=np.where(_d.WEI_2 > 0, _d[GRP].sum(1)/(_d.WEI_2/100), np.nan))
_Rsy = _d.pivot_table(index=['SB', 'Quarter'], columns='Year_', values='R')
_fsb = np.array([f.split('|')[0] for f in frags]); _fsea = np.array([coastal.get(x, False) for x in _fsb])
_farea = fq.area_share.to_numpy().reshape(NF, NQ)[:, 0]
Rshare_cq, r_cq = np.zeros((C, NQ)), np.ones((C, NQ, H))
for i in range(C):
    m = fc == i
    if not m.any():
        continue
    sel = np.where(m & _fsea)[0] if (m & _fsea).any() else np.where(m)[0]
    ser = [np.nansum([_Rsy.loc[(_fsb[f], f'Q{q+1}')].to_numpy()*_farea[f] for f in sel if (_fsb[f], f'Q{q+1}') in _Rsy.index], axis=0)
           for q in range(NQ)]
    Rshare_cq[i] = np.array([x.mean() for x in ser])/sum(x.mean() for x in ser)
    for q, x in enumerate(ser):
        if len(x) >= 8 and x.min() > 0 and x.std() > 0:
            sh_, _, sc_ = stats.lognorm.fit(x, floc=0)
            r_cq[i, q] = [stats.lognorm.ppf(np.linspace(a, b, 402)[1:-1], sh_, 0, sc_).mean()/x.mean() for a, b in ((0, .2), (.2, .8), (.8, 1))]
WEI_cq = np.divide(U_cq, R_nat[:, None]*Rshare_cq, out=np.zeros((C, NQ)), where=Rshare_cq > 0)
W17_cq = np.einsum('cqg,cg->cq', A_cq, W17G.reshape(C, NG)); W30_cq = np.einsum('cqg,cg->cq', A_cq, W0G.reshape(C, NG))
with np.errstate(divide='ignore', invalid='ignore'):
    cap_nq = np.minimum(np.where(WEI_cq[..., None] > 0, W17_cq[..., None]*THETA*r_cq/WEI_cq[..., None], np.inf),
                        (W30_cq*(1+INFRA))[..., None])                       # C x NQ x H, without alternative water
Ay_cq = np.einsum('cqg,cgn->cqn', A_cq, MwG.reshape(C, NG, nE))             # national quarterly withdrawal = Ay y + Az z + Ac
Az_cq = np.einsum('cqg,cgk->cqk', A_cq, MwG_zc.reshape(C, NG, K)); Ac_cq = np.einsum('cqg,cg->cq', A_cq, MwG_c.reshape(C, NG))

# ---- land indicators (reported, not constrained: 2030 wind/solar substitution needs <0.1% of EU farmland)
# agricultural land (cropland + permanent pasture, EXIOBASE 2017) per unit output, rescaled so the reference-year
# baseline keeps the 2017 area (yield growth); land for wind/solar from replaced fossil MWh (EXIOBASE fossil-power
# output / Ref2020 fossil generation) x 1.46 ha/GWh for ground PV and 0.14 ha/GWh direct for wind (NREL-type values)
_land = pd.read_csv(EXIO/'land/F.txt', sep='\t', header=[0, 1], index_col=0)
_agri17 = _land.loc[_land.index.str.startswith(('Cropland', 'Permanent pastures'))].to_numpy().sum(0)[iE]  # km2
_xref = LEE@y0+cst
lam_agri = np.divide(_agri17, _xref, out=np.zeros(nE), where=_xref > 1e-9)
_ref = pd.read_parquet(REF2020)
def _gen(item, yr):
    return _ref[(_ref.block == 'gen') & (_ref.item == item) & (_ref.year == yr)].set_index('country').value.reindex(EU).fillna(0)
_fos = lambda yr: _gen('Solid fossil fuels', yr)+_gen('Natural and manufactured gases', yr)+_gen('Petroleum products', yr)
_fos17 = (_fos(2015)+(_fos(2025)-_fos(2015))*0.2).to_numpy()                              # GWh
_fosout17 = d17['x'][iE].reshape(C, S)[:, [95, 96, 100]].sum(1)                            # bn EUR
_gwh_per_bn = np.divide(_fos17, _fosout17, out=np.zeros(C), where=_fosout17 > 0)
_sol = (_gen('Solar', 2030)-_gen('Solar', 2015)).to_numpy(); _wnd = (_gen('Wind power', 2030)-_gen('Wind power', 2015)).to_numpy()
_spv = np.clip(np.divide(_sol, _sol+_wnd, out=np.full(C, 0.5), where=(_sol+_wnd) > 0), 0, 1)
ren_km2 = np.where(tech['name'] == 'REN', (_gwh_per_bn*(_spv*1.46+(1-_spv)*0.14)/100)[cE[tnode]], 0.0)   # km2 per bn EUR z

lo = np.minimum(y0*(1-DELTA), y0*(1+DELTA)); hi = np.maximum(y0*(1-DELTA), y0*(1+DELTA))
lo[mE] = hi[mE] = y0[mE]

NW, NSB = (C*NG, NF*NQ*H) if BASIN else (0, 0)
NNQ = C*NQ*H if (NATQ and not BASIN) else 0
NV = nE+K+C+H*C+NW+NSB+NNQ+1
iy, iz, iEc, iS = 0, nE, nE+K, nE+K+C
iW, iSB = iS+H*C, iS+H*C+NW
iNQ, iL = iSB+NSB, iSB+NSB+NNQ
subnat = ~basin_c if (BASIN or NNQ) else np.ones(C, bool)                   # countries under the national annual rule
pen_nq = (pen[:, None, None]*np.ones((C, NQ, H))*PR*basin_c[:, None, None]).ravel()
pen_b = np.repeat(pen[fc], NQ*H).reshape(NF, NQ, H)*PR      # expected penalty per basin shortage unit

def vec(y=None, E=None, s=None, lam=0., z=None, sb=None, nq=None):
    v = np.zeros(NV)
    if y is not None: v[iy:iy+nE] = y
    if z is not None: v[iz:iz+K] = z
    if E is not None: v[iEc:iEc+C] = E
    if s is not None: v[iS:iS+H*C] = s
    if sb is not None: v[iSB:iSB+NSB] = sb
    if nq is not None: v[iNQ:iNQ+NNQ] = nq
    v[iL] = lam
    return v

C_UP = vec(y=g_gdp, z=g_gdp_z, s=-np.kron(PR, pen), sb=-pen_b.ravel() if BASIN else None, nq=-pen_nq if NNQ else None)
C_FOOD = vec(y=g_food, z=g_food_z)
C_CO2 = vec(y=g_co2, z=g_co2_z)
if EFLOW:
    # ecology follower: minimise expected e-flow exceedance and eutrophication (each relative to the baseline);
    # the extra reserve E is not used because THETA already reserves (1-THETA) of renewable water for ecology
    _nat = subnat
    _cap_n0 = np.array([np.minimum(W17*THETA*(flow[h]/Rbar)/WEI_nat, W0*(1+INFRA)) for h in range(H)])
    _PRn = np.array([.2, .6, .2]) if FIXNORM == 2 else PR     # FIXNORM=2: baseline exceedance under the historical regime
    D0 = float((_PRn[:, None]*np.maximum(W0-_cap_n0, 0)*_nat).sum())
    if BASIN:
        D0 += float((_PRn*np.maximum(base_fq[..., None]-cap_e, 0)).sum())
    if NNQ:
        D0 += float((_PRn*np.maximum(W30_cq[..., None]-cap_nq, 0)*basin_c[:, None, None]).sum())
    C_ECO = vec(y=-g_ep/EP0, z=-g_ep_z/EP0, s=-np.kron(PR, _nat.astype(float))/D0,
                sb=-np.tile(PR, NF*NQ)/D0 if BASIN else None,
                nq=-(np.ones((C, NQ, H))*PR*basin_c[:, None, None]).ravel()/D0 if NNQ else None)
else:
    C_ECO = vec(y=-g_ep/EP0, z=-g_ep_z/EP0, E=np.ones(C)/Emax.sum())
OBJ = {'U': (C_UP, 1), 'L1': (C_FOOD, 1), 'L2': (C_CO2, -1), 'L3': (C_ECO, 1)}

def base_constraints(sc):
    alt = sc['alt']
    rows, rhs = [], []
    nat = subnat                                                 # national annual rule where no basin layer
    for h in range(H):
        A = np.zeros((C, NV)); A[:, iy:iy+nE] = Mw; A[:, iz:iz+K] = Mw_zc; A[:, iEc:iEc+C] = np.eye(C)
        A[:, iS+h*C:iS+(h+1)*C] = -np.eye(C)
        if EFLOW:                                                # national annual e-flow rule
            cap_n = np.minimum(W17*THETA*(flow[h]/Rbar)/WEI_nat, W0*(1+INFRA))+alt
        else:
            cap_n = supply(flow[h], alt)
        rows.append(A[nat]); rhs.append((cap_n-dom-Mw_c)[nat])
    if BASIN:
        # W[c,g] = country water by EEA sector group (equality as two inequalities)
        A = np.zeros((NW, NV)); A[:, iy:iy+nE] = -MwG; A[:, iz:iz+K] = -MwG_zc; A[:, iW:iW+NW] = np.eye(NW)
        rows += [A, -A]; rhs += [MwG_c, -MwG_c]
        # fragment x quarter x scenario: sum_g a W[c,g] + share E_c - s <= cap
        alt_fq = sc['desal'][fc][:, None]*w_desal+sc['reuse'][fc][:, None]*w_reuse if EFLOW else alt[fc][:, None]*share_fq
        r_, c_, v_, b_ = [], [], [], []
        row = 0
        for f in range(NF):
            for q in range(NQ):
                for h in range(H):
                    for g in range(NG):
                        r_.append(row); c_.append(iW+fc[f]*NG+g); v_.append(acoef[f, q, g])
                    r_ += [row, row]; c_ += [iEc+fc[f], iSB+(f*NQ+q)*H+h]; v_ += [share_fq[f, q], -1.0]
                    b_.append((cap_e[f, q, h] if EFLOW else
                               base_fq[f, q]*(1+head_b[f, q])*(1-phi_b[f, q]+phi_b[f, q]*rlev[f, q, h]))+alt_fq[f, q])
                    row += 1
        rows.append(csr_matrix((v_, (r_, c_)), shape=(row, NV))); rhs.append(np.array(b_))
    if NNQ:
        # national x quarter x state: national quarterly withdrawal - s <= cap + a quarter of the alternative water
        for c in np.where(basin_c)[0]:
            for q in range(NQ):
                A = np.zeros((H, NV)); A[:, iy:iy+nE] = Ay_cq[c, q]; A[:, iz:iz+K] = Az_cq[c, q]
                A[np.arange(H), iNQ+(c*NQ+q)*H+np.arange(H)] = -1
                rows.append(A); rhs.append(cap_nq[c, q]+alt[c]/NQ-Ac_cq[c, q])
    for c, q in (sc['levels'].items() if CHANCE else []):
        r = EU.index(c)
        name, par = mpar[c]
        if EFLOW and NNQ:
            # national quarterly analogue of the summer basin rule: Q3, only where the national summer WEI+ <= THETA
            dist = getattr(stats, name)
            qr = dist.ppf(q, *par)/dist.mean(*par)
            if 0 < WEI_cq[r, 2] <= THETA:
                lim = W17_cq[r, 2]*THETA*qr/WEI_cq[r, 2]
                capc = min(max(lim, W17_cq[r, 2]) if CHANCE_ND else lim, W30_cq[r, 2]*(1+INFRA))+alt[r]/NQ
                A = np.zeros((1, NV)); A[0, iy:iy+nE] = Ay_cq[r, 2]; A[0, iz:iz+K] = Az_cq[r, 2]
                rows.append(A); rhs.append(np.array([capc-Ac_cq[r, 2]]))
            continue
        if EFLOW and not BASIN:
            # national annual analogue of the summer basin rule below
            dist = getattr(stats, name)
            qr = dist.ppf(q, *par)/dist.mean(*par)
            capc = min(max(W17[r]*THETA*qr/WEI_nat[r], W17[r]), W0[r]*(1+INFRA))+alt[r]
            A = np.zeros((1, NV)); A[0, iy:iy+nE] = Mw[r]; A[0, iz:iz+K] = Mw_zc[r]; A[0, iEc+r] = 1
            rows.append(A); rhs.append(np.array([capc-dom[r]-Mw_c[r]]))
            continue
        if EFLOW:
            # summer (Q3) allocation in every basin of country c must respect the e-flow limit at the
            # national drought quantile q (reliability 1-q); 2017 abstraction is grandfathered (WFD non-deterioration)
            dist = getattr(stats, name)
            qr = dist.ppf(q, *par)/dist.mean(*par)
            r_, c_, v_, b_ = [], [], [], []
            # only basins meeting the e-flow standard at mean hydrology (WEI+ <= THETA); over-abstracted
            # basins enter through the penalised deficit instead
            for k_, f in enumerate(np.where((fc == r) & (WEIb[:, 2] <= THETA))[0]):
                lim = base17_fq[f, 2]*THETA*qr/WEIb[f, 2] if WEIb[f, 2] > 0 else np.inf
                capc = min(max(lim, base17_fq[f, 2]) if CHANCE_ND else lim, base_fq[f, 2]*(1+INFRA))
                for g in range(NG):
                    r_.append(k_); c_.append(iW+r*NG+g); v_.append(acoef[f, 2, g])
                r_.append(k_); c_.append(iEc+r); v_.append(share_fq[f, 2])
                b_.append(capc+sc['desal'][r]*w_desal[f, 2]+sc['reuse'][r]*w_reuse[f, 2])
            rows.append(csr_matrix((v_, (r_, c_)), shape=(k_+1, NV))); rhs.append(np.array(b_))
            continue
        cap = supply(getattr(stats, name).ppf(q, *par), alt[r], r)
        A = np.zeros((1, NV)); A[0, iy:iy+nE] = Mw[r]; A[0, iz:iz+K] = Mw_zc[r]; A[0, iEc+r] = 1
        rows.append(A); rhs.append(np.array([cap-dom[r]-Mw_c[r]]))
    A = np.zeros((1, NV)); A[0, iy:iy+nE] = -g_food; A[0, iz:iz+K] = -g_food_z
    rows.append(A); rhs.append(np.array([-FOOD_FLOOR*(g_food@y0)]))
    if CO2CAP:
        A = np.zeros((1, NV)); A[0, iy:iy+nE] = g_co2; A[0, iz:iz+K] = g_co2_z
        rows.append(A); rhs.append(np.array([CO2CAP*(g_co2@y0+gE@cst)-gE@cst]))
    if CO2CAP_POWER:
        A = np.zeros((1, NV)); A[0, iy:iy+nE] = g_pow; A[0, iz:iz+K] = g_pow_z
        rows.append(A); rhs.append(np.array([CO2CAP_POWER*(g_pow@y0+gP@cst)-gP@cst]))
    if CAPBOUND:
        zb = [(0, c_*xr) for c_, xr in zip(tech['cap'], (LEE@y0+cst)[tnode])]
    else:
        A = np.zeros((K, NV))
        A[:, iy:iy+nE] = -tech['cap'][:, None]*LEE[tnode]
        A[:, iz:iz+K] = np.eye(K)-tech['cap'][:, None]*G[tnode]
        rows.append(A); rhs.append(tech['cap']*cst[tnode])
        zb = [(0, None)]*K
    bounds = [(lo[i], hi[i]) for i in range(nE)] + zb + [(0, 0 if EFLOW else e) for e in Emax] \
           + [(0, None)]*(H*C) + [(0, 0)]
    bounds = bounds[:iW] + [(0, None)]*(NW+NSB) + [(0, None) if basin_c[c] else (0, 0) for c in range(C) for _ in range(NQ*H) if NNQ] + [(0, 0)]
    dense = [csr_matrix(r) if isinstance(r, np.ndarray) else r for r in rows]
    return svstack(dense).tocsr(), np.concatenate(rhs), bounds

def solve(c, A, b, bounds):
    # HiGHS default first; on numerical failure retry with dual simplex, then interior point
    for method in ('highs', 'highs-ds', 'highs-ipm'):
        r = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method=method)
        if r.status == 0:
            return r.x
        if r.status == 2:                                        # proven infeasible: no point retrying
            return None
    return None

def tolerance_rows(idx, center, t, coupled=True):
    m = len(idx)
    T = csr_matrix((np.ones(m), (np.arange(m), idx)), shape=(m, NV))
    lam = csr_matrix((t if coupled else np.zeros(m), (np.arange(m), np.full(m, iL))), shape=(m, NV))
    return [T+lam, -T+lam], [center+t, -center+t]

def run(sc):
    A, b, bnd = base_constraints(sc)
    sols = {}
    for k, (c, sgn) in OBJ.items():
        z = solve(sgn*c, A, b, bnd)
        if z is None:
            return None
        sols[k] = z
    best = {k: c@sols[k] for k, (c, _) in OBJ.items()}
    worst = {k: (min if sgn > 0 else max)(c@sols[j] for j in sols) for k, (c, sgn) in OBJ.items()}
    if FIXNORM:
        f = OUT/'norm_endpoints'/f"{sc['scenario']}_{sc['transfer']}_{int(sc['unconventional'])}.json"
        if FIXNORM == 1:
            f.parent.mkdir(exist_ok=True); json.dump(dict(best=best, worst=worst), open(f, 'w'))
        else:
            e = json.load(open(f)); best, worst = e['best'], e['worst']
    rows, rhs = [A], [b]
    for k, (c, sgn) in OBJ.items():
        span = best[k]-worst[k]
        if abs(span) < 1e-9:
            continue
        r_ = -c/span; r_[iL] = 1
        rows.append(csr_matrix(r_[None])); rhs.append(np.array([-worst[k]/span]))
    t = TOL*np.abs(y0)
    li = np.where(lead_nodes & (np.abs(y0) > 0))[0] if LEADTOL else np.array([], int)  # TOL=0 fixes the leader at its optimum
    lz = np.where(tech['owner'] == 'U')[0] if LEADTOL else np.array([], int)
    tz, cy, cz = TOL*x0[tnode[lz]], sols['U'][iy+li], sols['U'][iz+lz]
    fref = OUT/'leader_ref'/f"{sc['scenario']}_{sc['transfer']}_{int(sc['unconventional'])}.npz"
    if LEADFIX:
        ref = np.load(fref)
        t, tz = (1-ref['lam'])*t, (1-ref['lam'])*tz
        if LEADFIX == 2:
            cy, cz = ref['yU'][li], ref['zU'][lz]
    if len(li):
        R_, b_ = tolerance_rows(iy+li, cy, t[li], not LEADFIX); rows += R_; rhs += b_
    if len(lz):
        R_, b_ = tolerance_rows(iz+lz, cz, tz, not LEADFIX); rows += R_; rhs += b_
    bnd = list(bnd); bnd[iL] = (None, None) if FIXNORM == 2 else (0, 1)     # fixed endpoints: memberships may leave [0, 1]
    A2, b2 = svstack(rows).tocsr(), np.concatenate(rhs)
    z = solve(vec(lam=1), A2, b2, bnd)
    if z is not None and FIXNORM == 1:
        fref.parent.mkdir(exist_ok=True)
        np.savez(fref, lam=z[iL], yU=sols['U'][iy:iy+nE], zU=sols['U'][iz:iz+K])
    if z is not None:
        # phase 2: hold the max-min lambda, maximise the sum of memberships (Pareto-efficient compromise)
        bnd[iL] = (z[iL]-1e-7, None) if FIXNORM == 2 else (max(0, z[iL]-1e-7), 1)
        c2 = sum(c/(best[k]-worst[k]) for k, (c, _) in OBJ.items() if abs(best[k]-worst[k]) > 1e-9)
        z2 = solve(c2, A2, b2, bnd)
        z = z2 if z2 is not None else z
    return dict(z=z, single=sols['U'])

def indicators(z):
    y, t, E, s = z[iy:iy+nE], z[iz:iz+K], z[iEc:iEc+C], z[iS:iS+H*C].reshape(H, C)
    x = LEE@y+G@t+cst
    pw = Mw@y+Mw_z@t+Mw_c
    xn = oE*x; np.add.at(xn, tnode, tech['dw']*t)
    J = [EU.index(c) for c in JOINT]
    adopt = {f'z_{k}': float(t[tech['name'] == k].sum()) for k in OPTIONS}
    saved = {f'saved_{k}': float(-(tech['dw']*t)[tech['name'] == k].sum()) for k in OPTIONS}
    bs = basin_stats(z)
    return dict(GDP=float(vE@x+tech['dv']@t),
                penalty=float(np.kron(PR, pen)@z[iS:iS+H*C])+(float(pen_b.ravel()@z[iSB:iSB+NSB]) if BASIN else 0.0)
                        +(float(pen_nq@z[iNQ:iNQ+NNQ]) if NNQ else 0.0),
                food=float(x[food_nodes].sum()), CO2=float(gE@x+tech['dg']@t),
                RoW_CO2_delta=float(r_co2@(y-y0)+r_co2_z@t), EP=float(eE@x), eco_reserve=E.sum(),
                agri_land_km2=float(lam_agri@x), ren_land_km2=float(ren_km2@t),
                power_CO2=float(gP@x+(tech['dg']*isP[tnode])@t),
                prod_water=pw.sum(), agri_water=float(xn[food_nodes].sum()),
                energy_water=float(xn[energy_nodes].sum()), **adopt, **saved,
                exp_shortage=float(PR@s.sum(1))+bs.get('basin_exp', 0.0), shortage_L=float(s[0].sum())+bs.get('basin_L', 0.0),
                **bs,
                MED_water=float((pw+dom+E)[J].sum()),
                shortage_rate_L=float(s[0].sum()/(pw.sum()+dom.sum()+E.sum())))

def basin_stats(z):
    if NNQ:
        sn = z[iNQ:iNQ+NNQ].reshape(C, NQ, H)
        return dict(basin_exp=float((sn*PR).sum()), basin_L=float(sn[..., 0].sum()),
                    Q3_short_L=float(sn[:, 2, 0].sum()), Q3_short_M=float(sn[:, 2, 1].sum()),
                    n_bind_Q3_M=int((sn[:, 2, 1] > 1e-6).sum()), n_bind_Q3_L=int((sn[:, 2, 0] > 1e-6).sum()))
    if not BASIN:
        return {}
    sb = z[iSB:iSB+NSB].reshape(NF, NQ, H)
    return dict(basin_exp=float((sb*PR).sum()), basin_L=float(sb[..., 0].sum()),
                Q3_short_L=float(sb[:, 2, 0].sum()), Q3_short_M=float(sb[:, 2, 1].sum()),
                n_bind_Q3_M=int((sb[:, 2, 1] > 1e-6).sum()), n_bind_Q3_L=int((sb[:, 2, 0] > 1e-6).sum()))

WATER_OPTIONS = {'none': (0, False), 'desal': (3000, False), 'reuse': (0, True), 'both': (3000, True)}

def scenarios():
    nonagri = (agg*oE*(~food_nodes))@(LEE@y0+cst)
    if DESIGN == 2:
        js = pd.read_csv(OUT/'joint_violation_symmetric.csv')
        for (_, row), (wo, (tv, uc)) in itertools.product(js.iterrows(), WATER_OPTIONS.items()):
            desal = np.zeros(C)
            for c, w in DESAL_SHARE.items():
                desal[EU.index(c)] += tv*w/sum(DESAL_SHARE.values())
            reuse = REUSE_RATE*(nonagri+dom) if uc else np.zeros(C)
            yield dict(scenario=row.scenario, p=row.p, transfer=wo, unconventional=uc, alt=desal+reuse,
                       desal=desal, reuse=reuse, levels={c: row[f'p_{c}'] for c in JOINT})
        return
    js = pd.read_csv(OUT/'joint_violation_scenarios.csv')
    for (_, row), (tl, tv), uc in itertools.product(js.iterrows(), DESAL_LEVELS.items(), [True, False]):
        desal, reuse = np.zeros(C), np.zeros(C)
        for c, w in DESAL_SHARE.items():
            desal[EU.index(c)] += tv*w/sum(DESAL_SHARE.values())
        if uc:
            reuse = REUSE_RATE*(nonagri+dom)
        yield dict(scenario=row.scenario, p=row.p, transfer=tl, unconventional=uc, alt=desal+reuse,
                   desal=desal, reuse=reuse, levels={c: row[f'p_{c}'] for c in JOINT})

def one(sc):
    key = dict(scenario=sc['scenario'], p=sc['p'], transfer=sc['transfer'], unconventional=sc['unconventional'])
    r = run(sc)
    if r is None or r['z'] is None:
        return [{**key, 'feasible': False}], []
    z = r['z']
    rows = [{**key, 'feasible': True, 'lambda': z[iL], 'model': 'SGBDSP', **indicators(z)},
            {**key, 'feasible': True, 'lambda': np.nan, 'model': 'single', **indicators(r['single'])}]
    y, t, E = z[iy:iy+nE], z[iz:iz+K], z[iEc:iEc+C]
    s = z[iS:iS+H*C].reshape(H, C)
    x = LEE@y+G@t+cst
    prov_rows = [{**key, 'province': c, 'alloc': (Mw@y+Mw_z@t+Mw_c+dom+E)[i], 'eco': E[i],
                  'shortage_L': s[0, i], 'exp_shortage': PR@s[:, i],
                  'gdp': float((agg*vE)[i]@x+(onehot*tech['dv'])[i]@t),
                  'tech_saved': float(-(onehot*tech['dw'])[i]@t)} for i, c in enumerate(EU)]
    if BASIN:
        sb = z[iSB:iSB+NSB].reshape(NF, NQ, H)
        prov_rows += [{**key, 'province': 'frag:'+frags[f], 'quarter': q+1, 'base': base_fq[f, q],
                       'shortage_L': sb[f, q, 0], 'exp_shortage': PR@sb[f, q]}
                      for f in range(NF) for q in range(NQ) if sb[f, q].max() > 1e-6]
    if NNQ:
        sn = z[iNQ:iNQ+NNQ].reshape(C, NQ, H)
        prov_rows += [{**key, 'province': 'natq:'+EU[c], 'quarter': q+1, 'base': W30_cq[c, q],
                       'shortage_L': sn[c, q, 0], 'exp_shortage': PR@sn[c, q]}
                      for c in range(C) for q in range(NQ) if sn[c, q].max() > 1e-6]
    return rows, prov_rows

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    tag = '_'.join(a.replace('=', '') for a in sys.argv[1:] if not a.startswith('JOBS')) or 'central'
    base = indicators(vec(y=y0, s=np.zeros(H*C)))
    json.dump(dict(base, WEI=dict(zip(EU, np.round(WEI, 3).tolist())), phi=dict(zip(EU, phi_c.round(3).tolist()))),
              open(OUT/'baseline_indicators.json', 'w'), indent=1, default=float)
    rows, prov_rows = [], []
    with ProcessPoolExecutor(int(JOBS)) as ex:
        for r_, p_ in ex.map(one, scenarios()):
            rows += r_; prov_rows += p_
            print(r_[0]['scenario'], r_[0]['transfer'], r_[0]['unconventional'], r_[0].get('lambda'), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT/f'scenario_results_{tag}.csv', index=False)
    pd.DataFrame(prov_rows).to_csv(OUT/f'province_results_{tag}.csv', index=False)
    print('infeasible:', (~df.feasible).sum(), 'of', len(df.drop_duplicates(['scenario', 'transfer', 'unconventional'])))
