"""Compare the 2002-2014 EEA sub-basin data used in the model with the EEA WEI+ release of January 2026 (2000-2023,
country and sub-unit scale; data/eea_wei_2000_2023/), and check Cyprus and Malta.

1. Country scale, 2002-2014: net consumption from the model's fragments (area-weighted EEA sector use) against the new
   country totals, annual and summer (Q3).
2. New data, 2002-2014 against 2015-2023: country Q3 WEI+ and the share of sub-units with Q3 WEI+ above 0.2 and 0.4.
3. Cyprus and Malta (national rule in the model): the model's national WEI (EXIOBASE withdrawal / Eurostat renewable
   resources) against the EEA WEI+ and Eurostat abstraction, and the expected exceedance of the central plans recomputed
   ex post with EEA-consistent inputs; effect on the headline scale numbers.
Writes results/eea2023_country.csv, results/eea2023_subunits.csv and results/cyprus_malta_check.csv.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES, NEW = HERE/'results', HERE/'data/eea_wei_2000_2023'
EU = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE',
      'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
GRP = ['AgricForestFish', 'ElecGasSteam', 'MiningConstruct', 'Service', 'WaterCollectTreat']
P1, P2 = range(2002, 2015), range(2015, 2024)

cn = pd.read_excel(NEW/'WEI_2000_2023_SDI_CountryScale.xlsx', sheet_name='seasonal wei country scale')
su = pd.read_excel(NEW/'WEI_2000_2023_SDI_SubunitScale.xlsx', sheet_name='Seasonal wei_subunit_scale')
for d in (cn, su):
    d['c'] = d.countryCode.replace({'EL': 'GR'})
    d['year'], d['q'] = d.phenomenonTimePeriod_year, d.phenomenonTimePeriod_quarter
su = su.rename(columns={'WEI+ (%)': 'wei'}); cn = cn.rename(columns={'WEI+': 'wei'})

# ---- 1. country consumption, old fragments vs new country data (2002-2014)
fq = pd.read_csv(RES/'basins/fragment_quarter_coefficients.csv')
old = fq.assign(U=fq[['U_'+g for g in GRP]].sum(1)).groupby(['country', 'Quarter']).U.sum().unstack()
new = cn[cn.year.isin(P1)].groupby(['c', 'q', 'year']).netWaterConsumption.sum().groupby(['c', 'q']).mean().unstack()
rows = []
for c in EU:
    r = dict(country=c)
    if c in old.index:
        r['old_U_annual'], r['old_U_Q3'] = old.loc[c].sum(), old.loc[c, 'Q3']
    if c in new.index:
        r['new_U_annual'], r['new_U_Q3'] = new.loc[c].sum(), new.loc[c, 'Q3']
    g = cn[cn.c == c]
    def ann(yrs):                                                      # annual consumption / annual renewable resources
        a = g[g.year.isin(yrs)].groupby('year')[['netWaterConsumption', 'renewableWaterResources']].sum()
        return float((a.netWaterConsumption/a.renewableWaterResources).mean()) if len(a) else np.nan
    q3 = lambda yrs: g[g.year.isin(yrs) & (g.q == 'Q3')].wei.mean()/100
    if len(g):
        r |= dict(new_WEI_annual_0214=ann(P1), new_WEI_annual_1523=ann(P2), new_WEI_Q3_0214=q3(P1), new_WEI_Q3_1523=q3(P2))
    rows.append(r)
ct = pd.DataFrame(rows)
ct['ratio_Q3_new_old'] = ct.new_U_Q3/ct.old_U_Q3
ct.to_csv(RES/'eea2023_country.csv', index=False)

# ---- 2. sub-units: share with Q3 WEI+ above 0.2 / 0.4 (period mean per unit), 2002-2014 vs 2015-2023
def shares(yrs):
    m = su[su.year.isin(yrs) & (su.q == 'Q3')].groupby(['c', 'spatialUnitIdentifier']).wei.mean().reset_index()
    return m.groupby('c').wei.agg(n='size', above20=lambda w: (w > 20).mean(), above40=lambda w: (w > 40).mean())
s1, s2 = shares(P1), shares(P2)
oldsb = pd.read_csv(RES/'basins/basin_quarter_hydrology.csv').query("Quarter == 'Q3'")
oldsb = oldsb.merge(fq[['SB', 'country']].drop_duplicates(), on='SB')
s0 = oldsb.groupby('country').WEI.agg(n='size', above20=lambda w: (w > .2).mean(), above40=lambda w: (w > .4).mean())
sub = pd.concat({'old_subbasins_0214': s0, 'new_subunits_0214': s1, 'new_subunits_1523': s2}, axis=1).loc[lambda x: x.index.isin(EU)]
sub.to_csv(RES/'eea2023_subunits.csv')

# ---- 3. Cyprus and Malta
sys.argv = ['x', 'YEAR=2030', 'EFLOW=1', 'DELTA=0.05', 'CAPBOUND=1', 'DESIGN=2', 'CO2CAP=1', 'BASIN=1', 'CO2CAP_POWER=0', 'PRDRY=0']
import model_eu as M
va = pd.read_csv(RES/'validation_water_exio_vs_eurostat.csv').set_index('c')
prov = pd.read_csv(RES/'province_results_YEAR2030_EFLOW1_DELTA0.05_CAPBOUND1_DESIGN2_CO2CAP1_BASIN1_CO2CAP_POWER0_PRDRY0.csv')
nat = pd.read_csv(RES/'province_results_YEAR2030_EFLOW1_DELTA0.05_CAPBOUND1_DESIGN2_CO2CAP1_BASIN0_CO2CAP_POWER0_PRDRY0.csv')
scs = {(s['scenario'], s['transfer'], s['unconventional']): s for s in M.scenarios()}
crow = []
for c in ('CY', 'MT'):
    i = M.EU.index(c)
    wei_eea = ct.set_index('country').loc[c, 'new_WEI_annual_0214']
    scale = va.loc[c, 'euro_tot']/va.loc[c, 'exio_tot']                     # Eurostat abstraction / EXIOBASE withdrawal
    variants = {'model (EXIOBASE withdrawal / Eurostat resources)': (M.WEI_nat[i], 1.0),
                'EEA WEI+ (consumption-based)': (wei_eea, 1.0),
                'withdrawal scaled to Eurostat': (M.WEI_nat[i]*scale, scale),
                'EEA WEI+ and withdrawal scaled to Eurostat': (wei_eea, scale)}
    for name, (wei, k) in variants.items():
        for lab, df in (('basin plan', prov), ('national plan', nat)):
            p = df[df.province == c]
            ex = []
            for _, r in p.iterrows():
                sc = scs[(r.scenario, r.transfer, r.unconventional)]
                cap = np.array([min(k*M.W17[i]*M.THETA*(M.flow[h, i]/M.Rbar[i])/wei, k*M.W0[i]*(1+M.INFRA)) for h in range(M.H)])+sc['alt'][i]
                ex.append(float(M.PR@np.maximum(k*r.alloc-cap, 0)))
            crow.append(dict(country=c, variant=name, plan=lab, WEI_used=wei, withdrawal_scale=k, exp_exceedance_Mm3=np.mean(ex),
                             reported_Mm3=p.exp_shortage.mean()))
cm = pd.DataFrame(crow); cm.to_csv(RES/'cyprus_malta_check.csv', index=False)

pd.set_option('display.width', 220, 'display.max_columns', 30)
focus = ['ES', 'PT', 'IT', 'GR', 'FR', 'CY', 'MT', 'NL', 'DE']
print('--- 1/2. country scale (consumption Mm3; WEI as fraction)')
print(ct.set_index('country').loc[focus].round(3).to_string())
print('EU (countries in both): Q3 consumption new/old =', round(ct.new_U_Q3.sum()/ct.dropna(subset=['old_U_Q3', 'new_U_Q3']).old_U_Q3.sum(), 3))
print('--- sub-units / sub-basins with Q3 WEI+ above 0.2 / 0.4 (period mean)')
print(sub.loc[[c for c in focus if c in sub.index]].round(3).to_string())
print('--- Cyprus and Malta: expected exceedance (Mm3), central Ref plans')
print(cm.round(3).to_string(index=False))
