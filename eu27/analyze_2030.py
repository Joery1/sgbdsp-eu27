"""2030 EU27 experiment: consistency checks, tables and English figures (PNG 300 dpi + PDF).

Runs (run_2030.sh), each with 12 scenarios = 3 joint-reliability levels (P05/P10/P20) x 4 water options:
  scale  BASIN=1 (sub-basin x quarter) / BASIN=0 (national annual), same e-flow rule and EEA hydrology
  policy CO2CAP_POWER=0 (Ref2020 cap) / 0.8138 (Fit-for-55 tightening of power CO2)
  regime PRDRY=0 (historical 0.2/0.6/0.2) / 1 (drier 0.4/0.5/0.1)
  plus no-technology, technology effect without reliability rows (CHANCE=0) and sensitivities.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

HERE = Path(__file__).resolve().parent
RES, DATA = HERE/'results', HERE/'data'
OUT, FIG = RES/'y2030', HERE/'figures_2030'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
PRE = 'YEAR2030_EFLOW1_DELTA0.05_CAPBOUND1_DESIGN2_CO2CAP1_'
KEY = ['scenario', 'transfer']
WATER = ['none', 'desal', 'reuse', 'both']
WLAB = {'none': 'None', 'desal': 'Desalination', 'reuse': 'Reuse', 'both': 'Both'}
PLAB = {'P05': 'p = 0.05', 'P10': 'p = 0.10', 'P20': 'p = 0.20'}
TECH = {'IRR': 'Efficient irrigation', 'AIR': 'Dry cooling', 'REN': 'Wind/solar for fossil power', 'REC': 'Industrial recycling'}

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
RED = '#e34948'
BLUES = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
         '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('seq', BLUES)
plt.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'axes.titleweight': 'bold',
    'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': GRID, 'grid.linewidth': 0.5, 'legend.frameon': False, 'legend.fontsize': 7.5,
    'axes.titlelocation': 'left', 'axes.titlepad': 8})

def panel(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')

def save(fig, name):
    fig.savefig(FIG/f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)

def hgrid(ax):
    ax.yaxis.grid(True); ax.set_axisbelow(True)

# ---------------------------------------------------------------- load
def load(suffix):
    f = RES/f'scenario_results_{PRE}{suffix}.csv'
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if 'model' not in d:
        d['model'] = None
    return d

def tag(scale=1, pol='0', dry=0, extra=''):
    return f'BASIN{scale}_CO2CAP_POWER{pol}_PRDRY{dry}' + (f'_{extra}' if extra else '')

def bi(d):
    return d[(d.model == 'SGBDSP') & d.feasible]

def sg(d):
    return d[(d.model == 'single') & d.feasible]

MAIN = {(s, p, r): load(tag(s, p, r)) for s in [1, 0] for p in ['0', '0.8138'] for r in [0, 1]}
central = MAIN[(1, '0', 0)]
base = json.load(open(RES/'baseline_indicators.json'))

# ---------------------------------------------------------------- checks
checks = []
for (s, p, r), d in MAIN.items():
    b = bi(d)
    checks.append(dict(run=tag(s, p, r), feasible=len(b), of=12, lambda_min=b['lambda'].min(), lambda_max=b['lambda'].max()))
    for sc, g in b.groupby('scenario'):
        e = g.set_index('transfer').exp_shortage
        ok = (e['none'] >= e['desal']-1e-6) and (e['none'] >= e['reuse']-1e-6) and (min(e['desal'], e['reuse']) >= e['both']-1e-6)
        checks.append(dict(run=tag(s, p, r), scenario=sc, monotone_alt_water=bool(ok)))
chk = pd.DataFrame(checks)
chk.to_csv(OUT/'checks.csv', index=False)
mono = chk.dropna(subset=['monotone_alt_water'])
print('feasibility:', chk.dropna(subset=['feasible'])[['run', 'feasible']].to_string(index=False))
print('alternative-water monotonicity holds in %d of %d (run x reliability) cells' % (mono.monotone_alt_water.sum(), len(mono)))

# ---------------------------------------------------------------- coupling coordination (paper eq. 5)
allb = []
for (s, p, r), d in MAIN.items():
    for model in ['SGBDSP', 'single']:
        x = d[(d.model == model) & d.feasible].copy()
        x['scale'], x['policy'], x['regime'] = ('Basin' if s else 'National'), ('FF55-power' if p != '0' else 'Ref'), ('Drier' if r else 'Historical')
        allb.append(x)
allb = pd.concat(allb, ignore_index=True)
allb['agri_wp'] = allb.food/allb.agri_water
SUB = {'F1_agri': {'food': 1, 'agri_wp': 1}, 'F2_energy': {'CO2': -1, 'energy_water': -1},
       'F3_eco': {'exp_shortage': -1, 'EP': -1}}
for f, ind in SUB.items():
    parts = []
    for col, sgn in ind.items():
        v = allb[col]; z = (v-v.min())/(v.max()-v.min()) if v.max() > v.min() else v*0+.5
        parts.append(z if sgn > 0 else 1-z)
    allb[f] = 0.01+0.99*np.mean(parts, axis=0)
F = allb[list(SUB)].to_numpy()
allb['C'] = np.prod(F, 1)**(1/3)/F.mean(1); allb['T'] = F.mean(1); allb['D'] = np.sqrt(allb.C*allb['T'])
allb.to_csv(OUT/'main_runs_with_CCD.csv', index=False)

# ---------------------------------------------------------------- tables
cols = ['lambda', 'GDP', 'food', 'CO2', 'power_CO2', 'exp_shortage', 'Q3_short_L', 'saved_IRR', 'saved_AIR',
        'saved_REN', 'saved_REC', 'agri_land_km2', 'ren_land_km2', 'RoW_CO2_delta', 'D']
t1 = allb.groupby(['scale', 'policy', 'regime', 'model'])[[c for c in cols if c in allb]].mean()
t1.to_csv(OUT/'T1_main_runs.csv')

# bi-level vs single (paired), basin / Ref / historical
b, s_ = bi(central), sg(central)
m = b.merge(s_, on=KEY, suffixes=('', '_s'))
pairs = {}
for c in ['GDP', 'food', 'CO2', 'power_CO2', 'energy_water', 'EP', 'exp_shortage', 'Q3_short_L', 'saved_IRR', 'saved_AIR', 'saved_REN']:
    pairs[c] = dict(bilevel=m[c].mean(), single=m[c+'_s'].mean(),
                    median_change_pct=100*((m[c]-m[c+'_s'])/m[c+'_s'].abs().where(lambda v: v > 1e-9)).median())
T3 = pd.DataFrame(pairs).T
T3.to_csv(OUT/'T3_bilevel_vs_single.csv')

# technology effect without the reliability rows
tech_runs = {pol: (load(tag(1, pol, 0, 'CHANCE0_TECH1')), load(tag(1, pol, 0, 'CHANCE0_TECH0'))) for pol in ['0', '0.8138']}
T4 = {}
for pol, (w, wo) in tech_runs.items():
    if w is None or wo is None:
        continue
    a, z = bi(w), bi(wo)
    lab = 'FF55-power' if pol != '0' else 'Ref'
    if z.empty:
        T4[lab] = dict(note='infeasible without technology options', feasible_without=0)
        continue
    mm = a.merge(z, on=KEY, suffixes=('', '_0'))
    T4[lab] = {c: 100*((mm[c]-mm[c+'_0'])/mm[c+'_0'].abs().where(lambda v: v > 1e-9)).mean()
               for c in ['GDP', 'food', 'CO2', 'power_CO2', 'prod_water', 'exp_shortage', 'Q3_short_L', 'lambda']}
    T4[lab]['feasible_without'] = len(z)
T4 = pd.DataFrame(T4).T
T4.to_csv(OUT/'T4_technology_effect.csv')

# sensitivities (bi-level means relative to the central run)
SENS = {'θ = 0.2': 'THETA0.2', 'θ = 0.6': 'THETA0.6', 'Wind/solar limit ×2': 'REN_CAP2', 'Irrigation cost ×2': 'IRR_COST2',
        'Irrigation limit ×0.5': 'IRR_CAP0.5', 'Capacity headroom 10%': 'INFRA0.1', 'Demand band ±10%': 'DELTA0.10',
        'Leader tolerance 6%': 'TOL0.06', 'Irrigation saving ×1.5': 'IRR_SAVE1.5', 'Wind/solar cost ×0.7': 'REN_COST0.7'}
cb = bi(central)
T5 = {}
for lab, ex in SENS.items():
    d = load(tag(1, '0', 0, ex))
    if d is None:
        continue
    x = bi(d)
    T5[lab] = {c: 100*(x[c].mean()/cb[c].mean()-1) for c in ['GDP', 'CO2', 'exp_shortage', 'saved_IRR', 'saved_REN', 'lambda']}
    T5[lab]['feasible'] = len(x)
T5 = pd.DataFrame(T5).T
T5.to_csv(OUT/'T5_sensitivity.csv')

# reliability x water options (basin / Ref / historical, bi-level)
T6 = cb.pivot_table(index='scenario', columns='transfer', values=['exp_shortage', 'saved_IRR', 'Q3_short_L'])
T6.to_csv(OUT/'T6_reliability_water.csv')

# deficits by country, national vs basin (bi-level mean over scenarios)
def country_deficit(suffix):
    pr = pd.read_csv(RES/f'province_results_{PRE}{suffix}.csv')
    p = pr.province.astype(str)
    nat = pr[~p.str.startswith(('frag:', 'natq:'))].groupby('province').exp_shortage.sum()
    fr = pr[p.str.startswith('frag:')]
    bas = fr.groupby(fr.province.str.split('|').str[1]).exp_shortage.sum() if len(fr) else pd.Series(dtype=float)
    nq = pr[p.str.startswith('natq:')]
    bas = bas.add(nq.groupby(nq.province.str.split(':').str[1]).exp_shortage.sum(), fill_value=0) if len(nq) else bas
    return nat.add(bas, fill_value=0)/12
T7 = pd.DataFrame({'national': country_deficit(tag(0, '0', 0)), 'national_quarterly': country_deficit(tag(0, '0', 0, 'NATQ1')),
                   'basin': country_deficit(tag(1, '0', 0))}).fillna(0)
T7 = T7[(T7 > 1).any(axis=1)].sort_values('basin', ascending=False)
T7.to_csv(OUT/'T7_country_deficits.csv')

# seasonal profile (basin, historical vs drier)
def seasonal(suffix):
    pr = pd.read_csv(RES/f'province_results_{PRE}{suffix}.csv')
    fr = pr[pr.province.astype(str).str.startswith('frag:')]
    return fr.groupby('quarter')[['exp_shortage', 'shortage_L']].sum()/12
T8 = pd.concat({'Historical': seasonal(tag(1, '0', 0)), 'Drier': seasonal(tag(1, '0', 1))}, axis=1)
T8.to_csv(OUT/'T8_seasonal.csv')

pd.set_option('display.width', 220)
for name, t in [('T1', t1), ('T3', T3), ('T4', T4), ('T5', T5), ('T6', T6), ('T7', T7), ('T8', T8)]:
    print('==', name); print(t.round(2).to_string())

# ---------------------------------------------------------------- figures
# Fig 1: maps
sb_rings = json.load(open(DATA/'wei_subbasin_rings.json'))
c_rings = json.load(open(DATA/'wei_country_rings.json'))
fq = pd.read_csv(RES/'basins/fragment_quarter_coefficients.csv')
modelled = set(fq.SB)
hyd = pd.read_csv(RES/'basins/basin_quarter_hydrology.csv')
q3 = hyd[hyd.Quarter == 'Q3'].set_index('SB')
dry_wei = (q3.WEI/q3.rL).clip(upper=2)
pr = pd.read_csv(RES/f'province_results_{PRE}{tag(1, "0", 0)}.csv')
fr = pr[pr.province.astype(str).str.startswith('frag:')].copy()
fr['SB'] = fr.province.str[5:].str.split('|').str[0]
q3def = fr[fr.quarter == 3].groupby('SB').exp_shortage.sum()/12

def draw_map(ax, values, bounds, label, only=None):
    norm = BoundaryNorm(bounds, SEQ.N)
    polys, face = [], []
    for sb, rings in sb_rings.items():
        v = values.get(sb, np.nan) if (only is None or sb in only) else np.nan
        for r in rings:
            polys.append(np.asarray(r)); face.append(SEQ(norm(v)) if np.isfinite(v) else '#ebeae6')
    ax.add_collection(PolyCollection(polys, facecolors=face, edgecolors=SURF, linewidths=0.15))
    ax.add_collection(PolyCollection([np.asarray(r) for rr in c_rings.values() for r in rr], facecolors='none',
                                     edgecolors=INK2, linewidths=0.35))
    ax.set_xlim(-11, 34); ax.set_ylim(34, 71.5); ax.set_aspect(1/np.cos(np.radians(50)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    cb_ = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), ax=ax, orientation='horizontal',
                       fraction=0.04, pad=0.02, aspect=35)
    cb_.outline.set_visible(False); cb_.ax.tick_params(length=0, labelsize=7, colors=INK2)
    cb_.set_label(label, fontsize=7.5)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.4))
draw_map(axs[0], dry_wei.to_dict(), [0, .05, .1, .2, .4, .6, 1.0, 2.0], 'Dry-summer WEI+ (consumption / renewable water)')
axs[0].set_title('Summer water stress in a dry year')
draw_map(axs[1], q3def.reindex(sb_rings.keys()).fillna(0).to_dict(), [0, 0.5, 2, 5, 10, 25, 50, 200],
         'Expected Q3 exceedance of the water-stress limit, 2030 (Mm³)', only=modelled)
axs[1].set_title('Where the 2030 plan exceeds the limit')
axs[1].text(0.02, 0.02, 'Grey: outside EU27 or no sub-basin record', transform=axs[1].transAxes, fontsize=6.5, color=INK2)
panel(axs[0], 'a'); panel(axs[1], 'b')
save(fig, 'fig1_basin_maps')

# Fig 2: national vs basin, and its robustness to the e-flow limit theta (T12)
def run_country(suffix):
    pr = pd.read_csv(RES/f'province_results_{PRE}{suffix}.csv')
    p = pr.province.astype(str)
    nat = pr[~p.str.startswith(('frag:', 'natq:'))].groupby('province').exp_shortage.sum()
    fr = pr[p.str.startswith('frag:')]
    bas = fr.groupby(fr.province.str.split('|').str[1]).exp_shortage.sum() if len(fr) else pd.Series(dtype=float)
    nq = pr[p.str.startswith('natq:')]
    bas = bas.add(nq.groupby(nq.province.str.split(':').str[1]).exp_shortage.sum(), fill_value=0) if len(nq) else bas
    return nat.add(bas, fill_value=0)/12
t12 = []
for th, ex in [(0.2, 'THETA0.2'), (0.4, ''), (0.6, 'THETA0.6')]:
    for sc, s, ex2 in [('National', 0, ex), ('National-quarterly', 0, '_'.join(x for x in ('NATQ1', ex) if x)), ('Sub-basin', 1, ex)]:
        d = load(tag(s, '0', 0, ex2))
        if d is None:
            continue
        b = bi(d); cty = run_country(tag(s, '0', 0, ex2))
        t12.append(dict(theta=th, scale=sc, exceedance=b.exp_shortage.mean(), irrigation=b.saved_IRR.mean(),
                        GDP=b.GDP.mean(), CO2=b.CO2.mean(), ES=cty.get('ES', 0.0), n_countries=int((cty > 1).sum())))
T12 = pd.DataFrame(t12)
T12.to_csv(OUT/'T12_scale_theta.csv', index=False)
print('== T12'); print(T12.round(2).to_string(index=False))
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
top = T7.head(8)
xx = np.arange(len(top))
axs[0].bar(xx-0.27, top.national, 0.26, color=S2, label='National, annual', edgecolor=SURF, linewidth=1)
axs[0].bar(xx, top.national_quarterly, 0.26, color=S3, label='National, quarterly', edgecolor=SURF, linewidth=1)
axs[0].bar(xx+0.27, top.basin, 0.26, color=S1, label='Sub-basin, quarterly', edgecolor=SURF, linewidth=1)
axs[0].set_xticks(xx, top.index); axs[0].set_ylabel('Expected exceedance (Mm³ per year)')
axs[0].legend(loc='upper right'); hgrid(axs[0]); axs[0].set_title('By country (θ = 0.4)', fontsize=8, fontweight='normal')
ths = sorted(T12.theta.unique()); xt = np.arange(len(ths))
for ax, col, lab in [(axs[1], 'exceedance', 'EU exceedance (bn m³)'), (axs[2], 'irrigation', 'Efficient irrigation (bn m³)')]:
    for k, (sc, colr, dx) in enumerate([('National', S2, -0.27), ('National-quarterly', S3, 0.0), ('Sub-basin', S1, 0.27)]):
        v = [T12[(T12.theta == th) & (T12.scale == sc)][col].iloc[0]/1e3 for th in ths]
        ax.bar(xt+dx, v, 0.26, color=colr, edgecolor=SURF, linewidth=1)
        for i, x in enumerate(v):
            ax.text(xt[i]+dx, x, f'{x:.1f}', ha='center', va='bottom', fontsize=5.5)
    ax.set_xticks(xt, [f'θ = {th}' for th in ths]); ax.set_title(lab, fontsize=8, fontweight='normal'); hgrid(ax)
    ax.set_xlabel('Water-stress limit θ (max. WEI+)', fontsize=7)
for ax, l in zip(axs, 'abc'):
    panel(ax, l)
fig.tight_layout(w_pad=1.4)
save(fig, 'fig2_scale')

# Fig 3: technology mix (basin, historical): single vs bi-level x Ref vs FF55
tm = {}
for pol, lab in [('0', 'Ref'), ('0.8138', 'FF55-power')]:
    d = MAIN[(1, pol, 0)]
    tm[f'{lab} · single-level'] = sg(d)[[f'saved_{k}' for k in TECH]].mean()/1e3
    tm[f'{lab} · bi-level'] = bi(d)[[f'saved_{k}' for k in TECH]].mean()/1e3
tm = pd.DataFrame(tm).T
tm.to_csv(OUT/'T2_technology_mix.csv')
fig, ax = plt.subplots(figsize=(7.2, 2.6))
order = list(tm.index)
left = np.zeros(len(order))
for k, c in zip(TECH, [S1, S2, S3, S4]):
    v = tm[f'saved_{k}'].to_numpy()
    ax.barh(range(len(order)), v, left=left, color=c, edgecolor=SURF, linewidth=1.5, height=0.55, label=TECH[k])
    left += v
for i, tot in enumerate(left):
    ax.text(tot+0.08, i, f'{tot:.1f}', va='center', fontsize=7.5)
ax.set_yticks(range(len(order)), order); ax.invert_yaxis()
ax.set_xlabel('Water saved by technology choice, 2030 (bn m³ per year, scenario mean)')
ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.45, 1.0))
save(fig, 'fig3_technology_mix')

# Fig 4: bi-level vs single (paired medians), basin / Ref / historical
pct = [('GDP', 'Value added'), ('food', 'Agricultural output'), ('CO2', 'EU CO₂'), ('power_CO2', 'Power CO₂'),
       ('energy_water', 'Energy-sector water'), ('EP', 'Eutrophication')]
fold = [('exp_shortage', 'Expected exceedance'), ('saved_IRR', 'Efficient irrigation'), ('saved_REN', 'Wind/solar')]
dd = pd.Series({lab: T3.loc[c, 'median_change_pct'] for c, lab in pct})
ff = pd.Series({lab: (m[c]/m[c+'_s'].where(m[c+'_s'] > 1e-6)).median() for c, lab in fold})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.5, 1]))
y = np.arange(len(dd)); col = [S1 if v >= 0 else RED for v in dd]
axs[0].axvline(0, color=INK2, lw=0.6)
axs[0].hlines(y, 0, dd.to_numpy(), color=col, lw=2)
axs[0].scatter(dd.to_numpy(), y, s=34, color=col, zorder=3, edgecolor=SURF, linewidth=1.2)
span = max(abs(dd.min()), abs(dd.max()), 1)
for i, v in enumerate(dd):
    axs[0].text(v+(0.04*span if v >= 0 else -0.04*span), i, f'{v:+.1f}%', va='center', ha='left' if v >= 0 else 'right', fontsize=7)
axs[0].set_yticks(y, dd.index); axs[0].invert_yaxis(); axs[0].set_xlim(-span*1.45, span*1.45)
axs[0].set_xlabel('Median paired change (%)'); axs[0].xaxis.grid(True); axs[0].set_axisbelow(True)
axs[0].set_title('Economy and environment')
y2 = np.arange(len(ff))
axs[1].axvline(1, color=INK2, lw=0.6)
axs[1].hlines(y2, 1, ff.to_numpy(), color=[S1 if v >= 1 else RED for v in ff], lw=2)
axs[1].scatter(ff.to_numpy(), y2, s=34, color=[S1 if v >= 1 else RED for v in ff], zorder=3, edgecolor=SURF, linewidth=1.2)
for i, v in enumerate(ff):
    axs[1].text(v-0.05*ff.max() if v < 1 else v+0.05*ff.max(), i, f'×{v:.2f}', va='center',
                ha='right' if v < 1 else 'left', fontsize=7)
axs[1].set_yticks(y2, ff.index); axs[1].invert_yaxis(); axs[1].set_xlim(0, ff.max()*1.35)
axs[1].set_ylim(len(ff)-0.5, -0.5)
axs[1].set_xlabel('Median ratio, bi-level / single-level'); axs[1].xaxis.grid(True); axs[1].set_axisbelow(True)
axs[1].set_title('Water outcomes')
panel(axs[0], 'a'); panel(axs[1], 'b')
fig.tight_layout(w_pad=2.5)
save(fig, 'fig4_bilevel_vs_single')

# Fig 5: policy x regime (basin, bi-level)
combo = [('Ref', 'Historical', (1, '0', 0)), ('Ref', 'Drier', (1, '0', 1)),
         ('FF55-power', 'Historical', (1, '0.8138', 0)), ('FF55-power', 'Drier', (1, '0.8138', 1))]
mets = [('GDP', 'Value added\n(tn EUR)', 1e3), ('CO2', 'EU CO₂\n(Mt)', 1), ('exp_shortage', 'Exceedance\n(bn m³)', 1e3),
        ('saved_IRR', 'Efficient irrigation\n(bn m³)', 1e3), ('lambda', 'Compromise\nlevel λ', 1)]
fig, axs = plt.subplots(1, len(mets), figsize=(7.2, 2.7))
for ax, (c, lab, div) in zip(axs, mets):
    for j, (pol, reg, k) in enumerate(combo):
        x = bi(MAIN[k])[c]/div
        colr = S1 if pol == 'Ref' else S2
        ax.scatter([j]*len(x), x, s=8, color=colr, alpha=0.35, linewidths=0)
        ax.scatter([j], [x.mean()], s=36, color=colr, edgecolor=SURF, linewidth=1.2, zorder=3,
                   marker='o' if reg == 'Historical' else 'D')
    ax.set_xticks(range(4), ['Ref\nhist', 'Ref\ndry', 'FF55\nhist', 'FF55\ndry'], fontsize=6.5)
    ax.set_title(lab, fontsize=7.5, fontweight='normal'); hgrid(ax)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker='o', ls='', color=S1, label='Ref2020 cap'),
                    Line2D([], [], marker='o', ls='', color=S2, label='Fit-for-55 power cap'),
                    Line2D([], [], marker='o', ls='', color=INK2, label='Historical regime (mean)'),
                    Line2D([], [], marker='D', ls='', color=INK2, label='Drier regime (mean)'),
                    Line2D([], [], marker='o', ls='', color=INK2, alpha=0.35, markersize=4, label='Individual scenarios')],
           ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=7)
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.93))
save(fig, 'fig5_policy_regime')

# Fig 6: sensitivity (bi-level, % change vs central)
if not T5.empty:
    met = [('exp_shortage', 'Expected exceedance'), ('saved_IRR', 'Efficient irrigation'), ('CO2', 'EU CO₂'), ('GDP', 'Value added')]
    fig, axs = plt.subplots(1, len(met), figsize=(7.2, 2.9), sharey=True)
    yy = np.arange(len(T5))
    for ax, (c, lab) in zip(axs, met):
        v = T5[c].to_numpy()
        ax.barh(yy, v, color=[S1 if x >= 0 else RED for x in v], height=0.6, edgecolor=SURF, linewidth=1)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title(lab, fontsize=8, fontweight='normal'); ax.xaxis.grid(True); ax.set_axisbelow(True)
        lim = max(abs(v).max()*1.15, 0.5); ax.set_xlim(-lim, lim)
        ax.set_xlabel('% vs central')
    axs[0].set_yticks(yy, T5.index); axs[0].invert_yaxis()
    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig6_sensitivity')

# Fig 7: reliability x water options (exceedance and irrigation required) + seasonal profile
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
for ax, col, lab in [(axs[0], 'exp_shortage', 'Expected exceedance (bn m³)'), (axs[1], 'saved_IRR', 'Efficient irrigation (bn m³)')]:
    grid = cb.pivot_table(index='scenario', columns='transfer', values=col).reindex(columns=WATER)/1e3
    G_ = grid.to_numpy()
    im = ax.imshow(G_, cmap=SEQ, aspect='auto', vmin=0, vmax=G_.max()*1.05)
    for (i, j), v in np.ndenumerate(G_):
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='white' if v > 0.6*G_.max() else INK)
    ax.set_xticks(range(4), [WLAB[w] for w in WATER], rotation=30, ha='right', fontsize=7)
    ax.set_yticks(range(len(grid)), [PLAB[i] for i in grid.index], fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(lab, fontsize=8, fontweight='normal')
axs[0].set_ylabel('Joint violation level (ES·FR·IT)')
qq = np.arange(1, 5)
for lab, colr, dx in [('Historical', S1, -0.18), ('Drier', S2, 0.18)]:
    v = T8[lab].exp_shortage.reindex(qq).fillna(0)/1e3
    axs[2].bar(qq+dx, v, 0.34, color=colr, label=f'{lab} regime', edgecolor=SURF, linewidth=1)
axs[2].set_xticks(qq, ['Q1', 'Q2', 'Q3', 'Q4'])
axs[2].set_ylabel('Expected exceedance (bn m³)'); hgrid(axs[2])
axs[2].set_ylim(0, axs[2].get_ylim()[1]*1.3)                     # room for the legend above the Q2/Q3 bars
axs[2].legend(loc='upper left', fontsize=6.5)
axs[2].set_title('Seasonal profile', fontsize=8, fontweight='normal')
for ax, l in zip(axs, 'abc'):
    panel(ax, l)
fig.tight_layout(w_pad=1.5)
save(fig, 'fig7_reliability_seasonal')
print('figures written to', FIG)
