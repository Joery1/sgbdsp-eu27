"""Shortage-penalty sensitivity for the 2030 EU27 experiment (run_penalty.sh).

Penalty settings, ordered by implied EUR per m3 of expected exceedance: agricultural VA per m3 x0.1,
  central (agricultural VA per m3), x10, economy-wide VA per m3 of withdrawal (PEN_MODE=0), and x10 of it.
Tests: (i) does the drier regime change decisions; (ii) do the main findings hold
(scale contrast, energy-follower veto, Fit-for-55 crowding-out of efficient irrigation);
(iii) how much of the efficient-irrigation potential is used.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RES, OUT, FIG = HERE/'results', HERE/'results/y2030', HERE/'figures_2030'
PRE = 'YEAR2030_EFLOW1_DELTA0.05_CAPBOUND1_DESIGN2_CO2CAP1_'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'      # same palette/style as analyze_2030.py
S1, S2, S3, RED = '#2a78d6', '#eb6834', '#1baf7a', '#e34948'
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

SETTINGS = [('Agri ×0.1', 'PEN_SCALE0.1'), ('Central (agri value)', ''), ('Agri ×10', 'PEN_SCALE10'),
            ('Economy-wide VA', 'PEN_MODE0'), ('Economy-wide ×10', 'PEN_MODE0_PEN_SCALE10')]
RUNS = {'Ref · historical': 'BASIN1_CO2CAP_POWER0_PRDRY0', 'Ref · drier': 'BASIN1_CO2CAP_POWER0_PRDRY1',
        'FF55 · historical': 'BASIN1_CO2CAP_POWER0.8138_PRDRY0'}

def load(run, pen, extra=''):
    suffix = run + (f'_{pen}' if pen else '') + (f'_{extra}' if extra else '')
    f = RES/f'scenario_results_{PRE}{suffix}.csv'
    return pd.read_csv(f) if f.exists() else None

rows = []
for lab, pen in SETTINGS:
    for rname, run in RUNS.items():
        d = load(run, pen)
        if d is None:
            continue
        for model in ['SGBDSP', 'single']:
            b = d[(d.model == model) & d.feasible]
            rows.append(dict(penalty_setting=lab, run=rname, model=model, n=len(b),
                             eur_per_m3=1e3*b.penalty.mean()/b.exp_shortage.mean(),
                             **b[['lambda', 'GDP', 'food', 'CO2', 'exp_shortage', 'Q3_short_L', 'saved_IRR',
                                  'saved_AIR', 'saved_REN', 'penalty']].mean().to_dict()))
T = pd.DataFrame(rows)
T.to_csv(OUT/'T9_penalty_sensitivity.csv', index=False)

# derived tests (bi-level)
bi = T[T.model == 'SGBDSP'].set_index(['penalty_setting', 'run'])
sg = T[T.model == 'single'].set_index(['penalty_setting', 'run'])
tests = []
for lab, _ in SETTINGS:
    try:
        h, dr, ff = bi.loc[(lab, 'Ref · historical')], bi.loc[(lab, 'Ref · drier')], bi.loc[(lab, 'FF55 · historical')]
        s = sg.loc[(lab, 'Ref · historical')]
    except KeyError:
        continue
    tests.append(dict(penalty_setting=lab, eur_per_m3=h.eur_per_m3,
                      drier_exceedance_pct=100*(dr.exp_shortage/h.exp_shortage-1),
                      drier_IRR_pct=100*(dr.saved_IRR/h.saved_IRR-1),
                      drier_GDP_pct=100*(dr.GDP/h.GDP-1),
                      veto_IRR_ratio_bi_single=h.saved_IRR/s.saved_IRR,
                      veto_AIR_bi=h.saved_AIR, veto_AIR_single=s.saved_AIR,
                      FF55_IRR_pct=100*(ff.saved_IRR/h.saved_IRR-1),
                      FF55_exceedance_pct=100*(ff.exp_shortage/h.exp_shortage-1),
                      penalty_share_GDP_pct=100*h.penalty/h.GDP))
TT = pd.DataFrame(tests)
TT.to_csv(OUT/'T10_penalty_tests.csv', index=False)

# scale contrast and technology rows under the agricultural-value penalty
nat = load('BASIN0_CO2CAP_POWER0_PRDRY0', '')
noch = load('BASIN1_CO2CAP_POWER0_PRDRY0', '', 'CHANCE0_TECH1')
extra = {}
if nat is not None:
    b = nat[(nat.model == 'SGBDSP') & nat.feasible]
    extra['national_agri: exceedance'] = b.exp_shortage.mean(); extra['national_agri: IRR'] = b.saved_IRR.mean()
if noch is not None:
    b = noch[(noch.model == 'SGBDSP') & noch.feasible]
    extra['basin_agri_no_reliability: IRR'] = b.saved_IRR.mean()
    extra['basin_agri_no_reliability: exceedance'] = b.exp_shortage.mean()

# efficient-irrigation potential used (bound formulation: z <= cap * baseline output)
sys.argv = ['x', 'YEAR=2030', 'EFLOW=1', 'DELTA=0.05', 'CAPBOUND=1', 'DESIGN=2', 'CO2CAP=1', 'BASIN=1']
import model_eu as m
irr = m.tech['name'] == 'IRR'
potential = float((m.tech['cap'][irr]*(m.LEE@m.y0+m.cst)[m.tnode[irr]]*(-m.tech['dw'][irr])).sum())
extra['IRR potential (Mm3)'] = potential

pd.set_option('display.width', 220)
print(T[T.model == 'SGBDSP'].round(2).to_string(index=False))
print(TT.round(2).to_string(index=False))
print({k: round(v, 1) for k, v in extra.items()})
pd.Series(extra).to_csv(OUT/'T11_penalty_extra.csv')

# ---------------------------------------------------------------- figure
labs = [l for l, _ in SETTINGS if l in set(T.penalty_setting)]
xx = np.arange(len(labs))
eur = [bi.loc[(l, 'Ref · historical')].eur_per_m3 for l in labs]
short = {'Central (agri value)': 'Central\n(agri value)', 'Economy-wide VA': 'Economy-\nwide VA', 'Economy-wide ×10': 'Economy-\nwide ×10'}
xt = [f'{short.get(l, l)}\n{e:.1f} €/m³' if e < 10 else f'{short.get(l, l)}\n{e:.0f} €/m³' for l, e in zip(labs, eur)]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.1))
styles = {'Ref · historical': (S1, 'o', '-'), 'Ref · drier': (S1, 'D', '--'), 'FF55 · historical': (S2, 'o', '-')}
for ax, col, lab, div in [(axs[0], 'exp_shortage', 'Expected exceedance (bn m³)', 1e3),
                          (axs[1], 'saved_IRR', 'Efficient irrigation (bn m³)', 1e3),
                          (axs[2], 'GDP', 'Value added (tn EUR)', 1e3)]:
    for rname, (c, mk, ls) in styles.items():
        v = [bi.loc[(l, rname)][col]/div for l in labs]
        ax.plot(xx, v, ls, color=c, marker=mk, ms=5, lw=1.4, label=rname)
    if col == 'saved_IRR':
        v = [sg.loc[(l, 'Ref · historical')][col]/div for l in labs]
        ax.plot(xx, v, ':', color=INK2, marker='s', ms=4, lw=1.2, label='Ref · historical, single-level')
        ax.axhline(potential/1e3, color=RED, lw=0.8)
        ax.text(xx[-1], potential/1e3, 'adoption limit', color=RED, fontsize=6.5, ha='right', va='bottom')
    ax.set_xticks(xx, xt, fontsize=6, rotation=0)
    ax.set_title(lab, fontsize=8, fontweight='normal'); hgrid(ax)
h_, l_ = axs[1].get_legend_handles_labels()
fig.legend(h_, l_, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=6.8)
for ax, l in zip(axs, 'abc'):
    panel(ax, l)
fig.supxlabel('Shortage penalty (implied value of an exceeded m³)', fontsize=7.5)
fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.93))
save(fig, 'fig8_penalty_sensitivity')
