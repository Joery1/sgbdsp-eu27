"""Leader-tolerance gradient: efficient irrigation of the bi-level compromise against the leader's tolerance
(0 = leader fixed at its individual optimum, 'none' = LEADTOL=0), Ref and Fit-for-55 power, sub-basin scale,
historical hydrology. (a) Water saved by efficient irrigation, bi-level line vs single-level reference;
(b) effect of the Fit-for-55 power cap on the bi-level plan (FF55 vs Ref, %)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG = HERE/'results', HERE/'figures_2030'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'axes.titleweight': 'bold',
    'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': GRID, 'grid.linewidth': 0.5, 'legend.frameon': False, 'legend.fontsize': 7.5})

P = 'scenario_results_YEAR2030_EFLOW1_DELTA0.05_CAPBOUND1_DESIGN2_CO2CAP1_'
TOLS = [('0', 0.0), ('0.015', 1.5), ('0.03', 3.0), ('0.06', 6.0), ('0.10', 10.0)]
X_NONE = 13.0                                              # position of the unconstrained compromise

def runs(pol):
    files = [(x, RES/f'{P}BASIN1_PRDRY0_CO2CAP_POWER{pol}_TOL{t}.csv') for t, x in TOLS]
    files.append((X_NONE, RES/f'{P}BASIN1_CO2CAP_POWER{pol}_PRDRY0_LEADTOL0.csv'))
    rows = []
    for x, f in files:
        df = pd.read_csv(f); df = df[df.feasible]
        b, s = df[df.model == 'SGBDSP'], df[df.model == 'single']
        rows.append(dict(x=x, irr=b.saved_IRR.mean()/1e3, lo=b.saved_IRR.min()/1e3, hi=b.saved_IRR.max()/1e3,
                         single=s.saved_IRR.mean()/1e3, exc=b.exp_shortage.mean()/1e3))
    return pd.DataFrame(rows)

ref, ff = runs('0'), runs('0.8138')

def axis(ax):
    ax.set_xticks([0, 1.5, 3, 6, 10, X_NONE], ['0', '1.5', '3', '6', '10', 'None'])
    ax.set_xlim(-0.6, X_NONE+0.8)
    ax.set_xlabel('Leader tolerance (% of reference final demand)')
    ax.axvline(3, color=INK2, lw=0.6, ls=':', zorder=0)
    ax.axvspan(11.0, 12.0, color=SURF, zorder=3)              # axis break before 'None'
    for xb in (11.0, 12.0):
        ax.plot([xb-0.15, xb+0.15], [-0.02, 0.02], transform=ax.get_xaxis_transform(), color=INK2, lw=0.6,
                clip_on=False, zorder=4)
    ax.yaxis.grid(True); ax.set_axisbelow(True)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.32))

ax = axs[0]
for d, col, mk, lab in [(ref, S1, 'o', 'Ref'), (ff, S2, 's', 'Fit-for-55 power')]:
    ax.fill_between(d.x, d.lo, d.hi, color=col, alpha=0.12, lw=0)
    ax.plot(d.x[:-1], d.irr[:-1], color=col, lw=2, marker=mk, ms=5, mec=SURF, mew=1, label=f'{lab}: bi-level')
    ax.plot(d.x[-2:], d.irr[-2:], color=col, lw=2, ls=(0, (2, 1.5)))
    ax.plot(d.x[-1:], d.irr[-1:], color=col, lw=0, marker=mk, ms=5, mec=SURF, mew=1)
    ax.axhline(d.single.iloc[0], color=col, lw=1, ls='--', zorder=1)
ax.text(-0.4, ref.single.iloc[0]+0.08, f'Single-level, Ref ({ref.single.iloc[0]:.2f})', ha='left', va='bottom', fontsize=7, color=INK2)
ax.text(10.7, ff.single.iloc[0]-0.08, f'Single-level, FF55 ({ff.single.iloc[0]:.2f})', ha='right', va='top', fontsize=7, color=INK2)
ax.text(3.15, 0.15, 'Paper\n(3%)', fontsize=7, color=INK2, va='bottom')
ax.set_ylim(0, 6.6)
ax.set_ylabel('Water saved by efficient irrigation (bn m³)')
ax.set_title('Bi-level vs single-level irrigation', loc='left')
ax.legend(loc='upper left', handlelength=2.2)
axis(ax)
ax.text(-0.02, 1.02, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')

ax = axs[1]
d_irr = 100*(ff.irr/ref.irr-1); d_exc = 100*(ff.exc/ref.exc-1)
ax.axhline(0, color=INK2, lw=0.6)
ax.plot(ref.x[:-1], d_irr[:-1], color=S3, lw=2, marker='o', ms=5, mec=SURF, mew=1)
ax.plot(ref.x[-2:], d_irr[-2:], color=S3, lw=2, ls=(0, (2, 1.5)))
ax.plot(ref.x[-1:], d_irr[-1:], color=S3, lw=0, marker='o', ms=5, mec=SURF, mew=1)
ax.plot(ref.x[:-1], d_exc[:-1], color=INK2, lw=2, marker='D', ms=4.5, mec=SURF, mew=1)
ax.plot(ref.x[-2:], d_exc[-2:], color=INK2, lw=2, ls=(0, (2, 1.5)))
ax.plot(ref.x[-1:], d_exc[-1:], color=INK2, lw=0, marker='D', ms=4.5, mec=SURF, mew=1)
ax.text(0.2, d_irr.iloc[0]+4, 'Efficient irrigation', fontsize=7, color=INK, va='bottom')
ax.text(0.2, 14, 'Expected exceedance', fontsize=7, color=INK, va='bottom')
for i, off in ((2, (5, 5)), (5, (6, -10))):
    ax.annotate(f'{d_irr.iloc[i]:+.0f}%', (ref.x.iloc[i], d_irr.iloc[i]), xytext=off, textcoords='offset points',
                fontsize=7, color=INK2, ha='right' if off[0] < 0 else 'left')
ax.set_ylim(-70, 25)
ax.set_ylabel('Change under Fit-for-55 power cap (%, bi-level)')
ax.set_title('Effect of the Fit-for-55 power cap', loc='left')
axis(ax)
ax.text(-0.02, 1.02, 'b', transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')

FIG.mkdir(exist_ok=True)
fig.savefig(FIG/'figS_leader_tolerance.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG/'figS_leader_tolerance.pdf', bbox_inches='tight')
pd.concat({'Ref': ref, 'FF55': ff}).to_csv(RES/'table_leader_tolerance.csv')
print(pd.concat({'Ref': ref, 'FF55': ff}).round(3))
