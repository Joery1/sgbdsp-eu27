"""Graphical abstract for Environmental Modelling & Software: 1328 x 531 px (w x h), readable at 13 x 5 cm.
Left: model set-up. Middle: over-abstraction of each plan judged at sub-basin-quarter scale vs what national
accounts report (Ref, bi-level; results/table_natq_crosseval.csv). Right: water saved by efficient irrigation under
value-added priority (tolerance 3%) and without it, Ref vs power-sector CO2 cap (results/table_leader_tolerance.csv)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
RES, FIG = HERE/'results', HERE/'figures_2030'
SURF, INK, INK2, GRID, PALE = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#c9c8c3'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 6.5, 'axes.edgecolor': INK2, 'axes.linewidth': 0.5,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.5,
    'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.labelsize': 6,
    'ytick.labelsize': 6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF})

W_CM, H_CM, W_PX = 13.0, 5.0, 1328
W_IN, H_IN = W_CM/2.54, W_CM/2.54*531/1328
DPI = W_PX/W_IN

nq = pd.read_csv(RES/'table_natq_crosseval.csv', keep_default_na=False)            # plan 'NA' is not missing
nq = nq[(nq.policy == 'Ref') & (nq.solution == 'bi-level')].set_index('plan')
tol = pd.read_csv(RES/'table_leader_tolerance.csv', index_col=0)
tol.columns = ['i'] + list(tol.columns[1:])

def irr(pol, x):
    return float(tol[(tol.index == pol) & (tol.x == x)].irr.iloc[0])

fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)

def box(x, y, w, h, text, fc, ec, size=6.2, weight='normal', color=INK):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.012',
                                      transform=fig.transFigure, fc=fc, ec=ec, lw=0.6))
    fig.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=size, weight=weight, color=color,
             linespacing=1.15)

def key(x, y, col, text):
    fig.patches.append(matplotlib.patches.Rectangle((x, y-0.025), 0.009, 0.05, transform=fig.transFigure,
                                                    fc=col, lw=0))
    fig.text(x+0.015, y, text, color=INK, fontsize=6, va='center')

def arrow(x0, y0, x1, y1):
    fig.add_artist(matplotlib.patches.FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                   arrowstyle='-|>', mutation_scale=5, lw=0.6, color=INK2))

# ---- left: model set-up ----
fig.text(0.012, 0.93, 'Bi-level MRIO model', fontsize=7, weight='bold', color=INK, va='top')
box(0.012, 0.62, 0.235, 0.13, 'EU leader: value added', '#e8f0fa', S1, weight='bold')
for k, lab in enumerate(['Food', 'Energy', 'Ecosystem']):
    box(0.012+k*0.08, 0.42, 0.075, 0.11, lab, SURF, INK2, size=5.6)
    arrow(0.1295, 0.62, 0.0495+k*0.08, 0.535)
box(0.012, 0.23, 0.235, 0.11, 'EXIOBASE EU27 MRIO, 2030', SURF, INK2, size=5.6)
box(0.012, 0.04, 0.235, 0.13, '412 sub-basins × 4 quarters\nWEI+ ≤ 0.4, summer reliability', '#e6f6ef', S3, size=5.6)

# ---- middle: scale blind spot ----
X0 = 0.285
fig.text(X0, 0.93, 'Scale blind spot', fontsize=7, weight='bold', color=INK, va='top')
fig.text(X0, 0.82, 'Over-abstraction of each plan, bn m³', fontsize=6, color=INK2, va='top')
ax = fig.add_axes([0.435, 0.29, 0.16, 0.42])
plans = [('NA', 'National–annual'), ('NQ', 'National–quarterly'), ('BQ', 'Basin–quarterly')]
for k, (p, lab) in enumerate(plans):
    y = 2-k
    ax.barh(y, nq.loc[p, 'eval_BQ'], height=0.6, color=S1, lw=0)
    ax.barh(y, nq.loc[p, 'eval_NA'], height=0.6, color=INK, lw=0)
    ax.text(nq.loc[p, 'eval_BQ']+0.1, y, f'{nq.loc[p, "eval_BQ"]:.2f}', va='center', fontsize=6, color=INK)
ax.set_yticks([2, 1, 0], [lab for _, lab in plans])
ax.tick_params(axis='y', length=0)
ax.set_xlim(0, 5.4); ax.set_xticks([0, 2, 4])
ax.spines['left'].set_visible(False)
gain = nq.loc['NA', 'eval_BQ']-nq.loc['BQ', 'eval_BQ']
key(X0, 0.12, S1, 'judged by sub-basin and quarter')
key(X0, 0.04, INK, f'seen in national accounts (≈{nq.loc["NA", "eval_NA"]:.2f} each)')

# ---- right: priority x carbon cap ----
X1 = 0.685
fig.text(X1, 0.93, 'Priority × carbon cap', fontsize=7, weight='bold', color=INK, va='top')
fig.text(X1, 0.82, 'Water saved by efficient irrigation, bn m³', fontsize=6, color=INK2, va='top')
ax = fig.add_axes([0.73, 0.29, 0.25, 0.42])
groups = [('Value-added priority', 3.0), ('No priority', 13.0)]
for g, (lab, x) in enumerate(groups):
    for j, (pol, col) in enumerate([('Ref', S1), ('FF55', S2)]):
        v = irr(pol, x)
        ax.bar(g+(j-0.5)*0.36, v, width=0.32, color=col, lw=0)
        ax.text(g+(j-0.5)*0.36, v+0.15, f'{v:.1f}', ha='center', fontsize=6, color=INK)
ax.set_xticks([0, 1], [lab for lab, _ in groups])
ax.tick_params(axis='x', length=0)
ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 6.4); ax.set_yticks([0, 2, 4, 6])
key(X1, 0.12, S1, 'Reference policy')
key(X1+0.15, 0.12, S2, 'Power CO₂ cap')
fig.text(X1, 0.04, 'The cap cuts irrigation only under priority', color=INK, fontsize=6, va='center')

for xs in (0.268, 0.668):
    fig.add_artist(matplotlib.lines.Line2D([xs, xs], [0.04, 0.93], transform=fig.transFigure, color=GRID, lw=0.6))

for ext in ('png', 'pdf', 'tiff'):
    fig.savefig(FIG/f'graphical_abstract.{ext}', dpi=DPI, **({'pil_kwargs': {'compression': 'tiff_lzw'}}
                                                              if ext == 'tiff' else {}))
print(f'NA {nq.loc["NA", "eval_BQ"]:.2f}  NQ {nq.loc["NQ", "eval_BQ"]:.2f}  BQ {nq.loc["BQ", "eval_BQ"]:.2f}  gain {gain:.2f}')
