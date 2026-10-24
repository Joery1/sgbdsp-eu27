"""Cross-scale evaluation: solve the plans of one model scale (BASIN=0 national-annual or BASIN=1
sub-basin-quarterly) and evaluate every plan ex post under BOTH exceedance rules, so that the plan
and the yardstick are separated (2x2: planning scale x evaluation scale).

Usage (same arguments as model_eu.py):
  python crosseval_scale.py YEAR=2030 EFLOW=1 DELTA=0.05 CAPBOUND=1 DESIGN=2 CO2CAP=1 BASIN=0 CO2CAP_POWER=0 PRDRY=0
Exceedance is recomputed from (y, z) as max(0, withdrawal - limit) (consumption-equivalent withdrawal when RHO_IRR < 1) for every unit and state, with the
same limits, alternative water and probabilities as the optimisation. Evaluating a plan at its own
scale reproduces the exceedance reported by model_eu.py (checked in the output).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import model_eu as M

def national_exceedance(y, t, alt):
    # national-annual e-flow rule for every country and state (main-text comparator)
    w = M.Mw@y+M.Mw_zc@t+M.Mw_c+M.dom
    ex = np.array([np.maximum(w-(np.minimum(M.W17*M.THETA*(M.flow[h]/M.Rbar)/M.WEI_nat, M.W0*(1+M.INFRA))+alt), 0)
                   for h in range(M.H)])                                     # H x C
    return ex

def basin_exceedance(y, t, sc):
    # sub-basin-quarter rule where a basin layer exists, national annual rule for Cyprus and Malta
    WG = (M.MwG@y+M.MwG_zc@t+M.MwG_c).reshape(M.C, M.NG)
    lhs = np.einsum('fqg,fg->fq', M.acoef, WG[M.fc])
    alt_fq = sc['desal'][M.fc][:, None]*M.w_desal+sc['reuse'][M.fc][:, None]*M.w_reuse
    exb = np.maximum(lhs[..., None]-M.cap_e-alt_fq[..., None], 0)              # NF x NQ x H
    exn = national_exceedance(y, t, sc['alt'])*(~M.basin_c)                     # H x C
    by_c = np.bincount(M.fc, (exb*M.PR).sum((1, 2)), minlength=M.C)+M.PR@exn
    return float((exb*M.PR).sum()+M.PR@exn.sum(1)), by_c

def natq_exceedance(y, t, sc):
    # national-quarterly rule where a basin layer exists, national annual rule for Cyprus and Malta
    lhs = np.einsum('cqn,n->cq', M.Ay_cq, y)+np.einsum('cqk,k->cq', M.Az_cq, t)+M.Ac_cq
    exq = np.maximum(lhs[..., None]-M.cap_nq-(sc['alt']/M.NQ)[:, None, None], 0)*M.basin_c[:, None, None]   # C x NQ x H
    exn = national_exceedance(y, t, sc['alt'])*(~M.basin_c)
    by_c = (exq*M.PR).sum((1, 2))+M.PR@exn
    return float(by_c.sum()), by_c

def one(sc):
    r = M.run(sc)
    if r is None or r['z'] is None:
        return []
    out = []
    for model, z in [('bi-level', r['z']), ('single', r['single'])]:
        y, t = z[M.iy:M.iy+M.nE], z[M.iz:M.iz+M.K]
        exn = national_exceedance(y, t, sc['alt'])
        b_tot, b_c = basin_exceedance(y, t, sc)
        q_tot, q_c = natq_exceedance(y, t, sc)
        ind = M.indicators(z)
        row = dict(scenario=sc['scenario'], water=sc['transfer'], unconventional=sc['unconventional'], model=model,
                   plan_scale='sub-basin' if M.BASIN else ('national-quarterly' if M.NNQ else 'national'), reported=ind['exp_shortage'],
                   eval_national=float(M.PR@exn.sum(1)), eval_natq=q_tot, eval_basin=b_tot, irrigation=ind['saved_IRR'],
                   GDP=ind['GDP'], CO2=ind['CO2'])
        row |= {f'basin_{c}': b_c[i] for i, c in enumerate(M.EU) if c in ('ES', 'PT', 'IT', 'GR', 'CY', 'MT')}
        row |= {f'natq_{c}': q_c[i] for i, c in enumerate(M.EU) if c in ('ES', 'PT', 'IT', 'GR', 'CY', 'MT')}
        out.append(row)
    return out

if __name__ == '__main__':
    tag = '_'.join(a.replace('=', '') for a in sys.argv[1:] if not a.startswith('JOBS'))
    rows = []
    with ProcessPoolExecutor(int(M.JOBS)) as ex:
        for r_ in ex.map(one, M.scenarios()):
            rows += r_
    df = pd.DataFrame(rows)
    (M.OUT/'crosseval').mkdir(exist_ok=True)
    df.to_csv(M.OUT/'crosseval'/f'crosseval_{tag}.csv', index=False)
    own = 'eval_basin' if M.BASIN else ('eval_natq' if M.NNQ else 'eval_national')
    print('max |own-scale evaluation - reported| (Mm3):', float((df[own]-df.reported).abs().max()))
    print(df.groupby('model')[['reported', 'eval_national', 'eval_natq', 'eval_basin', 'irrigation']].mean().div(1e3).round(3))
