"""Actual compliance of the summer joint reliability constraint (Section 2.5) for fixed plans.

For every scenario the bi-level plan is solved (model_eu.py, sub-basin scale, historical hydrology) and its summer (Q3)
withdrawal in each Spanish, French and Italian fragment is fixed. A fragment violates the water-stress limit in a year with
renewable-water ratio r if  withdrawal - alternative water > W2017 * theta * r / WEI+,  i.e. if r < r*_f. Three checks:
 A. as modelled: national ratios r_c drawn from the vine copula and the national marginals used in the model (1e6 draws);
    joint and per-country violation rates, by fragment group (compliant without / with the 2017 provision, over the limit);
 B. basin hydrology: each fragment's own Q3 renewable water (lognormal fit and the 13 observed years, 2002-2014) instead
    of the national annual quantile;
 C. parameter uncertainty: bootstrap of the 26 copula years and of the national marginal series (B = 200).
Usage: python reliability_check.py <model_eu.py arguments>  (writes results/reliability_check_<tag>.csv and _bootstrap.csv)
"""
import sys, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats, optimize
import model_eu as M
import copula as cp

JOINT = M.JOINT
NB, NSIM, NSIM_B = 200, 1_000_000, 200_000

def plan(sc):
    r = M.run(sc)
    if r is None or r['z'] is None:
        return None
    z = r['z']; y, t = z[M.iy:M.iy+M.nE], z[M.iz:M.iz+M.K]
    WG = (M.MwG@y+M.MwG_zc@t+M.MwG_c).reshape(M.C, M.NG)
    lhs = np.einsum('fg,fg->f', M.acoef[:, 2, :], WG[M.fc])                          # Q3 withdrawal by fragment
    alt = sc['desal'][M.fc]*M.w_desal[:, 2]+sc['reuse'][M.fc]*M.w_reuse[:, 2]
    return dict(scenario=sc['scenario'], p=sc['p'], water=sc['transfer'], reuse=sc['unconventional'],
                levels=sc['levels'], lhs=lhs, alt=alt, irrigation=float(-(M.tech['dw']*t)[M.tech['name'] == 'IRR'].sum()))

def ratio(c, u, par=None):
    name, p_ = M.mpar[c] if par is None else par
    d = getattr(stats, name)
    return d.ppf(u, *p_)/d.mean(*p_)

def groups(levels):
    """fragment index sets per country: compliant without / with the 2017 provision, over the limit (Q3 WEI+ > theta)"""
    out = {}
    for c in JOINT:
        i = M.EU.index(c); f = np.where(M.fc == i)[0]; w = M.WEIb[f, 2]
        qr = ratio(c, levels[c])
        comp = f[(w <= M.THETA) & (w > 0)]
        gf = comp[M.WEIb[comp, 2] > M.THETA*qr]                                      # limit raised to 2017 abstraction
        out[c] = dict(free=np.setdiff1d(comp, gf), grandfathered=gf, over=f[w > M.THETA])
    return out

def rstar(pl):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (pl['lhs']-pl['alt'])*M.WEIb[:, 2]/(M.base17_fq[:, 2]*M.THETA)        # ratio needed for compliance

def rates(R, rs, grp):
    """R: draws x 3 national ratios (order JOINT); violation rates of each fragment group"""
    out = {}
    for g in ('free', 'grandfathered', 'all_compliant', 'over'):
        viol_c = []
        for k, c in enumerate(JOINT):
            idx = np.r_[grp[c]['free'], grp[c]['grandfathered']] if g == 'all_compliant' else grp[c][g]
            T = rs[idx].max() if len(idx) else -np.inf
            viol_c.append(R[:, k] < T)
            out[f'{g}_{c}'] = float(viol_c[-1].mean())
        out[f'{g}_joint'] = float(np.any(viol_c, axis=0).mean())
    return out

def vine_sim(fl, n):
    U = stats.rankdata(fl, axis=0)/(len(fl)+1)
    return cp.simulate(cp.fit_vine(U), n)

if __name__ == '__main__':
    tag = '_'.join(a.replace('=', '') for a in sys.argv[1:] if not a.startswith('JOBS'))
    with ProcessPoolExecutor(int(M.JOBS)) as ex:
        plans = [p for p in ex.map(plan, M.scenarios()) if p is not None]
    res = pd.read_csv(M.HERE/'data/env_wat_res_long.csv')
    fl = res[(res.wat_proc == 'INTFL') & res.geo.isin(JOINT) & (res.value > 0)].pivot(index='time', columns='geo', values='value')[JOINT].dropna()
    S = vine_sim(fl.to_numpy(), NSIM)
    R = np.column_stack([ratio(c, S[:, k]) for k, c in enumerate(JOINT)])
    # B: fragment-level Q3 hydrology from the EEA records (same filter as basins_eu.py)
    d = pd.read_csv(M.HERE/'data/wei_subbasin_2002_2014.csv')
    d = d[d.groupby('SB').Year_.transform('size') == 52]
    d['R'] = np.where(d.WEI_2 > 0, d[M.GRP].sum(1)/(d.WEI_2/100), np.nan)
    q3 = d[d.Quarter == 'Q3'].pivot_table(index='Year_', columns='SB', values='R')
    sbs = [f.split('|')[0] for f in M.frags]
    rows, boot = [], []
    for pl in plans:
        rs, grp = rstar(pl), groups(pl['levels'])
        row = dict(scenario=pl['scenario'], p=pl['p'], water=pl['water'], reuse=pl['reuse'], q=pl['levels']['ES'],
                   irrigation=pl['irrigation'],
                   **{f'n_{g}_{c}': len(grp[c][g]) for c in JOINT for g in ('free', 'grandfathered', 'over')}, **rates(R, rs, grp))
        # B1: each fragment's own Q3 distribution (lognormal), mean violation probability of compliant fragments
        comp = np.concatenate([np.r_[grp[c]['free'], grp[c]['grandfathered']] for c in JOINT])
        pf = []
        for f in comp:
            Rf = q3[sbs[f]].dropna().to_numpy() if sbs[f] in q3 else np.array([])
            if len(Rf) >= 8 and Rf.min() > 0 and Rf.std() > 0:
                s, _, scl = stats.lognorm.fit(Rf, floc=0)
                pf.append(stats.lognorm.cdf(rs[f]*Rf.mean(), s, 0, scl))
        pf = np.array(pf)
        row |= dict(basin_mean_viol_prob=float(pf.mean()), basin_frags_above_q=int((pf > pl['levels']['ES']).sum()), basin_frags=len(pf))
        # B2: replay against the 13 observed summers
        yrs = q3.index
        viol_year = {c: np.zeros(len(yrs), bool) for c in JOINT}
        for c in JOINT:
            for f in np.r_[grp[c]['free'], grp[c]['grandfathered']]:
                if sbs[f] in q3:
                    Rf = q3[sbs[f]]
                    viol_year[c] |= (Rf/Rf.mean() < rs[f]).fillna(False).to_numpy()
        row |= {f'hist_years_viol_{c}': int(viol_year[c].sum()) for c in JOINT}
        row |= dict(hist_years_viol_joint=int(np.any([viol_year[c] for c in JOINT], axis=0).sum()), hist_years=len(yrs))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(M.OUT/f'reliability_check_{tag}.csv', index=False)
    # C: bootstrap of the copula years and of the national marginal series, for the P05 plans
    ser = pd.read_csv(M.OUT/'water_resources_series.csv', index_col=0)
    rng = np.random.default_rng(7)
    p05 = [pl for pl in plans if pl['scenario'] == 'P05']
    for b in range(NB):
        flb = fl.to_numpy()[rng.integers(0, len(fl), len(fl))]
        try:
            Sb = vine_sim(flb, NSIM_B)
            qb = optimize.brentq(lambda v: cp.joint_violation(Sb, [v]*3)-0.05, 1e-6, 0.05)
        except Exception:
            continue
        pars = {}
        for c in JOINT:
            v = ser[c].dropna().to_numpy(); vb = v[rng.integers(0, len(v), len(v))]
            nm, par, _ = cp.fit_marginal(vb); pars[c] = (nm, par)
        Rb = np.column_stack([ratio(c, Sb[:, k], pars[c]) for k, c in enumerate(JOINT)])
        rec = dict(b=b, q=qb, **{f'qr_{c}': float(ratio(c, qb, pars[c])) for c in JOINT})
        for pl in p05:
            r_ = rates(Rb, rstar(pl), groups(pl['levels']))
            rec[f"joint_free_{pl['water']}"] = r_['free_joint']; rec[f"joint_all_{pl['water']}"] = r_['all_compliant_joint']
        boot.append(rec)
    bt = pd.DataFrame(boot); bt.to_csv(M.OUT/f'reliability_check_{tag}_bootstrap.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 60)
    show = ['scenario', 'water', 'free_joint', 'grandfathered_joint', 'all_compliant_joint', 'over_joint',
            'basin_mean_viol_prob', 'basin_frags_above_q', 'hist_years_viol_joint']
    print(out[show].round(4).to_string(index=False))
    print('bootstrap (P05):', len(bt), 'replicates'); print(bt.drop(columns='b').quantile([.05, .5, .95]).round(4).T.to_string())
