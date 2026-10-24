"""Marginals for 31 provinces and a 3-D vine copula for the joint chance
constraint on Hebei, Shandong and Henan water resources (2003-2020)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats, optimize

OUT = Path(__file__).resolve().parent/'results'
JOINT = ['河北', '山东', '河南']
RNG = np.random.default_rng(20261021)
EPS = 1e-10

# ---------------- marginals ----------------
def fit_marginal(v):
    best = None
    for name, dist in [('norm', stats.norm), ('lognorm', stats.lognorm), ('gamma', stats.gamma)]:
        par = dist.fit(v, floc=0) if name != 'norm' else dist.fit(v)
        aic = 2*len(par) - 2*dist.logpdf(v, *par).sum() - (2 if name != 'norm' else 0)
        if best is None or aic < best[2]:
            best = (name, par, aic)
    return best

def band_mean(dist, par, lo, hi, k=2000):
    q = np.linspace(lo, hi, k+2)[1:-1]
    return dist.ppf(q, *par).mean()

# ---------------- bivariate families ----------------
def _clip(u):
    return np.clip(u, EPS, 1-EPS)

class Pair:
    def __init__(self, fam, th, rot=False):
        self.fam, self.th, self.rot = fam, th, rot

    def _pdf(self, u, v):
        t = self.th
        if self.fam == 'indep':
            return np.ones_like(u)
        if self.fam == 'gauss':
            a, b = stats.norm.ppf(u), stats.norm.ppf(v)
            return np.exp(-(t*t*(a*a+b*b)-2*t*a*b)/(2*(1-t*t)))/np.sqrt(1-t*t)
        if self.fam == 'clayton':
            return (1+t)*(u*v)**(-1-t)*(u**-t+v**-t-1)**(-2-1/t)
        if self.fam == 'frank':
            e = np.expm1(-t)
            return -t*e*np.exp(-t*(u+v))/(e+np.expm1(-t*u)*np.expm1(-t*v))**2
        if self.fam == 'gumbel':
            x, y = -np.log(u), -np.log(v)
            s = x**t+y**t
            C = np.exp(-s**(1/t))
            return C/(u*v)*(x*y)**(t-1)*s**(-2+1/t)*(s**(1/t)+t-1)

    def _h(self, u, v):                       # dC(u,v)/dv
        t = self.th
        if self.fam == 'indep':
            return u
        if self.fam == 'gauss':
            return stats.norm.cdf((stats.norm.ppf(u)-t*stats.norm.ppf(v))/np.sqrt(1-t*t))
        if self.fam == 'clayton':
            return v**(-t-1)*(u**-t+v**-t-1)**(-1-1/t)
        if self.fam == 'frank':
            return np.exp(-t*v)*np.expm1(-t*u)/(np.expm1(-t)+np.expm1(-t*u)*np.expm1(-t*v))
        if self.fam == 'gumbel':
            x, y = -np.log(u), -np.log(v)
            s = x**t+y**t
            return np.exp(-s**(1/t))/v*y**(t-1)*s**(1/t-1)

    def _hinv(self, w, v):
        t = self.th
        if self.fam == 'indep':
            return w
        if self.fam == 'gauss':
            return stats.norm.cdf(stats.norm.ppf(w)*np.sqrt(1-t*t)+t*stats.norm.ppf(v))
        if self.fam == 'clayton':
            return ((w*v**(t+1))**(-t/(1+t))+1-v**-t)**(-1/t)
        lo, hi = np.full_like(w, EPS), np.full_like(w, 1-EPS)   # bisection (frank, gumbel)
        for _ in range(60):
            mid = (lo+hi)/2
            big = self._h(mid, v) > w
            hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
        return (lo+hi)/2

    def pdf(self, u, v):
        u, v = _clip(u), _clip(v)
        return self._pdf(1-u, 1-v) if self.rot else self._pdf(u, v)

    def h(self, u, v):
        u, v = _clip(u), _clip(v)
        return _clip(1-self._h(1-u, 1-v) if self.rot else self._h(u, v))

    def hinv(self, w, v):
        w, v = _clip(w), _clip(v)
        return _clip(1-self._hinv(1-w, 1-v) if self.rot else self._hinv(w, v))

    def label(self):
        return ('survival ' if self.rot else '')+self.fam

BOUNDS = {'gauss': (-0.98, 0.98), 'clayton': (1e-3, 20), 'frank': (-30, 30), 'gumbel': (1.0001, 15)}

def fit_pair(u, v):
    cands = [(Pair('indep', 0), 0.0)]
    tau = stats.kendalltau(u, v)[0]
    for fam, (lo, hi) in BOUNDS.items():
        for rot in ([False, True] if fam in ('clayton', 'gumbel') else [False]):
            if fam in ('clayton', 'gumbel') and tau <= 0:
                continue
            nll = lambda th: -np.log(np.maximum(Pair(fam, th, rot).pdf(u, v), 1e-300)).sum()
            r = optimize.minimize_scalar(nll, bounds=(lo, hi), method='bounded')
            cands.append((Pair(fam, r.x, rot), 2+2*r.fun))
    return min(cands, key=lambda c: c[1])

# ---------------- C-vine (3-D) ----------------
def fit_vine(U):
    tau = np.abs(stats.kendalltau(U[:, 0], U[:, 1])[0]), np.abs(stats.kendalltau(U[:, 0], U[:, 2])[0]), \
          np.abs(stats.kendalltau(U[:, 1], U[:, 2])[0])
    T = np.zeros((3, 3)); T[0, 1], T[0, 2], T[1, 2] = tau; T = T+T.T
    c = int(np.argmax(T.sum(1)))                      # root = strongest overall dependence
    a, b = [i for i in range(3) if i != c]
    p_ac, aic1 = fit_pair(U[:, a], U[:, c])
    p_bc, aic2 = fit_pair(U[:, b], U[:, c])
    za, zb = p_ac.h(U[:, a], U[:, c]), p_bc.h(U[:, b], U[:, c])
    p_ab_c, aic3 = fit_pair(za, zb)
    return dict(root=c, a=a, b=b, p_ac=p_ac, p_bc=p_bc, p_ab_c=p_ab_c, aic=aic1+aic2+aic3)

def simulate(v, n):
    w = RNG.random((n, 3))
    U = np.empty((n, 3))
    U[:, v['root']] = w[:, 0]
    U[:, v['a']] = v['p_ac'].hinv(w[:, 1], U[:, v['root']])
    zb = v['p_ab_c'].hinv(w[:, 2], v['p_ac'].h(U[:, v['a']], U[:, v['root']]))
    U[:, v['b']] = v['p_bc'].hinv(zb, U[:, v['root']])
    return U

def joint_violation(S, levels):
    return np.mean((S <= np.asarray(levels)).any(1))

def solve_unknown(S, p, known, k):
    def gap(q):
        lv = list(known); lv.insert(k, q)
        return joint_violation(S, lv)-p
    return optimize.brentq(gap, 1e-6, p)

if __name__ == '__main__':
    d = np.load(OUT/'inputs.npz')
    meta = json.load(open(OUT/'meta.json'))
    prov = meta['provinces']
    res = d['resources']
    # marginals and FSP flow levels (L/M/H, prob 0.2/0.6/0.2) for every province
    rows, margins = [], {}
    for i, pr in enumerate(prov):
        name, par, aic = fit_marginal(res[i])
        dist = getattr(stats, name)
        margins[pr] = (dist, par)
        rows.append(dict(province=pr, dist=name, aic=aic, mean_obs=res[i].mean(),
                         L=band_mean(dist, par, 0, .2), M=band_mean(dist, par, .2, .8),
                         H=band_mean(dist, par, .8, 1), ks_p=stats.kstest(res[i], name, par).pvalue))
    mg = pd.DataFrame(rows)
    mg.to_csv(OUT/'marginals_flow_levels.csv', index=False, encoding='utf-8-sig')

    J = [prov.index(x) for x in JOINT]
    X = res[J].T
    U = (stats.rankdata(X, axis=0))/(X.shape[0]+1)
    vine = fit_vine(U)
    # self-check of h / h-inverse round trips
    for pc in (vine['p_ac'], vine['p_bc'], vine['p_ab_c']):
        ww, vv = RNG.random(1000), RNG.random(1000)
        assert np.max(np.abs(pc.h(pc.hinv(ww, vv), vv)-ww)) < 1e-6, pc.label()
    S = simulate(vine, 1_000_000)
    tau_obs = [stats.kendalltau(X[:, i], X[:, j])[0] for i, j in [(0, 1), (0, 2), (1, 2)]]
    tau_sim = [stats.kendalltau(S[:20000, i], S[:20000, j])[0] for i, j in [(0, 1), (0, 2), (1, 2)]]

    # scenario design mirroring the paper's Table 1: two levels known, third solved
    design = {.05: .004, .10: .03, .15: .05, .20: .004}
    out = []
    for tag, k in zip('XIN', [2, 1, 0]):      # X: Henan unknown, I: Shandong, N: Hebei
        for m, (p, kn) in enumerate(design.items(), 1):
            q = solve_unknown(S, p, [kn, kn], k)
            lv = [kn, kn]; lv.insert(k, q)
            out.append(dict(scenario=f'{tag}{m}', p=p, p_hebei=lv[0], p_shandong=lv[1], p_henan=lv[2],
                            p_indep=1-np.prod(1-np.array(lv)), check=joint_violation(S, lv)))
    sc = pd.DataFrame(out)
    sc.to_csv(OUT/'joint_violation_scenarios.csv', index=False, encoding='utf-8-sig')
    names = lambda i: JOINT[i]
    info = dict(root=names(vine['root']),
                pairs={f"{names(vine['a'])}-{names(vine['root'])}": [vine['p_ac'].label(), vine['p_ac'].th],
                       f"{names(vine['b'])}-{names(vine['root'])}": [vine['p_bc'].label(), vine['p_bc'].th],
                       f"{names(vine['a'])}-{names(vine['b'])}|{names(vine['root'])}":
                           [vine['p_ab_c'].label(), vine['p_ab_c'].th]},
                tau_obs=tau_obs, tau_sim=tau_sim, n_years=int(X.shape[0]))
    json.dump(info, open(OUT/'vine_fit.json', 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(info, ensure_ascii=False, indent=1, default=float))
    print(mg[mg.province.isin(JOINT)].to_string(index=False))
    print(sc.round(4).to_string(index=False))
