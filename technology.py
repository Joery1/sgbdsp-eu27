"""Technology-choice options (rectangular choice of technology in the IO-LP).

Each option k sits on one node (province r, sector j). z_k (亿元) is the part of
that node's output produced with the alternative technology. Its input column
differs by Da_k, so x = L(y + D z); water, CO2 and value added change by
dw_k, dg_k, dv_k = -sum(Da_k) per unit of z_k. Inputs are drawn from the
same province unless noted. z_k <= cap_k * x_node.
"""
import numpy as np

R, S = 31, 42
AGRI, COAL, OILGAS, PAPER, REFINE, CHEM, NONMET, METAL = 0, 1, 2, 9, 10, 11, 12, 13
GENEQ, SPECEQ, ELECEQ, POWER, GAS, ENVSERV = 15, 16, 18, 24, 25, 36
FUELS = [COAL, OILGAS, REFINE, GAS]

# central parameters; scaled by IRR_COST etc. from model.py CLI
OPTIONS = {
    # water-saving irrigation (drip/sprinkler/low-pressure pipe); ~1/3 of irrigated
    # area already efficient in 2017, so up to 30% more of output can switch
    'IRR': dict(sectors=[AGRI], saving=0.35, cap=0.30, cost_per_m3=2.0,
                inputs={SPECEQ: 0.7, POWER: 0.3}, owner='L1'),
    # dry (air) cooling of thermal power: ~70% less water, ~4% more coal per kWh
    'AIR': dict(sectors=[POWER], saving=0.70, cap=0.30, fuel_up=0.04, capex=0.01,
                fuel_sectors=[COAL], inputs={GENEQ: 1.0}, owner='L2'),
    # wind/solar replacing thermal output: fossil inputs removed, equipment at 1.1x fuel cost
    'REN': dict(sectors=[POWER], saving=0.90, cap=0.20, co2_cut=0.95, premium=1.10,
                fuel_sectors=FUELS, inputs={ELECEQ: 1.0}, owner='L2'),
    # industrial water recycling in water-intensive manufacturing
    'REC': dict(sectors=[PAPER, CHEM, NONMET, METAL], saving=0.30, cap=0.30, cost_per_m3=3.0,
                inputs={ENVSERV: 0.5, SPECEQ: 0.5}, owner='U'),
}

def build(A, omega, gamma, mask, scale=None, options=OPTIONS, regions=range(R), S=S):
    """A: technical coefficients over all nodes (inputs may come from any region);
    omega in water per monetary unit (so cost_per_m3 must be in the same units),
    gamma in Mt per monetary unit. Options are placed on `regions` only.
    Returns dict of arrays over K options."""
    scale = scale or {}
    n = A.shape[0]
    sec = np.tile(np.arange(S), n//S)
    cols, node, dw, dg, cap, owner, name = [], [], [], [], [], [], []
    for key, o in options.items():
        for r in regions:
            for j in o['sectors']:
                i = r*S+j
                if mask[i] or omega[i] <= 0:
                    continue
                da = np.zeros(n)
                w_saved = o['saving']*scale.get(key+'_SAVE', 1)*omega[i]   # water per monetary unit of output
                g = 0.0
                if 'cost_per_m3' in o:
                    # yuan/m3 * m3 saved per yuan of output = extra input share
                    c = o['cost_per_m3']*scale.get(key+'_COST', 1)*w_saved   # m3/yuan x yuan/m3
                    for s_, f in o['inputs'].items():
                        da[r*S+s_] += c*f
                if key == 'AIR':
                    fuel_rows = np.isin(sec, o['fuel_sectors'])
                    da[fuel_rows] += o['fuel_up']*A[fuel_rows, i]
                    for s_, f in o['inputs'].items():
                        da[r*S+s_] += o['capex']*scale.get('AIR_COST', 1)*f
                    g = o['fuel_up']*gamma[i]
                if key == 'REN':
                    fuel_rows = np.isin(sec, o['fuel_sectors'])
                    removed = A[fuel_rows, i].sum()
                    da[fuel_rows] -= A[fuel_rows, i]
                    for s_, f in o['inputs'].items():
                        da[r*S+s_] += o['premium']*scale.get('REN_COST', 1)*removed*f
                    g = -o['co2_cut']*gamma[i]
                cols.append(da); node.append(i); dw.append(-w_saved); dg.append(g)
                cap.append(o['cap']*scale.get(key+'_CAP', 1)); owner.append(o['owner']); name.append(key)
    return dict(D=np.array(cols).T, node=np.array(node), dw=np.array(dw), dg=np.array(dg),
                dv=-np.array(cols).sum(1), cap=np.array(cap), owner=np.array(owner), name=np.array(name))
