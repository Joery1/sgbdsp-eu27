"""Exact adoption-cap rows vs baseline-output bounds on the 2030 main configuration."""
import sys, time, json
ROOT = sys.argv[1]
sys.path.insert(0, ROOT)
base = ['YEAR=2030', 'EFLOW=1', 'DELTA=0.05', 'DESIGN=2', 'CO2CAP=1', 'BASIN=1', 'CO2CAP_POWER=0', 'PRDRY=0']
keys = ['GDP', 'food', 'CO2', 'power_CO2', 'exp_shortage', 'Q3_short_L', 'saved_IRR', 'saved_AIR', 'saved_REN', 'saved_REC']
out = {}
for mode in ['1', '0']:
    sys.argv = ['x']+base+[f'CAPBOUND={mode}']
    if 'model_eu' in sys.modules: del sys.modules['model_eu']
    import model_eu as m
    scs = list(m.scenarios())
    for idx in [0, 5, 11]:
        sc = scs[idx]; t0 = time.time(); r = m.run(sc); dt = time.time()-t0
        I, S = m.indicators(r['z']), m.indicators(r['single'])
        out[f'{mode}_{idx}'] = dict(time=dt, lam=float(r['z'][m.iL]), bi={k: float(I[k]) for k in keys}, single={k: float(S[k]) for k in keys},
                                   scenario=f"{sc['scenario']}-{sc['transfer']}")
        print(mode, sc['scenario'], sc['transfer'], 'time %.0fs' % dt, flush=True)
json.dump(out, open(ROOT+'/results/capcheck_2030.json', 'w'), indent=1)
for idx in [0, 5, 11]:
    a, b = out[f'0_{idx}'], out[f'1_{idx}']
    print(a['scenario'], 'speed-up x%.0f' % (a['time']/b['time']), 'lambda exact %.4f bound %.4f' % (a['lam'], b['lam']))
    for k in keys:
        for lv in ['bi', 'single']:
            va, vb = a[lv][k], b[lv][k]
            print('   %-12s %-6s exact %11.2f bound %11.2f rel %+.2e' % (k, lv, va, vb, (vb-va)/max(abs(va), 1e-9)))
