"""Eurostat inputs in data/: JSON-stat downloads and their long-format CSVs.

The files used for the paper are shipped (retrieval dates in README). By default this script only decodes the stored
JSON-stat files into the long CSVs read by prepare_eu.py, copula_eu.py and model_eu.py. With --download it first
fetches the same queries again; Eurostat revises its tables, so newer downloads can change the results slightly.
Usage: python fetch_eurostat.py [--download]
"""
import json, sys, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent/'data'
API = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/'
EU = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV',
      'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
# file -> (dataset, query, write long CSV)
QUERIES = {
    'env_wat_res': ('env_wat_res', [('unit', 'MIO_M3')], True),
    'env_wat_abs': ('env_wat_abs', [('unit', 'MIO_M3')]+[('geo', g) for g in EU+['EU27_2020']], True),
    'env_air_gge_co2': ('env_air_gge', [('airpol', 'CO2'), ('airpol', 'GHG'), ('src_crf', 'TOTX4_MEMO'),
                        ('src_crf', 'CRF1A1A'), ('unit', 'MIO_T')]+[('time', t) for t in ('1990', '2005', '2015', '2017', '2019')]
                        +[('geo', g) for g in EU+['EU27_2020']], True),
    'env_air_gge_crf': ('env_air_gge', [('airpol', 'CO2'), ('src_crf', 'TOTX4_MEMO'), ('src_crf', 'CRF1A1A'),
                        ('src_crf', 'CRF1A4B'), ('unit', 'MIO_T')]+[('time', t) for t in ('1990', '2015', '2017')]
                        +[('geo', g) for g in EU], False),
    'nama_gdp_clv15': ('nama_10_gdp', [('unit', 'CLV15_MEUR'), ('na_item', 'B1GQ')]+[('time', t) for t in ('2015', '2017')]
                       +[('geo', g) for g in EU], False),
}

def decode(path):
    """JSON-stat 2.0 -> long DataFrame (one column per dimension + value)"""
    j = json.load(open(path)); ids, size = j['id'], j['size']
    keys = [[k for k, _ in sorted(j['dimension'][d]['category']['index'].items(), key=lambda kv: kv[1])] for d in ids]
    rows = []
    for flat, v in j['value'].items():
        f, idx = int(flat), []
        for s in reversed(size):
            idx.append(f % s); f //= s
        idx = idx[::-1]
        rows.append({d: keys[i][idx[i]] for i, d in enumerate(ids)} | {'value': v})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    for name, (ds, q, long) in QUERIES.items():
        if '--download' in sys.argv:
            url = API+ds+'?'+urllib.parse.urlencode([('format', 'JSON'), ('lang', 'en')]+q)
            urllib.request.urlretrieve(url, DATA/f'{name}.json')
            print('downloaded', name)
        if long:
            decode(DATA/f'{name}.json').to_csv(DATA/f'{name}_long.csv', index=False)
            print('written', f'{name}_long.csv')
