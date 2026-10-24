"""Download EEA WEI+ sub-basin data (2002-2014, quarterly, water use by sector), sub-basin outlets (coastal or not)
and sub-basin / country outlines from the EEA ArcGIS service
WaterExploitationIndex/WEI (layer 2: sub-basins) and csi018_countries.
The files used for the paper are shipped in data/; running this script replaces them with the service's current content."""
from pathlib import Path
import json, time
import urllib.parse, urllib.request
import pandas as pd

DATA = Path(__file__).resolve().parent/'data'
SVC = 'https://water.discomap.eea.europa.eu/arcgis/rest/services/WaterExploitationIndex'
FIELDS = ['SB', 'SUB_BASIN', 'ZG', 'FRBD', 'SURFFinal', 'Year_', 'Quarter', 'Population', 'SurfaceWater',
          'Groundwater', 'AgricForestFish', 'ElecGasSteam', 'MiningConstruct', 'Service', 'WaterCollectTreat', 'WEI_2']

def query(url, **params):
    params.setdefault('f', 'json')
    body = urllib.parse.urlencode(params).encode()
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url+'/query', data=body), timeout=120) as r:
                j = json.load(r)
            if 'error' in j:
                raise RuntimeError(j['error'])
            return j
        except Exception as e:
            print('retry', attempt, e); time.sleep(3*(attempt+1))
    raise RuntimeError('query failed')

def attributes():
    url, rows, off = f'{SVC}/WEI/MapServer/2', [], 0
    while True:
        j = query(url, where='1=1', outFields=','.join(FIELDS), returnGeometry='false',
                  orderByFields='OBJECTID_12', resultOffset=off, resultRecordCount=1000)
        rows += [f['attributes'] for f in j['features']]
        off += len(j['features'])
        if not j.get('exceededTransferLimit') and len(j['features']) < 1000:
            break
    df = pd.DataFrame(rows)
    df.to_csv(DATA/'wei_subbasin_2002_2014.csv', index=False)
    print('attributes', df.shape, df.Year_.min(), df.Year_.max(), df.SB.nunique())

def outlets():
    j = query(f'{SVC}/WEI/MapServer/2', where="Year_=2006 AND Quarter='Q3'",
              outFields='SB,SUB_BASIN,OUTLET,SEA_CD,TYCOMP,SYSTEM_CD', returnGeometry='false')
    df = pd.DataFrame([f['attributes'] for f in j['features']])
    df.to_csv(DATA/'wei_subbasin_outlets.csv', index=False)
    print('outlets', df.shape, (df.OUTLET == '#SEA#').sum(), 'coastal')

def outlines():
    j = query(f'{SVC}/WEI/MapServer/2', where="Year_=2006 AND Quarter='Q3'", outFields='SB',
              returnGeometry='true', outSR=4326, maxAllowableOffset=0.01)
    json.dump({f['attributes']['SB']: f['geometry']['rings'] for f in j['features']},
              open(DATA/'wei_subbasin_rings.json', 'w'))
    print('sub-basin outlines', len(j['features']))
    j = query(f'{SVC}/csi018_countries/MapServer/0', where='1=1', outFields='Country',
              returnGeometry='true', outSR=4326, maxAllowableOffset=0.01)
    json.dump({f['attributes']['Country']: f['geometry']['rings'] for f in j['features']},
              open(DATA/'wei_country_rings.json', 'w'))
    print('countries', sorted(f['attributes']['Country'] for f in j['features']))

if __name__ == '__main__':
    attributes()
    outlets()
    outlines()
