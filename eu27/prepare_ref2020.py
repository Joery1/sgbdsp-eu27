"""EU Reference Scenario 2020 (PRIMES, DG ENER) country results -> data/eu_ref2020_long.parquet.

Input: ref2020_energy-transport-ghg.xlsx from
https://energy.ec.europa.eu/data-and-analysis/energy-modelling/eu-reference-scenario-2020_en
Usage: python prepare_ref2020.py <path to the xlsx> [output parquet]
Output columns: country, block (macro | fec | gen | co2), item, year (2005-2050, 5-year steps), value.
"""
import sys
from pathlib import Path
import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parent
YEARS = [2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
EU27 = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU',
        'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
SHEET = lambda r: 'EL' if r == 'GR' else r
# (sheet part, anchor row label, rows read after it, block)
BLOCKS = [('A', 'MACROECONOMIC INPUTS', 5, 'macro'),
          ('A', 'Final Energy Consumption (ktoe)', 19, 'fec'),
          ('B', 'Gross Electricity generation by source (GWh)', 16, 'gen'),
          ('B', 'Domestic energy-related CO2 Emissions (MtCO2)', 8, 'co2')]

def parse_country(wb, r):
    out = []
    for part, anchor, n, block in BLOCKS:
        rows = list(wb[f'{SHEET(r)}_{part}'].iter_rows(values_only=True))
        k = next(i for i, row in enumerate(rows) if row[0] and str(row[0]).strip() == anchor)
        sub = '' if block != 'fec' else None
        for row in rows[k:k+n+1]:
            label = str(row[0]).strip() if row[0] else ''
            if block == 'fec':                                   # final energy: only the rows after 'by fuel'
                if label == 'by fuel':
                    sub = 'fuel'; continue
                if sub != 'fuel' and label != anchor:
                    continue
            for y, v in zip(YEARS, row[1:11]):
                if isinstance(v, (int, float)):
                    out.append({'country': r, 'block': block, 'item': label, 'year': y, 'value': float(v)})
    return out

if __name__ == '__main__':
    wb = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE/'data'/'eu_ref2020_long.parquet'
    df = pd.DataFrame([x for r in EU27 for x in parse_country(wb, r)])
    df.to_parquet(out)
    print('written', out, df.shape)
