"""Locations of the external inputs that are not shipped with the code.

EXIOBASE_DIR  EXIOBASE 3.11.2 industry-by-industry tables for 2017 (folder IOT_2017_ixi, https://zenodo.org/records/23039074).
              Default: <repository>/external/IOT_2017_ixi.
EU_REF2020    EU Reference Scenario 2020 country results in long format (country, block, item, year, value), made by
              prepare_ref2020.py from the Commission's workbook. Default: data/eu_ref2020_long.parquet (shipped).
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXIO = Path(os.environ.get('EXIOBASE_DIR', HERE.parent/'external'/'IOT_2017_ixi'))
REF2020 = Path(os.environ.get('EU_REF2020', HERE/'data'/'eu_ref2020_long.parquet'))
