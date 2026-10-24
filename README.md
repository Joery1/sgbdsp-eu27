# SGBDSP-EU27: a basin-resolved bi-level input–output model for EU water, energy, food and ecosystem planning

Code and input data for:

> Zha, L., Chen, H., Chen, S. (2026). Scale and priority blind spots in water–energy–food–ecosystem planning: a basin-resolved bi-level input–output analysis for the European Union. *Environmental Modelling & Software* (submitted).

The model couples a Stackelberg-type bi-level decentralised stochastic programme (an EU-level leader maximising value added; food, energy and ecosystem followers) with the EU27 part of the EXIOBASE 3.11.2 multi-regional input–output table. It is solved as a hierarchical fuzzy compromise (max–min λ, then a Pareto phase) with linear programmes in HiGHS. Water constraints are set for 412 sub-basin fragments and four quarters:

- a water-stress limit based on the EEA Water Exploitation Index plus (WEI+ ≤ θ, θ = 0.4);
- a vine-copula joint summer reliability constraint for Spain, France and Italy;
- four water-saving technologies (efficient irrigation, dry cooling, wind and solar power, industrial water recycling) and alternative water (desalination, reuse);
- a 2030 baseline calibrated to the EU Reference Scenario 2020.

Repository: https://github.com/Joery1/sgbdsp-eu27 · Results archive: https://github.com/Joery1/sgbdsp-eu27/releases/tag/v1.0.0

## Layout

| Path | Content |
|---|---|
| `eu27/` | Model, input preparation, run scripts and analyses (Python and bash) |
| `eu27/data/` | Shipped public inputs: Eurostat, EEA WEI+ and sub-basin geometry, EU Reference Scenario 2020 extract |
| `copula.py` | Marginal fitting, three-dimensional vine copula, joint violation probabilities |
| `technology.py` | Technology-choice machinery (alternative input columns for adopting nodes); the EU parameters and their sources are set in `model_eu.py` |
| `external/` | Place for EXIOBASE 3.11.2 (not redistributed, see below) |
| `environment.yml`, `requirements.txt` | Python environment |

Generated at run time: `eu27/cache/` (about 0.9 GB of EXIOBASE-derived arrays), `eu27/results/` and `eu27/figures_2030/`.

## Requirements

- **Software:** Python 3.12 with NumPy, pandas, SciPy (≥ 1.9; HiGHS is called through `scipy.optimize.linprog`), Matplotlib, openpyxl and pyarrow; bash for the run scripts. All are free and open source; no commercial solver is needed.
- **Hardware:** no special hardware; the results were produced on a laptop with an Apple M5 processor (10 cores), 32 GB RAM and macOS. Disk: about 0.7 GB for the unzipped EXIOBASE 2017 tables and 0.9 GB for the cache.
- **Run time** on that machine: input preparation a few minutes; all 2030 runs and analyses (`run_all_2030.sh`, about 100 model configurations of 12 scenarios each, 4 scenario processes per configuration, two streams in parallel) about 40 minutes.

Create the environment with conda (or `pip install -r requirements.txt`):

```bash
conda env create -f environment.yml
conda activate sgbdsp-eu27
```

## External data: EXIOBASE 3.11.2

EXIOBASE is released under a customised CC BY-NC-SA licence and is not redistributed here. Download `IOT_2017_ixi.zip` from the EXIOBASE 3.11.2 Zenodo record (https://zenodo.org/records/23039074) and unzip it into `external/IOT_2017_ixi/`, or point the environment variable `EXIOBASE_DIR` to the unzipped folder. The folder must contain `Z.txt`, `Y.txt`, `x.txt` and the satellite folders (`water/`, `air_emissions/`, `nutrients/`, `factor_inputs/`, `land/`).

## Reproducing the results

Run from `eu27/`:

```bash
python prepare_eu.py        # EXIOBASE 2017 -> cache/inputs.npz, results/meta.json, national water-resource series
python basins_eu.py         # 412 sub-basin fragments x 4 quarters (EEA WEI+ 2002-2014) -> results/basins/
python copula_eu.py         # marginals, flow levels and the ES-FR-IT vine copula
python calib2030.py         # 2030 baseline (EU Reference Scenario 2020) -> cache/inputs_2030.npz
python validate_water.py    # EXIOBASE vs Eurostat water abstraction (Table S7)
bash run_all_2030.sh        # every 2030 run, the cross-scale evaluation and all analyses and figures
```

`run_all_2030.sh` calls `python`; set `PY=/path/to/python` to use another interpreter. Logs go to `results/logs2030/`.

The shipped inputs can be regenerated with the data scripts (the paper used the shipped versions; newer downloads may differ because the sources are revised):

```bash
python fetch_eurostat.py --download   # Eurostat env_wat_res, env_wat_abs, env_air_gge, nama_10_gdp
python fetch_wei.py                   # EEA WEI+ sub-basins 2002-2014, outlets and outlines (ArcGIS service)
python prepare_ref2020.py ref2020_energy-transport-ghg.xlsx   # EU Reference Scenario 2020 country results
```

### Running a single configuration

`model_eu.py` takes `KEY=value` arguments and writes `results/scenario_results_<tag>.csv` and `results/province_results_<tag>.csv`. The central configuration of the paper is

```bash
python model_eu.py YEAR=2030 EFLOW=1 DELTA=0.05 CAPBOUND=1 DESIGN=2 CO2CAP=1 BASIN=1 CO2CAP_POWER=0 PRDRY=0 JOBS=4
```

Main switches (defaults in the header of `model_eu.py`; the defaults there are those of the 2017 prototype, so always pass the values above):

| Switch | Meaning |
|---|---|
| `BASIN` | 1 = sub-basin × quarter water layer; 0 = national annual |
| `NATQ` | 1 (with `BASIN=0`) = national × quarter comparator |
| `CO2CAP_POWER` | power-sector CO₂ cap as share of baseline (0 = Reference policy; 0.8138 = Fit-for-55-derived cap) |
| `PRDRY` | 1 = drier hydrological regime (dry/normal/wet probabilities 0.4/0.5/0.1) |
| `THETA` | water-stress limit on WEI+ (0.4 central; 0.2 and 0.6 in sensitivity runs) |
| `TOL`, `LEADTOL` | leader tolerance (share of reference final demand); `LEADTOL=0` drops the leader band |
| `FIXNORM`, `LEADFIX` | save (1) or reuse (2) membership endpoints; fixed leader band |
| `PEN_MODE`, `PEN_SCALE` | shortage penalty basis and multiplier |
| `RHO_IRR` | share of irrigation savings that reduces net consumption (return-flow sensitivity) |
| `CHANCE`, `TECH` | 0 = drop the joint reliability rows / the technology options |

`crosseval_scale.py` takes the same arguments and judges each plan by the national–annual, national–quarterly and sub-basin–quarterly rules (`results/crosseval/`).

## Where the figures and tables come from

| Manuscript item | Script | Output |
|---|---|---|
| Fig. 1, 2, 3, 5, 6, 8; Fig. S1 | `analyze_2030.py` | `figures_2030/fig1_basin_maps`, `fig2_scale`, `fig3_technology_mix`, `fig7_reliability_seasonal`, `fig5_policy_regime`, `fig6_sensitivity`, `fig4_bilevel_vs_single`; tables `results/y2030/T*.csv` |
| Fig. 4 | `fig_tolerance.py` | `figS_leader_tolerance`, `results/table_leader_tolerance.csv` |
| Fig. 7 | `analyze_penalty.py` | `fig8_penalty_sensitivity`, `results/y2030/T9–T11*.csv` |
| Table 3 (3 × 3 cross-evaluation) | `analyze_natq.py` | `results/table_natq_crosseval.csv`, `table_natq_decomposition.csv` |
| Graphical abstract | `graphical_abstract.py` | `figures_2030/graphical_abstract.{png,pdf,tiff}` |
| Tables S3–S25 | analyses above plus `export_si.py`, `analyze_returnflow.py`, `analyze_fixednorm.py`, `analyze_leadfix.py`, `reliability_check.py`, `reliability_fragments.py`, `compare_eea_2023.py`, `capcheck2030.py` | CSV and JSON files in `results/` |

## Data sources and licences

| Data | Source | Licence |
|---|---|---|
| EXIOBASE 3.11.2, 2017 ixi | Zenodo record 23039074 (Stadler et al., 2018) | customised CC BY-NC-SA 4.0; not redistributed |
| EU Reference Scenario 2020, country results | European Commission, DG ENER (`ref2020_energy-transport-ghg.xlsx`); extract in `data/eu_ref2020_long.parquet` | Commission reuse policy (CC BY 4.0) |
| Renewable freshwater resources, water abstraction, GHG emissions, GDP (env_wat_res, env_wat_abs, env_air_gge, nama_10_gdp) | Eurostat API, retrieved 21 October 2026 | Eurostat reuse policy (CC BY 4.0) |
| WEI+ by sub-basin and quarter 2002–2014, outlets, sub-basin and country outlines | EEA ArcGIS service `WaterExploitationIndex`, retrieved 21 October 2026 | EEA (CC BY 4.0) |
| WEI+ country and sub-unit scale 2000–2023 | EEA SDI, doi:10.2909/3c13b52b-36e5-44bc-a07f-8389d43dfd3b and doi:10.2909/b16bd284-f2ec-4164-90b7-674c1de399ba, retrieved 23 October 2026 | EEA (CC BY 4.0) |

All scenario results, summary tables and figures are attached to release v1.0.0 as `sgbdsp-eu27-results-1.0.0.zip` (CC BY 4.0), so tables and figures can be checked without rerunning the model.

## Licence

Code: MIT licence (see `LICENSE`). Shipped data keep the licences of their sources listed above.

## Contact

Lei Zha (lei.zha@mailbox.unideb.hu), University of Debrecen; corresponding author of the article: Hantang Chen (chen.hantang@econ.unideb.hu), University of Debrecen. Please report problems through the repository's issue tracker.
