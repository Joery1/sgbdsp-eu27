#!/bin/bash
# shortage-penalty sensitivity (2030, basin): agricultural-value penalty and scale factors on the default penalty
cd "$(dirname "$0")"
PY=${PY:-python}
COMMON="YEAR=2030 EFLOW=1 DELTA=0.05 CAPBOUND=1 DESIGN=2 CO2CAP=1 JOBS=4"
mkdir -p results/logs2030
run() { local tag=$(echo "$*" | tr ' =' '_-'); $PY model_eu.py $COMMON "$@" > "results/logs2030/$tag.txt" 2>&1; echo "$tag: $(tail -n1 results/logs2030/$tag.txt)"; }
# central penalty = agricultural VA per m3 (PEN_MODE=1); variants span ~0.3 to ~520 EUR/m3
for pen in "PEN_SCALE=0.1" "PEN_SCALE=10" "PEN_MODE=0" "PEN_MODE=0 PEN_SCALE=10"; do
  run BASIN=1 CO2CAP_POWER=0 PRDRY=0 $pen
  run BASIN=1 CO2CAP_POWER=0 PRDRY=1 $pen
  run BASIN=1 CO2CAP_POWER=0.8138 PRDRY=0 $pen
done
