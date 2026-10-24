#!/bin/bash
# 2030 experiment (EU27): reference baseline (EU Ref 2020), e-flow limit, basin x quarter,
# 12 scenarios per run (3 joint-reliability levels x 4 alternative-water options).
# Run axes: scale (basin / national) x carbon policy (Ref / Fit-for-55 power) x hydrological regime.
cd "$(dirname "$0")"
PY=${PY:-python}
COMMON="YEAR=2030 EFLOW=1 DELTA=0.05 CAPBOUND=1 DESIGN=2 CO2CAP=1 JOBS=4"
mkdir -p results/logs2030
run() { local tag=$(echo "$*" | tr ' =' '_-'); $PY model_eu.py $COMMON "$@" > "results/logs2030/$tag.txt" 2>&1; echo "$tag: $(tail -n1 results/logs2030/$tag.txt)"; }

for scale in 1 0; do
  for pol in 0 0.8138; do
    for dry in 0 1; do
      run BASIN=$scale CO2CAP_POWER=$pol PRDRY=$dry
    done
  done
done
for pol in 0 0.8138; do
  for dry in 0 1; do
    run BASIN=1 CO2CAP_POWER=$pol PRDRY=$dry TECH=0
  done
done
# one-at-a-time sensitivities on the basin / Ref / historical run
for s in THETA=0.2 THETA=0.6 REN_CAP=2 IRR_COST=2 IRR_CAP=0.5 INFRA=0.1 DELTA=0.10 TOL=0.06 IRR_SAVE=1.5 REN_COST=0.7; do
  run BASIN=1 CO2CAP_POWER=0 PRDRY=0 $s
done
# technology effect without the reliability rows (the no-technology model is infeasible with them)
for tech in 1 0; do
  for pol in 0 0.8138; do
    run BASIN=1 CO2CAP_POWER=$pol PRDRY=0 CHANCE=0 TECH=$tech
  done
done
# national scale at the alternative e-flow limits (scale contrast across theta)
for th in 0.2 0.6; do
  run BASIN=0 CO2CAP_POWER=0 PRDRY=0 THETA=$th
done
