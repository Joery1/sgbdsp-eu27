#!/bin/bash
# National-quarterly comparator: main runs, three-way cross-scale evaluation and the e-flow limits (12 runs).
cd "$(dirname "$0")"
PY=${PY:-python}
COMMON="YEAR=2030 EFLOW=1 DELTA=0.05 CAPBOUND=1 DESIGN=2 CO2CAP=1 JOBS=4"
L=results/logs2030; mkdir -p $L
run() { local log=$L/$1.txt; shift; "$@" > $log 2>&1; local rc=$?; echo "$(date +%H:%M:%S) $(basename $log .txt): exit $rc | $(tail -n1 $log)"; }
for pol in 0 0.8138; do
  run crosseval_NA_P$pol $PY crosseval_scale.py $COMMON BASIN=0 CO2CAP_POWER=$pol PRDRY=0
  run crosseval_BQ_P$pol $PY crosseval_scale.py $COMMON BASIN=1 CO2CAP_POWER=$pol PRDRY=0
  run crosseval_NQ_P$pol $PY crosseval_scale.py $COMMON BASIN=0 CO2CAP_POWER=$pol PRDRY=0 NATQ=1
done
for pol in 0 0.8138; do
  for dry in 0 1; do run natq_P${pol}_D$dry $PY model_eu.py $COMMON BASIN=0 CO2CAP_POWER=$pol PRDRY=$dry NATQ=1; done
done
for th in 0.2 0.6; do run natq_theta$th $PY model_eu.py $COMMON BASIN=0 CO2CAP_POWER=0 PRDRY=0 NATQ=1 THETA=$th; done
