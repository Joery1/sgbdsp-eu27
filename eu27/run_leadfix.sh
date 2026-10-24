#!/bin/bash
# Decoupling test for Section 3.4: does the response to fixed normalisation come from the fixed reference points or
# from the leader's tolerance band (1-lambda) t widening as lambda falls? Runs the central and drier regimes (and the
# penalty x0.1 and x10) under standard and fixed normalisation with the leader's band fixed at its central width
# (LEADFIX=1) and additionally at its central centre (LEADFIX=2). Needs the FIXNORM=1 reference run, which is repeated
# first so that results/leader_ref/ exists. Then analyze_leadfix.py.
cd "$(dirname "$0")"
PY=${PY:-python}
C="YEAR=2030 EFLOW=1 DELTA=0.05 CAPBOUND=1 DESIGN=2 CO2CAP=1 JOBS=4 BASIN=1 CO2CAP_POWER=0"
L=results/logs_leadfix; mkdir -p $L
run() { local log=$L/$1.txt; shift; "$@" > $log 2>&1; local rc=$?; echo "$(date +%H:%M:%S) $(basename $log .txt): exit $rc | $(tail -n1 $log)"; }

run fixnorm1 $PY model_eu.py $C PRDRY=0 FIXNORM=1
for lf in 1 2; do
  for dry in 0 1; do
    run std_dry${dry}_lf${lf} $PY model_eu.py $C PRDRY=$dry LEADFIX=$lf
    run fix_dry${dry}_lf${lf} $PY model_eu.py $C PRDRY=$dry FIXNORM=2 LEADFIX=$lf
  done
  for pen in 0.1 10; do
    run std_pen${pen}_lf${lf} $PY model_eu.py $C PRDRY=0 PEN_SCALE=$pen LEADFIX=$lf
    run fix_pen${pen}_lf${lf} $PY model_eu.py $C PRDRY=0 FIXNORM=2 PEN_SCALE=$pen LEADFIX=$lf
  done
done
run analyze_leadfix $PY analyze_leadfix.py
