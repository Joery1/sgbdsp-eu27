#!/bin/bash
# Every 2030 run behind the manuscript and the Supplementary Material, in two parallel streams, then all analyses.
# Stream A: run_2030.sh (main, no technology, sensitivity, technology effect, national theta), run_penalty.sh,
#           return-flow sensitivity. Stream B: leader tolerance, cross-scale evaluation, fixed normalisation,
#           reliability compliance, adoption-limit check.
cd "$(dirname "$0")"
PY=${PY:-python}
COMMON="YEAR=2030 EFLOW=1 DELTA=0.05 CAPBOUND=1 DESIGN=2 CO2CAP=1 JOBS=4"
L=results/logs2030; mkdir -p $L
run() { local log=$L/$1.txt; shift; "$@" > $log 2>&1; local rc=$?; echo "$(date +%H:%M:%S) $(basename $log .txt): exit $rc | $(tail -n1 $log)"; }

streamA() {
  bash run_2030.sh
  bash run_penalty.sh
  for rho in 0.5 0.25; do
    for pol in 0 0.8138; do
      for b in 0 1; do run crosseval_B${b}_P${pol}_RHO${rho} $PY crosseval_scale.py $COMMON BASIN=$b CO2CAP_POWER=$pol PRDRY=0 RHO_IRR=$rho; done
      for t in TOL=0 TOL=0.06 LEADTOL=0; do run rho${rho}_P${pol}_${t/=/-} $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=$pol PRDRY=0 RHO_IRR=$rho $t; done
    done
  done
}
streamB() {
  for pol in 0 0.8138; do
    for tol in 0 0.015 0.03 0.06 0.10; do run tol_P${pol}_TOL${tol} $PY model_eu.py $COMMON BASIN=1 PRDRY=0 CO2CAP_POWER=$pol TOL=$tol; done
    run leadtol0_P${pol} $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=$pol PRDRY=0 LEADTOL=0
    for b in 0 1; do run crosseval_B${b}_P${pol} $PY crosseval_scale.py $COMMON BASIN=$b CO2CAP_POWER=$pol PRDRY=0; done
  done
  run fixnorm1 $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=0 PRDRY=0 FIXNORM=1
  run fixnorm2 $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=0 PRDRY=0 FIXNORM=2
  run fixnorm2_pen0.1 $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=0 PRDRY=0 FIXNORM=2 PEN_SCALE=0.1
  run fixnorm2_pen10 $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=0 PRDRY=0 FIXNORM=2 PEN_SCALE=10
  run fixnorm2_drier $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=0 PRDRY=1 FIXNORM=2
  for lf in 1 2; do                                                     # decoupling of the leader band (run_leadfix.sh)
    for d in 0 1; do
      run lf${lf}_std_dry$d $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=0 PRDRY=$d LEADFIX=$lf
      run lf${lf}_fix_dry$d $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=0 PRDRY=$d FIXNORM=2 LEADFIX=$lf
    done
    for pen in 0.1 10; do
      run lf${lf}_std_pen$pen $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=0 PRDRY=0 PEN_SCALE=$pen LEADFIX=$lf
      run lf${lf}_fix_pen$pen $PY model_eu.py $COMMON BASIN=1 CO2CAP_POWER=0 PRDRY=0 FIXNORM=2 PEN_SCALE=$pen LEADFIX=$lf
    done
  done
  for pol in 0 0.8138; do run reliability_P${pol} $PY reliability_check.py $COMMON BASIN=1 CO2CAP_POWER=$pol PRDRY=0; done
  run capcheck $PY capcheck2030.py "$(pwd)"
  cp $L/capcheck.txt results/capcheck_2030.txt
}
echo "start $(date +%H:%M:%S)"
streamA > $L/_streamA.txt 2>&1 &
streamB > $L/_streamB.txt 2>&1 &
wait
bash run_natq.sh > $L/_natq.txt 2>&1                                  # national-quarterly comparator
echo "runs done $(date +%H:%M:%S)"
C="YEAR=2030 EFLOW=1 DELTA=0.05 CAPBOUND=1 DESIGN=2 CO2CAP=1 BASIN=1 PRDRY=0"
run export_si $PY export_si.py
run analyze_2030 $PY analyze_2030.py
run analyze_natq $PY analyze_natq.py
run analyze_penalty $PY analyze_penalty.py
run fig_tolerance $PY fig_tolerance.py
run graphical_abstract $PY graphical_abstract.py
run analyze_returnflow $PY analyze_returnflow.py
run analyze_fixednorm $PY analyze_fixednorm.py
run analyze_leadfix $PY analyze_leadfix.py
run compare_eea_2023 $PY compare_eea_2023.py
run reliability_fragments $PY reliability_fragments.py
run membership_central $PY membership_check.py $C CO2CAP_POWER=0
run membership_pen10 $PY membership_check.py $C CO2CAP_POWER=0 FIXNORM=2 PEN_SCALE=10
run membership_drier $PY membership_check.py YEAR=2030 EFLOW=1 DELTA=0.05 CAPBOUND=1 DESIGN=2 CO2CAP=1 BASIN=1 PRDRY=1 CO2CAP_POWER=0 FIXNORM=2
echo "all done $(date +%H:%M:%S)"
