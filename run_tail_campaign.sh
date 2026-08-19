#!/usr/bin/env bash
# Large-L tail campaign, run case-parallel with the explicit-potential IMEX split.
#
# Every case is an independent evolution writing its own archive, so the ladder
# parallelizes exactly.  Threading is disabled per process because Dedalus is
# single-threaded here and oversubscription only adds contention.
set -u

PY=/home/govind/miniforge3/envs/dedalus3/bin/python
OUT=results/large_l_tail
LOG=$OUT/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$LOG"

run_case() {
    name=$1
    started=$(date +%s)
    if $PY -u -m black_hole.large_l_tail --output-dir "$OUT" run "$name" \
            > "$LOG/$name.log" 2>&1; then
        echo "OK   $name  $(( ($(date +%s)-started)/60 )) min"
    else
        echo "FAIL $name  (see $LOG/$name.log)"
    fi
}
export -f run_case
export PY OUT LOG

SCREEN="screen_schwarzschild_N1536_dt0p0025
screen_schwarzschild_N2048_dt0p0025
screen_sds_L640_N1536_dt0p0025
screen_sds_L640_N2048_dt0p0025
screen_sds_L1280_N1536_dt0p0025
screen_sds_L1280_N2048_dt0p0025
screen_sds_L2560_N1536_dt0p0025
screen_sds_L2560_N2048_dt0p0025
screen_sds_L5120_N1536_dt0p0025
screen_sds_L5120_N2048_dt0p0025"

# The long cases are ordered slowest-first so the tail of the schedule is short.
FINAL="final_sds_L5120_N2048_dt0p00125
final_schwarzschild_for_L5120_N2048_dt0p00125
final_sds_L5120_N3072_dt0p0025
final_schwarzschild_for_L5120_N3072_dt0p0025
final_sds_L5120_N2048_dt0p0025
final_schwarzschild_for_L5120_N2048_dt0p0025
final_sds_L5120_N1536_dt0p0025
final_schwarzschild_for_L5120_N1536_dt0p0025"

echo "=== screening ($(date)) ==="
echo "$SCREEN" | xargs -P 5 -I{} bash -c 'run_case {}'

echo "=== screening analysis ==="
for L in 640 1280 2560 5120; do
    $PY -u -m black_hole.large_l_tail --output-dir "$OUT" analyze-screen $L \
        > "$LOG/analyze_screen_$L.log" 2>&1 && echo "OK   analyze-screen $L" \
        || echo "FAIL analyze-screen $L"
done

echo "=== final ladder ($(date)) ==="
echo "$FINAL" | xargs -P 4 -I{} bash -c 'run_case {}'

echo "=== final report ($(date)) ==="
$PY -u -m black_hole.large_l_tail --output-dir "$OUT" report-final 5120 \
    > "$LOG/report_final_5120.log" 2>&1 && echo "OK   report-final" \
    || echo "FAIL report-final (see $LOG/report_final_5120.log)"

echo "=== campaign complete ($(date)) ==="
