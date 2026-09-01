#!/usr/bin/env bash
# Frozen production rerun of the paper-used tail cases.
#
# The screening and diagnostic campaign is run by run_tail_campaign.sh, which
# is kept as the record of what was measured while the length was being
# chosen.  This script is different in one respect that matters: every archive
# it writes is meant to carry production provenance, so it refuses to start
# unless the worktree it runs from is clean.
#
# Run it from a clean, immutable worktree and point OUT at the results tree:
#
#     git worktree add --no-checkout ../bh-frozen <commit>
#     cd ../bh-frozen
#     git sparse-checkout init --cone
#     git sparse-checkout set black_hole tests
#     git checkout
#     OUT=/mnt/e/Blackhole-Computation/results/large_l_tail \
#         bash /mnt/e/Blackhole-Computation/run_tail_production.sh
#
# The case list is generated from large_l_tail.PAPER_LENGTHS rather than
# written out here, so it cannot drift from the frozen contract that
# tail_manifest hashes.
set -u

PY=${PY:-/home/govind/miniforge3/envs/dedalus3/bin/python}
OUT=${OUT:-results/large_l_tail}
LOG=$OUT/logs
JOBS=${JOBS:-4}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "refusing to run: the worktree is dirty, so every archive would be"
    echo "recorded as screening grade.  Commit or stash first."
    git status --porcelain --untracked-files=no
    exit 1
fi
echo "worktree clean at $(git rev-parse --short HEAD)"

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

# Slowest first, so the tail of the schedule is short.  The halved-timestep
# cases are roughly twice the cost of their own resolution at the full step.
CASES=$($PY - <<'PYEOF'
from black_hole.large_l_tail import PAPER_LENGTHS, final_cases

ordered = sorted(
    (case for length in PAPER_LENGTHS for case in final_cases(length)),
    key=lambda case: -case.end_u / case.timestep * case.resolution,
)
print("\n".join(case.name for case in ordered))
PYEOF
)

echo "=== paper-used final ladders ($(date)) ==="
echo "$CASES" | sed 's/^/    /'
echo "$CASES" | xargs -P "$JOBS" -I{} bash -c 'run_case {}'

echo "=== final reports ($(date)) ==="
for L in $($PY -c "from black_hole.large_l_tail import PAPER_LENGTHS; print(' '.join(f'{v:g}' for v in PAPER_LENGTHS))"); do
    $PY -u -m black_hole.large_l_tail --output-dir "$OUT" report-final "$L" \
        > "$LOG/report_final_$L.log" 2>&1 && echo "OK   report-final $L" \
        || echo "FAIL report-final $L"
done

echo "=== regime map ($(date)) ==="
$PY -c "from pathlib import Path; from black_hole import tail_regime_map as m; m.build(Path('$OUT'))" \
    > "$LOG/regime_map.log" 2>&1 && echo OK || echo FAIL

echo "=== production complete ($(date)) ==="
echo "Now rebuild the manifest from the main tree:"
echo "    python -m black_hole.tail_manifest --output-dir $OUT"
echo "    python -m black_hole.tail_manifest --output-dir $OUT --verify"
