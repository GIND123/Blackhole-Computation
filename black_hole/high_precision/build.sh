#!/bin/sh
set -eu
# No downloads and no changes to the external QD checkout.  Invoke with sh.
backend_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$backend_dir" in /*/black_hole/high_precision) ;; *) echo "Unexpected backend path" >&2; exit 1;; esac
build_dir="$backend_dir/_build"
qd_revision=b1c8ddfd2d4a0f0901a88491524df728a26cbe8e
qd_source=${QD_SOURCE:-/Users/anilzenginoglu/Documents/projects/26-Sagnik-Silva/hyper_bhpt_ern_ct/src/hyperbhpt_ern_python/_build/qd-source}
mode=${1:-dd}
case "$mode" in dd|double) ;; *) echo "Usage: sh build.sh [dd|double]" >&2; exit 1;; esac
mkdir -p "$build_dir"
if [ "$mode" = double ]; then
    "${CXX:-c++}" -O3 -std=c++17 -ffp-contract=off -fno-fast-math -DUSE_DOUBLE \
        "$backend_dir/hermite_banded.cpp" -o "$build_dir/hermite_banded_double"
    echo "$build_dir/hermite_banded_double"
else
    if [ ! -d "$qd_source/.git" ]; then echo "Set QD_SOURCE to an existing pinned QD checkout." >&2; exit 1; fi
    actual_revision=$(git -C "$qd_source" rev-parse HEAD)
    if [ "$actual_revision" != "$qd_revision" ]; then echo "Expected QD commit $qd_revision" >&2; exit 1; fi
    "${CXX:-c++}" -O3 -std=c++17 -ffp-contract=off -fno-fast-math \
        -I"$backend_dir" -I"$qd_source/include" "$backend_dir/hermite_banded.cpp" \
        "$qd_source/src/dd_real.cpp" "$qd_source/src/dd_const.cpp" \
        "$qd_source/src/util.cpp" "$qd_source/src/bits.cpp" -o "$build_dir/hermite_banded_dd"
    echo "$build_dir/hermite_banded_dd (QD $qd_revision)"
fi
