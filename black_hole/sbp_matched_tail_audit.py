"""Compact manuscript audit accompanying the new matched-tail figure.

Signed-waveform refinement comparisons and frozen-code discrepancies are
kept separate.  A discrepancy with a frozen archive is not an exact error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .large_l_tail import rms_envelope
from .sbp_matched_tail_figure import FAMILIES, REVISION_ROOT, WINDOWS, load_members
from .sbp_pilot_analysis import (
    REPOSITORY, archived_path, exact_difference, interpolate, interpolation_sensitivity,
    metrics, read_archive, uniform_grid, write_csv,
)


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def build(root: Path, output: Path):
    root,output = root.resolve(),output.resolve()
    if root not in output.parents or any(output == root/n or root/n in output.parents
                                         for n in ("runs", "operators")):
        raise ValueError("Audit output must be a derived directory inside the revision root")
    members = load_members(root)
    intervals = {r["family"]:r for r in read_csv(output/"matched_tail_intervals.csv")}
    sweep = read_csv(output/"matched_tail_estimator_sweep.csv")
    refinement = read_csv(output/"matched_tail_refinement.csv")
    summary,legacy,accepted,inputs = {},[],[],{}
    for family in FAMILIES:
        stem=family.stem
        ladder=members[stem]
        fine=ladder["half_dt"]
        interval=intervals[stem]
        widths=[float(r["duration_over_M"]) for r in sweep if r["family"]==stem]
        controls={kind:[{k:float(row[k]) for k in ("start_U_over_M","end_U_over_M","relative_L2")}
                        for row in refinement if row["family"]==stem and row["comparison"]==kind]
                  for kind in ("spatial_coarse","spatial","timestep","output_cadence")}
        entry=dict(interval=interval,estimator_duration_range=[min(widths),max(widths)],
                   refinement_by_window=controls,
                   refinement_maxima={kind:max(row["relative_L2"] for row in rows)
                                      for kind,rows in controls.items()})
        archive=archived_path(fine)
        if archive is None:
            raise FileNotFoundError(f"No matched frozen archive: {stem}")
        old_times,old_signal,_=read_archive(archive)
        times=uniform_grid(0.,min(1000.,fine.times[-1],old_times[-1]),.05)
        new=interpolate(fine.times,fine.u,times)
        old=interpolate(old_times,old_signal,times)
        new_envelope=rms_envelope(times,new,10.,floor_multiplier=0.)
        old_envelope=rms_envelope(times,old,10.,floor_multiplier=0.)
        interpolation_change=interpolation_sensitivity(old_times,old_signal,times)
        for start,end in WINDOWS:
            waveform=metrics(times,new-old,old,start,end)
            envelope=metrics(times,new_envelope-old_envelope,old_envelope,start,end)
            cadence=metrics(times,interpolation_change,old,start,end)
            legacy.append(dict(family=stem,start_U_over_M=start,end_U_over_M=end,
                               relative_waveform_L2=waveform["relative_L2"],
                               relative_RMS_envelope_L2=envelope["relative_L2"],
                               frozen_half_cadence_sensitivity=cadence["relative_L2"]))
        own_legacy=[row for row in legacy if row["family"]==stem]
        entry["frozen_waveform_discrepancy_range"]= [min(r["relative_waveform_L2"] for r in own_legacy),
                                                     max(r["relative_waveform_L2"] for r in own_legacy)]
        entry["frozen_comparison_by_window"]=own_legacy
        if interval["passes_price_criterion"]=="True":
            start,end=float(interval["start_U_over_M"]),float(interval["end_U_over_M"])
            for kind,first,second in (("spatial_coarse","coarse","medium"),
                                      ("spatial","medium","fine"),
                                      ("timestep","fine","half_dt")):
                clock,difference,reference=exact_difference(ladder[first],ladder[second])
                accepted.append(dict(family=stem,comparison=kind,start_U_over_M=start,
                                     end_U_over_M=end,**metrics(clock,difference,reference,start,end)))
            entry["accepted_interval_refinement"]=[row for row in accepted if row["family"]==stem]
        inputs[str(archive.relative_to(REPOSITORY))]=hashlib.sha256(archive.read_bytes()).hexdigest()
        summary[stem]=entry
    write_csv(output/"matched_tail_frozen_comparison.csv",legacy)
    write_csv(output/"matched_tail_accepted_interval_refinement.csv",accepted)
    result=dict(
        statement="All numbers are fractions. Adjacent-degree and fixed-degree timestep waveform changes are observed comparisons, not rigorous absolute-error bounds. Frozen discrepancies use the frozen waveform in the denominator, with no fitted alignment.",
        physical_protocol="ell=1, same velocity bump, coupling, profile, minimal foliation and geometric U=tau-q",
        families=summary,
        additional_frozen_inputs_sha256=inputs,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output/"matched_tail_audit_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=REVISION_ROOT)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.root,args.output_dir or args.root/"analysis"),indent=2))


if __name__=="__main__":
    main()
