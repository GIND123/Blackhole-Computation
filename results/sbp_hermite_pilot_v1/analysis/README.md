# Independent SBP–Hermite pilot assessment

No production archive or manuscript figure was changed.

The new solver is tested against independent spatial, timestep, layout, and arithmetic controls where available. Missing controls are not inferred from high precision.

Waveforms use U=tau-q with no fitted shift or amplitude rescaling. New-run differences are subtracted as Decimal values on shared nominal tau samples (printed times matched within 1e-24 M). Archive comparisons use cubic interpolation and report the full/half-cadence sensitivity separately.

The 10M RMS / 40M logarithmic-fit estimator is reused unchanged. The signed instantaneous index -Uv/u is additional diagnostic information. Diagnostic band durations are not the full archived acceptance criterion: they do not include a reference comparison and measured refinement floor.

## Selected finest pilot members

- exterior640_minimal_p40_wave_radau3_dt0p05_U1000: p=40, 641 nodes, dt=0.05, double-double, radau3; U ends at 1000. Available control types: layout, spatial, timestep. See `exterior_L640_xi0_ell1_radau3.pdf`.
- exterior640_conformal_p40_wave_radau3_dt0p05_U1000: p=40, 641 nodes, dt=0.05, double-double, radau3; U ends at 1000. Available control types: arithmetic, layout, spatial, timestep. See `exterior_L640_xi1o6_ell1_radau3.pdf`.
- schwarzschild_p32_dt0p01_U500: p=32, 289 nodes, dt=0.01, double-double, hermite4; U ends at 500. Available control types: arithmetic, integrator, layout, spatial, timestep. See `schwarzschild_ell1_hermite4.pdf`.
- schwarzschild_p40_radau3_dt0p05_U1000: p=40, 361 nodes, dt=0.05, double-double, radau3; U ends at 1000. Available control types: arithmetic, integrator, layout, spatial, timestep. See `schwarzschild_ell1_radau3.pdf`.
- uniform3072_minimal_p40_radau3_dt0p05_U1000: p=40, 361 nodes, dt=0.05, double-double, radau3; U ends at 1000. Available control types: spatial, timestep. See `uniform_L3072_xi0_ell1_radau3.pdf`.
- uniform640_conformal_p40_radau3_dt0p05_U1000: p=40, 361 nodes, dt=0.05, double-double, radau3; U ends at 1000. Available control types: arithmetic, spatial, timestep. See `uniform_L640_xi1o6_ell1_radau3.pdf`.

## Explicitly excluded pilot runs

These raw histories remain untouched but are excluded from all tables, comparisons, and figure selection. Only figures listed in this report and its manifest belong to this assessment; older unlisted derived files are not evidence.

- `schwarzschild_p24_radau3_dt0p1_U1000`: Superseded preliminary backend (82c6390), before robust decimal parsing. Excluded conservatively together with the affected uniform run; original output retained.
- `uniform640_conformal_p24_radau3_dt0p1_U1000`: Preliminary backend (82c6390), before robust decimal parsing, disagrees with independent band-LU Radau applied to the identical operator. Rechecked with corrected backend aa347daa; original output retained. Not spatial truncation error.

## Reading the tables

- `independent_controls.csv`: one factor changed at a time; observed differences, not Richardson error estimates.
- `archived_waveform_differences.csv`: relative L2 differences to the finest frozen matched waveform. Values are fractions, not percentages.
- `power_index_samples.csv`: instantaneous and archived-estimator indices at U=200, 300, 500, and 900 where available.
- `diagnostic_price_intervals.csv`: the longest unrefined fit interval in each band, starting after U=100. These are not publication acceptance intervals.
- `run_inventory.csv`: completed runs, endpoint amplitudes, precision, and timing.

Replacing a figure requires resolved spatial, temporal, and interface/layout accuracy, not just a smoother curve or agreement of two high-precision outputs.
