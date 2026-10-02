# Private retained-operator scope audit

Configuration inspection for the final focused revision, 2026-09-26. No new
evolutions, production-data edits, manuscript edits, or commits were performed
for this audit. Static operator actions are measured separately by the new
`black_hole/uniform_operator_scope.py`, which stops the actual solver
construction before any timestep. New regression results belong in
`revision_notes_minor.md`; the observations below do not certify cutoff
insensitivity.

After the user reported completion, the single bounded N128 uniform-L640
static audit was verified complete under
`results/uniform_operator_scope_minor_v1/benchmark_L640_N128`. It tests
production, 1e-10/0, 1e-14/0, and exactly zero cutoffs. A representative
fine-grid N768 uniform-L640 production/tight static check then completed
under `flat_L640_N768`, without timesteps. Following the root agent's scope
instruction, no all-member flat-sequence or crossover operator campaign is
being launched. The broader finite-time cutoff-sensitivity question remains
open; neither initial-state operator agreement nor constraint preservation
certifies those waveforms or their extrapolants.

## Resolved paths and settings

| Retained calculation | Evidence and discretization | Operator cutoffs and IMEX split |
|---|---|---|
| Uniform fixed-mode sequence and Schwarzschild controls | `results/regulator_production_v3/raw/flat/{coarse,medium,fine}/*.npz`; archive metadata and `black_hole/regulator_suite.py:flat_numerical/run_case`; frozen source at local Git object `2460d976fd023f7bcae892d760436248d32d0290:black_hole/sds_solver.py` | Frozen source calls `build_solver(RK222)` without cutoff overrides. Archived Dedalus version is 3.0.5, whose actual defaults resolve to `ncc_cutoff=1e-6`, `entry_cutoff=1e-12`, `max_ncc_terms=None`. All transport and the potential are implicit; there is no explicit physical RHS. Chebyshev T, dealias 1.5; N=384,512,768 and dt=.005,.00375,.0025; tau endpoint 200, signal cadence .03. |
| Uniform minimal dipole L=3072 and its Schwarzschild controls | `results/large_l_tail/raw/final_{sds_L3072,schwarzschild_for_L3072}_N{1536,2048,3072}_dt0p0025.npz`; `dedalus_matrix_assembly`, `imex_split`, and `numerical` are explicitly stored | `ncc_cutoff=1e-6`, `entry_cutoff=1e-12`; implicit transport, explicit potential. RK222, dealias 1.5, dt=.0025, with N2048 half-dt=.00125. Signal cadence .05, snapshots 25, no constraint damping. Velocity bump r=6, half-width 3, u=psi=0, pi=G/A. Archives record Dedalus 3.0.5 and simulation commit d43f25ae; that Git object is not available locally, so the stored metadata is the direct configuration evidence. |
| Finite-radius crossover | `black_hole/crossover_final.py:Archives` chooses `results/sds_scalar/tails/crossover_final/raw` before `high_resolution_rates/raw`; the principal uniform L80/160 files are in the latter, while the extended Schwarzschild N2048 file is in the former | Source at locally available commit `166c01a:black_hole/sds_solver.py` calls the same standard-flux solver with defaults, potential implicit. `high_resolution_tail_rates.py:run_resolution_case` does not request explicit potential. Actual archives have RK222, dealias1.5, signal .05, snapshots20. Primary N2048 dt=.0025: SdS tau endpoints 410.495247 (L80), 810.240807 (L160); extended Schwarzschild tau830. These older archives do **not** store Dedalus version/cutoff/provenance fields. The code path resolves the usual defaults 1e-6/1e-12, but runtime provenance is less complete than the v3 or L3072 archive. |
| Localized-source production, including D1 | `results/regulator_production_v3/raw/source` and `black_hole/source_evolution.py`; fine archive explicitly says finite-difference order8 and Dedalus=null | Independent eighth-order finite-difference derivative and classical RK4, no Dedalus matrix or NCC assembly. Coarse/medium/fine (Nr,dt,lmax)=(1024,.001,42),(1536,1/1500,46),(2048,.0005,50). The Dedalus cutoff mechanism is not applicable to this production path. |
| Historical sourced Dedalus cross-check | `results/caustic_production_v2/cross_code/dedalus/{schwarzschild,sds_L80}.npz`, `black_hole/dedalus_source_evolution.py:219-225` | Separate implementation, Chebyshev T N512, dealias1.5, RK443, dt=.002, tau72; potential and transport implicit, source explicitly stage-evaluated. `build_solver` has no cutoff override; archive version3.0.5 therefore resolves 1e-6/1e-12. Numerical metadata retains an irrelevant generic finite_difference_order field; the explicit `radial_discretization.backend` is the correct backend discriminator. |
| Refined exterior ringdown and matched five-signal tails | `results/revision_sbp_ringdown_v1`, `results/sbp_matched_tail_revision_v1`, `black_hole/sbp_hermite.py`, `sbp_geometry.py`, and the high-precision backend | Independent continuous LGL/SBP multidomain weak form and linear Radau IIA3 propagation, not a Dedalus IMEX assembly. No NCC or matrix-entry pruning cutoffs. These validated campaigns are not to be repeated. |

The standard Dedalus uniform/Schwarzschild path uses separate assembled
equations for `u`, `psi`, `pi`, with outgoing flux `A*(B*psi+pi)` and
`C=psi-D_rho u`. It does not enter the exterior-only cancellation-free
`1+B` path or the newer shared-flux characteristic path. All use natural
regular/outflow endpoints, without added boundary conditions or tau boundary
rows. Thus the **assembly mechanism is shared**, but neither its size nor its
finite-time waveform consequence follows from the exterior result alone.

The coefficient cutoff applies to implicit nonconstant-coefficient spectral
expansions, and `entry_cutoff` is the actual Dedalus keyword for dropping
assembled matrix entries. They are not resolution, arithmetic precision,
dealiasing, or timestep controls.

## Version resolution

The active laptop environment contains Dedalus **3.0.4**, not the 3.0.5
recorded by the frozen v3 and L3072 archives. Its `core/solvers.py:59` gives
the same defaults. These defaults were independently checked against the
official upstream 3.0.5 source:

<https://raw.githubusercontent.com/DedalusProject/dedalus/v3.0.5/dedalus/core/solvers.py>

The archived v3 fixed-mode source itself is locally available and explicitly
contains the fully implicit potential. A new cutoff comparison against the
old archive must distinguish a library/runtime difference from a cutoff
difference; an in-environment baseline at production cutoffs is the cleanest
way to do so if archive differences are not demonstrably negligible.

The official GitHub `v3.0.4...v3.0.5` comparison was also inspected. It contains
six commits and no changes to solvers, timesteppers, arithmetic, or matrix
solvers. Basis changes concern disk/ball constant modes and spherical spin
coupling, not the radial Chebyshev basis. The only potentially shared assembly
change moves `dtype=int` from `np.concatenate` to `sparse.diags` when building
the valid-equation/variable selectors, leaving their values unchanged. An
operator change adds an explicit dtype to diagonal Fourier-symbol matrices,
not Chebyshev derivatives. No changed radial-operator mathematics was found;
platform and NumPy/SciPy differences still mean bitwise identity is not assumed.

## Existing evidence worth reusing, with limits

* The previous genuine cutoff sweep is the **exterior**, N128, initial/short
  diagnostic under `results/ringdown_constraint_revision_v1/diagnostic`.
  It establishes the exterior transport mechanism but is not a uniform or
  long-tail trajectory comparison. No compatible preexisting uniform cutoff
  sweep was found in the repository filename/configuration search.
* `black_hole/ringdown_constraint_diagnostic.py:instrument_solver` measures
  actual assembled implicit/explicit RHS and initial stages. It is reusable,
  but its S0 normalization is displacement-specific. For the dipole velocity
  data it vanishes and must not be reused. Archives preserve only u snapshots,
  not full evolved psi/pi states, so those snapshots alone cannot reconstruct
  actual historical evolved-state C or the full historical RHS.
* All 21 v3 fixed-mode archives have very small **stored endpoint** C norms:
  the overall maximum is 9.98989684e-9 (medium L40); fine Schwarzschild is
  3.37940688e-10 and fine L640 is 5.30043718e-10. These archives sample C only
  at tau0 and200 (medium also199.99875); these are not dense time histories,
  and their magnitude is not a waveform error bound.
* L3072 stored C samples through approximately U450 have maxima
  3.34509498e-9 for SdS N2048 and 2.41098491e-9 for its Schwarzschild control;
  corresponding N3072 values are 9.95867142e-9 and 8.03463039e-9. The whole
  stored records are similarly small, but this cannot certify tail-relative
  accuracy or replace the requested cutoff regression. Stored q values are
  2.7699892491613465 and 2.772588722239781, respectively.
* Finite-radius primary archived C maxima are 6.87272933e-9 (L80 N2048),
  2.67608107e-9 (L160 N2048), and 4.39906500e-9 (extended Schwarzschild).
  These are absolute reduction-constraint observations, not envelope errors.
* Historical sourced Dedalus cross-checks have sampled C maxima
  1.99533611e-10 (Schwarzschild) and 4.38031556e-10 (L80). The retained
  waveform comparison is independently checked against finite differences;
  it must not be relabeled as a dedicated cutoff study.
* `results/revision_sbp_ringdown_v1/analysis/compact_summary.csv` already
  compares refined independent uniform L320/640 and Schwarzschild waveforms
  to frozen fixed-mode controls. Maximum uniform frozen/new waveform change
  over the three ringdown windows is 4.47076413e-5 and maximum paired residual
  change 4.71251921e-5, normalized by the fine Schwarzschild window signal.
  Over U0--80 the uniform differences are 3.68484433e-5 (L320) and
  3.16632900e-5 (L640), and paired residual differences 2.56165875e-5 and
  3.11587304e-5. These include different spatial/time methods, not a pure
  cutoff perturbation. They provide finite-time independent evidence only
  through U80; they do not validate cumulative U160 or the complete nested
  extrapolants.
* The original L3072 envelope-refinement scales on the accepted Price window
  are directly available in `final_L3072_numerical_sensitivities.csv`: SdS
  N2048->3072 4.76901981e-5, N1536->2048 1.27098197e-4, half-dt
  3.29037117e-8; Schwarzschild 2.19936319e-5, 3.57924806e-4,
  3.35527681e-8. The new cutoff comparison must be judged against these
  existing scales and the unchanged interval rule, not a new chosen tolerance.

## Consequences for the final revision

The L3072 benchmark requires the targeted matched finite-time regression;
small archived absolute C or a roundoff-sized initial RHS defect is not
sufficient. A U450 check leaves the long record near U1984 unaudited for
cutoff sensitivity. For fixed-mode cumulative/extrapolated claims, a complete
waveform-weight calculation must include every tested triple member and its
Schwarzschild control; the U80 independent controls cannot stand in for
missing U160 or unchecked-member changes. There is no existing evidence
justifying a blanket sentence that all uniform runs are unaffected.

## Completed static uniform-L640 benchmark

The N128 fully implicit quadrupole check used the actual compact displacement
data and a smooth constraint-satisfying polynomial displacement with nonzero
independent momentum. It evaluated the actual assembled mass-matrix-inverted
implicit and explicit RHS and interpolated the resulting constraint defect
to both endpoints. No evolution steps were taken. Production settings gave
grid C_rhs norms 2.36093e-13 (actual data) and 2.23934e-16 (polynomial);
explicit forcing is exactly zero. Tight and zero cutoffs leave defects at
floating-point cancellation scale rather than revealing the exterior-like
transport defect.

This does **not** make the full operators cutoff-independent. The relative
Euclidean norms of the complete RHS changes, over all three grid fields, are:

| State | production -> 1e-10/0 | 1e-10/0 -> 1e-14/0 | 1e-14/0 -> zero |
|---|---:|---:|---:|
| Actual compact datum | 7.26475e-6 | 1.17092e-10 | 5.38475e-13 |
| Constrained polynomial/momentum | 2.43548e-5 | 1.38703e-9 | 3.06628e-11 |

These are state-dependent operator-action diagnostics, not waveform norms or
finite-time error bounds. Production-to-tight momentum RHS changes reach
5.57283e-4 on the actual datum, while the initial u/psi RHS changes are at
roundoff. The full equation can therefore change despite an accurately
preserved reduction constraint. Production, tight, near-roundoff, and zero
assemblies/actions took 92.1, 62.5, 62.3, and 75.3 seconds respectively.

## Completed representative fine-grid check; broader audit not completed

The actual fine N768, ell=2, uniform L640 fully implicit assembly gives
production C_rhs norms 2.14526e-13 (actual datum) and 3.12165e-16
(polynomial/momentum). With NCC=1e-10 and entry=0, these are 7.11331e-13
and 1.54617e-16; the endpoint values are of the same size. Explicit RHS is
exactly zero throughout. No exterior-like reduction-identity defect was found
on these two states, at either cutoff. The static actions nevertheless change:
complete RHS relative L2 differences are 8.16446e-6 and 2.43548e-5, respectively.
The actual-datum momentum RHS maximum change is 6.95751e-4; u/psi RHS
changes are 9.16e-17 and 1.54e-13. These numbers reinforce that reduction
constraint cancellation and cutoff insensitivity of the wave evolution are
different questions. The two assemblies/actions took 49.8 and128.0 seconds.

For both completed audit packages, the action archive hash is correct, all
stored arrays are finite, and the two input states are bitwise identical
across cutoffs. The N768 driver and production-solver hashes also verify.
After N128 finished, the driver gained only a `--production-tight-only`
selection option to avoid high-N near-roundoff matrix fill-in; its mathematical
`measure` implementation was unchanged. Both packages explicitly record zero
evolution steps. No existing raw production archives were overwritten.

Only this representative fine-grid member was audited. Remaining flat members,
the Schwarzschild fixed-mode operator, and crossover-specific operators were
not subjected to new static cutoff sweeps. No fixed-mode cutoff trajectory
through U160 or weighted nested-extrapolant sensitivity was measured. Those
questions remain unresolved rather than being inferred from the L640 initial
operator, the U80 SBP controls, or the separate L3072 dipole trajectories.
