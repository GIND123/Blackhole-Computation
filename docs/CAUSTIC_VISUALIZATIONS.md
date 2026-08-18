# Caustic visualization candidates

These figures reconstruct the angle dependent field from the final
\(\ell_{\max}=50\) localized source archives. They remain outside the
manuscript until one is selected. Every candidate has a high resolution PNG,
a PDF, a plain text caption, and a reproducible generation command.

## Sphere and time view

[sphere_time_caustic.png](../results/caustic_visualizations/sphere_time_caustic.png)
shows the outer extraction field at the measured direct pulse and first
antipodal caustic peak. The display position of each sphere encodes retarded
time. It is not a second spatial radius. A common signed symmetric log color
map preserves the amplitude decrease while keeping the echo visible.

The two peaks are measured from the final archive with the same tapered
analytic envelope used by the timing audit. The direct pulse is measured in
the source direction and the next pulse at the antipode. No incomplete later
pulse is labeled as a measured peak.

## Regulator comparison

[regulator_angular_comparison.png](../results/caustic_visualizations/regulator_angular_comparison.png)
compares the outer angular field at one common geometric retarded time for
\(L/M=80,160,320,640\) and Schwarzschild. The top row shows the fields on
\(\mathcal H_c^+\) beside the separately evolved field on \(\mathscr I^+\).
The lower row subtracts that Schwarzschild field from every finite length
field. Each title gives the cosmological horizon radius or the sphere norm of
the residual. The field row and residual row have separate common signed
color scales. Modal samples are linearly interpolated to the stated common
time. No clock translation is fitted.

## Dense cutaway

The cutaway candidate needs radial data at the caustic time. The production
archives contain dense time series at three observers but saved the radial
responses only at the initial and final states. The plotter therefore refuses
to invent a cutaway from those files.

The targeted runner measures the direct and first caustic peaks from the
final archive, converts them to bridge time with the analytic \(q_L\), rounds
only to an exact integration step, and reruns the same \(N_r=2048\),
\(\ell_{\max}=50\), \(\Delta\tau=0.0005M\) calculation. It stores 1024 radial
points at the selected steps. The resulting cutaway uses the physical field
\(\Phi=u/r\) and a documented logarithmic display radius. At the measured
peak phase the field is negative throughout the stored slice. Two negative
meridional contours are revolved about the source axis to form exact
isosurfaces, while the colored slice retains the direct two dimensional
reconstruction.

The existing candidates are regenerated with

    python -m black_hole.caustic_visualizations sphere-time
    python -m black_hole.caustic_visualizations regulator

The targeted \(L/M=80\) archive and cutaway are produced with

    python -m black_hole.caustic_visualizations run-snapshots 80
    python -m black_hole.caustic_visualizations cutaway results/caustic_visualizations/raw/sds_L80.npz

The reconstruction uses the spherical addition theorem applied to the stored
radial response for every \(\ell\). A regression test compares it with the
independent real spherical harmonic reconstruction at several angles to
eleven decimal places.
