# Refined SBP–Radau tail pilot

The overview uses p=40, dt=0.05M double-double Radau3 waveforms. Exterior calculations use the wave-resolved layout. Spatial controls compare p=32 and p=40 at dt=0.1M on that same layout; timestep controls compare dt=0.1M and 0.05M at p=40.

Transition-layer subdivisions plus areal-radius boundaries r/M=80,120,160,200 to resolve the incoming wave before the transition; physical profile unchanged.

The upper two panels show the signed instantaneous index -U v/u, not an RMS fit or a smoothed derivative. Zeros and immediately adjacent samples are masked. Excursions outside the displayed 0–8 range are clipped by the axes, not used to infer tail intervals. The shaded band is 3 ± 10%.

The bottom panel takes the larger observed spatial or timestep waveform difference and divides by the finest 10M RMS amplitude. This is an observed comparison, not a proved error bound. The CSV lists the independent spatial and temporal relative L2 differences separately.

The legacy comparison table reports both the signed-waveform L2 difference and the 10M RMS-envelope L2 difference from the actual finest frozen archive. It also reports the maximum difference between their 40M fitted indices. The old waveform is not replaced with a theoretical tail or a refitted curve.

Interval tests retain the archived 10M RMS / 40M logarithmic-fit estimator. A candidate and the independently computed Schwarzschild control must both lie within the specified band around 3 and agree with each other to the same absolute tolerance. Both require ≤1% envelope refinement change, ≤0.1 index refinement change, and amplitude exceeding ten times the observed envelope-refinement floor. Durations are measured on a common U grid through U=975M. These are pilot consistency checks, not replacements for the complete archived protocol or its parameter sweeps.

Curves that fail the numerical refinement gate anywhere over U=150–950M are explicitly labelled unresolved. A failed Price-interval test for such a curve is not by itself independent evidence of physical failure. The exterior comparison must not be promoted to a replacement production result while this sensitivity remains.

Quarantined preliminary backend outputs and Hermite stiff-mode diagnostics are excluded. No time translation or amplitude rescaling is fitted. High arithmetic precision alone does not establish spatial or temporal accuracy.
