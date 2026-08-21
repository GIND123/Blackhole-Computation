# Tail post-processing: repair, verification, and what the ladder supports

This note records the repairs made to the tail post-processing, the tests that
hold them in place, and what the completed `L/M = 5120` ladder does and does
not establish. No production simulation was rerun to produce anything below;
every number comes from the archived ladder.

## 1. The rolling envelope lost the tail

`rms_envelope` builds a centred root mean square over a moving window. The
window sums were formed by differencing one global cumulative sum:

```text
cumulative = concatenate(([0], cumsum(values)))
window     = cumulative[count:] - cumulative[:-count]
```

That is exact in real arithmetic and useless here. A tail waveform carries a
prompt pulse of order unity followed by a decay that approaches the
double-precision floor. Squaring doubles the ratio in the exponent, so the
cumulative sum is dominated by the prompt pulse while the quantity wanted is
the difference of two nearly equal partial sums far out in the tail.

Measured on a synthetic case with a prompt pulse and a `U^-3` tail:

| quantity | value |
|---|---:|
| cumulative sum at the sample point | `7.089815e+01` |
| true window sum | `1.010243e-16` |
| ratio | `7.0e+17` |
| reciprocal of machine epsilon | `4.5e+15` |
| **value returned by the differenced cumulative sum** | **`0.000000e+00`** |

The dynamic range exceeds `1/eps` by two orders of magnitude, so the answer is
not merely inaccurate, it is identically zero.

### The replacement

Window sums are now accumulated inside blocks of `count` samples. Prefix sums
run from the start of a block and suffix sums from its end, so a window that
straddles two blocks is one suffix plus one prefix and never references a
partial sum from outside those two blocks. The rounding error is proportional
to the local amplitude rather than to the largest amplitude anywhere in the
record. The cost stays linear in the number of samples.

Verification against extended-precision reference sums over the same windows:

| case | worst relative error |
|---|---:|
| random data, several window lengths | `1.5e-14` (absolute, order unity data) |
| prompt pulse with a `U^-3` tail, all valid windows | `1.6e-15` |
| the deep tail sample above | `0.0` |

### Effect on the result

The Price departure at the outer boundary moves from `584M` to
**`668.08M`**.

The screening results move too, and in the direction that matters. The outer
boundary, which is what selected the length, is unchanged: `L/M = 5120` still
gives an accepted interval of `154.13` to `474.98` with a continuous duration
of `320.85M`, and `640`, `1280`, `2560` still fail. But the two fixed radius
observers at `L/M = 5120` change from failing to passing:

| observer | before | after |
|---|---|---|
| `r = 8M` | `51.5` to `55.6`, duration `4.1M`, fails | `206.7` to `400.5`, duration `193.8M`, passes |
| `r = 16M` | `227.9` to `241.4`, duration `13.5M`, fails | `227.2` to `475.0`, duration `247.7M`, passes |
| outer | `154.1` to `475.0`, duration `320.8M`, passes | unchanged |

The fixed radius tails are two orders of magnitude weaker than the outer
boundary signal at the same retarded time, so they enter the destructive range
sooner and were being shredded by the differenced cumulative sum. All three
observers now establish the Price index at `L/M = 5120`, where previously only
the outer boundary did. The selection of the length is unaffected; the
evidence behind it is stronger.

### Regression tests

`tests/test_tail_dynamic_range.py` drives the estimators with signals whose
decay laws are known in closed form:

* the squared signal range is asserted to exceed `1/eps`, so the test cannot
  silently stop exercising the failure;
* the differenced cumulative sum is asserted to return exactly zero, which
  pins the regression rather than describing it;
* `p_eff` is recovered as `3.000` and `5.000` from planted `U^-3` and `U^-5`
  envelopes, to `2e-3` absolute;
* `gamma_eff / kappa_c` is recovered as `1.000` from a planted exponential, to
  `2e-3` relative;
* the envelope itself tracks a known power law to `5e-3` relative.

## 2. The floor is now measured, not assumed

The old validity floor was `1000 * eps * max|signal|` times a multiplier: a
statement about arithmetic, not about the calculation. It has been replaced by
a floor measured from the refinement ladder itself.

At each retarded time the floor is the larger of

* the spatial difference between the two finest grids, `|A(3072) - A(2048)|`,
  which bounds the error of the finer one, and
* the temporal difference `|A(dt) - A(dt/2)|` at `N = 2048`.

The difference between the two coarser grids, `|A(2048) - A(1536)|`, is
reported beside them so the ratio shows the differences are still falling.
A rate is read only where the envelope exceeds the floor by a factor of ten.

### What the ladder says

| observer | Price target | median convergence ratio | trusted to `U/M` | `kappa_c U` |
|---|---:|---:|---:|---:|
| `r = 8M` | 5 | 1205 | 2262 | 0.442 |
| `r = 16M` | 5 | 998 | 3313 | 0.647 |
| outer boundary | 3 | 3062 | 4501 | 0.879 |
| Schwarzschild reference, outer | 3 | 12 | 4649 | 0.908 |

The convergence ratio is the median of `|A(2048) - A(1536)| / |A(3072) -
A(2048)|` over the Price interval. Values near `10^3` mean the spatial
differences are still falling steeply with resolution.

Envelope differences over the Price interval at the outer boundary:

| comparison | SdS | Schwarzschild |
|---|---:|---:|
| `N = 1536` against `N = 2048` | `5.79e-3` | `2.49e-3` |
| `N = 2048` against `N = 3072` | `2.10e-6` | `1.61e-4` |
| `dt` against `dt/2` at `N = 2048` | `3.18e-8` | `1.99e-8` |

## 3. What the completed ladder establishes

### Price behaviour is confirmed, and cleanly

Measured `p_eff` inside the trusted window:

| observer | `U/M = 300` | `500` | `668` | `1000` | target |
|---|---:|---:|---:|---:|---:|
| `r = 8M` | 4.943 | 4.969 | 4.982 | 4.994 | 5 |
| `r = 16M` | 4.816 | 4.887 | 4.920 | 4.964 | 5 |
| outer boundary | 3.078 | 3.111 | 3.149 | 3.237 | 3 |

At `r = 8M` the measured index reaches `4.994`, within `0.1%` of the fixed
radius Price value `p = 2l + 3 = 5` for `l = 1`. The outer boundary sits on
`p = l + 2 = 3`. The independently evolved Schwarzschild reference holds
`p_eff = 3` across its whole trusted window, which is the control that makes
the SdS departure meaningful.

### The cosmological regime is not reached

`gamma_eff / kappa_c` falls monotonically but never approaches unity:

| `U/M` | 668 | 1000 | 2000 | 3000 | 4000 |
|---|---:|---:|---:|---:|---:|
| outer boundary | 35.1 | 18.1 | 9.42 | 7.35 | not read |

The reason is arithmetic rather than physics. While the decay is still a power
law, `gamma_eff = p / U`, so `gamma_eff / kappa_c = p / (kappa_c U)`. Reaching
unity requires `kappa_c U` of order `p`, that is `kappa_c U` of a few. The
ladder floor stops the measurement at `kappa_c U = 0.879`.

**No cosmological transition time is assigned.** The recorded status is
`no_cosmological_entry_before_ladder_floor`, and `U_dS`, the crossover
interval, and the fitted-law intersection are all null.

### Measured across the sequence

The single length result cannot say whether some other `L` would do better, so
the final ladder was run for `L/M = 640` as well, from a clean commit. It
answers the question directly, and it answers it the other way round:

| | `L/M = 640` | `L/M = 5120` |
|---|---|---|
| Price interval at the outer boundary | **not established** | established, departs at `U = 668.1` |
| record trusted to | `kappa_c U = 3.162` | `kappa_c U = 0.879` |
| cosmological entry | **`kappa_c U = 2.267`**, persisting to `3.162` | none, even unanchored |
| scaled duration of the entry | `0.895` against `0.4` required | not applicable |

At `L/M = 640` the normalized rate settles on `gamma_eff / kappa_c = 1` within
the ten per cent tolerance from `kappa_c U = 2.267` onward and stays there to
the end of the record. The cosmological decay is resolved. What is missing is
the Schwarzschild power law: the cosmological horizon interferes too early for
a Price plateau to establish after ringdown.

At `L/M = 5120` the situation is exactly reversed.

**No tested length resolves both regimes in the same waveform.** The two
requirements pull apart: the power law needs the cosmological influence to
arrive late, which wants large `L`, and the exponential needs `kappa_c U` of
order a few, which at large `L` lies beyond where the tail has fallen under
the ladder floor.

### Two estimator faults found by plotting the curve

Neither of these was visible in the scalar summaries, and both changed a
conclusion.

`trusted_interval_end` returned the end of the *first* contiguous run of
trusted samples. Where the power law hands over to the exponential the
envelope passes through a local minimum, and at `L/M = 640` the ratio to the
floor dips `10.13`, `9.97`, then recovers, for three samples. The record was
being truncated at `kappa_c U = 1.083` when the waveform is usable to `3.162`:

| run | `kappa_c U` | samples |
|---:|---|---:|
| 1 | `0.156` to `1.083` | 11907 |
| 2 | `1.092` to `1.096` | 49 |
| 3 | `1.105` to `3.162` | 26407 |

Between the first and last trusted sample, `99.4%` are trusted. The estimator
now allows a brief excursion, requiring at least `95%` continuity, and falls
back to the longest unbroken run otherwise. At `L/M = 5120` the trusted
fraction is `83.7%`, so that case still stops at `0.879` and its reported
value does not move.

Second, the cosmological entry was only computed when a Price departure
existed, because the ordered test asks for both regimes in that order. That
made a length which reaches the exponential regime without ever showing a
power law report silence rather than a result. Entry is now also measured
anchored at the start of the resolved record, and the anchor used is recorded
beside it. The ordered both-regimes test still requires the Price anchor.

### The obstruction is structural

The two requirements pull in opposite directions. Establishing the Price index
needs the tail to be resolved for a few hundred `M` after ringdown, which
favours large `L`. Reaching the cosmological regime needs `kappa_c U` of a few,
that is `U` of order a few `L`, by which time a `U^-3` tail has fallen by
another factor of order `(3L / U_P)^3`. Raising `L` postpones the cosmological
regime and lowers the amplitude at which it would have to be measured.

The screened lengths show the same tension from the other side: `L/M = 2560`
reaches a continuous outer Price interval of `130.2M` against the `150M`
requirement, and `640` and `1280` do worse. So `5120` is the smallest tested
dyadic length that establishes the Price regime, and it is already too large
for the cosmological regime to be reached above the floor in double precision.

## 4. Reporting path

Both sensitivity columns are now exercised. The cosmological column was
previously always empty, because the interval it uses is degenerate whenever
no entry is found, so that half of the path had never run. All three grids
enter the sensitivity table; the previous version compared only `N = 2048`
with `N = 3072` and never used `N = 1536`.

The three-panel figure is truncated at the ladder floor for both backgrounds.
Before this, the Schwarzschild reference was drawn to `U = 16000M` using the
round-off floor, roughly three times beyond the point where its own
resolutions stop agreeing; the wander it develops there would otherwise read
as a measured Price index.

## 5. Manifest completeness

Verification checked that every listed file exists and hashes correctly, but
never that the package contains nothing unlisted. A campaign that is extended
after the manifest is written would still verify while holding unrecorded
results. Verification now compares the directory against the manifest in both
directions and reports `unlisted:` failures, and the manifest records its own
output directory so the check has something to scan. Schema version is now 2.

## Reproduction

```text
python -m black_hole.large_l_tail --output-dir results/large_l_tail analyze-screen 5120
python -m black_hole.large_l_tail --output-dir results/large_l_tail report-final 5120
python -m black_hole.tail_manifest --output-dir results/large_l_tail
python -m black_hole.tail_manifest --output-dir results/large_l_tail --verify
python -m pytest tests/test_tail_dynamic_range.py tests/test_large_l_tail.py
```
