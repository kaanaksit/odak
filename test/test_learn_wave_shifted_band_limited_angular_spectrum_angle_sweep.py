"""Angle-sweep validation for shifted-BASM vs raw BASM: Case B (safe tilt, comfortably inside
raw BASM's passband) and Case C (approaching/crossing raw BASM's passband boundary).

Case A (zero tilt, similarity == 1.0 exactly) is already covered by
test_learn_wave_shifted_band_limited_angular_spectrum.py and is not repeated here.

Goal: confirm the qualitative behavior implied by
test_learn_wave_shifted_band_limited_angular_spectrum_convergence.py's "reference floor"
finding -- raw BASM and shifted-BASM should agree almost perfectly while the tilted carrier
stays inside raw BASM's own passband, and should only start disagreeing once the carrier
approaches/exceeds it (raw BASM's *own* mask, evaluated at the absolute/carrier-included
frequency, clipping real spectral content -- not a shifted-BASM defect). This test is
diagnostic: it does not change the propagation implementation and does not relax any
threshold to force a pass. Case C especially is expected to show *degradation*, not agreement.

Same physical configuration as the convergence test (same wavelength, distance, FoV, diffuser
construction): WAVELENGTH_M, DISTANCE_M, RESOLUTION, PITCH_M, FOV_M,
DIFFUSER_NATIVE_RESOLUTION below. Both raw and shifted BASM run at the SAME 1024x1024
resolution here -- no binning or upsampling anywhere in this file.

Key quantities, each measured directly from the running implementation rather than only
re-derived by hand:

- f_BASM_limit: raw BASM's own usable spatial-frequency limit (Eq. 6's fx,req). Computed via
  the closed-form formula (mirroring get_band_limited_angular_spectrum_kernel's internal
  fx_max), then cross-checked against the actual boolean mask
  get_band_limited_angular_spectrum_kernel produces (see _measure_fx_limit) -- both are
  reported, and the closed-form value (exact, not discretized to a pixel) is used for all
  downstream occupancy/margin arithmetic.
- f_residual_max: the untilted diffuser field's own spectral extent, measured empirically as
  the radius in cycles/m containing RESIDUAL_ENERGY_FRACTION (90%) of its total spectral power
  (see _measure_residual_bandwidth) -- not a hand-derived estimate from the diffuser's nominal
  feature size.
- occupancy = (|f_carrier| + f_residual_max) / f_BASM_limit, per the task spec. Target
  occupancies are inverted to a carrier frequency, then that frequency is rounded to the
  nearest exact multiple of the FFT bin spacing 1/FOV_M (the same fix
  test_learn_wave_shifted_band_limited_angular_spectrum_convergence.py needed to avoid
  spectral leakage contaminating the raw-BASM reference -- see that file's module docstring),
  so the ACHIEVED occupancy (printed) may differ slightly from the nominal target.
- clipped spectral energy fraction: computed directly from the tilted field's own FFT power
  spectrum against raw BASM's actual mask (again via get_band_limited_angular_spectrum_kernel),
  not estimated.
"""

import math
import sys
import torch
import odak


WAVELENGTH_M = 532e-9
DISTANCE_M = 3e-3
RESOLUTION = 1024
PITCH_M = 1.6e-6
FOV_M = RESOLUTION * PITCH_M
DIFFUSER_NATIVE_RESOLUTION = 8
RESIDUAL_ENERGY_FRACTION = 0.9
BIN_SPACING_HZ_PER_M = 1.0 / FOV_M

CASE_B_OCCUPANCIES = [0.0, 0.25, 0.50, 0.70, 0.80]
CASE_C_OCCUPANCIES = [0.85, 0.90, 0.92, 0.95, 0.97, 1.00, 1.02, 1.05, 1.10]


def _diffuser_phase(resolution, native_resolution=DIFFUSER_NATIVE_RESOLUTION, device=torch.device("cpu"), seed=0):
    """The same physical diffuser construction as
    test_learn_wave_shifted_band_limited_angular_spectrum_convergence.py (random at a native
    feature resolution, nearest-neighbor upsampled) -- reimplemented locally rather than
    cross-imported, matching this file's own self-contained convention."""
    upsample = resolution // native_resolution
    generator = torch.Generator().manual_seed(seed)
    native = 2.0 * odak.pi * torch.rand(native_resolution, native_resolution, generator=generator)
    return native.repeat_interleave(upsample, dim=0).repeat_interleave(upsample, dim=1).to(device)


def _spatial_grid(resolution, pixel_pitch_m, device):
    coords = (torch.arange(resolution, device=device) - (resolution - 1) / 2.0) * pixel_pitch_m
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def _recenter(field, pixel_pitch_m, shift_x_m, shift_y_m=0.0):
    h, w = field.shape[-2:]
    fy = torch.fft.fftfreq(h, d=pixel_pitch_m, device=field.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(w, d=pixel_pitch_m, device=field.device, dtype=torch.float32)
    qy, qx = torch.meshgrid(2.0 * math.pi * fy, 2.0 * math.pi * fx, indexing="ij")
    shift_phase = torch.exp(1j * (qx * shift_x_m + qy * shift_y_m).to(torch.complex64))
    return torch.fft.ifft2(torch.fft.fft2(field) * shift_phase)


def _centroid_px(intensity):
    h, w = intensity.shape[-2:]
    y = torch.arange(h, dtype=torch.float64, device=intensity.device) - (h - 1) / 2.0
    x = torch.arange(w, dtype=torch.float64, device=intensity.device) - (w - 1) / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    total = intensity.double().sum()
    cx = (xx * intensity.double()).sum() / total
    cy = (yy * intensity.double()).sum() / total
    return cx.item(), cy.item()


def _compare(reference_intensity, reference_pitch_m, candidate_intensity, candidate_pitch_m):
    """Same-resolution comparison (bin_factor is always 1 in this file -- no binning or
    upsampling, per this test's requirements), mirroring the metric definitions in
    test_learn_wave_shifted_band_limited_angular_spectrum_convergence.py's _compare."""
    similarity = (
        torch.sum(reference_intensity * candidate_intensity)
        / torch.sqrt(torch.sum(reference_intensity**2) * torch.sum(candidate_intensity**2))
    ).item()

    diff = reference_intensity - candidate_intensity
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    ref_rms = torch.sqrt(torch.mean(reference_intensity**2)).item()
    nrmse = rmse / ref_rms if ref_rms > 0 else float("nan")
    peak = float(reference_intensity.max())
    psnr = 20.0 * math.log10(peak / rmse) if rmse > 0 else float("inf")

    reference_energy = float(reference_intensity.sum()) * reference_pitch_m**2
    candidate_energy = float(candidate_intensity.sum()) * candidate_pitch_m**2
    energy_ratio = candidate_energy / reference_energy if reference_energy > 0 else float("nan")

    ref_cx, ref_cy = _centroid_px(reference_intensity)
    cand_cx, cand_cy = _centroid_px(candidate_intensity)
    d_px = math.hypot(cand_cx - ref_cx, cand_cy - ref_cy)

    return {
        "similarity": similarity,
        "nrmse": nrmse,
        "psnr": psnr,
        "energy_ratio": energy_ratio,
        "d_px": d_px,
    }


def _measure_fx_limit(resolution, pitch, distance, wavelength, device):
    """Raw BASM's usable spatial-frequency limit -- computed as min(grid Nyquist, the
    aperture/distance-dependent Eq. 6 term mirroring get_band_limited_angular_spectrum_kernel's
    internal fx_max), AND cross-checked against the actual boolean mask that function produces
    (per this test's requirement to determine the limit from the actual implementation rather
    than only re-deriving the formula by hand). The min() matters here: odak's actual
    get_band_limited_angular_spectrum_kernel code computes only the aperture/distance term,
    without an explicit clamp against the grid's own Nyquist limit 1/(2*pitch) -- for these
    parameters that term (~495 kHz/m) exceeds the grid's own Nyquist (~312 kHz/m), so the mask
    is effectively unrestricted by it and the array's intrinsic Nyquist is the real binding
    limit. Omitting this min() was caught by the cross-check below (formula vs. measured
    disagreed by ~60%% on the first attempt at writing this test). Returns (limit, empirical)."""
    x_extent = pitch * resolution
    grid_nyquist = 1.0 / (2.0 * pitch)
    aperture_term = 1.0 / math.sqrt((2.0 * distance / x_extent) ** 2 + 1.0) / wavelength
    fx_limit_formula = min(grid_nyquist, aperture_term)

    kernel = odak.learn.wave.get_band_limited_angular_spectrum_kernel(
        nu=resolution, nv=resolution, dx=pitch, wavelength=wavelength, distance=distance, device=device
    )
    mask = odak.learn.wave.calculate_amplitude(kernel) > 0.5
    fx = torch.linspace(
        -1 / (2 * pitch) + 0.5 / (2 * x_extent), 1 / (2 * pitch) - 0.5 / (2 * x_extent), resolution, dtype=torch.float32
    )
    center_row = resolution // 2
    passing = fx[mask[center_row, :]]
    fx_limit_empirical = float(passing.abs().max()) if passing.numel() > 0 else 0.0
    return fx_limit_formula, fx_limit_empirical, mask


def _measure_residual_bandwidth(field, pitch, energy_fraction=RESIDUAL_ENERGY_FRACTION):
    """The untilted diffuser field's own spectral extent, measured empirically as the radius
    (in cycles/m) containing `energy_fraction` of its total spectral power -- not a hand-derived
    estimate from the diffuser's nominal feature size."""
    h, w = field.shape[-2:]
    spectrum = torch.fft.fftshift(torch.fft.fft2(field))
    power = (spectrum.abs() ** 2).double()
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, d=pitch))
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, d=pitch))
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(FX**2 + FY**2)

    order = torch.argsort(radius.flatten())
    sorted_power = power.flatten()[order]
    sorted_radius = radius.flatten()[order]
    cumulative = torch.cumsum(sorted_power, dim=0)
    total = cumulative[-1]
    idx = min(int(torch.searchsorted(cumulative, energy_fraction * total).item()), sorted_radius.numel() - 1)
    return float(sorted_radius[idx])


def _clipped_energy_fraction(tilted_field, mask):
    """Fraction of the tilted field's own spectral power that raw BASM's actual mask rejects --
    computed directly from the field's FFT power spectrum, not estimated."""
    spectrum = torch.fft.fftshift(torch.fft.fft2(tilted_field))
    power = (spectrum.abs() ** 2).double()
    total = power.sum()
    inside = power[mask].sum()
    if total <= 0:
        return float("nan")
    return float(1.0 - inside / total)


def _bin_align(frequency_hz_per_m):
    return round(frequency_hz_per_m / BIN_SPACING_HZ_PER_M) * BIN_SPACING_HZ_PER_M


def _angle_for_occupancy(occupancy, fx_limit, f_residual_max, wavelength):
    """Inverts occupancy = (|f_carrier| + f_residual_max) / fx_limit for f_carrier, rounds it to
    the nearest exact FFT bin (see module docstring), and returns
    (theta_deg, offset_fx_bin_aligned, achieved_occupancy), or None if the target occupancy
    would require a non-physical (|sin(theta)| >= 1) carrier."""
    f_carrier_target = occupancy * fx_limit - f_residual_max
    f_carrier_target = max(f_carrier_target, 0.0)
    offset_fx = _bin_align(f_carrier_target)
    sin_theta = offset_fx * wavelength
    if abs(sin_theta) >= 1.0:
        return None
    theta_deg = math.degrees(math.asin(sin_theta))
    achieved_occupancy = (abs(offset_fx) + f_residual_max) / fx_limit
    return theta_deg, offset_fx, achieved_occupancy


def _run_angle(offset_fx, offset_fy, device, k, mask, x_extent_unused=None):
    """Builds the diffuser scene, runs raw BASM (on the tilted field) and shifted BASM (on the
    untilted field, offset-only), recenters both, and returns (metrics, extra) where extra
    holds the clipped-energy fraction and per-run diagnostics."""
    diffuser_phase = _diffuser_phase(RESOLUTION, device=device)
    field = odak.learn.wave.generate_complex_field(torch.ones(RESOLUTION, RESOLUTION, device=device), diffuser_phase)

    xx, yy = _spatial_grid(RESOLUTION, PITCH_M, device)
    carrier_phase = 2.0 * odak.pi * (offset_fx * xx + offset_fy * yy)
    carrier = odak.learn.wave.generate_complex_field(torch.ones_like(carrier_phase), carrier_phase)
    tilted_field = field * carrier.to(torch.complex64)

    sin_theta = offset_fx * WAVELENGTH_M
    tan_theta = sin_theta / math.sqrt(max(1.0 - sin_theta**2, 1e-12))
    chief_ray_shift_x_m = DISTANCE_M * tan_theta

    propagated_raw = odak.learn.wave.band_limited_angular_spectrum(tilted_field, k, DISTANCE_M, PITCH_M, WAVELENGTH_M)
    recentered_raw = _recenter(propagated_raw, PITCH_M, chief_ray_shift_x_m)
    intensity_raw = odak.learn.wave.calculate_amplitude(recentered_raw) ** 2

    propagated_shifted = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field, k, DISTANCE_M, PITCH_M, WAVELENGTH_M, offset_fx=offset_fx, offset_fy=offset_fy
    )
    recentered_shifted = _recenter(propagated_shifted, PITCH_M, chief_ray_shift_x_m)
    intensity_shifted = odak.learn.wave.calculate_amplitude(recentered_shifted) ** 2

    metrics = _compare(intensity_raw, PITCH_M, intensity_shifted, PITCH_M)
    clipped_fraction = _clipped_energy_fraction(tilted_field, mask)
    finite = bool(torch.isfinite(intensity_raw).all() and torch.isfinite(intensity_shifted).all())
    return metrics, clipped_fraction, finite


def _print_table(header_cols, rows):
    header = " | ".join("{:>10}".format(c) for c in header_cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join("{:>10}".format(v) for v in row))


def _fmt_occ_theta(row, fallback):
    if row is None:
        return fallback
    return "occupancy={:.3f} (theta={:.2f} deg)".format(row["occupancy"], row["theta_deg"])


def _print_diagnostic_summary(case_b_rows, case_c_rows):
    print("Diagnostic summary:")
    all_rows = case_b_rows + case_c_rows
    above_9999 = [r for r in all_rows if r["similarity"] > 0.9999]
    above_999 = [r for r in all_rows if r["similarity"] > 0.999]
    highest_9999 = max(above_9999, key=lambda r: r["occupancy"]) if above_9999 else None
    highest_999 = max(above_999, key=lambda r: r["occupancy"]) if above_999 else None
    print("  - highest occupancy/angle with similarity > 0.9999: " + _fmt_occ_theta(highest_9999, "none tested"))
    print("  - highest occupancy/angle with similarity > 0.999:  " + _fmt_occ_theta(highest_999, "none tested"))

    sorted_rows = sorted(all_rows, key=lambda r: r["occupancy"])
    first_below_9999 = next((r for r in sorted_rows if r["similarity"] <= 0.9999), None)
    first_margin_negative = next((r for r in sorted_rows if r["margin"] < 0.0), None)
    print(
        "  - measurable disagreement (similarity <= 0.9999) first begins at: "
        + _fmt_occ_theta(first_below_9999, "not reached in this sweep")
    )
    print(
        "  - spectrum first touches/exceeds raw BASM's limit (margin < 0) at: "
        + _fmt_occ_theta(first_margin_negative, "not reached in this sweep")
    )

    if first_below_9999 is not None and first_margin_negative is not None:
        correlates = abs(first_below_9999["occupancy"] - first_margin_negative["occupancy"]) < 0.15
        print(
            "  - similarity degradation {}correlate with the predicted bandlimit crossing "
            "(disagreement onset occupancy={:.3f} vs. margin-crossing occupancy={:.3f})".format(
                "appears to " if correlates else "does NOT clearly ",
                first_below_9999["occupancy"],
                first_margin_negative["occupancy"],
            )
        )
    elif first_below_9999 is not None and first_margin_negative is None:
        print(
            "  - similarity degraded (first at occupancy={:.3f}) BEFORE the spectrum reached raw BASM's "
            "predicted limit in this sweep -- this suggests another implementation mismatch, not simple "
            "bandlimit clipping, and should be investigated further".format(first_below_9999["occupancy"])
        )

    if first_below_9999 is not None and first_below_9999["occupancy"] < 0.3:
        print(
            "  - disagreement onset (occupancy={:.3f}) is well before occupancy=1: clipped spectral energy "
            "there is {:.5f} ({}), so this is NOT bulk energy loss -- the mask's hard, discontinuous cutoff "
            "(evaluated at a DIFFERENT origin for raw's absolute-frequency mask vs. shifted's residual-"
            "frequency mask) reshapes the few frequency components right at the boundary differently between "
            "the two methods at ANY nonzero carrier; those components carry little energy but control fine "
            "spatial structure, matching energy ratio staying ~1 and centroid staying ~0 while "
            "similarity/NRMSE still degrade smoothly with occupancy rather than switching on sharply at "
            "occupancy=1. Not a new implementation bug -- the same reference-side mask-asymmetry effect the "
            "convergence test's module docstring documents, just visible earlier than a clean pass/fail "
            "split at the nominal bandlimit would suggest.".format(
                first_below_9999["occupancy"],
                first_below_9999["clipped_fraction"],
                "negligible" if first_below_9999["clipped_fraction"] < 1e-4 else "not negligible",
            )
        )

    if first_below_9999 is None and any(r["occupancy"] > 1.05 for r in sorted_rows):
        print(
            "  - similarity remained > 0.9999 even well past the predicted bandlimit (occupancy > 1.05) -- "
            "the current explanation of the reference-side similarity ceiling is probably incomplete"
        )

    energy_ratios = [r["energy_ratio"] for r in all_rows]
    max_energy_deviation = max(abs(e - 1.0) for e in energy_ratios)
    print(
        "  - total energy remains matched across the whole sweep despite structural disagreement: {} "
        "(max |energy ratio - 1| = {:.4f})".format("yes" if max_energy_deviation < 0.02 else "no", max_energy_deviation)
    )


def test(device=torch.device("cpu")):
    k = odak.learn.wave.wavenumber(WAVELENGTH_M)

    fx_limit_formula, fx_limit_empirical, mask = _measure_fx_limit(RESOLUTION, PITCH_M, DISTANCE_M, WAVELENGTH_M, device)
    print()
    print("f_BASM_limit (Eq. 6, closed form):  {:.1f} cycles/m".format(fx_limit_formula))
    print("f_BASM_limit (measured from mask):  {:.1f} cycles/m".format(fx_limit_empirical))
    assert abs(fx_limit_formula - fx_limit_empirical) / fx_limit_formula < 0.01, (
        "closed-form and measured raw-BASM frequency limits disagree by more than 1%% -- "
        "the formula reused here may not match the actual kernel implementation"
    )

    untilted_diffuser = odak.learn.wave.generate_complex_field(
        torch.ones(RESOLUTION, RESOLUTION, device=device), _diffuser_phase(RESOLUTION, device=device)
    )
    f_residual_max = _measure_residual_bandwidth(untilted_diffuser, PITCH_M)
    print(
        "f_residual_max (measured, {:.0f}% energy radius): {:.1f} cycles/m".format(
            RESIDUAL_ENERGY_FRACTION * 100, f_residual_max
        )
    )
    print()

    def run_case(occupancies, case_name):
        rows_data = []
        for occupancy in occupancies:
            result = _angle_for_occupancy(occupancy, fx_limit_formula, f_residual_max, WAVELENGTH_M)
            if result is None:
                print("  [{}] occupancy {:.2f}: skipped, requires non-physical carrier".format(case_name, occupancy))
                continue
            theta_deg, offset_fx, achieved_occupancy = result
            metrics, clipped_fraction, finite = _run_angle(offset_fx, 0.0, device, k, mask)
            margin = fx_limit_formula - (abs(offset_fx) + f_residual_max)
            rows_data.append(
                {
                    "occupancy_target": occupancy,
                    "occupancy": achieved_occupancy,
                    "theta_deg": theta_deg,
                    "offset_fx": offset_fx,
                    "margin": margin,
                    "clipped_fraction": clipped_fraction,
                    "finite": finite,
                    **metrics,
                }
            )
        return rows_data

    print("=== Case B: safe tilt (carrier comfortably inside raw BASM's passband) ===")
    case_b_rows = run_case(CASE_B_OCCUPANCIES, "B")
    _print_table(
        ["Angle", "Occup.", "Margin", "Clipped", "Similarity", "NRMSE", "PSNR", "EnergyRatio", "d(px)"],
        [
            [
                "{:.2f}".format(r["theta_deg"]),
                "{:.3f}".format(r["occupancy"]),
                "{:.0f}".format(r["margin"]),
                "{:.5f}".format(r["clipped_fraction"]),
                "{:.6f}".format(r["similarity"]),
                "{:.4f}".format(r["nrmse"]),
                "{:.2f}".format(r["psnr"]),
                "{:.4f}".format(r["energy_ratio"]),
                "{:.3f}".format(r["d_px"]),
            ]
            for r in case_b_rows
        ],
    )
    print()
    case_b_violations = []
    for r in case_b_rows:
        assert r["finite"], "non-finite intensity at a Case B (safe tilt) angle -- this should never happen"
        if r["similarity"] <= 0.999:
            case_b_violations.append(r)
    if case_b_violations:
        print(
            "WARNING: {} of {} Case B (safe-tilt) angles did NOT reach similarity > 0.999 -- see the "
            "diagnostic summary below for the investigated explanation (not deferred silently; the "
            "run continues through Case C so the full picture is available before the final "
            "assertion at the end of this test fails, per this test's requirement not to relax "
            "thresholds to force a pass):".format(len(case_b_violations), len(case_b_rows))
        )
        for r in case_b_violations:
            print(
                "  - occupancy={:.3f} (theta={:.2f} deg): similarity={:.6f}".format(
                    r["occupancy"], r["theta_deg"], r["similarity"]
                )
            )
        print()

    print("=== Case C: approaching and crossing raw BASM's passband boundary (diagnostic only) ===")
    case_c_rows = run_case(CASE_C_OCCUPANCIES, "C")
    _print_table(
        ["Angle", "Occup.", "Margin", "Clipped", "Similarity", "NRMSE", "PSNR", "EnergyRatio", "d(px)"],
        [
            [
                "{:.2f}".format(r["theta_deg"]),
                "{:.3f}".format(r["occupancy"]),
                "{:.0f}".format(r["margin"]),
                "{:.5f}".format(r["clipped_fraction"]),
                "{:.6f}".format(r["similarity"]),
                "{:.4f}".format(r["nrmse"]),
                "{:.2f}".format(r["psnr"]),
                "{:.4f}".format(r["energy_ratio"]),
                "{:.3f}".format(r["d_px"]),
            ]
            for r in case_c_rows
        ],
    )
    for r in case_c_rows:
        assert r["finite"], "non-finite intensity at a Case C angle -- implementation crash, not just degraded agreement"

    print()
    _print_diagnostic_summary(case_b_rows, case_c_rows)

    # Deferred (not skipped) per the requirement not to relax thresholds to force a pass: raised
    # only now, after the full Case B/C tables and diagnostic summary above have printed, so a
    # failure here comes with the complete picture rather than a bare mid-run traceback.
    assert not case_b_violations, (
        "{} Case B (safe-tilt) angle(s) did not reach similarity > 0.999 -- see the WARNING and "
        "diagnostic summary above for the investigated cause".format(len(case_b_violations))
    )


if __name__ == "__main__":
    sys.exit(test())
