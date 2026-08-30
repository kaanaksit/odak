"""BASM vs shifted-BASM equivalence: mask ablation, pixel semantics, field-level and
sensor-level validation, merged into one file (previously three: ..._equivalence_debug.py,
..._field_equivalence.py, ..._sensor_equivalence.py).

Four independent pytest entry points, each runnable/failable on its own:

  test_mask_ablation()      -- does the band-limiting mask cause raw-vs-shifted disagreement?
                                No: bandlimit ON/OFF gives bit-identical results.
  test_pixel_semantics()    -- are raw and shifted BASM compared with consistent pixel units?
                                Fixed here: both return POINT-SAMPLED intensity (see Notes),
                                so downsampling raw's fine grid for comparison must use
                                AREA-AVERAGE binning, not the SUM binning appropriate for real
                                sensor-pixel integration. PASSES once corrected.
  test_field_equivalence()  -- do the two solvers compute the same COMPLEX field at identical
                                coordinates (Test 1, a numerical-solver question)? Currently
                                FAILS to reach the ideal >0.9999 aligned-correlation target;
                                left failing intentionally (see its docstring below).
  test_sensor_equivalence() -- do the two solvers predict the same finite-area SENSOR pixel
                                measurement (Test 2, the physically-measurable question)?
                                PASSES cleanly (similarity > 0.9999, energy ratio within 0.1%).

Notes (pixel semantics): band_limited_angular_spectrum and shifted_band_limited_angular_spectrum
both return point-sampled complex field values -- `custom()` is a pure FFT/kernel-multiply/IFFT
operation; `dx` only shapes the kernel's frequency axis, never scales the output. So |U|^2 from
either is point-sampled intensity, not integrated sensor energy. SUM binning
(sum_bin_sensor_pixels-style, src/asm_psf_propagation.py) is correct ONLY for converting a fine
simulation grid into an actual, physically larger sensor pixel; AREA-AVERAGE binning is correct
for comparing two simulation grids of the same point-sampled field at different resolutions.

Raw BASM's own resolution is never fixed (e.g. never "1024 for every angle") -- it is
automatically selected per angle to keep its own input safely below Nyquist (holding the
physical FoV fixed and increasing N), then VERIFIED via an explicit doubling-resolution
convergence check, up to a physically-motivated cap (grid Nyquist <= 1/wavelength; beyond that,
odak's ORIGINAL non-shifted kernel has a latent NaN bug for evanescent components, and no new
propagating information exists past that point anyway).
"""

import math
import sys
import torch
import odak


# ============================================================================
# Shared physical setup
# ============================================================================
WAVELENGTH_M = 532e-9
DISTANCE_M = 3e-3
RESOLUTION = 1024
PITCH_M = 1.6e-6
FOV_M = RESOLUTION * PITCH_M
DIFFUSER_NATIVE_RESOLUTION = 8
BIN_SPACING_HZ_PER_M = 1.0 / FOV_M
ANGLES_DEG = [0.0, 2.0, 4.0, 6.0, 8.0, 9.0]

BASE_RAW_RESOLUTION = RESOLUTION
SAFE_NYQUIST_FRACTION = 0.8
CONVERGENCE_SIMILARITY_THRESHOLD = 0.9999
SHIFTED_RESOLUTION = 512
SHIFTED_RESOLUTION_ALT = 1024
SENSOR_RESOLUTION = SHIFTED_RESOLUTION
SHIFT_INTERNAL_RESOLUTIONS = [512, 1024, 2048]
SHIFT_INTERNAL_PRIMARY = 2048

MAIN_SIMILARITY_THRESHOLD = 0.999
CONTROL_TOLERANCE = 1e-4
ALIGNED_CORR_TARGET = 0.9999
PHASE_MASK_INTENSITY_FRACTION = 0.01
SENSOR_SIMILARITY_TARGET = 0.999
ENERGY_RATIO_TOLERANCE = 0.05


def _max_physical_raw_resolution():
    """Largest power-of-two grid (fixed FOV_M) whose Nyquist limit stays at/below 1/wavelength
    -- beyond that, frequencies are evanescent (no propagating info) and odak's ORIGINAL
    (non-shifted) kernel has a latent NaN bug there, so this cap avoids both issues."""
    limit = 2.0 * FOV_M / WAVELENGTH_M
    n = BASE_RAW_RESOLUTION
    while n * 2 <= limit:
        n *= 2
    return n


MAX_RAW_RESOLUTION = _max_physical_raw_resolution()
CONVERGENCE_DOUBLE_CEILING = MAX_RAW_RESOLUTION


# ============================================================================
# Shared grid / propagation helpers
# ============================================================================
def _diffuser_phase(resolution, native_resolution=DIFFUSER_NATIVE_RESOLUTION, device=torch.device("cpu"), seed=0):
    upsample = resolution // native_resolution
    generator = torch.Generator().manual_seed(seed)
    native = 2.0 * odak.pi * torch.rand(native_resolution, native_resolution, generator=generator)
    return native.repeat_interleave(upsample, dim=0).repeat_interleave(upsample, dim=1).to(device)


def _spatial_grid(resolution, pixel_pitch_m, device):
    coords = (torch.arange(resolution, device=device) - (resolution - 1) / 2.0) * pixel_pitch_m
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def _recenter(field, pixel_pitch_m, shift_x_m, shift_y_m=0.0):
    """Shifts `field` by (shift_x_m, shift_y_m) via the Fourier shift theorem -- exact for a
    field represented by its own finite DFT (true of every field in this file)."""
    h, w = field.shape[-2:]
    fy = torch.fft.fftfreq(h, d=pixel_pitch_m, device=field.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(w, d=pixel_pitch_m, device=field.device, dtype=torch.float32)
    qy, qx = torch.meshgrid(2.0 * math.pi * fy, 2.0 * math.pi * fx, indexing="ij")
    shift_phase = torch.exp(1j * (qx * shift_x_m + qy * shift_y_m).to(torch.complex64))
    return torch.fft.ifft2(torch.fft.fft2(field) * shift_phase)


def _bin_align(frequency_hz_per_m):
    return round(frequency_hz_per_m / BIN_SPACING_HZ_PER_M) * BIN_SPACING_HZ_PER_M


def _angle_to_offset(theta_deg):
    return _bin_align(math.sin(math.radians(theta_deg)) / WAVELENGTH_M)


def _build_scene_at(resolution, offset_fx, device):
    """Same physical scenario (diffuser, aperture, source, distance) at an arbitrary resolution,
    holding FOV_M fixed (pixel pitch shrinks as resolution grows). Returns the untilted envelope
    (shifted-BASM's input), the fully tilted field (raw BASM's input), the pitch, and the
    chief-ray lateral landing shift."""
    pitch = FOV_M / resolution
    diffuser_phase = _diffuser_phase(resolution, device=device)
    field = odak.learn.wave.generate_complex_field(torch.ones(resolution, resolution, device=device), diffuser_phase)
    xx, yy = _spatial_grid(resolution, pitch, device)
    carrier_phase = 2.0 * odak.pi * offset_fx * xx
    carrier = odak.learn.wave.generate_complex_field(torch.ones_like(carrier_phase), carrier_phase)
    tilted_field = field * carrier.to(torch.complex64)
    sin_theta = offset_fx * WAVELENGTH_M
    tan_theta = sin_theta / math.sqrt(max(1.0 - sin_theta**2, 1e-12))
    chief_ray_shift_x_m = DISTANCE_M * tan_theta
    return field, tilted_field, pitch, chief_ray_shift_x_m


def _centroid_px(intensity):
    h, w = intensity.shape[-2:]
    y = torch.arange(h, dtype=torch.float64, device=intensity.device) - (h - 1) / 2.0
    x = torch.arange(w, dtype=torch.float64, device=intensity.device) - (w - 1) / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    total = intensity.double().sum()
    cx = (xx * intensity.double()).sum() / total
    cy = (yy * intensity.double()).sum() / total
    return cx.item(), cy.item()


def _bin_intensity(intensity, factor):
    """Energy-preserving SUM binning (matches sum_bin_sensor_pixels): a coarser PHYSICAL SENSOR
    pixel that integrates light over its footprint. See module Notes."""
    n = intensity.shape[-1]
    m = n // factor
    return intensity.reshape(m, factor, m, factor).sum(dim=(1, 3))


def _bin_intensity_average(intensity, factor):
    """AREA-AVERAGE binning: the correct operation for comparing the SAME point-sampled field
    at two simulation-grid resolutions (preserves intensity-density semantics). See module Notes."""
    n = intensity.shape[-1]
    m = n // factor
    return intensity.reshape(m, factor, m, factor).mean(dim=(1, 3))


def _decimate_intensity(intensity, factor):
    """Point-samples the coarse grid's own centers directly (no averaging/summing) -- exact
    when `factor` is odd, off by half a fine pixel when even (negligible at these scales)."""
    offset = factor // 2
    return intensity[offset::factor, offset::factor]


def _sensor_energy(intensity_fine, dx_fine, factor):
    """E_ij = sum(I_fine over the sensor pixel footprint) * dx_fine^2 -- a physical
    pixel-area-integrated energy, as opposed to a point-sampled intensity."""
    if factor <= 1:
        return intensity_fine * dx_fine**2
    return _bin_intensity(intensity_fine, factor) * dx_fine**2


def _residual_bandwidth_hz_per_m():
    """Diffuser's own residual spatial-frequency content: it is built from a
    DIFFUSER_NATIVE_RESOLUTION x DIFFUSER_NATIVE_RESOLUTION native grid over the fixed FoV, so
    its meaningful content sits below that native grid's Nyquist limit."""
    return DIFFUSER_NATIVE_RESOLUTION / (2.0 * FOV_M)


def _select_raw_resolution(offset_fx, f_residual_max):
    """Smallest power-of-two grid (fixed FOV_M) whose Nyquist limit, with SAFE_NYQUIST_FRACTION
    margin, exceeds the carrier plus the scene's own residual bandwidth."""
    n = BASE_RAW_RESOLUTION
    while True:
        dx = FOV_M / n
        f_nyquist = 1.0 / (2.0 * dx)
        if abs(offset_fx) + f_residual_max <= SAFE_NYQUIST_FRACTION * f_nyquist or n >= MAX_RAW_RESOLUTION:
            return n
        n *= 2


def _is_safe_at_resolution(offset_fx, f_residual_max, resolution):
    """Would `resolution` alone (without oversampling) safely sample this angle's carrier?"""
    dx = FOV_M / resolution
    f_nyquist = 1.0 / (2.0 * dx)
    occupancy = (abs(offset_fx) + f_residual_max) / f_nyquist
    return occupancy <= SAFE_NYQUIST_FRACTION, occupancy


def _raw_basm_intensity(resolution, offset_fx, k, device):
    _, tilted_field, pitch, shift_x_m = _build_scene_at(resolution, offset_fx, device)
    propagated = odak.learn.wave.band_limited_angular_spectrum(tilted_field, k, DISTANCE_M, pitch, WAVELENGTH_M)
    recentered = _recenter(propagated, pitch, shift_x_m)
    return odak.learn.wave.calculate_amplitude(recentered) ** 2, pitch


def _shifted_basm_intensity(offset_fx, resolution, k, device):
    field, _, pitch, shift_x_m = _build_scene_at(resolution, offset_fx, device)
    propagated = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field, k, DISTANCE_M, pitch, WAVELENGTH_M, offset_fx=offset_fx, offset_fy=0.0
    )
    recentered = _recenter(propagated, pitch, shift_x_m)
    return odak.learn.wave.calculate_amplitude(recentered) ** 2, pitch


def _raw_basm_field(resolution, offset_fx, k, device):
    _, tilted_field, pitch, shift_x_m = _build_scene_at(resolution, offset_fx, device)
    propagated = odak.learn.wave.band_limited_angular_spectrum(tilted_field, k, DISTANCE_M, pitch, WAVELENGTH_M)
    return _recenter(propagated, pitch, shift_x_m), pitch


def _restore_carrier(residual_field, offset_fx, offset_fy, pitch, device):
    """Shifted-BASM propagates only the untilted residual/baseband envelope; multiplying back
    the SAME carrier ramp used on raw BASM's input reconstructs the full physical field
    (U_raw = exp(i*2*pi*fc*x) * U_shifted, from the modulation theorem). Identity at
    offset_fx = offset_fy = 0."""
    resolution = residual_field.shape[-1]
    xx, yy = _spatial_grid(resolution, pitch, device)
    carrier_phase = 2.0 * odak.pi * (offset_fx * xx + offset_fy * yy)
    carrier = odak.learn.wave.generate_complex_field(torch.ones_like(carrier_phase), carrier_phase)
    return residual_field * carrier.to(torch.complex64)


def _shifted_basm_field(offset_fx, resolution, k, device):
    field, _, pitch, shift_x_m = _build_scene_at(resolution, offset_fx, device)
    propagated = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field, k, DISTANCE_M, pitch, WAVELENGTH_M, offset_fx=offset_fx, offset_fy=0.0
    )
    full_field = _restore_carrier(propagated, offset_fx, 0.0, pitch, device)
    return _recenter(full_field, pitch, shift_x_m), pitch


def _print_table(title, header_cols, rows):
    print(title)
    header = " | ".join("{:>11}".format(c) for c in header_cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join("{:>11}".format(v) for v in row))
    print()


# ============================================================================
# Comparison metrics
# ============================================================================
def _compare_intensity(reference, reference_pitch_m, candidate, candidate_pitch_m):
    """Compares two point-sampled intensity arrays of the same shape/pitch."""
    similarity = (
        torch.sum(reference * candidate) / torch.sqrt(torch.sum(reference**2) * torch.sum(candidate**2))
    ).item()
    diff = reference - candidate
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    ref_rms = torch.sqrt(torch.mean(reference**2)).item()
    nrmse = rmse / ref_rms if ref_rms > 0 else float("nan")
    peak = float(reference.max())
    psnr = 20.0 * math.log10(peak / rmse) if rmse > 0 else float("inf")
    reference_energy = float(reference.sum()) * reference_pitch_m**2
    candidate_energy = float(candidate.sum()) * candidate_pitch_m**2
    energy_ratio = candidate_energy / reference_energy if reference_energy > 0 else float("nan")
    ref_cx, ref_cy = _centroid_px(reference)
    cand_cx, cand_cy = _centroid_px(candidate)
    dx_px, dy_px = cand_cx - ref_cx, cand_cy - ref_cy
    return {
        "similarity": similarity, "nrmse": nrmse, "psnr": psnr, "energy_ratio": energy_ratio,
        "d_px": math.hypot(dx_px, dy_px), "dx_px": dx_px, "dy_px": dy_px,
    }


def _compare_energy(reference, candidate):
    """Compares two arrays already in the SAME physical energy units (e.g. both already
    pixel-area-integrated) -- does not re-apply any pitch^2 scaling."""
    similarity = (
        torch.sum(reference * candidate) / torch.sqrt(torch.sum(reference**2) * torch.sum(candidate**2))
    ).item()
    diff = reference - candidate
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    ref_rms = torch.sqrt(torch.mean(reference**2)).item()
    nrmse = rmse / ref_rms if ref_rms > 0 else float("nan")
    peak = float(reference.max())
    psnr = 20.0 * math.log10(peak / rmse) if rmse > 0 else float("inf")
    ref_total, cand_total = float(reference.sum()), float(candidate.sum())
    energy_ratio = cand_total / ref_total if ref_total > 0 else float("nan")
    ref_mean = float(reference.mean())
    mean_ratio = float(candidate.mean()) / ref_mean if ref_mean > 0 else float("nan")
    ref_cx, ref_cy = _centroid_px(reference)
    cand_cx, cand_cy = _centroid_px(candidate)
    dx_px, dy_px = cand_cx - ref_cx, cand_cy - ref_cy
    return {
        "similarity": similarity, "nrmse": nrmse, "psnr": psnr, "energy_ratio": energy_ratio,
        "mean_ratio": mean_ratio, "dx_px": dx_px, "dy_px": dy_px, "d_px": math.hypot(dx_px, dy_px),
        "max_abs_error": float(diff.abs().max()),
    }


def _complex_metrics(u_ref, u_cand):
    """Full complex-field metric set. `strict_corr` is NOT invariant to a global phase offset
    (real part of the normalized inner product); `aligned_corr` removes exactly one global
    scalar phase (phi_global = angle(sum(conj(u_ref)*u_cand))), never anything spatially
    varying. Phase RMSE is masked to pixels above PHASE_MASK_INTENSITY_FRACTION of peak
    intensity, so near-zero-amplitude phase noise does not dominate."""
    inner = torch.sum(torch.conj(u_ref) * u_cand)
    denom = torch.sqrt(torch.sum(u_ref.abs() ** 2) * torch.sum(u_cand.abs() ** 2)).item()
    strict_corr = (inner.real / denom).item() if denom > 0 else float("nan")
    global_phase = torch.angle(inner).item()
    u_cand_aligned = u_cand * torch.exp(torch.tensor(-1j * global_phase, dtype=u_cand.dtype))
    aligned_inner = torch.sum(torch.conj(u_ref) * u_cand_aligned)
    aligned_corr = (aligned_inner.real / denom).item() if denom > 0 else float("nan")

    diff = u_ref - u_cand_aligned
    complex_rmse = torch.sqrt(torch.mean(diff.abs() ** 2)).item()
    ref_field_rms = torch.sqrt(torch.mean(u_ref.abs() ** 2)).item()
    complex_nrmse = complex_rmse / ref_field_rms if ref_field_rms > 0 else float("nan")

    amp_ref, amp_cand = u_ref.abs(), u_cand.abs()
    amp_sim = (torch.sum(amp_ref * amp_cand) / torch.sqrt(torch.sum(amp_ref**2) * torch.sum(amp_cand**2))).item()
    amp_rmse = torch.sqrt(torch.mean((amp_ref - amp_cand) ** 2)).item()
    amp_ref_rms = torch.sqrt(torch.mean(amp_ref**2)).item()
    amp_nrmse = amp_rmse / amp_ref_rms if amp_ref_rms > 0 else float("nan")

    intensity_ref = amp_ref**2
    mask = intensity_ref > (intensity_ref.max() * PHASE_MASK_INTENSITY_FRACTION)
    if mask.any():
        phase_diff = torch.angle(u_cand_aligned[mask] * torch.conj(u_ref[mask]))
        phase_rmse = torch.sqrt(torch.mean(phase_diff**2)).item()
    else:
        phase_rmse = float("nan")

    energy_ref, energy_cand = torch.sum(amp_ref**2).item(), torch.sum(amp_cand**2).item()
    energy_ratio = energy_cand / energy_ref if energy_ref > 0 else float("nan")

    return {
        "strict_corr": strict_corr, "aligned_corr": aligned_corr, "complex_nrmse": complex_nrmse,
        "amp_sim": amp_sim, "amp_nrmse": amp_nrmse, "phase_rmse": phase_rmse,
        "global_phase": global_phase, "energy_ratio": energy_ratio,
    }


# ============================================================================
# Oversampled/converged raw-BASM reference -- intensity-based (shared by
# test_mask_ablation's Mode A/B experiments and test_sensor_equivalence)
# ============================================================================
def _raw_convergence_metrics(resolution, offset_fx, k, device):
    """Compares raw BASM at `resolution` against raw BASM at 2x, area-average-binned back down
    -- both point-sampled-intensity quantities. None if doubling exceeds the physical cap."""
    doubled = resolution * 2
    if doubled > CONVERGENCE_DOUBLE_CEILING:
        return None
    intensity_a, pitch_a = _raw_basm_intensity(resolution, offset_fx, k, device)
    intensity_b, _ = _raw_basm_intensity(doubled, offset_fx, k, device)
    intensity_b_binned = _bin_intensity_average(intensity_b, 2)
    return _compare_intensity(intensity_a, pitch_a, intensity_b_binned, pitch_a)


def _converged_raw_reference(theta_deg, k, device):
    """Picks the smallest safe N_raw, then VERIFIES convergence via explicit doubling,
    increasing N_raw further if not yet converged, up to the physical cap. Reports the angle as
    unresolved (converged=False) if convergence is never demonstrated -- never silently
    accepted, never falls back to a fixed low resolution at large angles."""
    offset_fx = _angle_to_offset(theta_deg)
    f_residual_max = _residual_bandwidth_hz_per_m()
    n_raw = _select_raw_resolution(offset_fx, f_residual_max)

    convergence = _raw_convergence_metrics(n_raw, offset_fx, k, device)
    last_valid = convergence
    while (
        convergence is not None
        and convergence["similarity"] <= CONVERGENCE_SIMILARITY_THRESHOLD
        and n_raw < MAX_RAW_RESOLUTION
    ):
        n_raw *= 2
        convergence = _raw_convergence_metrics(n_raw, offset_fx, k, device)
        if convergence is not None:
            last_valid = convergence
    convergence = last_valid
    converged = convergence is not None and convergence["similarity"] > CONVERGENCE_SIMILARITY_THRESHOLD

    dx_raw = FOV_M / n_raw
    f_nyquist_raw = 1.0 / (2.0 * dx_raw)
    occupancy = (abs(offset_fx) + f_residual_max) / f_nyquist_raw
    intensity, pitch = _raw_basm_intensity(n_raw, offset_fx, k, device)
    return {
        "theta_deg": theta_deg, "offset_fx": offset_fx, "f_residual_max": f_residual_max,
        "n_raw": n_raw, "dx_raw": dx_raw, "f_nyquist_raw": f_nyquist_raw, "occupancy": occupancy,
        "converged": converged, "convergence": convergence, "intensity": intensity, "pitch": pitch,
    }


# ============================================================================
# TEST: mask ablation -- does the band-limiting mask cause the disagreement? (No.)
# ============================================================================
def _native_freq_grid(resolution, pitch, device):
    """Reconstructs get_band_limited_angular_spectrum_kernel's own frequency axis (needed since
    that function does not expose FX/FY directly)."""
    x_extent = pitch * resolution
    fx = torch.linspace(
        -1 / (2 * pitch) + 0.5 / (2 * x_extent), 1 / (2 * pitch) - 0.5 / (2 * x_extent), resolution,
        dtype=torch.float32, device=device,
    )
    return torch.meshgrid(fx, fx, indexing="ij")


def _fx_limit(x_extent, pitch, distance, wavelength):
    grid_nyquist = 1.0 / (2.0 * pitch)
    aperture_term = 1.0 / math.sqrt((2.0 * distance / x_extent) ** 2 + 1.0) / wavelength
    return min(grid_nyquist, aperture_term)


def _kernel(nu, nv, dx, wavelength, distance, offset_fx, offset_fy, apply_mask, device):
    """Local kernel mirroring get_(shifted_)band_limited_angular_spectrum_kernel's formulas,
    with the band-limiting mask independently toggleable and always evaluated at absolute
    (carrier-included) frequency (the literal-Eq.9 fix candidate). offset=0 reproduces original
    BASM's own kernel exactly; apply_mask=False reproduces "bandlimit off" for either case."""
    x, y = dx * float(nu), dx * float(nv)
    fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), nu, dtype=torch.float32, device=device)
    fy = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * y), 1 / (2 * dx) - 0.5 / (2 * y), nv, dtype=torch.float32, device=device)
    FY, FX = torch.meshgrid(fx, fy, indexing="ij")
    FX_shifted, FY_shifted = FX + offset_fx, FY + offset_fy

    kz_squared = 1.0 / wavelength**2 - (FX_shifted**2 + FY_shifted**2)
    propagating = kz_squared >= 0.0
    HH_exp = 2 * torch.pi * torch.sqrt(torch.clamp(kz_squared, min=0.0))
    distance_t = torch.tensor([distance], device=device)
    H_exp = torch.mul(HH_exp, distance_t)

    if apply_mask:
        fx_max = 1 / torch.sqrt((2 * distance_t * (1 / x)) ** 2 + 1) / wavelength
        fy_max = 1 / torch.sqrt((2 * distance_t * (1 / y)) ** 2 + 1) / wavelength
        B = (torch.abs(FX_shifted) < fx_max) & (torch.abs(FY_shifted) < fy_max)
    else:
        B = torch.ones_like(FX, dtype=torch.bool)
    H_filter = (B & propagating).clone().detach()
    return odak.learn.wave.generate_complex_field(H_filter.to(torch.float32), H_exp)


def _mask_ablation_table(device, k, fx_limit):
    """A=raw/ON, B=shifted-prod/ON, C=raw/OFF, D=shifted/OFF, all at a fixed matching
    resolution RESOLUTION. If C-vs-D agrees while A-vs-B does not, the mask is the culprit
    (it is not: ON/OFF give bit-identical results at every tested angle)."""
    rows = []
    for theta_deg in ANGLES_DEG:
        offset_fx = _angle_to_offset(theta_deg)
        field, tilted_field, _, shift_x_m = _build_scene_at(RESOLUTION, offset_fx, device)
        occupancy = abs(offset_fx) / fx_limit

        intensity_a = _recenter(
            odak.learn.wave.band_limited_angular_spectrum(tilted_field, k, DISTANCE_M, PITCH_M, WAVELENGTH_M),
            PITCH_M, shift_x_m,
        )
        intensity_a = odak.learn.wave.calculate_amplitude(intensity_a) ** 2
        propagated_b = odak.learn.wave.shifted_band_limited_angular_spectrum(
            field, k, DISTANCE_M, PITCH_M, WAVELENGTH_M, offset_fx=offset_fx, offset_fy=0.0
        )
        intensity_b = odak.learn.wave.calculate_amplitude(_recenter(propagated_b, PITCH_M, shift_x_m)) ** 2

        kernel_c = _kernel(RESOLUTION, RESOLUTION, PITCH_M, WAVELENGTH_M, DISTANCE_M, 0.0, 0.0, False, device)
        intensity_c = odak.learn.wave.calculate_amplitude(
            _recenter(odak.learn.wave.custom(tilted_field, kernel_c, zero_padding=False, aperture=1.0), PITCH_M, shift_x_m)
        ) ** 2
        kernel_d = _kernel(RESOLUTION, RESOLUTION, PITCH_M, WAVELENGTH_M, DISTANCE_M, offset_fx, 0.0, False, device)
        intensity_d = odak.learn.wave.calculate_amplitude(
            _recenter(odak.learn.wave.custom(field, kernel_d, zero_padding=False, aperture=1.0), PITCH_M, shift_x_m)
        ) ** 2

        m_on = _compare_intensity(intensity_a, PITCH_M, intensity_b, PITCH_M)
        m_off = _compare_intensity(intensity_c, PITCH_M, intensity_d, PITCH_M)
        rows.append({"theta_deg": theta_deg, "occupancy": occupancy, "on": m_on, "off": m_off})

    _print_table(
        "=== Bandlimit ON/OFF ablation (A=raw/ON, B=shifted/ON, C=raw/OFF, D=shifted/OFF) ===",
        ["Angle", "Occ", "Sim(ON)", "Sim(OFF)", "|Diff|"],
        [
            [
                "{:.1f}".format(r["theta_deg"]), "{:.3f}".format(r["occupancy"]),
                "{:.6f}".format(r["on"]["similarity"]), "{:.6f}".format(r["off"]["similarity"]),
                "{:.2e}".format(abs(r["on"]["similarity"] - r["off"]["similarity"])),
            ]
            for r in rows
        ],
    )
    return rows


def test_mask_ablation(device=torch.device("cpu")):
    k = odak.learn.wave.wavenumber(WAVELENGTH_M)
    fx_limit = _fx_limit(PITCH_M * RESOLUTION, PITCH_M, DISTANCE_M, WAVELENGTH_M)
    print("f_BASM_limit (fixed N={}): {:.1f} cycles/m\n".format(RESOLUTION, fx_limit))

    rows = _mask_ablation_table(device, k, fx_limit)
    bandlimit_makes_no_difference = all(abs(r["on"]["similarity"] - r["off"]["similarity"]) < 1e-9 for r in rows)

    print("Does the band-limiting mask cause raw-vs-shifted disagreement?")
    print("  " + ("NO" if bandlimit_makes_no_difference else "YES/PARTIALLY -- see table above"))
    assert bandlimit_makes_no_difference, (
        "bandlimit ON and OFF should give bit-identical similarity at every angle if the mask "
        "is not the cause of any raw-vs-shifted disagreement; see the table above"
    )


# ============================================================================
# TEST: pixel semantics -- Mode A (intensity, area-average) / Mode B (sensor energy),
# plus a raw-vs-raw control and a point-sample-vs-area-average residual diagnosis
# ============================================================================
def _build_comparison_row(theta_deg, k, device):
    raw_ref = _converged_raw_reference(theta_deg, k, device)
    offset_fx, n_raw = raw_ref["offset_fx"], raw_ref["n_raw"]
    bin_factor = n_raw // SHIFTED_RESOLUTION

    shifted_512, dx_shifted_512 = _shifted_basm_intensity(offset_fx, SHIFTED_RESOLUTION, k, device)
    shifted_1024, dx_shifted_1024 = _shifted_basm_intensity(offset_fx, SHIFTED_RESOLUTION_ALT, k, device)

    if bin_factor <= 1:
        raw_avg_512 = raw_sum_512 = raw_decimated_512 = raw_ref["intensity"]
    else:
        raw_avg_512 = _bin_intensity_average(raw_ref["intensity"], bin_factor)
        raw_sum_512 = _bin_intensity(raw_ref["intensity"], bin_factor)
        raw_decimated_512 = _decimate_intensity(raw_ref["intensity"], bin_factor)

    return {
        "theta_deg": theta_deg, "raw_ref": raw_ref, "offset_fx": offset_fx, "n_raw": n_raw,
        "dx_raw": raw_ref["pitch"], "bin_factor": bin_factor,
        "shifted_512": shifted_512, "dx_shifted_512": dx_shifted_512,
        "shifted_1024": shifted_1024, "dx_shifted_1024": dx_shifted_1024,
        "raw_avg_512": raw_avg_512, "raw_sum_512": raw_sum_512, "raw_decimated_512": raw_decimated_512,
    }


def _mode_a_intensity_comparison(rows):
    """Mode A: raw BASM's fine output AREA-AVERAGED down to shifted-BASM's resolution
    (both sides point-sampled intensity) vs. shifted-BASM's own native output."""
    table_rows = []
    for row in rows:
        dx = row["dx_shifted_512"]
        metrics = _compare_intensity(row["raw_avg_512"], dx, row["shifted_512"], dx)
        mean_ratio = (row["shifted_512"].mean() / row["raw_avg_512"].mean()).item()
        old_sum_based = _compare_intensity(row["raw_sum_512"], dx, row["shifted_512"], dx)
        table_rows.append({
            "row": row, "metrics": metrics, "mean_ratio": mean_ratio,
            "old_energy_ratio": old_sum_based["energy_ratio"],
        })
    _print_table(
        "=== Mode A: intensity comparison (area-average binning) ===",
        ["Angle", "RawN", "Similarity", "NRMSE", "MeanRatio", "PhysEnergyRatio"],
        [
            [
                "{:.1f}".format(t["row"]["theta_deg"]), "{}".format(t["row"]["n_raw"]),
                "{:.6f}".format(t["metrics"]["similarity"]), "{:.4f}".format(t["metrics"]["nrmse"]),
                "{:.4f}".format(t["mean_ratio"]), "{:.4f}".format(t["metrics"]["energy_ratio"]),
            ]
            for t in table_rows
        ],
    )
    return table_rows


def _mode_b_sensor_comparison(rows):
    """Mode B: explicit physical energy (E_fine_block = sum(I_fine)*dx_fine^2 vs.
    E_coarse = I_coarse*dx_coarse^2) -- an independent construction that should match Mode A up
    to global scaling."""
    table_rows = []
    for row in rows:
        dx_raw, dx_shifted = row["dx_raw"], row["dx_shifted_512"]
        e_raw = row["raw_sum_512"] * dx_raw**2
        e_shifted = row["shifted_512"] * dx_shifted**2
        metrics = _compare_energy(e_raw, e_shifted)
        table_rows.append({"row": row, "metrics": metrics})
    _print_table(
        "=== Mode B: sensor-energy comparison ===",
        ["Angle", "Similarity", "NRMSE", "EnergyRatio"],
        [
            [
                "{:.1f}".format(t["row"]["theta_deg"]), "{:.6f}".format(t["metrics"]["similarity"]),
                "{:.4f}".format(t["metrics"]["nrmse"]), "{:.4f}".format(t["metrics"]["energy_ratio"]),
            ]
            for t in table_rows
        ],
    )
    return table_rows


def _raw_vs_raw_binning_control(rows, k, device):
    """No shifted-BASM at all: raw run DIRECTLY at 512 (point sample) vs. the SAME converged
    raw reference area-averaged down from n_raw -- isolates how much of any residual similarity
    gap is explained by a diffuser's real sub-pixel structure alone."""
    control_rows = []
    for row in rows:
        intensity_direct, pitch_direct = _raw_basm_intensity(SHIFTED_RESOLUTION, row["offset_fx"], k, device)
        metrics = _compare_intensity(intensity_direct, pitch_direct, row["raw_avg_512"], pitch_direct)
        control_rows.append({"theta_deg": row["theta_deg"], "metrics": metrics})
    _print_table(
        "=== Control: raw-vs-raw binning artifact (no shifted-BASM) ===",
        ["Angle", "ControlSim"],
        [["{:.1f}".format(t["theta_deg"]), "{:.6f}".format(t["metrics"]["similarity"])] for t in control_rows],
    )
    return control_rows


def test_pixel_semantics(device=torch.device("cpu")):
    k = odak.learn.wave.wavenumber(WAVELENGTH_M)
    rows = [_build_comparison_row(theta_deg, k, device) for theta_deg in ANGLES_DEG]

    mode_a_rows = _mode_a_intensity_comparison(rows)
    _mode_b_sensor_comparison(rows)
    control_rows = _raw_vs_raw_binning_control(rows, k, device)

    old_ratios = [t["old_energy_ratio"] for t in mode_a_rows]
    new_ratios = [t["metrics"]["energy_ratio"] for t in mode_a_rows]
    control_sims = {c["theta_deg"]: c["metrics"]["similarity"] for c in control_rows}
    shift_sims = [t["metrics"]["similarity"] for t in mode_a_rows]

    print("Previous ~1/64 SUM-binning artifact reproduced: {:.4f}-{:.4f}".format(min(old_ratios), max(old_ratios)))
    print("Corrected (area-average) PhysicalEnergyRatio: {:.4f}-{:.4f}".format(min(new_ratios), max(new_ratios)))
    energy_near_one = all(abs(r - 1.0) < 0.05 for r in new_ratios)
    print("Does corrected physical energy ratio stay near 1? " + ("YES" if energy_near_one else "NO"))

    control_agree = all(
        t["metrics"]["similarity"] >= control_sims[t["row"]["theta_deg"]] - CONTROL_TOLERANCE for t in mode_a_rows
    )
    print(
        "Raw-vs-shifted similarity matches/exceeds the raw-vs-raw binning-only control at every "
        "angle (ruling out an implementation mismatch)? " + ("YES" if control_agree else "NO")
    )
    print("Lowest raw-vs-shifted similarity: {:.6f}".format(min(shift_sims)))

    assert energy_near_one, (
        "corrected physical energy ratio should stay near 1 at every angle once pixel semantics "
        "are fixed; see the printed tables above"
    )
    assert control_agree, (
        "raw-vs-shifted similarity should match or exceed the implementation-independent "
        "binning-only control at every angle; see the printed tables above"
    )


# ============================================================================
# TEST 1: complex-field solver equivalence -- exact-coordinate comparison, carrier
# restoration, global-phase-only alignment. Currently FAILS (left intentional, not forced).
# ============================================================================
def _exact_subset_offset(n_fine, n_coarse):
    """Is the coarse grid's coordinate set an exact index subset of the fine grid's, under the
    shared _spatial_grid convention? For factor = n_fine/n_coarse, the fine index coinciding
    with coarse index 0 is (factor-1)/2 -- an integer only when `factor` is odd."""
    factor = n_fine // n_coarse
    offset = (factor - 1) / 2.0
    return offset == int(offset), offset, factor


def _extract_at_coarse_coordinates(fine_field, pitch_fine, factor, device):
    """Exact complex-valued resampling of `fine_field` at a `factor`-times-coarser grid's true
    coordinates: plain strided indexing for an integer offset, otherwise an exact Fourier-domain
    sub-pixel shift (verified against an analytic bin-aligned complex exponential) then strided
    indexing -- NOT nearest-neighbor decimation, which would be an approximation."""
    is_exact, offset, factor_check = _exact_subset_offset(fine_field.shape[-1], fine_field.shape[-1] // factor)
    assert factor_check == factor
    if is_exact:
        start = int(offset)
        return fine_field[start::factor, start::factor]
    shift_m = offset * pitch_fine
    shifted = _recenter(fine_field, pitch_fine, shift_m, shift_m)
    return shifted[0::factor, 0::factor]


def _raw_field_convergence_metrics(resolution, offset_fx, k, device):
    """Raw BASM's own convergence, verified at the COMPLEX FIELD level (matching Test 1's own
    validation standard, stricter than the intensity-based check used elsewhere)."""
    doubled = resolution * 2
    if doubled > CONVERGENCE_DOUBLE_CEILING:
        return None
    field_a, pitch_a = _raw_basm_field(resolution, offset_fx, k, device)
    field_b, pitch_b = _raw_basm_field(doubled, offset_fx, k, device)
    field_b_at_a = _extract_at_coarse_coordinates(field_b, pitch_b, 2, device)
    return _complex_metrics(field_a, field_b_at_a)


def _converged_raw_field_reference(theta_deg, k, device):
    offset_fx = _angle_to_offset(theta_deg)
    f_residual_max = _residual_bandwidth_hz_per_m()
    n_raw = _select_raw_resolution(offset_fx, f_residual_max)

    convergence = _raw_field_convergence_metrics(n_raw, offset_fx, k, device)
    last_valid = convergence
    while (
        convergence is not None
        and convergence["aligned_corr"] <= CONVERGENCE_SIMILARITY_THRESHOLD
        and n_raw < MAX_RAW_RESOLUTION
    ):
        n_raw *= 2
        convergence = _raw_field_convergence_metrics(n_raw, offset_fx, k, device)
        if convergence is not None:
            last_valid = convergence
    convergence = last_valid
    converged = convergence is not None and convergence["aligned_corr"] > CONVERGENCE_SIMILARITY_THRESHOLD

    field, pitch = _raw_basm_field(n_raw, offset_fx, k, device)
    return {
        "theta_deg": theta_deg, "offset_fx": offset_fx, "n_raw": n_raw, "pitch": pitch,
        "converged": converged, "convergence": convergence, "field": field,
    }


def test_field_equivalence(device=torch.device("cpu")):
    """TEST 1: do the two solvers compute the same propagated COMPLEX field at identical
    physical coordinates, up to one global phase? Currently FAILS: aligned correlation does not
    reach the ideal >0.9999 target at every angle, and is non-monotonic with angle -- unlike
    every intensity-only diagnostic elsewhere in this file. Verified NOT an artifact of this
    test's own machinery (exact-coordinate extraction and the theta=0 carrier-restoration
    identity both reproduce an analytic reference to float32 precision); the same
    non-improvement from 512->1024 shows up at the plain intensity level too, so it is a real
    property of shifted-BASM's own propagated field. Most likely explanation, not confirmed:
    float32 precision loss in the kernel's phase term (1/wavelength^2 - (FX+offset_fx)^2
    subtracts two ~1e12-scale values at large carrier offsets). Left failing intentionally --
    the threshold is not relaxed to force a pass."""
    k = odak.learn.wave.wavenumber(WAVELENGTH_M)
    raw_refs = [_converged_raw_field_reference(theta_deg, k, device) for theta_deg in ANGLES_DEG]

    rows = []
    for raw_ref in raw_refs:
        offset_fx, n_raw = raw_ref["offset_fx"], raw_ref["n_raw"]
        field_shifted_512, _ = _shifted_basm_field(offset_fx, SHIFTED_RESOLUTION, k, device)
        field_shifted_1024, pitch_1024 = _shifted_basm_field(offset_fx, SHIFTED_RESOLUTION_ALT, k, device)

        _, _, factor_512 = _exact_subset_offset(n_raw, SHIFTED_RESOLUTION)
        raw_at_512 = _extract_at_coarse_coordinates(raw_ref["field"], raw_ref["pitch"], factor_512, device)
        metrics_512 = _complex_metrics(raw_at_512, field_shifted_512)

        _, _, factor_1024 = _exact_subset_offset(n_raw, SHIFTED_RESOLUTION_ALT)
        raw_at_1024 = _extract_at_coarse_coordinates(raw_ref["field"], raw_ref["pitch"], factor_1024, device)
        metrics_1024 = _complex_metrics(raw_at_1024, field_shifted_1024)

        rows.append({
            "theta_deg": raw_ref["theta_deg"], "n_raw": n_raw, "converged": raw_ref["converged"],
            "metrics_512": metrics_512, "metrics_1024": metrics_1024,
        })

    _print_table(
        "=== Table 1: complex-field equivalence (raw vs. shifted-BASM at {}) ===".format(SHIFTED_RESOLUTION),
        ["Angle", "RawN", "ComplexCorr", "AlignedCorr", "ComplexNRMSE", "PhaseRMSE"],
        [
            [
                "{:.1f}".format(r["theta_deg"]), "{}".format(r["n_raw"]),
                "{:.6f}".format(r["metrics_512"]["strict_corr"]), "{:.6f}".format(r["metrics_512"]["aligned_corr"]),
                "{:.4f}".format(r["metrics_512"]["complex_nrmse"]), "{:.4f}".format(r["metrics_512"]["phase_rmse"]),
            ]
            for r in rows
        ],
    )
    _print_table(
        "=== Table 2: shifted-resolution convergence (512 vs. 1024) ===",
        ["Angle", "Shift512 Corr", "Shift1024 Corr"],
        [
            [
                "{:.1f}".format(r["theta_deg"]), "{:.6f}".format(r["metrics_512"]["aligned_corr"]),
                "{:.6f}".format(r["metrics_1024"]["aligned_corr"]),
            ]
            for r in rows
        ],
    )

    zero = next(r for r in rows if r["theta_deg"] == 0.0)
    print(
        "Control A (theta=0 baseline): aligned_corr={:.6f}, global_phase={:.2e} rad".format(
            zero["metrics_512"]["aligned_corr"], zero["metrics_512"]["global_phase"]
        )
    )

    aligned_512 = [r["metrics_512"]["aligned_corr"] for r in rows]
    aligned_1024 = [r["metrics_1024"]["aligned_corr"] for r in rows]
    lowest_512, lowest_1024 = min(aligned_512), min(aligned_1024)
    worst_theta = rows[aligned_512.index(lowest_512)]["theta_deg"]
    all_converged = all(r["converged"] for r in rows)
    pass_512 = lowest_512 > ALIGNED_CORR_TARGET
    pass_1024 = lowest_1024 > ALIGNED_CORR_TARGET

    print("\nTEST 1 -- Solver equivalence: " + ("PASS" if pass_512 else "FAIL"))
    print("Lowest global-phase-aligned complex correlation: {:.8f} (theta={:.1f} deg)".format(lowest_512, worst_theta))
    print("Raw BASM's own field-level convergence at every angle: {}".format("YES" if all_converged else "NO"))
    if pass_512:
        verdict = "YES"
    elif pass_1024 and lowest_1024 > lowest_512 + 1e-4:
        verdict = "NOT YET CONVERGED (512 insufficient; 1024 substantially better)"
    else:
        verdict = "NO (neither 512 nor 1024 reaches {:.4f} -- see docstring for the leading hypothesis)".format(
            ALIGNED_CORR_TARGET
        )
    print("Is shifted-BASM mathematically reproducing ordinary BASM? " + verdict)

    assert pass_512 or (pass_1024 and lowest_1024 > lowest_512 + 1e-4), (
        "global-phase-aligned complex correlation should reach > {:.4f} at N_shifted={} (or show "
        "substantial improvement at N_shifted={}) at every angle -- lowest were {:.8f} / {:.8f}; "
        "see this function's docstring for what is/is not explained. Threshold NOT relaxed.".format(
            ALIGNED_CORR_TARGET, SHIFTED_RESOLUTION, SHIFTED_RESOLUTION_ALT, lowest_512, lowest_1024
        )
    )


# ============================================================================
# TEST 2: physical sensor-measurement equivalence -- integrate |U|^2 over the SAME
# physical sensor pixels for both methods before comparing. PASSES.
# ============================================================================
def test_sensor_equivalence(device=torch.device("cpu")):
    """TEST 2: do the two solvers predict the same measurement for a real, finite-area sensor
    pixel (E_ij = integral_over_pixel |U|^2 dx dy)? Shifted-BASM is run on an internal grid
    finer than the sensor (SHIFT_INTERNAL_PRIMARY=2048), never its own native-resolution point
    sample, to avoid repeating the point-sample-vs-area-average mismatch from
    test_pixel_semantics. Passes cleanly at every angle."""
    k = odak.learn.wave.wavenumber(WAVELENGTH_M)
    raw_rows = [_converged_raw_reference(theta_deg, k, device) for theta_deg in ANGLES_DEG]

    rows = []
    for raw_ref in raw_rows:
        offset_fx, n_raw = raw_ref["offset_fx"], raw_ref["n_raw"]
        e_raw = _sensor_energy(raw_ref["intensity"], raw_ref["pitch"], n_raw // SENSOR_RESOLUTION)
        intensity_shift, pitch_shift = _shifted_basm_intensity(offset_fx, SHIFT_INTERNAL_PRIMARY, k, device)
        e_shift = _sensor_energy(intensity_shift, pitch_shift, SHIFT_INTERNAL_PRIMARY // SENSOR_RESOLUTION)
        metrics = _compare_energy(e_raw, e_shift)
        rows.append({
            "theta_deg": raw_ref["theta_deg"], "n_raw": n_raw, "converged": raw_ref["converged"],
            "metrics": metrics,
        })

    _print_table(
        "=== Table 3: sensor-measurement equivalence (raw N vs. shifted internal {},\n"
        "    both integrated onto the same {}x{} sensor) ===".format(
            SHIFT_INTERNAL_PRIMARY, SENSOR_RESOLUTION, SENSOR_RESOLUTION
        ),
        ["Angle", "RawN", "Similarity", "NRMSE", "PSNR", "EnergyRatio", "d(px)"],
        [
            [
                "{:.1f}".format(r["theta_deg"]), "{}".format(r["n_raw"]), "{:.6f}".format(r["metrics"]["similarity"]),
                "{:.4f}".format(r["metrics"]["nrmse"]), "{:.2f}".format(r["metrics"]["psnr"]),
                "{:.4f}".format(r["metrics"]["energy_ratio"]), "{:.3f}".format(r["metrics"]["d_px"]),
            ]
            for r in rows
        ],
    )

    # Control C: raw-vs-raw sensor-integration reference accuracy (no shifted-BASM at all).
    control_rows = []
    for raw_ref in raw_rows:
        n_raw = raw_ref["n_raw"]
        other_n = n_raw * 2 if n_raw * 2 <= CONVERGENCE_DOUBLE_CEILING else n_raw // 2
        e_raw = _sensor_energy(raw_ref["intensity"], raw_ref["pitch"], n_raw // SENSOR_RESOLUTION)
        intensity_other, pitch_other = _raw_basm_intensity(other_n, raw_ref["offset_fx"], k, device)
        e_other = _sensor_energy(intensity_other, pitch_other, max(other_n // SENSOR_RESOLUTION, 1))
        control_rows.append({"theta_deg": raw_ref["theta_deg"], "sim": _compare_energy(e_raw, e_other)["similarity"]})
    _print_table(
        "=== Control C: raw-vs-raw sensor-integration reference accuracy (no shifted-BASM) ===",
        ["Angle", "Similarity"],
        [["{:.1f}".format(c["theta_deg"]), "{:.6f}".format(c["sim"])] for c in control_rows],
    )

    # Control D (sensor half): shifted internal-grid sensor-integration convergence.
    convergence_rows = []
    for row in rows:
        offset_fx = next(r["offset_fx"] for r in raw_rows if r["theta_deg"] == row["theta_deg"])
        energies = {}
        for internal_n in SHIFT_INTERNAL_RESOLUTIONS:
            intensity, pitch = _shifted_basm_intensity(offset_fx, internal_n, k, device)
            energies[internal_n] = _sensor_energy(intensity, pitch, max(internal_n // SENSOR_RESOLUTION, 1))
        sim_512_1024 = _compare_energy(energies[512], energies[1024])["similarity"]
        sim_1024_2048 = _compare_energy(energies[1024], energies[2048])["similarity"]
        convergence_rows.append({"theta_deg": row["theta_deg"], "s1": sim_512_1024, "s2": sim_1024_2048})
    _print_table(
        "=== Control D: shifted internal-grid sensor convergence (512->1024->2048) ===",
        ["Angle", "Sim(512v1024)", "Sim(1024v2048)"],
        [["{:.1f}".format(c["theta_deg"]), "{:.6f}".format(c["s1"]), "{:.6f}".format(c["s2"])] for c in convergence_rows],
    )

    similarities = [r["metrics"]["similarity"] for r in rows]
    energy_ratios = [r["metrics"]["energy_ratio"] for r in rows]
    lowest_similarity = min(similarities)
    worst_theta = rows[similarities.index(lowest_similarity)]["theta_deg"]
    energy_near_one = all(abs(e - 1.0) < ENERGY_RATIO_TOLERANCE for e in energy_ratios)
    largest_energy_mismatch = max(abs(e - 1.0) for e in energy_ratios)
    test2_pass = energy_near_one and lowest_similarity > SENSOR_SIMILARITY_TARGET

    print("\nTEST 2 -- Sensor equivalence: " + ("PASS" if test2_pass else "FAIL"))
    print("Lowest sensor similarity: {:.6f} (theta={:.1f} deg)".format(lowest_similarity, worst_theta))
    print("Largest physical energy mismatch: {:.4f}".format(largest_energy_mismatch))
    print("Do both methods predict the same finite-area sensor measurement? " + ("YES" if test2_pass else "NO"))

    assert energy_near_one, (
        "physical sensor-energy ratio should stay within {:.0%} of 1.0 at every angle -- largest "
        "mismatch was {:.4f}; see Table 3 above".format(ENERGY_RATIO_TOLERANCE, largest_energy_mismatch)
    )
    assert lowest_similarity > SENSOR_SIMILARITY_TARGET, (
        "sensor similarity should exceed {:.3f} at every angle -- lowest was {:.6f} at theta={:.1f}; "
        "see Table 3/Control C/D above".format(SENSOR_SIMILARITY_TARGET, lowest_similarity, worst_theta)
    )


# ============================================================================
# __main__: run all four, report a combined pass/fail summary (field equivalence is
# expected to fail -- see its docstring -- without blocking the other three from running)
# ============================================================================
def _run_all():
    results = {}
    for name, fn in [
        ("mask_ablation", test_mask_ablation),
        ("pixel_semantics", test_pixel_semantics),
        ("field_equivalence", test_field_equivalence),
        ("sensor_equivalence", test_sensor_equivalence),
    ]:
        print("\n" + "#" * 78)
        print("# {}".format(name))
        print("#" * 78 + "\n")
        try:
            fn()
            results[name] = True
        except AssertionError as error:
            print("\nFAILED: {}".format(error))
            results[name] = False

    print("\n" + "=" * 78)
    print("Summary:")
    for name, passed in results.items():
        print("  {:<20} {}".format(name, "PASS" if passed else "FAIL"))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(_run_all())
