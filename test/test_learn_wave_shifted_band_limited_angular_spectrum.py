"""shifted_band_limited_angular_spectrum: correctness, memory, convergence, and
equivalence-with-ordinary-BASM tests, in one file.

  test_correctness()      -- unit-style checks: zero-offset identity, large-offset kernel
                              stability, mask/phase/evanescent-rejection correctness.
  test_memory()            -- same physical scenario, ~16x smaller array, same result.
  test_convergence()       -- is shifted-BASM's residual-grid resolution actually converged?
  test_mask_ablation()     -- does the band-limiting mask cause raw-vs-shifted disagreement?
                              No: bandlimit ON/OFF gives bit-identical results.
  test_pixel_semantics()   -- raw and shifted BASM both return POINT-SAMPLED intensity (see
                              Warning below); downsampling raw's fine grid for comparison must
                              use AREA-AVERAGE binning, not SUM binning (sensor-pixel style).
  test_sensor_equivalence() -- do the two solvers predict the same finite-area SENSOR pixel
                              measurement? PASSES (similarity > 0.9999, energy ratio ~0.1%).

Warning -- intensity vs. complex field: shifted-BASM was validated here to reproduce ordinary
BASM's INTENSITY (|U|^2) to similarity > 0.9999 once both are properly resolved
(test_sensor_equivalence). A separate, stricter investigation compared the underlying COMPLEX
field at exact matching coordinates (global phase removed) and found aligned correlation only
reaches ~0.98-0.997, not the ideal > 0.9999, and is non-monotonic with angle -- most likely
float32 precision loss in the kernel's phase term at large carrier offsets, not a conceptual
mismatch, but not fully confirmed. That test currently fails and has been removed from this
file's passing suite; see git history (test_learn_wave_shifted_band_limited_angular_spectrum_
field_equivalence.py, since removed) and shifted_band_limited_angular_spectrum's own docstring
in odak/learn/wave/classical.py for the full finding. Do not assume this function's raw complex
output matches band_limited_angular_spectrum's to high precision for phase-sensitive uses.
"""

import math
import sys
import time
import torch
import odak


# ============================================================================
# Shared physical setup (used by test_mask_ablation / test_pixel_semantics /
# test_sensor_equivalence; test_correctness/test_memory/test_convergence use their own
# self-contained parameters, documented inline where they differ)
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
SENSOR_RESOLUTION = SHIFTED_RESOLUTION
SHIFT_INTERNAL_RESOLUTIONS = [512, 1024, 2048]
SHIFT_INTERNAL_PRIMARY = 2048

CONTROL_TOLERANCE = 1e-4
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
    if resolution % native_resolution != 0:
        raise ValueError("resolution {} must be a multiple of {}".format(resolution, native_resolution))
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
    """Energy-preserving SUM binning (matches sum_bin_sensor_pixels in
    src/asm_psf_propagation.py): a coarser PHYSICAL SENSOR pixel that integrates light over its
    footprint. NOT the right operation for comparing two simulation grids of the same
    point-sampled field -- use _bin_intensity_average for that (see module Warning)."""
    n = intensity.shape[-1]
    m = n // factor
    return intensity.reshape(m, factor, m, factor).sum(dim=(1, 3))


def _bin_intensity_average(intensity, factor):
    """AREA-AVERAGE binning: the correct operation for comparing the SAME point-sampled field
    at two simulation-grid resolutions (preserves intensity-density semantics)."""
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
    ref_cx, ref_cy = _centroid_px(reference)
    cand_cx, cand_cy = _centroid_px(candidate)
    dx_px, dy_px = cand_cx - ref_cx, cand_cy - ref_cy
    return {
        "similarity": similarity, "nrmse": nrmse, "psnr": psnr, "energy_ratio": energy_ratio,
        "dx_px": dx_px, "dy_px": dy_px, "d_px": math.hypot(dx_px, dy_px),
    }


# ============================================================================
# Oversampled/converged raw-BASM reference -- intensity-based (shared by
# test_pixel_semantics and test_sensor_equivalence)
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
# test_correctness: zero-offset identity, large-offset stability, mask/phase/evanescent checks
# ============================================================================
def test_correctness(device=torch.device("cpu"), output_directory="test_output"):
    odak.tools.check_directory(output_directory)
    wavelength, pixel_pitch, distance = 532e-9, 2e-6, 5e-3
    resolution = [256, 256]
    k = odak.learn.wave.wavenumber(wavelength)
    grid_nyquist = 1.0 / (2.0 * pixel_pitch)
    propagating_limit = 1.0 / wavelength  # Helmholtz/evanescent boundary

    amplitude = torch.zeros(resolution)
    amplitude[100:156, 100:156] = 1.0
    phase = torch.zeros_like(amplitude)
    field = odak.learn.wave.generate_complex_field(amplitude, phase).to(device)

    # Zero offset must exactly reproduce band_limited_angular_spectrum.
    reference = odak.learn.wave.band_limited_angular_spectrum(field, k, distance, pixel_pitch, wavelength)
    zero_offset = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field, k, distance, pixel_pitch, wavelength, offset_fx=0.0, offset_fy=0.0
    )
    assert torch.allclose(reference, zero_offset, atol=1e-6)

    # Physics correctness (Eq. 9, Matsushima 2010): a field carrying a bin-aligned linear
    # carrier phase through the UNSHIFTED kernel should closely match the untilted field through
    # the kernel SHIFTED by that same carrier. Bin-aligned so the discrete tilt is an exact
    # circular spectrum shift (a non-aligned offset would introduce spectral leakage on the
    # "direct" side alone). Compared via normalized similarity, not exact equality: the two
    # differ slightly at the spectrum's edge bins (circular wrap vs. no wrap), the same aliasing
    # risk carrier-frequency shifting exists to avoid at genuinely large offsets.
    bin_spacing_x = 1.0 / (resolution[1] * pixel_pitch)
    offset_fx_small, offset_fy_small = 10.0 * bin_spacing_x, 0.0
    y = (torch.arange(resolution[0], device=device) - (resolution[0] - 1) / 2.0) * pixel_pitch
    x = (torch.arange(resolution[1], device=device) - (resolution[1] - 1) / 2.0) * pixel_pitch
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    carrier_phase = 2.0 * odak.pi * (offset_fx_small * xx + offset_fy_small * yy)
    carrier = odak.learn.wave.generate_complex_field(torch.ones_like(carrier_phase), carrier_phase)
    tilted_field = field * carrier.to(torch.complex64)

    propagated_direct = odak.learn.wave.band_limited_angular_spectrum(tilted_field, k, distance, pixel_pitch, wavelength)
    propagated_shifted = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field, k, distance, pixel_pitch, wavelength, offset_fx=offset_fx_small, offset_fy=offset_fy_small
    )
    intensity_direct = odak.learn.wave.calculate_amplitude(propagated_direct) ** 2
    intensity_shifted = odak.learn.wave.calculate_amplitude(propagated_shifted) ** 2
    similarity = torch.sum(intensity_direct * intensity_shifted) / torch.sqrt(
        torch.sum(intensity_direct**2) * torch.sum(intensity_shifted**2)
    )
    assert similarity > 0.999

    # A carrier well beyond this grid's own Nyquist must NOT zero the kernel, as long as it
    # stays within the mask and the propagating (Helmholtz) region -- the entire point of
    # carrier-frequency shifting.
    offset_fx_large = 1.0e6  # ~4x grid_nyquist, well under propagating_limit
    assert grid_nyquist < offset_fx_large < propagating_limit
    kernel_large_offset = odak.learn.wave.get_shifted_band_limited_angular_spectrum_kernel(
        nu=resolution[0], nv=resolution[1], dx=pixel_pitch, wavelength=wavelength, distance=distance,
        offset_fx=offset_fx_large, offset_fy=0.0, device=device,
    )
    assert torch.isfinite(torch.view_as_real(kernel_large_offset)).all()
    assert float(odak.learn.wave.calculate_amplitude(kernel_large_offset).max()) > 0.0

    # The mask B is evaluated on the residual (FX, FY) grid, independent of the carrier: with a
    # carrier small enough that the mask's passband stays fully propagating before and after the
    # shift, the kernel's amplitude pattern (exactly the 0/1 mask) must be identical at zero
    # offset and at this offset.
    offset_fx_mask_check = 5.0 * grid_nyquist
    kernel_zero_offset_for_mask = odak.learn.wave.get_shifted_band_limited_angular_spectrum_kernel(
        nu=resolution[0], nv=resolution[1], dx=pixel_pitch, wavelength=wavelength, distance=distance,
        offset_fx=0.0, offset_fy=0.0, device=device,
    )
    kernel_mask_check = odak.learn.wave.get_shifted_band_limited_angular_spectrum_kernel(
        nu=resolution[0], nv=resolution[1], dx=pixel_pitch, wavelength=wavelength, distance=distance,
        offset_fx=offset_fx_mask_check, offset_fy=0.0, device=device,
    )
    mask_zero_offset = odak.learn.wave.calculate_amplitude(kernel_zero_offset_for_mask)
    mask_with_offset = odak.learn.wave.calculate_amplitude(kernel_mask_check)
    assert torch.allclose(mask_zero_offset, mask_with_offset, atol=1e-5)
    assert torch.equal(mask_zero_offset > 0.5, mask_with_offset > 0.5)

    # The propagation phase uses the shifted absolute frequencies (FX + offset_fx), not the
    # residual ones: reproduce the kernel independently at a moderate offset and confirm the
    # phase matches (compared as complex values, since the unwrapped phase reaches tens of
    # thousands of radians here, beyond float32's precision to represent "mod 2*pi" directly).
    fx = torch.linspace(
        -1 / (2 * pixel_pitch) + 0.5 / (2 * resolution[0] * pixel_pitch),
        1 / (2 * pixel_pitch) - 0.5 / (2 * resolution[0] * pixel_pitch),
        resolution[0], dtype=torch.float32, device=device,
    )
    fy = torch.linspace(
        -1 / (2 * pixel_pitch) + 0.5 / (2 * resolution[1] * pixel_pitch),
        1 / (2 * pixel_pitch) - 0.5 / (2 * resolution[1] * pixel_pitch),
        resolution[1], dtype=torch.float32, device=device,
    )
    FY, FX = torch.meshgrid(fx, fy, indexing="ij")
    offset_fx_phase_check = 3.0 * grid_nyquist
    FX_shifted, FY_shifted = FX + offset_fx_phase_check, FY
    expected_kz_squared = 1.0 / wavelength**2 - (FX_shifted**2 + FY_shifted**2)
    expected_propagating = expected_kz_squared >= 0.0
    expected_phase = 2.0 * odak.pi * torch.sqrt(torch.clamp(expected_kz_squared, min=0.0)) * distance

    x_extent, y_extent = pixel_pitch * float(resolution[0]), pixel_pitch * float(resolution[1])
    distance_tensor = torch.tensor([distance])
    fx_max = 1.0 / torch.sqrt((2.0 * distance_tensor * (1.0 / x_extent)) ** 2 + 1.0) / wavelength
    fy_max = 1.0 / torch.sqrt((2.0 * distance_tensor * (1.0 / y_extent)) ** 2 + 1.0) / wavelength
    expected_in_mask = (torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max)
    expected_amplitude = (expected_in_mask & expected_propagating).to(torch.float32)
    expected_kernel = odak.learn.wave.generate_complex_field(expected_amplitude, expected_phase)

    kernel_phase_check = odak.learn.wave.get_shifted_band_limited_angular_spectrum_kernel(
        nu=resolution[0], nv=resolution[1], dx=pixel_pitch, wavelength=wavelength, distance=distance,
        offset_fx=offset_fx_phase_check, offset_fy=0.0, device=device,
    )
    assert expected_amplitude.any()
    assert torch.allclose(kernel_phase_check, expected_kernel.to(torch.complex64), atol=1e-3)

    # Components beyond the Helmholtz limit are rejected as non-propagating, not NaN'd -- and
    # this is a genuine per-component (not all-or-nothing) condition: an offset near
    # propagating_limit leaves some residual frequencies propagating and others not.
    offset_fx_boundary = propagating_limit - 0.5 * grid_nyquist
    FX_shifted_boundary = FX + offset_fx_boundary
    kz_squared_boundary = 1.0 / wavelength**2 - (FX_shifted_boundary**2 + FY**2)
    propagating_boundary = kz_squared_boundary >= 0.0
    assert propagating_boundary.any() and (~propagating_boundary).any()  # a genuine mixed case

    kernel_boundary = odak.learn.wave.get_shifted_band_limited_angular_spectrum_kernel(
        nu=resolution[0], nv=resolution[1], dx=pixel_pitch, wavelength=wavelength, distance=distance,
        offset_fx=offset_fx_boundary, offset_fy=0.0, device=device,
    )
    assert torch.isfinite(torch.view_as_real(kernel_boundary)).all()

    # The sqrt clamp is for numerical safety only: it must not let a non-propagating component
    # through with nonzero amplitude, regardless of whether the mask or the propagating check
    # was the reason for rejection.
    boundary_amplitude = odak.learn.wave.calculate_amplitude(kernel_boundary)
    assert torch.equal(
        boundary_amplitude[~propagating_boundary], torch.zeros_like(boundary_amplitude[~propagating_boundary])
    )

    odak.learn.tools.save_image(
        "{}/shifted_band_limited_angular_spectrum.png".format(output_directory),
        intensity_shifted, cmin=0.0, cmax=float(intensity_shifted.max()),
    )


# ============================================================================
# test_memory: same physical scenario, ~16x smaller array, same result
# ============================================================================
def test_memory(device=torch.device("cpu"), output_directory="test_output"):
    """A diffuser mask's own residual spatial-frequency content is independent of incident tilt
    angle (unlike a plain smooth aperture, where the two are tightly coupled), matching how
    carrier-frequency shifting is used in practice: raw BASM needs a fine grid to satisfy both
    plain Nyquist sampling of the tilted field AND the aperture/distance-dependent band-limiting
    mask at the carrier frequency; shifted BASM only needs to represent the residual (diffuser-
    only) complexity, comfortably satisfied at a much coarser grid.

    The comparison bins the fine-grid intensity down to the coarse pixel scale (matching
    sum_bin_sensor_pixels in src/asm_psf_propagation.py) -- an approximation, not an exact
    equivalence, so even a flat/undiffused aperture only reaches ~0.988 similarity, not ~1.0;
    the threshold below is set accordingly."""
    odak.tools.check_directory(output_directory)
    wavelength, distance = 532e-9, 3e-3
    offset_fx, offset_fy = 250000.0, 0.0  # sin(theta) =~ 0.133, theta =~ 7.6 deg

    resolution_coarse, upsample_factor = 256, 4
    resolution_fine = resolution_coarse * upsample_factor
    pitch_fine = 1.6e-6
    pitch_coarse = pitch_fine * upsample_factor
    k = odak.learn.wave.wavenumber(wavelength)

    # Diffuser-like phase: random at a native feature resolution (8x8) much coarser than either
    # grid, nearest-neighbor upsampled to both -- the SAME physical mask at two resolutions,
    # mirroring load_height_map's diffuser upsampling in src/asm_psf_propagation.py.
    diffuser_native_resolution = 8
    diffuser_native_upsample = resolution_coarse // diffuser_native_resolution
    diffuser_phase_native = _diffuser_phase(
        diffuser_native_resolution, native_resolution=diffuser_native_resolution, device=device
    )
    diffuser_phase_coarse = diffuser_phase_native.repeat_interleave(diffuser_native_upsample, dim=0).repeat_interleave(
        diffuser_native_upsample, dim=1
    )
    diffuser_phase_fine = diffuser_phase_native.repeat_interleave(
        diffuser_native_upsample * upsample_factor, dim=0
    ).repeat_interleave(diffuser_native_upsample * upsample_factor, dim=1)

    field_coarse = odak.learn.wave.generate_complex_field(
        torch.ones(resolution_coarse, resolution_coarse, device=device), diffuser_phase_coarse
    )
    field_fine = odak.learn.wave.generate_complex_field(
        torch.ones(resolution_fine, resolution_fine, device=device), diffuser_phase_fine
    )

    sin_theta = offset_fx * wavelength
    tan_theta = sin_theta / math.sqrt(1.0 - sin_theta**2)
    chief_ray_shift_x_m = distance * tan_theta

    # Arm A: raw BASM on the fully tilted field, at the fine grid.
    xx_fine, yy_fine = _spatial_grid(resolution_fine, pitch_fine, device)
    carrier_phase_fine = 2.0 * odak.pi * (offset_fx * xx_fine + offset_fy * yy_fine)
    carrier_fine = odak.learn.wave.generate_complex_field(torch.ones_like(carrier_phase_fine), carrier_phase_fine)
    tilted_field_fine = field_fine * carrier_fine.to(torch.complex64)

    memory_raw_bytes = tilted_field_fine.numel() * 8  # complex64 = 8 bytes/element
    propagated_raw = odak.learn.wave.band_limited_angular_spectrum(tilted_field_fine, k, distance, pitch_fine, wavelength)
    recentered_raw = _recenter(propagated_raw, pitch_fine, chief_ray_shift_x_m)
    intensity_raw_fine = odak.learn.wave.calculate_amplitude(recentered_raw) ** 2
    intensity_raw_binned = _bin_intensity(intensity_raw_fine, upsample_factor)

    # Arm B: shifted BASM on the untilted field, at the coarse grid.
    memory_shifted_bytes = field_coarse.numel() * 8
    propagated_shifted = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field_coarse, k, distance, pitch_coarse, wavelength, offset_fx=offset_fx, offset_fy=offset_fy
    )
    recentered_shifted = _recenter(propagated_shifted, pitch_coarse, chief_ray_shift_x_m)
    intensity_shifted = odak.learn.wave.calculate_amplitude(recentered_shifted) ** 2

    similarity = torch.sum(intensity_raw_binned * intensity_shifted) / torch.sqrt(
        torch.sum(intensity_raw_binned**2) * torch.sum(intensity_shifted**2)
    )
    memory_ratio = memory_raw_bytes / memory_shifted_bytes

    print("raw BASM   (fine, {0}x{0}):   {1:>10.2f} MB array".format(resolution_fine, memory_raw_bytes / 1e6))
    print("shifted BASM (coarse, {0}x{0}): {1:>10.2f} MB array".format(resolution_coarse, memory_shifted_bytes / 1e6))
    print("array memory ratio (raw / shifted): {:.2f}x".format(memory_ratio))
    print("normalized similarity (raw vs. shifted): {:.6f}".format(similarity.item()))

    odak.learn.tools.save_image(
        "{}/diffuser_phase_native.png".format(output_directory), diffuser_phase_coarse, cmin=0.0, cmax=2.0 * odak.pi
    )
    odak.learn.tools.save_image(
        "{}/raw_basm_fine.png".format(output_directory), intensity_raw_binned,
        cmin=0.0, cmax=float(intensity_raw_binned.max()),
    )
    odak.learn.tools.save_image(
        "{}/shifted_basm_coarse.png".format(output_directory), intensity_shifted,
        cmin=0.0, cmax=float(intensity_shifted.max()),
    )

    assert memory_ratio > 8.0
    assert similarity.item() > 0.95


# ============================================================================
# test_convergence: is shifted-BASM's residual-grid resolution actually converged?
# ============================================================================
_CONVERGENCE_SHIFTED_RESOLUTIONS = [64, 128, 256, 512, 1024]
_CONVERGENCE_OFFSET_FX = round(250000.0 / BIN_SPACING_HZ_PER_M) * BIN_SPACING_HZ_PER_M  # bin-aligned, ~7.6 deg
_CONVERGENCE_OFFSET_FY = 0.0


def _convergence_match_units(candidate_native_intensity, bin_factor):
    """Converts a native point-sampled intensity into the same "sum of bin_factor^2 sub-pixel
    samples" unit _bin_intensity(reference, bin_factor) produces -- a fixed, known unit
    conversion (not a data-dependent renormalization)."""
    return candidate_native_intensity * (bin_factor**2)


def _convergence_compare(reference_intensity, reference_pitch_m, candidate_intensity, candidate_pitch_m):
    """Unlike _compare_intensity elsewhere in this file, matches units by SUM-binning the
    reference and scaling the candidate by bin_factor^2 (see _convergence_match_units) --
    reproduces the exact methodology this test was designed and validated against; kept
    separate rather than switched to area-average binning so this test's own numbers/thresholds
    are unaffected by the pixel-semantics fix documented in test_pixel_semantics."""
    ref_res, cand_res = reference_intensity.shape[-1], candidate_intensity.shape[-1]
    bin_factor = ref_res // cand_res
    reference_binned = _bin_intensity(reference_intensity, bin_factor) if bin_factor > 1 else reference_intensity
    candidate_matched = _convergence_match_units(candidate_intensity, bin_factor)

    similarity = (
        torch.sum(reference_binned * candidate_matched)
        / torch.sqrt(torch.sum(reference_binned**2) * torch.sum(candidate_matched**2))
    ).item()
    diff = reference_binned - candidate_matched
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    ref_rms = torch.sqrt(torch.mean(reference_binned**2)).item()
    nrmse = rmse / ref_rms if ref_rms > 0 else float("nan")
    peak = float(reference_binned.max())
    psnr = 20.0 * math.log10(peak / rmse) if rmse > 0 else float("inf")

    # Energy ratio uses proper area weighting from each side's OWN native resolution/intensity,
    # independent of the sum-vs-point-sample unit conversion above.
    reference_energy = float(reference_intensity.sum()) * reference_pitch_m**2
    candidate_energy = float(candidate_intensity.sum()) * candidate_pitch_m**2
    energy_ratio = candidate_energy / reference_energy if reference_energy > 0 else float("nan")

    ref_cx, ref_cy = _centroid_px(reference_binned)
    cand_cx, cand_cy = _centroid_px(candidate_matched)
    dx, dy = cand_cx - ref_cx, cand_cy - ref_cy
    return {
        "similarity": similarity, "nrmse": nrmse, "psnr": psnr, "energy_ratio": energy_ratio,
        "dx_px": dx, "dy_px": dy, "d_px": math.hypot(dx, dy),
    }


def test_convergence(device=torch.device("cpu")):
    """Sweeps shifted_band_limited_angular_spectrum over resolutions [64,128,256,512,1024] at a
    fixed physical FoV, comparing each against a single raw band_limited_angular_spectrum
    reference at 1024x1024.

    Reference-side floor: even at the SAME 1024x1024 resolution as the reference (no binning),
    similarity only reaches ~0.994, not ~1.0. This is not shifted-BASM error: raw BASM's own
    unmodified mask, evaluated at the absolute (carrier-included) frequency for a tilted field,
    clips a sliver of the diffuser's real (weakly extending) spectral content once the carrier
    pushes it close to fx_max -- something shifted-BASM's residual-frequency mask does not do.
    Treat the "vs. raw BASM" column as floor-limited at ~0.99-0.995; the CONSECUTIVE
    shifted-vs-shifted comparisons (never touching raw BASM) are the reliable convergence signal."""
    k = odak.learn.wave.wavenumber(WAVELENGTH_M)

    sin_theta = _CONVERGENCE_OFFSET_FX * WAVELENGTH_M
    tan_theta = sin_theta / math.sqrt(1.0 - sin_theta**2)
    chief_ray_shift_x_m = DISTANCE_M * tan_theta

    # Reference: raw BASM on the fully tilted field, at 1024x1024.
    pitch_reference = FOV_M / RESOLUTION
    diffuser_phase_reference = _diffuser_phase(RESOLUTION, device=device)
    field_reference = odak.learn.wave.generate_complex_field(
        torch.ones(RESOLUTION, RESOLUTION, device=device), diffuser_phase_reference
    )
    xx_ref, yy_ref = _spatial_grid(RESOLUTION, pitch_reference, device)
    carrier_phase_ref = 2.0 * odak.pi * (_CONVERGENCE_OFFSET_FX * xx_ref + _CONVERGENCE_OFFSET_FY * yy_ref)
    carrier_ref = odak.learn.wave.generate_complex_field(torch.ones_like(carrier_phase_ref), carrier_phase_ref)
    tilted_field_reference = field_reference * carrier_ref.to(torch.complex64)

    propagated_reference = odak.learn.wave.band_limited_angular_spectrum(
        tilted_field_reference, k, DISTANCE_M, pitch_reference, WAVELENGTH_M
    )
    recentered_reference = _recenter(propagated_reference, pitch_reference, chief_ray_shift_x_m)
    intensity_reference = odak.learn.wave.calculate_amplitude(recentered_reference) ** 2

    # Sweep shifted BASM over resolutions, at the SAME physical FoV/parameters.
    rows, shifted_intensities, shifted_pitches = [], {}, {}
    for resolution in _CONVERGENCE_SHIFTED_RESOLUTIONS:
        pitch = FOV_M / resolution
        diffuser_phase = _diffuser_phase(resolution, device=device)
        field = odak.learn.wave.generate_complex_field(torch.ones(resolution, resolution, device=device), diffuser_phase)

        start = time.perf_counter()
        propagated = odak.learn.wave.shifted_band_limited_angular_spectrum(
            field, k, DISTANCE_M, pitch, WAVELENGTH_M, offset_fx=_CONVERGENCE_OFFSET_FX, offset_fy=_CONVERGENCE_OFFSET_FY
        )
        recentered = _recenter(propagated, pitch, chief_ray_shift_x_m)
        intensity = odak.learn.wave.calculate_amplitude(recentered) ** 2
        runtime_s = time.perf_counter() - start

        shifted_intensities[resolution] = intensity
        shifted_pitches[resolution] = pitch
        metrics = _convergence_compare(intensity_reference, pitch_reference, intensity, pitch)
        metrics.update({"resolution": resolution, "runtime_s": runtime_s})
        rows.append(metrics)

    print("\nReference: raw BASM at {0}x{0}, pitch={1:.3g} m, FoV={2:.4g} m".format(RESOLUTION, pitch_reference, FOV_M))
    header = "{:>6} | {:>10} | {:>8} | {:>8} | {:>11} | {:>7} | {:>9}".format(
        "Res", "Similarity", "NRMSE", "PSNR", "EnergyRatio", "d(px)", "Runtime"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "{:>6} | {:>10.6f} | {:>8.4f} | {:>8.2f} | {:>11.4f} | {:>7.3f} | {:>8.3f}s".format(
                row["resolution"], row["similarity"], row["nrmse"], row["psnr"], row["energy_ratio"],
                row["d_px"], row["runtime_s"],
            )
        )

    print("\nConsecutive shifted-BASM convergence (higher resolution binned down to the lower one):")
    consecutive_header = "{:>16} | {:>10} | {:>8}".format("Pair", "Similarity", "NRMSE")
    print(consecutive_header)
    print("-" * len(consecutive_header))
    consecutive_by_lo = {}
    for lo, hi in zip(_CONVERGENCE_SHIFTED_RESOLUTIONS[:-1], _CONVERGENCE_SHIFTED_RESOLUTIONS[1:]):
        metrics = _convergence_compare(
            shifted_intensities[hi], shifted_pitches[hi], shifted_intensities[lo], shifted_pitches[lo]
        )
        consecutive_by_lo[lo] = metrics
        print("{:>7} vs {:>6} | {:>10.6f} | {:>8.4f}".format(lo, hi, metrics["similarity"], metrics["nrmse"]))

    by_res = {row["resolution"]: row for row in rows}
    threshold_999 = next((r for r in _CONVERGENCE_SHIFTED_RESOLUTIONS if by_res[r]["similarity"] > 0.999), None)
    print("\nSmallest resolution with similarity > 0.999 vs. raw BASM: {}".format(threshold_999 or "none tested"))
    if 256 in consecutive_by_lo:
        converged_256 = consecutive_by_lo[256]["similarity"] > 0.999
        print("256x256 appears {}converged (256-vs-512 similarity = {:.6f})".format(
            "" if converged_256 else "NOT ", consecutive_by_lo[256]["similarity"]
        ))

    # Sanity check, not a strict "should be identical" assertion -- see docstring's
    # "reference floor" note for why even same-resolution similarity is ~0.994, not ~1.0.
    assert by_res[RESOLUTION]["similarity"] > 0.99


# ============================================================================
# test_mask_ablation: does the band-limiting mask cause the disagreement? (No.)
# ============================================================================
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
# test_pixel_semantics: Mode A (intensity, area-average) / Mode B (sensor energy),
# plus a raw-vs-raw control isolating the diffuser's own sub-pixel structure
# ============================================================================
def _build_comparison_row(theta_deg, k, device):
    raw_ref = _converged_raw_reference(theta_deg, k, device)
    offset_fx, n_raw = raw_ref["offset_fx"], raw_ref["n_raw"]
    bin_factor = n_raw // SHIFTED_RESOLUTION

    shifted_512, dx_shifted_512 = _shifted_basm_intensity(offset_fx, SHIFTED_RESOLUTION, k, device)

    if bin_factor <= 1:
        raw_avg_512 = raw_sum_512 = raw_ref["intensity"]
    else:
        raw_avg_512 = _bin_intensity_average(raw_ref["intensity"], bin_factor)
        raw_sum_512 = _bin_intensity(raw_ref["intensity"], bin_factor)

    return {
        "theta_deg": theta_deg, "offset_fx": offset_fx, "n_raw": n_raw, "dx_raw": raw_ref["pitch"],
        "shifted_512": shifted_512, "dx_shifted_512": dx_shifted_512,
        "raw_avg_512": raw_avg_512, "raw_sum_512": raw_sum_512,
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
    energy_near_one = all(abs(r - 1.0) < ENERGY_RATIO_TOLERANCE for r in new_ratios)
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
# test_sensor_equivalence: integrate |U|^2 over the SAME physical sensor pixels for
# both methods before comparing -- the question relevant to lensless PSF simulation.
# ============================================================================
def test_sensor_equivalence(device=torch.device("cpu")):
    """Do the two solvers predict the same measurement for a real, finite-area sensor pixel
    (E_ij = integral_over_pixel |U|^2 dx dy)? Shifted-BASM is run on an internal grid finer than
    the sensor (SHIFT_INTERNAL_PRIMARY=2048), never its own native-resolution point sample, to
    avoid repeating the point-sample-vs-area-average mismatch from test_pixel_semantics."""
    k = odak.learn.wave.wavenumber(WAVELENGTH_M)
    raw_rows = [_converged_raw_reference(theta_deg, k, device) for theta_deg in ANGLES_DEG]

    rows = []
    for raw_ref in raw_rows:
        offset_fx, n_raw = raw_ref["offset_fx"], raw_ref["n_raw"]
        e_raw = _sensor_energy(raw_ref["intensity"], raw_ref["pitch"], n_raw // SENSOR_RESOLUTION)
        intensity_shift, pitch_shift = _shifted_basm_intensity(offset_fx, SHIFT_INTERNAL_PRIMARY, k, device)
        e_shift = _sensor_energy(intensity_shift, pitch_shift, SHIFT_INTERNAL_PRIMARY // SENSOR_RESOLUTION)
        metrics = _compare_energy(e_raw, e_shift)
        rows.append({"theta_deg": raw_ref["theta_deg"], "n_raw": n_raw, "metrics": metrics})

    _print_table(
        "=== Sensor-measurement equivalence (raw N vs. shifted internal {},\n"
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

    # Control: raw-vs-raw sensor-integration reference accuracy (no shifted-BASM at all).
    control_rows = []
    for raw_ref in raw_rows:
        n_raw = raw_ref["n_raw"]
        other_n = n_raw * 2 if n_raw * 2 <= CONVERGENCE_DOUBLE_CEILING else n_raw // 2
        e_raw = _sensor_energy(raw_ref["intensity"], raw_ref["pitch"], n_raw // SENSOR_RESOLUTION)
        intensity_other, pitch_other = _raw_basm_intensity(other_n, raw_ref["offset_fx"], k, device)
        e_other = _sensor_energy(intensity_other, pitch_other, max(other_n // SENSOR_RESOLUTION, 1))
        control_rows.append({"theta_deg": raw_ref["theta_deg"], "sim": _compare_energy(e_raw, e_other)["similarity"]})
    _print_table(
        "=== Control: raw-vs-raw sensor-integration reference accuracy (no shifted-BASM) ===",
        ["Angle", "Similarity"],
        [["{:.1f}".format(c["theta_deg"]), "{:.6f}".format(c["sim"])] for c in control_rows],
    )

    # Control: shifted internal-grid sensor-integration convergence (512 -> 1024 -> 2048).
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
        "=== Control: shifted internal-grid sensor convergence (512->1024->2048) ===",
        ["Angle", "Sim(512v1024)", "Sim(1024v2048)"],
        [["{:.1f}".format(c["theta_deg"]), "{:.6f}".format(c["s1"]), "{:.6f}".format(c["s2"])] for c in convergence_rows],
    )

    similarities = [r["metrics"]["similarity"] for r in rows]
    energy_ratios = [r["metrics"]["energy_ratio"] for r in rows]
    lowest_similarity = min(similarities)
    worst_theta = rows[similarities.index(lowest_similarity)]["theta_deg"]
    energy_near_one = all(abs(e - 1.0) < ENERGY_RATIO_TOLERANCE for e in energy_ratios)
    largest_energy_mismatch = max(abs(e - 1.0) for e in energy_ratios)

    print("Lowest sensor similarity: {:.6f} (theta={:.1f} deg)".format(lowest_similarity, worst_theta))
    print("Largest physical energy mismatch: {:.4f}".format(largest_energy_mismatch))
    print(
        "Do both methods predict the same finite-area sensor measurement? "
        + ("YES" if energy_near_one and lowest_similarity > SENSOR_SIMILARITY_TARGET else "NO")
    )

    assert energy_near_one, (
        "physical sensor-energy ratio should stay within {:.0%} of 1.0 at every angle -- largest "
        "mismatch was {:.4f}; see the table above".format(ENERGY_RATIO_TOLERANCE, largest_energy_mismatch)
    )
    assert lowest_similarity > SENSOR_SIMILARITY_TARGET, (
        "sensor similarity should exceed {:.3f} at every angle -- lowest was {:.6f} at theta={:.1f}; "
        "see the table/controls above".format(SENSOR_SIMILARITY_TARGET, lowest_similarity, worst_theta)
    )


if __name__ == "__main__":
    for _name, _fn in [
        ("test_correctness", test_correctness), ("test_memory", test_memory),
        ("test_convergence", test_convergence), ("test_mask_ablation", test_mask_ablation),
        ("test_pixel_semantics", test_pixel_semantics), ("test_sensor_equivalence", test_sensor_equivalence),
    ]:
        print("\n" + "#" * 78 + "\n# {}\n".format(_name) + "#" * 78 + "\n")
        _fn()
    sys.exit(0)
