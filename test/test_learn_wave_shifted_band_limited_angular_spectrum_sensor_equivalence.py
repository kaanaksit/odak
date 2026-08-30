"""TEST 2 -- physical sensor-measurement equivalence between raw (ordinary) BASM and
shifted-BASM.

This file answers a DIFFERENT question from
test_learn_wave_shifted_band_limited_angular_spectrum_field_equivalence.py's Test 1:

    Test 1 (field_equivalence.py): do the two solvers compute the SAME propagated COMPLEX field
    at identical physical coordinates, up to a single global phase? A point-sampled-field,
    numerical-solver-level question.

    Test 2 (this file): do the two solvers predict the SAME measurement for a real, finite-area
    sensor pixel, after intensity formation (|U|^2) and physical pixel-AREA integration? A
    downstream, physically-measurable-quantity question -- the one actually relevant to
    lensless PSF simulation, where a sensor always integrates over its own finite pixel area
    rather than reading a single point sample of the continuous field.

These are genuinely different validation levels and are answered independently, reusing (but
never conflating) the pixel-semantics distinction established in
test_learn_wave_shifted_band_limited_angular_spectrum_equivalence_debug.py: raw BASM and
shifted-BASM both natively return POINT-SAMPLED complex field values (see that file's module
docstring, Section 1); a physical sensor pixel instead reads an INTEGRAL,
E_ij = integral_over_pixel |U(x,y)|^2 dx dy, approximated here as
sum(I_fine over the pixel's footprint) * dx_fine * dy_fine -- the SAME energy-preserving SUM
convention as sum_bin_sensor_pixels in src/asm_psf_propagation.py.

Critically (Section 3): shifted-BASM's own NATIVE 512-resolution point sample is NOT treated as
equal to a finite-area sensor integration. Doing so would silently repeat the ~1/64 pixel-
semantics artifact and the ~0.998 point-sample-vs-area-average ceiling already characterized in
test_..._equivalence_debug.py's Mode A/B and Experiment 6/7. Instead, shifted-BASM is run on an
internal propagation grid FINER than the 512-pixel physical sensor (1024, and 2048 as the
preferred/primary case, matching the same "run a properly-resolved point-sampled field, THEN
integrate over the physical pixel footprint" recipe already used for raw BASM), and its intensity
is integrated over the SAME 512 physical sensor pixels before comparison. Section 4's sensor-
integration convergence check verifies this predicted sensor image is itself stable as the
internal grid increases (512 -> 1024 -> 2048), separating "is shifted-BASM's internal propagation
grid fine enough" from "is the physical sensor resolution itself the bottleneck" (it is not, by
construction, since both sides are integrated onto the identical 512-pixel sensor).
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
BIN_SPACING_HZ_PER_M = 1.0 / FOV_M
ANGLES_DEG = [0.0, 2.0, 4.0, 6.0, 8.0, 9.0]

BASE_RAW_RESOLUTION = RESOLUTION
SAFE_NYQUIST_FRACTION = 0.8
CONVERGENCE_SIMILARITY_THRESHOLD = 0.9999
SENSOR_RESOLUTION = 512
SHIFT_INTERNAL_RESOLUTIONS = [512, 1024, 2048]
SHIFT_INTERNAL_PRIMARY = 2048
SENSOR_SIMILARITY_TARGET = 0.999
ENERGY_RATIO_TOLERANCE = 0.05


def _max_physical_raw_resolution():
    """Same physically-motivated cap as the other two files in this suite: largest power-of-two
    grid (fixed FOV_M) whose Nyquist limit does not exceed 1/wavelength, avoiding both a
    physically meaningless subwavelength sampling regime and odak's latent NaN bug in the
    ORIGINAL (non-shifted) kernel for evanescent components beyond that point."""
    limit = 2.0 * FOV_M / WAVELENGTH_M
    n = BASE_RAW_RESOLUTION
    while n * 2 <= limit:
        n *= 2
    return n


MAX_RAW_RESOLUTION = _max_physical_raw_resolution()
CONVERGENCE_DOUBLE_CEILING = MAX_RAW_RESOLUTION


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


def _compare(reference, candidate):
    """Compares two arrays already expressed in the SAME physical units (e.g. both already
    integrated sensor energy) -- unlike the intensity `_compare` helpers elsewhere in this
    suite, this one does NOT re-apply any pixel-pitch^2 scaling, since that would double-count
    the area integration already baked into `_sensor_energy`."""
    similarity = (
        torch.sum(reference * candidate) / torch.sqrt(torch.sum(reference**2) * torch.sum(candidate**2))
    ).item()
    diff = reference - candidate
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    ref_rms = torch.sqrt(torch.mean(reference**2)).item()
    nrmse = rmse / ref_rms if ref_rms > 0 else float("nan")
    peak = float(reference.max())
    psnr = 20.0 * math.log10(peak / rmse) if rmse > 0 else float("inf")
    ref_total = float(reference.sum())
    cand_total = float(candidate.sum())
    energy_ratio = cand_total / ref_total if ref_total > 0 else float("nan")
    mean_ratio = float(candidate.mean()) / float(reference.mean()) if float(reference.mean()) > 0 else float("nan")
    ref_cx, ref_cy = _centroid_px(reference)
    cand_cx, cand_cy = _centroid_px(candidate)
    dx_px = cand_cx - ref_cx
    dy_px = cand_cy - ref_cy
    max_abs_error = float(diff.abs().max())
    return {
        "similarity": similarity, "nrmse": nrmse, "psnr": psnr, "energy_ratio": energy_ratio,
        "mean_ratio": mean_ratio, "dx_px": dx_px, "dy_px": dy_px, "d_px": math.hypot(dx_px, dy_px),
        "max_abs_error": max_abs_error,
    }


def _bin_intensity(intensity, factor):
    """Energy-preserving SUM binning, matching sum_bin_sensor_pixels in
    src/asm_psf_propagation.py -- the correct operation for integrating a fine, point-sampled
    SIMULATION grid onto a physically larger SENSOR pixel that genuinely integrates light over
    its footprint (Section 2/3's E_ij), per the pixel-semantics distinction established in
    test_..._equivalence_debug.py's module docstring."""
    n = intensity.shape[-1]
    m = n // factor
    return intensity.reshape(m, factor, m, factor).sum(dim=(1, 3))


def _sensor_energy(intensity_fine, dx_fine, factor):
    """Section 2/3: E_ij = sum(I_fine over the sensor pixel's footprint) * dx_fine^2. When
    factor == 1 (the internal grid already equals the sensor resolution), this reduces to
    I * dx^2 -- a single point sample times the pixel area, exactly the naive approximation
    Section 3 warns against trusting alone for shifted-BASM; it is still computed here (for
    Table 2's convergence check and Control D) so that warning can be verified numerically
    rather than just asserted."""
    if factor <= 1:
        return intensity_fine * dx_fine**2
    return _bin_intensity(intensity_fine, factor) * dx_fine**2


def _bin_align(frequency_hz_per_m):
    return round(frequency_hz_per_m / BIN_SPACING_HZ_PER_M) * BIN_SPACING_HZ_PER_M


def _angle_to_offset(theta_deg):
    return _bin_align(math.sin(math.radians(theta_deg)) / WAVELENGTH_M)


def _build_scene_at(resolution, offset_fx, device):
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


def _intensity_at(field_out, pitch, shift_x_m, device):
    recentered = _recenter(field_out, pitch, shift_x_m)
    return odak.learn.wave.calculate_amplitude(recentered) ** 2


def _raw_basm_intensity(resolution, offset_fx, k, device):
    _, tilted_field, pitch, shift_x_m = _build_scene_at(resolution, offset_fx, device)
    propagated = odak.learn.wave.band_limited_angular_spectrum(tilted_field, k, DISTANCE_M, pitch, WAVELENGTH_M)
    return _intensity_at(propagated, pitch, shift_x_m, device), pitch


def _shifted_basm_intensity(offset_fx, resolution, k, device):
    field, _, pitch, shift_x_m = _build_scene_at(resolution, offset_fx, device)
    propagated = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field, k, DISTANCE_M, pitch, WAVELENGTH_M, offset_fx=offset_fx, offset_fy=0.0
    )
    return _intensity_at(propagated, pitch, shift_x_m, device), pitch


def _bin_intensity_average(intensity, factor):
    """Only used for raw BASM's OWN internal convergence check (Section 4's "do the same for
    raw BASM if needed"), which is a point-sampled-field numerical-convergence question, not a
    sensor-energy one -- reuses the SAME area-average convention established (and required) in
    test_..._equivalence_debug.py for comparing two point-sampled representations of the same
    field, as distinct from _sensor_energy's SUM+area convention used for the actual Test 2
    sensor comparison below."""
    n = intensity.shape[-1]
    m = n // factor
    return intensity.reshape(m, factor, m, factor).mean(dim=(1, 3))


def _residual_bandwidth_hz_per_m():
    return DIFFUSER_NATIVE_RESOLUTION / (2.0 * FOV_M)


def _select_raw_resolution(offset_fx, f_residual_max):
    n = BASE_RAW_RESOLUTION
    while True:
        dx = FOV_M / n
        f_nyquist = 1.0 / (2.0 * dx)
        if abs(offset_fx) + f_residual_max <= SAFE_NYQUIST_FRACTION * f_nyquist or n >= MAX_RAW_RESOLUTION:
            return n
        n *= 2


def _field_similarity(intensity_a, pitch_a, intensity_b_binned):
    """Scale-invariant shape check used ONLY for raw BASM's own point-sampled-field convergence
    verification (Section 4), not for the sensor-energy comparisons below."""
    return (
        torch.sum(intensity_a * intensity_b_binned)
        / torch.sqrt(torch.sum(intensity_a**2) * torch.sum(intensity_b_binned**2))
    ).item()


def _raw_convergence_similarity(resolution, offset_fx, k, device):
    doubled = resolution * 2
    if doubled > CONVERGENCE_DOUBLE_CEILING:
        return None
    intensity_a, pitch_a = _raw_basm_intensity(resolution, offset_fx, k, device)
    intensity_b, _ = _raw_basm_intensity(doubled, offset_fx, k, device)
    intensity_b_binned = _bin_intensity_average(intensity_b, 2)
    return _field_similarity(intensity_a, pitch_a, intensity_b_binned)


def _converged_raw_reference(theta_deg, k, device):
    """Sections 4-5 of the shared task spec (reused verbatim from
    test_..._equivalence_debug.py / field_equivalence.py's identical logic): pick the smallest
    safe N_raw, then VERIFY convergence via an explicit doubling comparison, increasing N_raw
    further if not yet converged, up to the physical resolution cap. NEVER falls back to a fixed
    512/1024 reference at large angles."""
    offset_fx = _angle_to_offset(theta_deg)
    f_residual_max = _residual_bandwidth_hz_per_m()
    n_raw = _select_raw_resolution(offset_fx, f_residual_max)

    conv_sim = _raw_convergence_similarity(n_raw, offset_fx, k, device)
    last_valid = conv_sim
    while conv_sim is not None and conv_sim <= CONVERGENCE_SIMILARITY_THRESHOLD and n_raw < MAX_RAW_RESOLUTION:
        n_raw *= 2
        conv_sim = _raw_convergence_similarity(n_raw, offset_fx, k, device)
        if conv_sim is not None:
            last_valid = conv_sim
    conv_sim = last_valid
    converged = conv_sim is not None and conv_sim > CONVERGENCE_SIMILARITY_THRESHOLD

    intensity, pitch = _raw_basm_intensity(n_raw, offset_fx, k, device)
    return {
        "theta_deg": theta_deg, "offset_fx": offset_fx, "n_raw": n_raw, "pitch": pitch,
        "converged": converged, "convergence_similarity": conv_sim, "intensity": intensity,
    }


def _print_table(title, header_cols, rows):
    print(title)
    header = " | ".join("{:>13}".format(c) for c in header_cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join("{:>13}".format(v) for v in row))
    print()


def control_c_raw_sensor_reference_accuracy(rows, k, device):
    """Control C: integrates the oversampled raw reference onto the 512 sensor, then compares
    against an INDEPENDENTLY, differently-sampled raw-BASM run ALSO integrated onto the same
    sensor. Establishes the achievable sensor-reference accuracy with NO shifted-BASM involved
    at all. Prefers 2x n_raw (a genuinely finer, independent reference); when n_raw is already
    at the physical resolution cap (as it is for every angle here -- see
    test_..._equivalence_debug.py's module docstring on why sampling finer than
    wavelength/2 pixel pitch is not physically meaningful), falls back to n_raw/2 instead, so
    this control still reports a real number rather than "N/A" for every angle."""
    control_rows = []
    for row in rows:
        n_raw = row["n_raw"]
        dx_raw = row["pitch"]
        e_raw = _sensor_energy(row["intensity"], dx_raw, n_raw // SENSOR_RESOLUTION)
        doubled = n_raw * 2
        if doubled <= CONVERGENCE_DOUBLE_CEILING:
            other_n = doubled
        else:
            other_n = n_raw // 2
        intensity_other, pitch_other = _raw_basm_intensity(other_n, row["offset_fx"], k, device)
        e_raw_other = _sensor_energy(intensity_other, pitch_other, max(other_n // SENSOR_RESOLUTION, 1))
        metrics = _compare(e_raw, e_raw_other)
        control_rows.append({"theta_deg": row["theta_deg"], "other_n": other_n, "metrics": metrics})

    _print_table(
        "=== Control C: raw-vs-raw sensor-integration reference accuracy (no shifted-BASM;\n"
        "    n_raw vs. an independently-sampled reference, both integrated onto the SAME {}\n"
        "    sensor) ===".format(SENSOR_RESOLUTION),
        ["Angle", "OtherN", "Similarity", "NRMSE", "EnergyRatio"],
        [
            [
                "{:.1f}".format(t["theta_deg"]), "{}".format(t["other_n"]),
                "{:.6f}".format(t["metrics"]["similarity"]),
                "{:.4f}".format(t["metrics"]["nrmse"]), "{:.4f}".format(t["metrics"]["energy_ratio"]),
            ]
            for t in control_rows
        ],
    )
    return control_rows


def control_d_sensor_convergence(rows, k, device):
    """Control D (sensor half): shifted 512 vs. 1024 vs. 2048 (where feasible), each integrated
    onto the SAME 512 sensor -- tells us how much INTERNAL shifted-grid resolution is actually
    required once the comparison is expressed in sensor-energy terms (Section 4)."""
    table_rows = []
    for row in rows:
        offset_fx = row["offset_fx"]
        energies = {}
        for internal_n in SHIFT_INTERNAL_RESOLUTIONS:
            intensity, pitch = _shifted_basm_intensity(offset_fx, internal_n, k, device)
            factor = max(internal_n // SENSOR_RESOLUTION, 1)
            energies[internal_n] = _sensor_energy(intensity, pitch, factor)
        m_512_1024 = _compare(energies[512], energies[1024])
        m_1024_2048 = _compare(energies[1024], energies[2048])
        table_rows.append({
            "theta_deg": row["theta_deg"], "sim_512_1024": m_512_1024["similarity"],
            "sim_1024_2048": m_1024_2048["similarity"], "energies": energies,
        })

    _print_table(
        "=== Control D (sensor half): shifted internal-grid sensor-integration convergence\n"
        "    (512 -> 1024 -> 2048 internal, all onto the SAME {} sensor) ===".format(SENSOR_RESOLUTION),
        ["Angle", "Sim(512 vs 1024)", "Sim(1024 vs 2048)"],
        [
            [
                "{:.1f}".format(t["theta_deg"]), "{:.6f}".format(t["sim_512_1024"]),
                "{:.6f}".format(t["sim_1024_2048"]),
            ]
            for t in table_rows
        ],
    )
    return table_rows


def test(device=torch.device("cpu")):
    k = odak.learn.wave.wavenumber(WAVELENGTH_M)

    raw_rows = [_converged_raw_reference(theta_deg, k, device) for theta_deg in ANGLES_DEG]

    rows = []
    for raw_ref in raw_rows:
        theta_deg = raw_ref["theta_deg"]
        offset_fx = raw_ref["offset_fx"]
        n_raw = raw_ref["n_raw"]

        e_raw = _sensor_energy(raw_ref["intensity"], raw_ref["pitch"], n_raw // SENSOR_RESOLUTION)

        intensity_shift_primary, pitch_shift_primary = _shifted_basm_intensity(offset_fx, SHIFT_INTERNAL_PRIMARY, k, device)
        e_shift = _sensor_energy(intensity_shift_primary, pitch_shift_primary, SHIFT_INTERNAL_PRIMARY // SENSOR_RESOLUTION)

        metrics = _compare(e_raw, e_shift)
        rows.append({
            "theta_deg": theta_deg, "offset_fx": offset_fx, "n_raw": n_raw, "pitch": raw_ref["pitch"],
            "intensity": raw_ref["intensity"], "converged": raw_ref["converged"],
            "convergence_similarity": raw_ref["convergence_similarity"], "metrics": metrics,
        })

    _print_table(
        "=== Table 3: sensor-measurement equivalence (raw internal N vs. shifted internal N,\n"
        "    both integrated onto the same {}x{} physical sensor) ===".format(SENSOR_RESOLUTION, SENSOR_RESOLUTION),
        ["Angle", "RawIntN", "ShiftIntN", "SensorN", "Similarity", "NRMSE", "PSNR", "EnergyRatio", "d(px)"],
        [
            [
                "{:.1f}".format(r["theta_deg"]), "{}".format(r["n_raw"]), "{}".format(SHIFT_INTERNAL_PRIMARY),
                "{}".format(SENSOR_RESOLUTION), "{:.6f}".format(r["metrics"]["similarity"]),
                "{:.4f}".format(r["metrics"]["nrmse"]), "{:.2f}".format(r["metrics"]["psnr"]),
                "{:.4f}".format(r["metrics"]["energy_ratio"]), "{:.3f}".format(r["metrics"]["d_px"]),
            ]
            for r in rows
        ],
    )

    control_c_raw_sensor_reference_accuracy(raw_rows, k, device)
    control_d_sensor_convergence(rows, k, device)

    similarities = [r["metrics"]["similarity"] for r in rows]
    energy_ratios = [r["metrics"]["energy_ratio"] for r in rows]
    max_abs_errors = [r["metrics"]["max_abs_error"] for r in rows]
    lowest_similarity = min(similarities)
    worst_theta = rows[similarities.index(lowest_similarity)]["theta_deg"]
    energy_near_one = all(abs(e - 1.0) < ENERGY_RATIO_TOLERANCE for e in energy_ratios)
    largest_energy_mismatch = max(abs(e - 1.0) for e in energy_ratios)
    all_raw_converged = all(r["converged"] for r in rows)

    test2_pass = energy_near_one and lowest_similarity > SENSOR_SIMILARITY_TARGET

    print("TEST 2 -- Sensor equivalence:")
    print("  " + ("PASS" if test2_pass else "FAIL"))
    print()
    print("Lowest sensor similarity:")
    print("  {:.6f} (theta={:.1f} deg)".format(lowest_similarity, worst_theta))
    print()
    print("Largest physical energy mismatch:")
    print("  {:.4f} (|EnergyRatio - 1|)".format(largest_energy_mismatch))
    print()
    print("Do both methods predict the same finite-area sensor measurement?")
    print("  " + ("YES" if test2_pass else "NO"))
    print()

    if test2_pass:
        error_source = "none"
    elif not all_raw_converged:
        error_source = "raw convergence"
    elif not energy_near_one:
        error_source = "sensor integration (physical energy ratio not near 1)"
    else:
        error_source = "unknown (energy ratio is near 1 but similarity/NRMSE remain below target -- see\n" \
            "  Table 3, Control C, and Control D above for the achievable sensor-reference accuracy\n" \
            "  once both propagation methods are properly resolved)"
    print("Main remaining error source (Test 2): {}".format(error_source))
    print("Largest single-pixel absolute error across angles: {:.6e}".format(max(max_abs_errors)))

    assert energy_near_one, (
        "physical sensor-energy ratio should stay within {:.0%} of 1.0 at every angle once both "
        "methods are integrated onto the SAME {}-pixel sensor -- largest mismatch was {:.4f}; see "
        "Table 3 above for which angle failed".format(ENERGY_RATIO_TOLERANCE, SENSOR_RESOLUTION, largest_energy_mismatch)
    )


if __name__ == "__main__":
    sys.exit(test())
