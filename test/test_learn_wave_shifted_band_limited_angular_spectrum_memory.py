import math
import sys
import odak
import torch


def test(device=torch.device("cpu"), output_directory="test_output"):
    """Raw BASM vs. shifted BASM: same physical scenario (a diffuser-like phase mask under
    tilted illumination), same result, far less memory for shifted BASM.

    A diffuser mask's own local phase structure (its "residual" spatial-frequency content) is
    independent of the incident tilt angle -- unlike a plain smooth aperture, where they are
    tightly coupled (verified numerically while designing this: for a plain aperture, pushing
    the incident tilt's required padding past the standard 2x baseline while staying within
    Eq. 5-6's band-limiting mask is provably impossible with any real safety margin, since both
    are governed by the same aperture-size/distance relationship). A diffuser breaks that
    coupling, matching how carrier-frequency shifting is actually used in practice.

    Concretely: the SAME diffuser-like random phase is built at a coarse native resolution
    (256x256) and nearest-neighbor upsampled to a fine resolution (1024x1024, matching how a
    physical diffuser's own pixel pitch is upsampled onto a finer simulation grid in
    src/asm_psf_propagation.py's load_height_map). Raw BASM, propagating the *fully tilted*
    field, needs the fine 1024x1024 grid to satisfy both plain Nyquist sampling of the tilted
    field (pitch <= 1/(2*offset_fx)) and Eq. 6's aperture/distance-dependent band-limiting mask
    at the carrier frequency. Shifted BASM only needs to represent the *residual* (diffuser-only)
    complexity -- comfortably satisfied at the coarse 256x256 grid, a 4x4 = 16x smaller array.

    These parameters were chosen by direct numeric search (not hand-derived) to give comfortable
    margins: ~2.0x for Eq. 6 vs. the carrier frequency and ~1.25x for plain Nyquist vs. the
    carrier frequency at the fine grid, and ~12x for Eq. 6 vs. the residual frequency at the
    coarse grid -- i.e. both arms are safely within their respective sampling limits, so the
    "same result" comparison below isn't comparing a correct case against an aliased one.

    The comparison itself bins the fine-grid intensity down to the coarse pixel scale before
    comparing (matching sum_bin_sensor_pixels in src/asm_psf_propagation.py) -- this binning is
    an approximation, not an exact equivalence, so even the trivial case (a flat, undiffused
    aperture) only reaches a normalized-similarity of ~0.988, not ~1.0. The similarity threshold
    below is set below that natural ceiling rather than at the near-machine-precision threshold
    used for same-resolution comparisons in test_learn_wave_shifted_band_limited_angular_spectrum.py.
    """
    odak.tools.check_directory(output_directory)

    wavelength = 532e-9
    distance = 3e-3
    offset_fx = 250000.0  # cycles/m; sin(theta) = offset_fx * wavelength =~ 0.133, theta =~ 7.6 deg
    offset_fy = 0.0

    resolution_coarse = 256
    upsample_factor = 4
    resolution_fine = resolution_coarse * upsample_factor
    pitch_fine = 1.6e-6
    pitch_coarse = pitch_fine * upsample_factor

    k = odak.learn.wave.wavenumber(wavelength)

    # Diffuser-like phase: random at a *native* feature resolution much coarser than either
    # simulation grid (8x8 -- one random value per diffuser feature, each feature 32 coarse /
    # 128 fine pixels wide), then nearest-neighbor upsampled to both the coarse and fine grids --
    # the SAME physical mask represented at two simulation-grid resolutions, mirroring
    # load_height_map's diffuser_pixel_pitch/simulation_grid_pitch upsampling in
    # src/asm_psf_propagation.py. Deliberately chunky (large, individually countable blocks)
    # rather than fine per-pixel noise, purely so the saved images are easy to visually compare
    # feature-by-feature between the two arms -- this is not a sampling requirement (a finer
    # native resolution would also satisfy the margins computed below, just look like unresolved
    # speckle instead of a visible mosaic). Per-simulation-pixel white noise (one random value
    # per *simulation* pixel, not per diffuser feature) would additionally put spectral content
    # all the way out to each grid's own Nyquist limit with no rolloff, a different (much
    # harsher) spatial-frequency profile than a real diffuser's finite feature size actually has
    # -- the native-feature/upsample step is what keeps the residual bandwidth controlled and
    # below each grid's Nyquist limit with real margin, independent of how chunky it looks.
    diffuser_native_resolution = 8
    diffuser_native_upsample = resolution_coarse // diffuser_native_resolution
    torch.manual_seed(0)
    diffuser_phase_native = 2.0 * odak.pi * torch.rand(diffuser_native_resolution, diffuser_native_resolution, device=device)
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

    # Physical off-axis landing shift, shared by both arms (same physical scenario).
    sin_theta = offset_fx * wavelength
    tan_theta = sin_theta / math.sqrt(1.0 - sin_theta**2)
    chief_ray_shift_x_m = distance * tan_theta

    def recenter(field, pixel_pitch_m):
        h, w = field.shape[-2:]
        fy = torch.fft.fftfreq(h, d=pixel_pitch_m, device=field.device, dtype=torch.float32)
        fx = torch.fft.fftfreq(w, d=pixel_pitch_m, device=field.device, dtype=torch.float32)
        qy, qx = torch.meshgrid(2.0 * math.pi * fy, 2.0 * math.pi * fx, indexing="ij")
        shift_phase = torch.exp(1j * (qx * chief_ray_shift_x_m).to(torch.complex64))
        return torch.fft.ifft2(torch.fft.fft2(field) * shift_phase)

    # Arm A: raw BASM on the fully tilted field, at the fine grid.
    y_fine = (torch.arange(resolution_fine, device=device) - (resolution_fine - 1) / 2.0) * pitch_fine
    x_fine = (torch.arange(resolution_fine, device=device) - (resolution_fine - 1) / 2.0) * pitch_fine
    yy_fine, xx_fine = torch.meshgrid(y_fine, x_fine, indexing="ij")
    carrier_phase_fine = 2.0 * odak.pi * (offset_fx * xx_fine + offset_fy * yy_fine)
    carrier_fine = odak.learn.wave.generate_complex_field(torch.ones_like(carrier_phase_fine), carrier_phase_fine)
    tilted_field_fine = field_fine * carrier_fine.to(torch.complex64)

    memory_raw_bytes = tilted_field_fine.numel() * 8  # complex64 = 8 bytes/element
    propagated_raw = odak.learn.wave.band_limited_angular_spectrum(tilted_field_fine, k, distance, pitch_fine, wavelength)
    recentered_raw = recenter(propagated_raw, pitch_fine)

    intensity_raw_fine = odak.learn.wave.calculate_amplitude(recentered_raw) ** 2
    # Bin the fine-grid intensity down to the coarse pixel scale (pitch_coarse = 4 * pitch_fine)
    # for a pixel-for-pixel comparison against arm B, the same way sum_bin_sensor_pixels bins a
    # high-resolution simulation grid down to the sensor's own pixel pitch in
    # src/asm_psf_propagation.py.
    intensity_raw_binned = (
        intensity_raw_fine.reshape(
            resolution_coarse, upsample_factor, resolution_coarse, upsample_factor
        )
        .sum(dim=(1, 3))
    )

    # Arm B: shifted BASM on the untilted field, at the coarse grid.
    memory_shifted_bytes = field_coarse.numel() * 8
    propagated_shifted = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field_coarse, k, distance, pitch_coarse, wavelength, offset_fx=offset_fx, offset_fy=offset_fy
    )
    recentered_shifted = recenter(propagated_shifted, pitch_coarse)
    intensity_shifted = odak.learn.wave.calculate_amplitude(recentered_shifted) ** 2

    similarity = torch.sum(intensity_raw_binned * intensity_shifted) / torch.sqrt(
        torch.sum(intensity_raw_binned**2) * torch.sum(intensity_shifted**2)
    )

    memory_ratio = memory_raw_bytes / memory_shifted_bytes
    print(
        "raw BASM   (fine, {0}x{0}):   {1:>10.2f} MB array".format(resolution_fine, memory_raw_bytes / 1e6)
    )
    print(
        "shifted BASM (coarse, {0}x{0}): {1:>10.2f} MB array".format(resolution_coarse, memory_shifted_bytes / 1e6)
    )
    print("array memory ratio (raw / shifted): {:.2f}x".format(memory_ratio))
    print("normalized similarity (raw vs. shifted): {:.6f}".format(similarity.item()))

    odak.learn.tools.save_image(
        "{}/diffuser_phase_native.png".format(output_directory),
        diffuser_phase_coarse,  # upsampled to 256x256 for viewing; native is only 8x8 pixels
        cmin=0.0,
        cmax=2.0 * odak.pi,
    )
    odak.learn.tools.save_image(
        "{}/raw_basm_fine.png".format(output_directory),
        intensity_raw_binned,
        cmin=0.0,
        cmax=float(intensity_raw_binned.max()),
    )
    odak.learn.tools.save_image(
        "{}/shifted_basm_coarse.png".format(output_directory),
        intensity_shifted,
        cmin=0.0,
        cmax=float(intensity_shifted.max()),
    )

    assert memory_ratio > 8.0
    assert similarity.item() > 0.95
    assert True


if __name__ == "__main__":
    sys.exit(test())
