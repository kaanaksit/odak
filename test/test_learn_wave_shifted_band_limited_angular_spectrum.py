import sys
import odak
import torch


def test(device=torch.device("cpu"), output_directory="test_output"):
    odak.tools.check_directory(output_directory)
    wavelength = 532e-9  # (1)
    pixel_pitch = 2e-6  # (2)
    distance = 5e-3  # (3)
    resolution = [256, 256]  # (4)
    k = odak.learn.wave.wavenumber(wavelength)  # (5)
    grid_nyquist = 1.0 / (2.0 * pixel_pitch)  # 2.5e5 cycles/m for this grid
    propagating_limit = 1.0 / wavelength  # ~1.88e6 cycles/m -- the Helmholtz/evanescent boundary

    amplitude = torch.zeros(resolution)
    amplitude[100:156, 100:156] = 1.0  # (6)
    phase = torch.zeros_like(amplitude)
    field = odak.learn.wave.generate_complex_field(amplitude, phase).to(device)  # (7)

    # (1) On-axis case is unchanged: offset_fx = offset_fy = 0 must reproduce
    # band_limited_angular_spectrum exactly, since a zero carrier shift is a no-op
    # generalization of the existing function.
    reference = odak.learn.wave.band_limited_angular_spectrum(
        field, k, distance, pixel_pitch, wavelength
    )  # (8)
    zero_offset = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field, k, distance, pixel_pitch, wavelength, offset_fx=0.0, offset_fy=0.0
    )  # (9)
    assert torch.allclose(reference, zero_offset, atol=1e-6)

    # Physics correctness (the modulation/Fourier-shift derivation behind the shifted kernel,
    # Eq. 9 of Matsushima 2010): propagating a field that already carries a linear carrier
    # phase through the *unshifted* kernel should closely match propagating the untilted field
    # through the kernel *shifted* by the matching carrier frequency. The offset is chosen as an
    # exact multiple of the FFT bin spacing 1/(N*dx) so the discrete tilt is an exact circular
    # shift of the field's discrete spectrum (a non-bin-aligned offset would introduce spectral
    # leakage on the "direct" side that has nothing to do with the shifted kernel itself).
    #
    # The two are compared with a normalized-similarity metric rather than element-wise
    # equality: they are expected to differ slightly near the spectrum's edge bins, because the
    # "direct" side's circular bin shift silently wraps frequencies that fall off one edge back
    # onto the other (a discrete-FFT artifact), while the shifted kernel evaluates the
    # continuous propagation phase at those same absolute frequencies without wrapping. This
    # small discrepancy is expected and is exactly the aliasing risk carrier-frequency shifting
    # is meant to avoid for genuinely large offsets; for the modest offset used here it is small
    # (normalized inner product > 0.999 in practice).
    bin_spacing_x = 1.0 / (resolution[1] * pixel_pitch)
    offset_fx_small = 10.0 * bin_spacing_x  # comfortably inside this grid's Nyquist limit (128 bins)
    offset_fy_small = 0.0
    y = (torch.arange(resolution[0], device=device) - (resolution[0] - 1) / 2.0) * pixel_pitch
    x = (torch.arange(resolution[1], device=device) - (resolution[1] - 1) / 2.0) * pixel_pitch
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    carrier_phase = 2.0 * odak.pi * (offset_fx_small * xx + offset_fy_small * yy)
    carrier = odak.learn.wave.generate_complex_field(torch.ones_like(carrier_phase), carrier_phase)
    tilted_field = field * carrier.to(torch.complex64)  # (10)

    propagated_direct = odak.learn.wave.band_limited_angular_spectrum(
        tilted_field, k, distance, pixel_pitch, wavelength
    )
    propagated_shifted = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field, k, distance, pixel_pitch, wavelength, offset_fx=offset_fx_small, offset_fy=offset_fy_small
    )
    intensity_direct = odak.learn.wave.calculate_amplitude(propagated_direct) ** 2
    intensity_shifted = odak.learn.wave.calculate_amplitude(propagated_shifted) ** 2
    similarity = torch.sum(intensity_direct * intensity_shifted) / torch.sqrt(
        torch.sum(intensity_direct**2) * torch.sum(intensity_shifted**2)
    )
    assert similarity > 0.999

    # (2) A carrier well beyond this grid's own Nyquist limit (grid_nyquist) must NOT zero the
    # kernel, as long as (a) the residual frequency lies within the band-limiting mask B and (b)
    # the shifted absolute frequency is still within the Helmholtz propagating region. This is
    # the entire point of carrier-frequency shifting -- see the paper's claim that "all incident
    # angles share the same on-axis sampling condition" once the carrier is absorbed into the
    # kernel, quoted in the kernel builder's docstring.
    offset_fx_large = 1.0e6  # 4x grid_nyquist, well under propagating_limit (~1.88e6)
    assert offset_fx_large > grid_nyquist
    assert offset_fx_large < propagating_limit
    kernel_large_offset = odak.learn.wave.get_shifted_band_limited_angular_spectrum_kernel(
        nu=resolution[0], nv=resolution[1], dx=pixel_pitch, wavelength=wavelength, distance=distance,
        offset_fx=offset_fx_large, offset_fy=0.0, device=device,
    )
    assert torch.isfinite(torch.view_as_real(kernel_large_offset)).all()
    assert float(odak.learn.wave.calculate_amplitude(kernel_large_offset).max()) > 0.0

    # (3) The band-limiting mask B is evaluated only on the residual (FX, FY) grid and is
    # independent of the carrier: with a carrier small enough that every point inside B's
    # passband remains propagating both before and after the shift (guaranteed here since
    # grid_nyquist << propagating_limit), the kernel's amplitude pattern -- which is exactly the
    # 0/1 mask, since the phase term always has unit magnitude -- must be identical at zero
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
    assert torch.equal(mask_zero_offset > 0.5, mask_with_offset > 0.5)  # identical pass/reject pattern

    # (4) The propagation phase uses the shifted absolute frequencies (FX + offset_fx,
    # FY + offset_fy), not the residual ones: reproduce the kernel independently at a moderate
    # offset and confirm the phase matches within the region that both the mask and the
    # propagating condition keep valid.
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
    FX_shifted = FX + offset_fx_phase_check
    FY_shifted = FY
    expected_kz_squared = 1.0 / wavelength**2 - (FX_shifted**2 + FY_shifted**2)
    expected_propagating = expected_kz_squared >= 0.0
    expected_phase = 2.0 * odak.pi * torch.sqrt(torch.clamp(expected_kz_squared, min=0.0)) * distance

    x_extent = pixel_pitch * float(resolution[0])
    y_extent = pixel_pitch * float(resolution[1])
    distance_tensor = torch.tensor([distance])
    fx_max = 1.0 / torch.sqrt((2.0 * distance_tensor * (1.0 / x_extent)) ** 2 + 1.0) / wavelength
    fy_max = 1.0 / torch.sqrt((2.0 * distance_tensor * (1.0 / y_extent)) ** 2 + 1.0) / wavelength
    expected_in_mask = (torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max)
    expected_amplitude = (expected_in_mask & expected_propagating).to(torch.float32)
    # Compare complex kernel values directly (as test (1) above does) rather than raw phase
    # angles: the unwrapped propagation phase reaches tens of thousands of radians at this
    # distance, where float32 no longer has enough precision to represent "phase mod 2*pi" on
    # its own -- comparing via cos/sin (bounded in [-1, 1]) sidesteps that entirely, and
    # amplitude=0 at masked-out points makes the phase there irrelevant on both sides.
    expected_kernel = odak.learn.wave.generate_complex_field(expected_amplitude, expected_phase)

    kernel_phase_check = odak.learn.wave.get_shifted_band_limited_angular_spectrum_kernel(
        nu=resolution[0], nv=resolution[1], dx=pixel_pitch, wavelength=wavelength, distance=distance,
        offset_fx=offset_fx_phase_check, offset_fy=0.0, device=device,
    )
    assert expected_amplitude.any()
    assert torch.allclose(kernel_phase_check, expected_kernel.to(torch.complex64), atol=1e-3)

    # (5) Components with (FX + offset_fx)^2 + (FY + offset_fy)^2 > 1/wavelength^2 are rejected
    # as non-propagating rather than producing NaNs, and this is a genuine per-component
    # (not all-or-nothing) condition: choose an offset near propagating_limit so that some
    # residual frequencies within the array remain propagating after the shift and others do
    # not, and confirm both halves of that split are handled correctly.
    offset_fx_boundary = propagating_limit - 0.5 * grid_nyquist
    FX_shifted_boundary = FX + offset_fx_boundary
    kz_squared_boundary = 1.0 / wavelength**2 - (FX_shifted_boundary**2 + FY**2)
    propagating_boundary = kz_squared_boundary >= 0.0
    assert propagating_boundary.any()  # some components remain propagating ...
    assert (~propagating_boundary).any()  # ... and some do not: a genuine mixed case

    kernel_boundary = odak.learn.wave.get_shifted_band_limited_angular_spectrum_kernel(
        nu=resolution[0], nv=resolution[1], dx=pixel_pitch, wavelength=wavelength, distance=distance,
        offset_fx=offset_fx_boundary, offset_fy=0.0, device=device,
    )
    assert torch.isfinite(torch.view_as_real(kernel_boundary)).all()  # no NaN/Inf anywhere

    # (6) The sqrt clamp is for numerical safety only: it must not let a non-propagating
    # component through with nonzero amplitude. Wherever the true (unclamped) condition rejects
    # a component, this kernel's amplitude there must be exactly zero -- the aperture mask B
    # only widens the exclusion (residual frequencies outside B are excluded even if they would
    # still be propagating), so this must hold regardless of whether B or the propagating check
    # was the reason for rejection.
    boundary_amplitude = odak.learn.wave.calculate_amplitude(kernel_boundary)
    assert torch.equal(boundary_amplitude[~propagating_boundary], torch.zeros_like(boundary_amplitude[~propagating_boundary]))

    odak.learn.tools.save_image(
        "{}/shifted_band_limited_angular_spectrum.png".format(output_directory),
        intensity_shifted,
        cmin=0.0,
        cmax=float(intensity_shifted.max()),
    )  # (11)
    assert True


if __name__ == "__main__":
    sys.exit(test())
