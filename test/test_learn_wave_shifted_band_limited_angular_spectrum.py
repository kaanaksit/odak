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

    amplitude = torch.zeros(resolution)
    amplitude[100:156, 100:156] = 1.0  # (6)
    phase = torch.zeros_like(amplitude)
    field = odak.learn.wave.generate_complex_field(amplitude, phase).to(device)  # (7)

    # offset_fx = offset_fy = 0 must reproduce band_limited_angular_spectrum exactly, since a
    # zero carrier shift is a no-op generalization of the existing function.
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
    # is meant to avoid for genuinely large offsets (see the module docstring); for the modest
    # offset used here it is small (normalized inner product > 0.9999 in practice).
    bin_spacing_x = 1.0 / (resolution[1] * pixel_pitch)
    offset_fx = 10.0 * bin_spacing_x  # comfortably inside this grid's Nyquist limit (128 bins)
    offset_fy = 0.0
    y = (torch.arange(resolution[0], device=device) - (resolution[0] - 1) / 2.0) * pixel_pitch
    x = (torch.arange(resolution[1], device=device) - (resolution[1] - 1) / 2.0) * pixel_pitch
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    carrier_phase = 2.0 * odak.pi * (offset_fx * xx + offset_fy * yy)
    carrier = odak.learn.wave.generate_complex_field(torch.ones_like(carrier_phase), carrier_phase)
    tilted_field = field * carrier.to(torch.complex64)  # (10)

    propagated_direct = odak.learn.wave.band_limited_angular_spectrum(
        tilted_field, k, distance, pixel_pitch, wavelength
    )
    propagated_shifted = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field, k, distance, pixel_pitch, wavelength, offset_fx=offset_fx, offset_fy=offset_fy
    )
    intensity_direct = odak.learn.wave.calculate_amplitude(propagated_direct) ** 2
    intensity_shifted = odak.learn.wave.calculate_amplitude(propagated_shifted) ** 2
    similarity = torch.sum(intensity_direct * intensity_shifted) / torch.sqrt(
        torch.sum(intensity_direct**2) * torch.sum(intensity_shifted**2)
    )
    assert similarity > 0.999

    # Documented discrepancy: the band-limiting mask (Eq. 5-6 of Kang et al., "Geometry-aware
    # phase compensation for sampling-efficient angular spectrum method," Opt. Express 34(8),
    # 15244 (2026)) is evaluated at the literal (absolute, shifted) frequency per the bare
    # notation of Eq. 9. When the carrier frequency alone exceeds fx,req the mask is false
    # everywhere and the kernel is identically zero, even if the *residual* (post-carrier)
    # bandwidth would easily fit this grid. This is captured here rather than silently worked
    # around.
    large_offset_fx = 0.9 / pixel_pitch  # far beyond this grid's Nyquist limit of 1/(2*pixel_pitch)
    zeroed = odak.learn.wave.shifted_band_limited_angular_spectrum(
        field, k, distance, pixel_pitch, wavelength, offset_fx=large_offset_fx, offset_fy=0.0
    )
    zeroed_intensity = odak.learn.wave.calculate_amplitude(zeroed) ** 2
    assert float(zeroed_intensity.max()) < 1e-12

    odak.learn.tools.save_image(
        "{}/shifted_band_limited_angular_spectrum.png".format(output_directory),
        intensity_shifted,
        cmin=0.0,
        cmax=float(intensity_shifted.max()),
    )  # (11)
    assert True


if __name__ == "__main__":
    sys.exit(test())
