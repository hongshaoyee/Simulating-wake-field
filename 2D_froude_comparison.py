import numpy as np
import matplotlib.pyplot as plt

# Try to use scipy FFT if available; otherwise fall back to numpy FFT
try:
    from scipy.fft import rfft2, irfft2
    HAS_SCIPY_FFT = True
except ImportError:
    from numpy.fft import rfft2, irfft2
    HAS_SCIPY_FFT = False


# ============================================================
# 1. PHYSICAL CONSTANTS (fixed constants)
# ============================================================
RHO_WATER = 1025.0              # seawater density (kg/m^3)
GRAVITY = 9.81                  # gravitational acceleration (m/s^2)
PRESSURE_AMPLITUDE = 1.0e5      # Gaussian pressure amplitude (Pa)

# Classical Kelvin half-angle: arcsin(1/3) ≈ 19.47 deg
KELVIN_ANGLE_DEG = np.degrees(np.arcsin(1.0 / 3.0))


# ============================================================
# 2. USER-EDITABLE MODEL PARAMETERS
#    Put everything you may want to change here
# ============================================================

# ---------- Physical reference scales ----------
REF_LENGTH = 10.0               # reference hull half-length used in Fr definition (m)
SOURCE_WIDTH = 3.0              # Gaussian pressure-source half-width (m)

# ---------- Numerical domain ----------
GRID_SIZE = 1024                # number of grid points per side
DOMAIN_SIZE = 1200.0            # physical box size in x and y (m)

# ---------- Damping / absorbing boundary ----------
DAMPING_COEFF = 0.005           # small linear damping coefficient (1/s)
SPONGE_FRACTION = 0.15          # fraction of box used as absorbing boundary
RAMP_STEPS = 100                # number of startup steps for source ramp-up

# ---------- FFT ----------
FFT_WORKERS = -1                # scipy.fft workers if available

# ---------- Cases to compare ----------
FROUDE_LOW = 0.50
FROUDE_HIGH = 2.00

# ---------- Plot settings ----------
PLOT_CROP_X = (-450, 50)        # in ship-centered coordinates
PLOT_CROP_Y = (-160, 160)
SHOW_MEASURED_HIGH_FR_ANGLE = True
OUTPUT_FIGURE = "poster_low_high_split.png"


# ============================================================
# 3. Sponge layer (absorbing boundary)
# ============================================================
def make_sponge_mask(grid_size, sponge_fraction):
    """
    Create a 2D sponge mask that is ~1 in the interior and tapers
    smoothly to 0 near the boundaries.

    Parameters
    ----------
    grid_size : int
        Number of grid points along one dimension.
    sponge_fraction : float
        Fraction of the domain width used for the sponge layer.

    Returns
    -------
    sponge_mask : 2D ndarray
        Multiplicative mask used to absorb outgoing waves near boundaries.
    """
    taper_1d = np.ones(grid_size)
    sponge_points = int(sponge_fraction * grid_size)

    if sponge_points > 0:
        # Cosine taper: near boundary -> 0, interior -> 1
        cosine_ramp = 0.5 * (1 + np.cos(np.linspace(np.pi, 0, sponge_points)))
        taper_1d[:sponge_points] = cosine_ramp
        taper_1d[-sponge_points:] = cosine_ramp[::-1]

    sponge_mask = np.outer(taper_1d, taper_1d)
    return sponge_mask


# ============================================================
# 4. Unified Kelvin wake solver
# ============================================================
def run_kelvin_case(
    froude_number,
    grid_size,
    domain_size,
    ref_length,
    source_width,
    pressure_amplitude,
    fluid_density,
    gravity_accel,
    damping_coeff,
    sponge_fraction,
    ramp_steps,
    fft_workers=-1,
):
    """
    Run one linear deep-water Kelvin wake simulation using a moving
    Gaussian pressure source and a pseudospectral Fourier method.

    Froude number definition:
        Fr = U / sqrt(g * L_ref)

    Parameters
    ----------
    froude_number : float
        Target Froude number.
    grid_size : int
        Number of grid points per side.
    domain_size : float
        Physical domain size in x and y (m).
    ref_length : float
        Reference length used in Fr definition.
    source_width : float
        Gaussian source half-width (m).
    pressure_amplitude : float
        Pressure amplitude (Pa).
    fluid_density : float
        Fluid density (kg/m^3).
    gravity_accel : float
        Gravitational acceleration (m/s^2).
    damping_coeff : float
        Linear damping coefficient (1/s).
    sponge_fraction : float
        Fraction of box used for absorbing boundary.
    ramp_steps : int
        Number of startup steps for gradual source ramp-up.
    fft_workers : int
        Number of FFT workers if scipy.fft is available.

    Returns
    -------
    result : dict
        Contains the final computed field and metadata.
    """
    fft_kwargs = dict(workers=fft_workers) if HAS_SCIPY_FFT else {}

    # Convert Fr to physical speed
    ship_speed = froude_number * np.sqrt(gravity_accel * ref_length)

    # Numerical resolution in space and time
    dx = domain_size / grid_size
    dt = dx / ship_speed
    num_steps = int(0.6 * grid_size)

    print(
        f"\nRunning case: Fr = {froude_number:.2f}, "
        f"U = {ship_speed:.2f} m/s, N = {grid_size}, D = {domain_size:.1f} m"
    )
    print(f"dx = {dx:.4f} m, dt = {dt:.5f} s, steps = {num_steps}")

    # ------------------------------------------------------------
    # Physical grid
    # ------------------------------------------------------------
    x_coords = np.linspace(-domain_size / 2, domain_size / 2, grid_size, endpoint=False)
    y_coords = np.linspace(-domain_size / 2, domain_size / 2, grid_size, endpoint=False)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")

    # ------------------------------------------------------------
    # Fourier grid
    # ------------------------------------------------------------
    kx_coords = np.fft.fftfreq(grid_size, d=dx) * 2 * np.pi
    ky_coords = np.fft.rfftfreq(grid_size, d=dx) * 2 * np.pi
    KX_grid, KY_grid = np.meshgrid(kx_coords, ky_coords, indexing="ij")
    k_magnitude = np.sqrt(KX_grid**2 + KY_grid**2)

    # Deep-water dispersion relation: omega^2 = g |k|
    omega = np.sqrt(gravity_accel * k_magnitude)
    omega[0, 0] = 0.0

    # Exact propagator for the linear free-surface system
    cos_omega_dt = np.cos(omega * dt)
    sin_omega_dt = np.sin(omega * dt)

    with np.errstate(divide="ignore", invalid="ignore"):
        k_over_omega = np.where(k_magnitude > 0, k_magnitude / omega, 0.0)
        omega_over_k = np.where(k_magnitude > 0, omega / k_magnitude, 0.0)

    # ------------------------------------------------------------
    # Sponge / damping
    # ------------------------------------------------------------
    sponge_mask = make_sponge_mask(grid_size, sponge_fraction)

    # ------------------------------------------------------------
    # State variables
    # eta = free-surface elevation
    # phi = surface velocity potential
    # ------------------------------------------------------------
    eta_field = np.zeros((grid_size, grid_size), dtype=float)
    phi_field = np.zeros((grid_size, grid_size), dtype=float)

    damping_factor = np.exp(-damping_coeff * dt)
    eta_decay = damping_factor * sponge_mask
    forcing_coeff = dt / (2.0 * fluid_density)

    inv_source_width_sq = 1.0 / (source_width**2)
    y_term = Y_grid**2 * inv_source_width_sq

    # ------------------------------------------------------------
    # Time stepping: Strang splitting
    # ------------------------------------------------------------
    for step_idx in range(num_steps):
        # Ship moves in +x direction
        x_ship_pos = -domain_size / 4 + step_idx * ship_speed * dt

        # 1) Half-step pressure forcing
        pressure_field = pressure_amplitude * np.exp(
            -((X_grid - x_ship_pos) ** 2 * inv_source_width_sq + y_term)
        )

        # Ramp source gradually at start
        ramp_value = min(1.0, (step_idx + 1) / ramp_steps)
        pressure_field *= ramp_value

        phi_field -= pressure_field * forcing_coeff

        # 2) Exact linear propagation in Fourier space
        eta_hat = rfft2(eta_field, **fft_kwargs)
        phi_hat = rfft2(phi_field, **fft_kwargs)

        eta_hat_new = cos_omega_dt * eta_hat + k_over_omega * sin_omega_dt * phi_hat
        phi_hat_new = -omega_over_k * sin_omega_dt * eta_hat + cos_omega_dt * phi_hat

        eta_field = irfft2(eta_hat_new, s=(grid_size, grid_size), **fft_kwargs)
        phi_field = irfft2(phi_hat_new, s=(grid_size, grid_size), **fft_kwargs)

        # 3) Damping and sponge
        eta_field *= eta_decay
        phi_field *= damping_factor

        # 4) Second half-step forcing
        phi_field -= pressure_field * forcing_coeff

        # 5) Additional sponge on phi
        phi_field *= sponge_mask

        if (step_idx + 1) % 200 == 0:
            print(f"  step {step_idx + 1:4d}/{num_steps}")

    result = {
        "Fr": froude_number,
        "U": ship_speed,
        "eta": eta_field,
        "x1d": x_coords,
        "y1d": y_coords,
        "x_ship": x_ship_pos,
    }
    return result


# ============================================================
# 5. Measure apparent wake angle
# ============================================================
def measure_wake_angle(
    eta_field,
    x_coords,
    y_coords,
    x_ship_pos,
    ref_length,
    sponge_fraction,
):
    """
    Estimate the apparent wake half-angle from a computed wave field.

    Method:
    - sample |eta| along many radial rays behind the ship
    - compute RMS along each ray
    - find the outermost angle whose RMS stays above half the peak

    Parameters
    ----------
    eta_field : 2D ndarray
        Surface elevation field.
    x_coords, y_coords : 1D arrays
        Physical coordinates.
    x_ship_pos : float
        Ship x-position.
    ref_length : float
        Reference length scale.
    sponge_fraction : float
        Fraction of the domain excluded near boundaries.

    Returns
    -------
    apparent_angle_deg : float
        Estimated apparent wake half-angle in degrees.
    """
    domain_size = x_coords[-1] - x_coords[0]
    grid_size = len(x_coords)

    scan_angles_deg = np.linspace(1.0, 45.0, 400)
    scan_angles_rad = np.radians(scan_angles_deg)

    sponge_margin = sponge_fraction * domain_size
    r_max = domain_size / 2 - sponge_margin
    r_min = max(5.0 * ref_length, 0.12 * domain_size)
    r_min = min(r_min, 0.5 * r_max)

    dx = x_coords[1] - x_coords[0]
    num_radial_samples = max(int((r_max - r_min) / dx), 50)
    radii = np.linspace(r_min, r_max, num_radial_samples)

    cos_angles = np.cos(scan_angles_rad)[:, None]
    sin_angles = np.sin(scan_angles_rad)[:, None]
    radii_row = radii[None, :]

    ray_x = x_ship_pos - radii_row * cos_angles

    total_sq = np.zeros_like(scan_angles_deg)
    count = np.zeros_like(scan_angles_deg)

    for sign in [1, -1]:
        ray_y = sign * radii_row * sin_angles

        ray_x_flat = ray_x.ravel()
        ray_y_flat = ray_y.ravel()

        ix = np.searchsorted(x_coords, ray_x_flat) - 1
        iy = np.searchsorted(y_coords, ray_y_flat) - 1
        ix = np.clip(ix, 0, grid_size - 1)
        iy = np.clip(iy, 0, grid_size - 1)

        ix_next = np.clip(ix + 1, 0, grid_size - 1)
        iy_next = np.clip(iy + 1, 0, grid_size - 1)

        use_next_x = np.abs(x_coords[ix_next] - ray_x_flat) < np.abs(x_coords[ix] - ray_x_flat)
        use_next_y = np.abs(y_coords[iy_next] - ray_y_flat) < np.abs(y_coords[iy] - ray_y_flat)

        ix = np.where(use_next_x, ix_next, ix)
        iy = np.where(use_next_y, iy_next, iy)

        valid_mask = (
            (ray_x_flat >= x_coords[0]) & (ray_x_flat <= x_coords[-1]) &
            (ray_y_flat >= y_coords[0]) & (ray_y_flat <= y_coords[-1])
        )

        sampled_vals = np.abs(eta_field[ix, iy])
        sampled_vals = np.where(valid_mask, sampled_vals, 0.0)

        vals_2d = sampled_vals.reshape(len(scan_angles_deg), num_radial_samples)
        valid_2d = valid_mask.reshape(len(scan_angles_deg), num_radial_samples)

        total_sq += np.sum(vals_2d**2, axis=1)
        count += np.sum(valid_2d, axis=1)

    rms_vals = np.zeros_like(scan_angles_deg)
    valid_rms = count > 0
    rms_vals[valid_rms] = np.sqrt(total_sq[valid_rms] / count[valid_rms])

    # Light Gaussian smoothing in angle space
    sigma = 2
    half_width = 3 * sigma
    kernel_axis = np.arange(-half_width, half_width + 1, dtype=float)
    gaussian_kernel = np.exp(-0.5 * (kernel_axis / sigma) ** 2)
    gaussian_kernel /= gaussian_kernel.sum()

    rms_smooth = np.convolve(rms_vals, gaussian_kernel, mode="same")

    peak_val = np.max(rms_smooth)
    if peak_val <= 0:
        return scan_angles_deg[0]

    threshold = 0.5 * peak_val
    above_threshold = np.where(rms_smooth >= threshold)[0]

    apparent_angle_deg = (
        scan_angles_deg[above_threshold[-1]]
        if len(above_threshold) > 0
        else scan_angles_deg[0]
    )
    return apparent_angle_deg


# ============================================================
# 6. Plot comparison: low Fr on top, high Fr on bottom
# ============================================================
def plot_split_low_high(
    low_result,
    high_result,
    ref_length,
    sponge_fraction,
    kelvin_angle_deg=19.47,
    crop_x=(-450, 50),
    crop_y=(-160, 160),
    add_high_angle=True,
    savepath="poster_low_high_split.png",
):
    """
    Plot a single compact comparison figure:
    - upper half-plane (y >= 0): low-Froude wake
    - lower half-plane (y <= 0): high-Froude wake

    Coordinates are re-centered so the ship is at x=0.
    """
    # Low-Fr data
    x_low = low_result["x1d"]
    y_low = low_result["y1d"]
    eta_low = low_result["eta"]
    ship_x_low = low_result["x_ship"]

    # High-Fr data
    x_high = high_result["x1d"]
    y_high = high_result["y1d"]
    eta_high = high_result["eta"]
    ship_x_high = high_result["x_ship"]

    # Re-center each case so the ship sits at x = 0
    xrel_low = x_low - ship_x_low
    xrel_high = x_high - ship_x_high

    X_low, Y_low = np.meshgrid(xrel_low, y_low, indexing="ij")
    X_high, Y_high = np.meshgrid(xrel_high, y_high, indexing="ij")

    # Keep only top half for low Fr and bottom half for high Fr
    eta_top = np.ma.masked_where(Y_low < 0, eta_low)
    eta_bottom = np.ma.masked_where(Y_high > 0, eta_high)

    # Shared color scale
    vmax = max(
        np.percentile(np.abs(eta_low), 99),
        np.percentile(np.abs(eta_high), 99)
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=220)

    cf = ax.pcolormesh(
        xrel_low, y_low, eta_top.T,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
    )

    ax.pcolormesh(
        xrel_high, y_high, eta_bottom.T,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
    )

    ax.axhline(0, color="black", lw=0.9)
    ax.plot(0, 0, "ko", ms=5)

    # Kelvin angle guide lines
    x_back = np.linspace(crop_x[0], 0, 400)
    y_kelvin_up = np.tan(np.radians(kelvin_angle_deg)) * (-x_back)
    y_kelvin_down = -np.tan(np.radians(kelvin_angle_deg)) * (-x_back)

    ax.plot(x_back, y_kelvin_up, "k--", lw=1.0)
    ax.plot(x_back, y_kelvin_down, "k--", lw=1.0)

    # Measured high-Fr angle
    if add_high_angle:
        measured_angle = measure_wake_angle(
            eta_field=high_result["eta"],
            x_coords=high_result["x1d"],
            y_coords=high_result["y1d"],
            x_ship_pos=high_result["x_ship"],
            ref_length=ref_length,
            sponge_fraction=sponge_fraction,
        )

        y_measured = -np.tan(np.radians(measured_angle)) * (-x_back)
        ax.plot(x_back, y_measured, color="red", lw=1.3)

    ax.text(
        crop_x[0] + 20, crop_y[1] - 28,
        f"Low Fr = {low_result['Fr']:.2f}",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
    )

    ax.text(
        crop_x[0] + 20, crop_y[0] + 18,
        f"High Fr = {high_result['Fr']:.2f}",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
    )

    ax.text(
        crop_x[1] - 135, crop_y[1] - 28,
        "Kelvin angle",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
    )

    if add_high_angle:
        ax.text(
            crop_x[1] - 195, crop_y[0] + 18,
            "Measured high-Fr angle",
            color="red",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
        )

    ax.set_xlim(crop_x)
    ax.set_ylim(crop_y)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x - x_{\mathrm{ship}}$ (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Low vs high Froude wake", fontsize=14)

    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Surface elevation η")

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved {savepath}")


# ============================================================
# 7. Main
# ============================================================
if __name__ == "__main__":
    low_case = run_kelvin_case(
        froude_number=FROUDE_LOW,
        grid_size=GRID_SIZE,
        domain_size=DOMAIN_SIZE,
        ref_length=REF_LENGTH,
        source_width=SOURCE_WIDTH,
        pressure_amplitude=PRESSURE_AMPLITUDE,
        fluid_density=RHO_WATER,
        gravity_accel=GRAVITY,
        damping_coeff=DAMPING_COEFF,
        sponge_fraction=SPONGE_FRACTION,
        ramp_steps=RAMP_STEPS,
        fft_workers=FFT_WORKERS,
    )

    high_case = run_kelvin_case(
        froude_number=FROUDE_HIGH,
        grid_size=GRID_SIZE,
        domain_size=DOMAIN_SIZE,
        ref_length=REF_LENGTH,
        source_width=SOURCE_WIDTH,
        pressure_amplitude=PRESSURE_AMPLITUDE,
        fluid_density=RHO_WATER,
        gravity_accel=GRAVITY,
        damping_coeff=DAMPING_COEFF,
        sponge_fraction=SPONGE_FRACTION,
        ramp_steps=RAMP_STEPS,
        fft_workers=FFT_WORKERS,
    )

    plot_split_low_high(
        low_result=low_case,
        high_result=high_case,
        ref_length=REF_LENGTH,
        sponge_fraction=SPONGE_FRACTION,
        kelvin_angle_deg=KELVIN_ANGLE_DEG,
        crop_x=PLOT_CROP_X,
        crop_y=PLOT_CROP_Y,
        add_high_angle=SHOW_MEASURED_HIGH_FR_ANGLE,
        savepath=OUTPUT_FIGURE,
    )