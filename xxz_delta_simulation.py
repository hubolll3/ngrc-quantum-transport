import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tenpy.algorithms import tebd
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tqdm import tqdm

# ==============================================================================
# 1. PROFILE FUNCTIONS (Scaled strictly for <sigma_z> in [-1.0, +1.0])[cite: 4]
# ==============================================================================

def random_smooth_gaussian_filter(L: int, sigma: float = 4.0, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.randn(L)
    smoothed = gaussian_filter1d(noise, sigma=sigma, mode="nearest")
    return smoothed / np.max(np.abs(smoothed))


def step_profile(L: int, left_val: float = 1.0, right_val: float = -1.0) -> np.ndarray:
    """Step function (domain wall): Spin Up (+1.0) on left, Spin Down (-1.0) on right."""
    profile = np.ones(L) * left_val
    profile[L // 2 :] = right_val
    return profile


def gaussian_profile(L: int, amp: float = 2.0, sigma: float = 10.0, center: float = None) -> np.ndarray:
    """Gaussian pulse of spin-up excitation on a spin-down (-1.0) background."""
    if center is None:
        center = L / 2.0
    j = np.arange(L)
    return -1.0 + amp * np.exp(-((j - center) ** 2) / (2 * sigma**2))


def delta_profile(L: int, site: int = None, val_background: float = -1.0, val_peak: float = 1.0) -> np.ndarray:
    """Single-site (delta) excitation on a uniform background bounded in [-1.0, +1.0]."""
    if site is None:
        site = L // 2
    profile = np.ones(L) * val_background
    profile[site] = val_peak
    return profile


def sine_profile(L: int, amp: float = 1.0, k: float = 4.0) -> np.ndarray:
    """Sinusoidal magnetization profile bounded in [-1.0, +1.0]."""
    j = np.arange(L)
    return amp * np.sin(2 * np.pi * k * j / L)


def random_rough_uniform(L: int, seed: int = None) -> np.ndarray:
    """Generates raw, un-smoothed white noise strictly in (-1, 1)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-1.0, 1.0, L)


def create_mps_from_sigmaz_profile(sites, bc_MPS: str, sigmaz_profile: np.ndarray) -> MPS:
    """
    Converts a <sigma_z> expectation array (values in [-1.0, 1.0])
    into a TeNPy Matrix Product State (MPS) product state[cite: 4].
    """
    sigmaz_profile = np.clip(sigmaz_profile, -1.0, 1.0)

    product_state = []
    for Z in sigmaz_profile:
        c_up = np.sqrt((1.0 + Z) / 2.0)
        c_down = np.sqrt((1.0 - Z) / 2.0)

        # TeNPy basis ordering is [down (0), up (1)][cite: 4]
        product_state.append(np.array([c_down, c_up]))

    return MPS.from_product_state(
        sites, product_state, bc=bc_MPS, unit_cell_width=len(sites)
    )


# ==============================================================================
# 2. XXZ HEISENBERG SIMULATION ENGINE
# ==============================================================================

def simulate_xxz_dynamics(
    L: int = 100,
    t_final: float = 50.0,
    dt: float = 0.05,
    J: float = 1.0,
    Delta: float = 0.5,
    profile_func=None,
    profile_label: str = "step",
    data_dir: str = "og_data_xxz",
    plots_dir: str = "og_plots_xxz",
):
    """
    Simulates quantum dynamics for the XXZ Heisenberg model:
      H = J * sum_j (S_j^x S_{j+1}^x + S_j^y S_{j+1}^y + Delta * S_j^z S_{j+1}^z)
    
    Measures <sigma_z>, saves datasets and plots with 'xxz' and 'Delta' in filenames[cite: 4].
    """
    if profile_func is None:
        profile_func = lambda L: step_profile(L)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Dynamic filename generation incorporating 'xxz' and 'Delta'
    delta_str = f"{Delta:.2f}".replace(".", "p")
    output_txt = f"xxz_delta_{delta_str}_{profile_label}_sigmaz_data.txt"
    output_pdf = f"xxz_delta_{delta_str}_{profile_label}_dynamics.pdf"

    # TeNPy SpinChain Model Setup for XXZ Heisenberg Chain
    model_params = {
        "L": L,
        "S": 0.5,
        "Jx": J,
        "Jy": J,
        "Jz": J * Delta,  # Anisotropy parameter Delta
        "hx": 0.0,        # No transverse field (conserves Sz)
        "hz": 0.0,        # No longitudinal field
        "bc_MPS": "finite",
        "conserve": None, # Kept None to support single-site product state superpositions[cite: 4]
    }
    model = SpinChain(model_params)

    # Initial MPS State
    sigmaz_init = profile_func(L)
    psi = create_mps_from_sigmaz_profile(
        model.lat.mps_sites(), model.lat.bc_MPS, sigmaz_init
    )

    # TEBD Engine Configuration[cite: 4]
    tebd_params = {
        "dt": dt,
        "N_steps": 2,
        "order": 2,
        "trunc_params": {
            "chi_max": 128,
            "svd_min": 1e-10,
        },
    }
    engine = tebd.TEBDEngine(psi, model, tebd_params)

    # Time Evolution Loop (<sigma_z> = 2 * <S_z>)[cite: 4]
    times = [0.0]
    sigmaz_profiles = [2.0 * psi.expectation_value("Sz")]

    total_iterations = int(t_final / (dt * tebd_params["N_steps"]))

    for _ in tqdm(
        range(total_iterations),
        desc=f"Simulating XXZ (Δ={Delta}, {profile_label})",
        unit="step",
    ):
        engine.run()
        times.append(engine.evolved_time)
        sigmaz_profiles.append(2.0 * psi.expectation_value("Sz"))

    times = np.array(times)
    sigmaz_profiles = np.array(sigmaz_profiles)

    # Save Data Matrix for ML Pipeline[cite: 4]
    ml_data_matrix = np.column_stack([times, sigmaz_profiles])
    site_headers = [f"site_{j}" for j in range(L)]
    header_str = "time " + " ".join(site_headers)

    data_path = os.path.join(data_dir, output_txt)
    np.savetxt(
        data_path,
        ml_data_matrix,
        fmt="%.8e",
        header=header_str,
        comments="# ",
    )
    print(f"Dataset successfully saved to '{data_path}'.")

    # Plotting Dynamics[cite: 4]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    snapshot_indices = np.linspace(0, len(times) - 1, 10, dtype=int)
    for idx in snapshot_indices:
        t_val = times[idx]
        axes[0].plot(
            range(L),
            sigmaz_profiles[idx],
            marker="o",
            linestyle="-",
            markersize=3,
            label=f"$t = {t_val:.2f}$",
        )

    axes[0].set_xlabel("Site Index $j$", fontsize=12)
    axes[0].set_ylabel("$\\langle \\sigma^z_j \\rangle$", fontsize=12)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_title(
        f"XXZ ($\\Delta={Delta}$) Spatial Profile $\\langle \\sigma^z_j \\rangle$",
        fontsize=12,
    )
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=9)

    chosen_sites = np.linspace(0, L - 1, 10, dtype=int)
    for j in chosen_sites:
        axes[1].plot(
            times,
            sigmaz_profiles[:, j],
            label=f"Site $j = {j}$",
            linewidth=1.5,
        )

    axes[1].set_xlabel("Time $t$", fontsize=12)
    axes[1].set_ylabel("$\\langle \\sigma^z_j \\rangle$", fontsize=12)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].set_title(
        f"XXZ ($\\Delta={Delta}$) Time Dynamics $\\langle \\sigma^z_j \\rangle(t)$",
        fontsize=12,
    )
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, output_pdf)
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Plot saved to '{plot_path}'.\n")


# ==============================================================================
# 3. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    L = 100
    t_f = 50.0
    J_val = 1.0
    
    # Set anisotropic parameter Delta (e.g. Delta = 1.5 for easy-axis diffusive regime)
    Delta_val = 1.5

    # 1. Domain-wall step profile
    simulate_xxz_dynamics(
        L=L,
        t_final=t_f,
        J=J_val,
        Delta=Delta_val,
        profile_func=lambda L: 0.5 * step_profile(L, left_val=1.0, right_val=-1.0),
        profile_label="half_step",
    )

    # 2. Gaussian wavepacket profile
    simulate_xxz_dynamics(
        L=L,
        t_final=t_f,
        J=J_val,
        Delta=Delta_val,
        profile_func=lambda L: gaussian_profile(L, amp=2.0, sigma=10.0, center=L / 2),
        profile_label="gaussian",
    )

    # 3. Delta single-site excitation
    simulate_xxz_dynamics(
        L=L,
        t_final=t_f,
        J=J_val,
        Delta=Delta_val,
        profile_func=lambda L: delta_profile(L, site=L // 2, val_background=-1.0, val_peak=1.0),
        profile_label="delta",
    )

    # 4. Sinusoidal perturbation profile
    simulate_xxz_dynamics(
        L=L,
        t_final=t_f,
        J=J_val,
        Delta=Delta_val,
        profile_func=lambda L: sine_profile(L, amp=1.0, k=4.0),
        profile_label="sine",
    )

    # 5. Random smooth profiles
    for seed_val in [200, 300]:
        simulate_xxz_dynamics(
            L=L,
            t_final=t_f,
            J=J_val,
            Delta=Delta_val,
            profile_func=lambda L: random_smooth_gaussian_filter(L, sigma=4, seed=seed_val),
            profile_label=f"random_gauss_{seed_val}",
        )