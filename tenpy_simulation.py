import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tenpy.algorithms import tebd
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tqdm import tqdm
# ==============================================================================
# 1. PROFILE FUNCTIONS (Scaled strictly for <sigma_z> in [-1.0, +1.0])
# ==============================================================================

def random_smooth_gaussian_filter(L: int, sigma: float = 4.0, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    # Generate 1D white noise and smooth via Gaussian convolution
    noise = np.random.randn(L)
    smoothed = gaussian_filter1d(noise, sigma=sigma, mode="nearest")
    
    return smoothed / np.max(np.abs(smoothed))

def step_profile(
    L: int, left_val: float = 1.0, right_val: float = -1.0
) -> np.ndarray:
    """Step function (domain wall): Spin Up (+1.0) on left, Spin Down (-1.0) on right."""
    profile = np.ones(L) * left_val
    profile[L // 2 :] = right_val
    return profile


def gaussian_profile(
    L: int, amp: float = 2.0, sigma: float = 10.0, center: float = None
) -> np.ndarray:
    """Gaussian pulse of spin-up excitation (peak at +1.0) on a spin-down (-1.0) background."""
    if center is None:
        center = L / 2.0
    j = np.arange(L)
    return -1.0 + amp * np.exp(-((j - center) ** 2) / (2 * sigma**2))


def sine_profile(L: int, amp: float = 1.0, k: float = 4.0) -> np.ndarray:
    """Sinusoidal magnetization profile bounded in [-1.0, +1.0]."""
    j = np.arange(L)
    return amp * np.sin(2 * np.pi * k * j / L)


def random_smooth_profile(L: int, num_modes: int = 5, seed: int = None) -> np.ndarray:
    """Generates a random smooth continuous initial profile bounded in [-1, 1]."""
    if seed is not None:
        np.random.seed(seed)
    j = np.arange(L)
    profile = np.zeros(L)
    for _ in range(num_modes):
        k = np.random.randint(1, 5)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.2, 1.0)
        profile += amp * np.sin(2 * np.pi * k * j / L + phase)
    # Normalize strictly to [-1, 1]
    return profile / np.max(np.abs(profile))

def random_rough_uniform(L: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    # Generate raw, un-smoothed white noise strictly in (-1, 1)
    return np.random.uniform(-1.0, 1.0, L)

def create_mps_from_sigmaz_profile(
    sites, bc_MPS: str, sigmaz_profile: np.ndarray
) -> MPS:
    """Converts a <sigma_z> expectation array (values in [-1.0, 1.0])

    into a TeNPy Matrix Product State (MPS) product state.
    """
    sigmaz_profile = np.clip(sigmaz_profile, -1.0, 1.0)

    product_state = []
    for Z in sigmaz_profile:
        c_up = np.sqrt((1.0 + Z) / 2.0)
        c_down = np.sqrt((1.0 - Z) / 2.0)

        # FIXED: TeNPy basis ordering is [down (0), up (1)]
        product_state.append(np.array([c_down, c_up]))

    return MPS.from_product_state(
        sites, product_state, bc=bc_MPS, unit_cell_width=len(sites)
    )


# ==============================================================================
# 2. MAIN SIMULATION ENGINE
# ==============================================================================

def simulate_xx_dynamics(
    L: int = 200,
    t_final: float = 50.0,
    dt: float = 0.05,
    profile_func=None,
    output_pdf: str = "xx_model_dynamics.pdf",
    output_txt: str = "xx_sigmaz_data.txt",
):
    """Simulates XX dynamics, measures <sigma_z>, saves data for ML, and plots results."""
    if profile_func is None:
        profile_func = lambda L: step_profile(L)

    os.makedirs("og_data", exist_ok=True)
    os.makedirs("og_plots", exist_ok=True)

    model_params = {
        "L": L,
        "S": 0.5,
        "Jx": 1.0,
        "Jy": 1.0,
        "Jz": 0.0,
        "bc_MPS": "finite",
        "conserve": None,
    }
    model = SpinChain(model_params)

    # Initial MPS state
    sigmaz_init = profile_func(L)
    psi = create_mps_from_sigmaz_profile(
        model.lat.mps_sites(), model.lat.bc_MPS, sigmaz_init
    )

    # TEBD Engine
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

    # Time Evolution Loop (<sigma_z> = 2 * <S_z>)
    times = [0.0]
    sigmaz_profiles = [2.0 * psi.expectation_value("Sz")]

    total_iterations = int(t_final / (dt * tebd_params["N_steps"]))

    # ADDED TQDM PROGRESS BAR HERE
    for _ in tqdm(
        range(total_iterations),
        desc=f"Simulating ({output_txt})",
        unit="step",
    ):
        engine.run()
        times.append(engine.evolved_time)
        sigmaz_profiles.append(2.0 * psi.expectation_value("Sz"))

    times = np.array(times)
    sigmaz_profiles = np.array(sigmaz_profiles)

    # Save Data for ML
    ml_data_matrix = np.column_stack([times, sigmaz_profiles])
    site_headers = [f"site_{j}" for j in range(L)]
    header_str = "time " + " ".join(site_headers)

    data_path = os.path.join("og_data", output_txt)
    np.savetxt(
        data_path,
        ml_data_matrix,
        fmt="%.8e",
        header=header_str,
        comments="# ",
    )
    print(f"Dataset successfully saved to '{data_path}'.")

    # Plotting
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
        "Spatial Profile $\\langle \\sigma^z_j \\rangle$ at Fixed Times",
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
        "Time Dynamics $\\langle \\sigma^z_j \\rangle(t)$ at Selected Sites",
        fontsize=12,
    )
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join("og_plots", output_pdf)
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Plot saved to '{plot_path}'.\n")



# ==============================================================================
# 3. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Step Function Profile
    L = 50
    t_f = 20.0
    """
    simulate_xx_dynamics(
        L=L,
        t_final=t_f,
        profile_func=lambda L: step_profile(L, left_val=1.0, right_val=-1.0),
        output_pdf="xx_step_dynamics.pdf",
        output_txt="xx_step_sigmaz_data.txt",
    )
    
    # Gaussian Wavepacket Profile (amp=2.0 reaches +1.0 peak from -1.0 background)
    simulate_xx_dynamics(
        L=L,
        t_final=t_f,
        profile_func=lambda L: gaussian_profile(
            L, amp=2.0, sigma=10.0, center=L / 2
        ),
        output_pdf="xx_gaussian_dynamics.pdf",
        output_txt="xx_gaussian_sigmaz_data.txt",
    )
    
    # Sine Profile
    simulate_xx_dynamics(
        L=L,
        t_final=t_f,
        profile_func=lambda L: sine_profile(L, amp=1.0, k=4.0),
        output_pdf="xx_sine_dynamics.pdf",
        output_txt="xx_sine_sigmaz_data.txt",
    )
    """
    simulate_xx_dynamics(
                L=L,
                t_final=t_f,
                profile_func=lambda L: random_rough_uniform(L, seed=42),
                output_pdf="xx_random_discrete_dynamics.pdf",
                output_txt="xx_random_discrete_sigmaz_data.txt",
            )
    #Random smooth profile
    simulate_xx_dynamics(
            L=L,
            t_final=t_f,
            profile_func=lambda L: random_smooth_gaussian_filter(L, sigma=4, seed=500),
            output_pdf="xx_random_gauss_5_dynamics.pdf",
            output_txt="xx_random_gauss_5_sigmaz_data.txt",
        )
    simulate_xx_dynamics(
                L=L,
                t_final=t_f,
                profile_func=lambda L: random_smooth_gaussian_filter(L, sigma=4, seed=200),
                output_pdf="xx_random_gauss_2_dynamics.pdf",
                output_txt="xx_random_gauss_2_sigmaz_data.txt",
        )
    simulate_xx_dynamics(
                    L=L,
                    t_final=t_f,
                    profile_func=lambda L: random_smooth_gaussian_filter(L, sigma=4, seed=300),
                    output_pdf="xx_random_gauss_3_dynamics.pdf",
                    output_txt="xx_random_gauss_3_sigmaz_data.txt",
            )
    simulate_xx_dynamics(
                    L=L,
                    t_final=t_f,
                    profile_func=lambda L: random_smooth_gaussian_filter(L, sigma=4, seed=400),
                    output_pdf="xx_random_gauss_4_dynamics.pdf",
                    output_txt="xx_random_gauss_4_sigmaz_data.txt",
            )