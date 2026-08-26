import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import savgol_filter

# ==============================================================================
# 1. HELPER: DELTA EXTRACTION
# ==============================================================================

def extract_delta_from_filename(filename: str) -> float:
    """
    Parses Delta anisotropy value from filenames (e.g. 'delta_1p50' -> 1.5).
    Returns None if pattern is not found.
    """
    match = re.search(r"delta_(\d+[p\.]\d+)", filename, re.IGNORECASE)
    if match:
        raw_str = match.group(1).replace("p", ".")
        return float(raw_str)
    return None

# ==============================================================================
# 2. DIFFUSION CONSTANT EXTRACTION
# ==============================================================================

def compute_spatial_msd_magnetization(sigmaz_matrix: np.ndarray) -> np.ndarray:
    """Computes MSD using direct excess magnetization density."""
    N_times, L = sigmaz_matrix.shape
    sites = np.arange(L)
    msd = np.zeros(N_times)

    for t_idx in range(N_times):
        profile = sigmaz_matrix[t_idx, :]
        profile_shifted = profile - np.min(profile)
        total_mag = np.sum(profile_shifted)

        if total_mag < 1e-12:
            P = np.ones(L) / L
        else:
            P = profile_shifted / total_mag

        mu = np.sum(sites * P)
        msd[t_idx] = np.sum(((sites - mu) ** 2) * P)

    return msd


def extract_diffusion_constant(
    times: np.ndarray, 
    sigmaz_matrix: np.ndarray, 
    t_min: float = 5.0, 
    t_max: float = None,
    smooth_window: int = 11,   # Smoothing window length (must be odd integer)
    poly_order: int = 2        # Polynomial order for Savitzky-Golay filter
):
    """
    Extracts diffusion constant D by fitting MSD(t) = 2*D*t + C over [t_min, t_max].
    """
    msd = compute_spatial_msd_magnetization(sigmaz_matrix)

    if smooth_window is not None and smooth_window > 3:
        w_len = min(smooth_window, len(msd))
        if w_len % 2 == 0:
            w_len -= 1
        if w_len > poly_order:
            msd = savgol_filter(msd, window_length=w_len, polyorder=poly_order)

    if t_max is None:
        t_max = times[-1]

    eval_indices = np.where((times >= t_min) & (times <= t_max))[0]
    
    if len(eval_indices) < 2:
        raise ValueError(f"Insufficient time steps in window [{t_min}, {t_max}].")

    t_eval = times[eval_indices]
    msd_eval = msd[eval_indices]

    slope, intercept, _, _, _ = linregress(t_eval, msd_eval)
    D = slope / 2.0
    fit_line = slope * t_eval + intercept

    return D, msd, fit_line, eval_indices

# ==============================================================================
# 3. COMPARISON PIPELINE
# ==============================================================================

def compare_diffusion_dynamics(
    true_dir: str = "og_data_2",
    pred_dir: str = "ngrc_boundary_data",
    output_dir: str = "diffusion_results",
    t_short: float = 3.0,
    t_min: float = None,
    t_max: float = None,
    min_delta: float = 1.0  # Only process files with Delta > min_delta
):
    """
    Loads simulated true dynamics and predicted dynamics for Delta > min_delta,
    calculates D for true (full window), predicted NG-RC, and baseline (t < t_short),
    and outputs comparison plots and a summary table.
    """
    os.makedirs(output_dir, exist_ok=True)

    t_min_base = t_min if (t_min is not None and t_min < t_short) else 1.0

    if t_min is None:
        t_min = t_short

    suffix = f"_{t_short:.1f}_ngrc_predicted_sigmaz.txt"
    pred_pattern = os.path.join(pred_dir, f"*{suffix}")
    pred_files = sorted(glob.glob(pred_pattern))

    if not pred_files:
        suffix = f"_{t_short}_ngrc_predicted_sigmaz.txt"
        pred_pattern = os.path.join(pred_dir, f"*{suffix}")
        pred_files = sorted(glob.glob(pred_pattern))

    if not pred_files:
        raise FileNotFoundError(f"No predicted files matching pattern '{pred_pattern}' were found in '{pred_dir}'!")

    results = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        
        # 1. Parse Delta from filename
        delta_val = extract_delta_from_filename(filename)
        
        if delta_val is None:
            print(f"Warning: Could not parse Delta from '{filename}', skipping.")
            continue

        # 2. Filter: Only use files with Delta > min_delta (Diffusive regime)
        if delta_val <= min_delta:
            print(f"Skipping '{filename}': Delta = {delta_val:.2f} <= {min_delta} (non-diffusive / superdiffusive regime).")
            continue

        base_prefix = filename.replace(suffix, "")
        true_path = os.path.join(true_dir, f"{base_prefix}_sigmaz_data.txt")
        
        if not os.path.exists(true_path):
            print(f"Warning: Skipping '{filename}', true data missing at '{true_path}'")
            continue

        raw_true = np.loadtxt(true_path)
        raw_pred = np.loadtxt(pred_path)

        times_true, sigmaz_true = raw_true[:, 0], raw_true[:, 1:]
        times_pred, sigmaz_pred = raw_pred[:, 0], raw_pred[:, 1:]

        # Extract D values
        D_true, msd_true, _, _ = extract_diffusion_constant(times_true, sigmaz_true, t_min=t_min, t_max=t_max)
        D_pred, msd_pred, _, _ = extract_diffusion_constant(times_pred, sigmaz_pred, t_min=t_min, t_max=t_max)
        D_base, _, _, _ = extract_diffusion_constant(times_true, sigmaz_true, t_min=t_min_base, t_max=t_short)

        rel_error_pred = np.abs(D_true - D_pred) / np.abs(D_true) * 100.0 if D_true != 0 else 0.0
        rel_error_base = np.abs(D_true - D_base) / np.abs(D_true) * 100.0 if D_true != 0 else 0.0

        label = base_prefix.replace("xx_x_z_", "").replace("_sigmaz", "")
        results.append({
            "label": label,
            "delta": delta_val,
            "D_true": D_true,
            "D_base": D_base,
            "D_pred": D_pred,
            "rel_error_pred": rel_error_pred,
            "rel_error_base": rel_error_base
        })

        # Plot MSD Curves
        mask_true = times_true[:] < (t_max if t_max is not None else times_true[-1])
        mask_pred = times_pred[:] < (t_max if t_max is not None else times_pred[-1])
        axes[0].plot(times_true[mask_true], msd_true[mask_true], label=f"$\Delta={delta_val:.2f}$ (True)", linestyle="-")
        axes[0].plot(times_pred[mask_pred], msd_pred[mask_pred], label=f"$\Delta={delta_val:.2f}$ (NG-RC)", linestyle="--")

    # Formatting - MSD Dynamics
    axes[0].axvline(x=t_short, color="red", linestyle="--", label=f"$t_{{short}}$ ({t_short})")
    axes[0].set_xlabel("Time $t$", fontsize=12)
    axes[0].set_ylabel("Spatial Variance $\\sigma^2(t)$ (MSD)", fontsize=12)
    axes[0].set_title(f"Diffusive Spreading ($\Delta > {min_delta}$, $t_{{short}} = {t_short}$)", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=8, loc="upper left")

    # Formatting - Bar Chart Comparison
    if results:
        # Sort results by Delta value for clean plotting
        results = sorted(results, key=lambda r: r["delta"])

        x_labels = [f"$\Delta={r['delta']:.2f}$\n({r['label']})" for r in results]
        d_trues = [r["D_true"] for r in results]
        d_bases = [r["D_base"] for r in results]
        d_preds = [r["D_pred"] for r in results]

        x = np.arange(len(x_labels))
        width = 0.25

        axes[1].bar(x - width, d_trues, width, label="True TEBD (Full)", color="navy", alpha=0.85)
        axes[1].bar(x, d_bases, width, label=f"Baseline ($t < t_{{short}}$)", color="slategray", alpha=0.85)
        axes[1].bar(x + width, d_preds, width, label="Predicted NG-RC", color="crimson", alpha=0.85)

        axes[1].set_ylabel("Diffusion Constant $D$", fontsize=12)
        axes[1].set_title("Diffusion Constant $D$ Comparison", fontsize=12)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(x_labels, rotation=0, ha="center")
        axes[1].grid(True, linestyle="--", alpha=0.6, axis="y")
        axes[1].legend(fontsize=9)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f"diffusion_comparison_t{t_short}.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.clf()
    plt.close()

    # Print Summary Table
    print("\n" + "=" * 105)
    print(f"DIFFUSIVE REGIME RESULTS (Delta > {min_delta}) FOR t_short = {t_short}")
    print("=" * 105)
    print(f"{'Initial Condition':<18} | {'Delta':<6} | {'D (True)':<10} | {f'D (t < {t_short})':<12} | {'D (NG-RC)':<10} | {'Err Base (%)':<12} | {'Err NG-RC (%)':<12}")
    print("=" * 105)
    for r in results:
        print(f"{r['label']:<18} | {r['delta']:<6.2f} | {r['D_true']:<10.6f} | {r['D_base']:<12.6f} | {r['D_pred']:<10.6f} | {r['rel_error_base']:<12.2f}% | {r['rel_error_pred']:<12.2f}%")
    print("=" * 105)
    print(f"Summary plot saved to '{pdf_path}'.\n")


if __name__ == "__main__":
    for t_short in [5.0]:
        compare_diffusion_dynamics(
            true_dir="og_data_xxz",
            pred_dir="ngrc_boundary_data",
            output_dir="diffusion_results_xxz",
            t_short=t_short,
            t_min=1.0, 
            t_max=30.0,
            min_delta=1.0  # Only includes Delta > 1.0
        )