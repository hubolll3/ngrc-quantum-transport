import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# ==============================================================================
# 1. ENHANCED LOCAL NG-RC FEATURES (EXTENDED SPATIAL RADIUS)
# ==============================================================================
def build_local_ngrc_features(X_window, k=4, s=2, boundary="obc", radius=2):
    """
    Builds local linear and quadratic features with extended spatial neighborhood (radius=2)
    to capture multi-site bond interactions required for local energy density (h_i) dynamics.
    """
    n_steps, L_sites = X_window.shape
    O_lin = np.column_stack([X_window[-1 - d * s] for d in range(k)])  # (L, k)
    
    feature_rows = []
    for j in range(L_sites):
        neighbors = []
        for r in range(-radius, radius + 1):
            if boundary == "obc":
                idx = max(0, min(L_sites - 1, j + r))
            else:
                idx = (j + r) % L_sites
            neighbors.append(idx)
            
        local_lin = O_lin[neighbors, :].ravel()
        
        # Non-linear quadratic combinations
        outer_quad = np.outer(local_lin, local_lin)
        triu_quad = outer_quad[np.triu_indices(len(local_lin))]
        
        # Normalized spatial position [-1, 1]
        pos_feat = (j - (L_sites - 1) / 2.0) / ((L_sites - 1) / 2.0)
        
        feature_rows.append([1.0, pos_feat] + list(local_lin) + list(triu_quad))
    return np.array(feature_rows)


# ==============================================================================
# 2. NG-RC RUNNER WITH STRICT CONSERVATION & DYNAMIC ADAPTATION
# ==============================================================================
def run_ngrc_boundary(data_name, t_short, t_start=1.0, plot=True):
    """
    Trains a boundary-stabilized NG-RC model with exact conservation projection
    and dataset-aware range/step handling.
    """
    output_dir_data = "ngrc_boundary_data"
    output_dir_plots = "ngrc_boundary_plots"
    
    if not os.path.exists(data_name):
        raise FileNotFoundError(f"File '{data_name}' not found!")

    base_name = os.path.basename(data_name)
    prefix = base_name.replace("_sigmaz_data.txt", "").replace("_data.txt", "")
    is_energy_dataset = "energy" in base_name

    # 1. Load raw data
    raw_data = np.loadtxt(data_name)
    times = raw_data[:, 0]
    sigmaz_full = raw_data[:, 1:]
    N_times, L = sigmaz_full.shape

    # 2. Preprocessing & Windowing
    train_indices = np.where((times >= t_start) & (times <= t_short))[0]
    if len(train_indices) == 0:
        raise ValueError(f"No time steps found between t_start={t_start} and t_short={t_short}")

    cutoff_idx = train_indices[-1]
    
    spatial_window = 1
    if spatial_window > 1:
        sigmaz_coarse = uniform_filter1d(sigmaz_full, size=spatial_window, axis=1, mode="nearest")
    else:
        sigmaz_coarse = np.copy(sigmaz_full)

    # 3. Model Hyperparameters & Dataset Adaptation
    k, s = 6, 2             # Delay memory k=4, stride s=2 (dt=0.05 -> delay step 0.1s)
    radius = 1              # Neighborhood radius (R=2 captures 5-site coupling)
    if is_energy_dataset: radius = 2
    alpha = 1e-1            # Ridge regularization
    jitter_std = 2e-2       # Feature noise for feedback stabilization
    warmup = (k - 1) * s

    # Adapt step limit based on physical quantity dynamics
    max_step_delta = 0.10 if is_energy_dataset else 0.03

    sigmaz_train = sigmaz_coarse[train_indices, :]
    N_train = len(train_indices)

    # Store total conserved sum at initial state (M_0 for spin, E_0 for energy)
    total_conserved_quantity = np.sum(sigmaz_train[0, :])

    X_feats_list, Y_targets_list = [], []
    for i in range(warmup, N_train - 1):
        window = sigmaz_train[i - warmup : i + 1]
        feat = build_local_ngrc_features(window, k=k, s=s, boundary="obc", radius=radius)
        
        delta_x = sigmaz_train[i + 1] - sigmaz_train[i]
        
        X_feats_list.append(feat)
        Y_targets_list.append(delta_x)

    X_mat = np.vstack(X_feats_list)
    Y_mat = np.concatenate(Y_targets_list)

    # Ridge Fit with Jitter Regularization
    np.random.seed(42)
    X_mat_noisy = X_mat + np.random.normal(0, jitter_std, size=X_mat.shape)
    n_feats = X_mat.shape[1]
    
    A = X_mat_noisy.T @ X_mat_noisy + alpha * np.eye(n_feats)
    B = X_mat_noisy.T @ Y_mat
    W_out = np.linalg.solve(A, B)

    # 4. Autoregressive Rollout with Exact Conservation Projection
    sigmaz_pred_full = np.copy(sigmaz_coarse)
    history_buffer = list(sigmaz_coarse[cutoff_idx - warmup : cutoff_idx + 1])

    for i in range(cutoff_idx, N_times - 1):
        curr_window = np.array(history_buffer[-(warmup + 1):])
        feat = build_local_ngrc_features(curr_window, k=k, s=s, boundary="obc", radius=radius)
        
        delta_x_raw = feat @ W_out
        
        # --- CONSERVATION STEP 1: Zero-mean projection on raw increments ---
        delta_x_raw -= np.mean(delta_x_raw)
        
        # Apply smooth tanh step-bounding
        delta_x_bounded = max_step_delta * np.tanh(delta_x_raw / max_step_delta)
        
        # --- CONSERVATION STEP 2: Re-zero mean after non-linear bounding ---
        delta_x_bounded -= np.mean(delta_x_bounded)
        
        x_next = curr_window[-1] + delta_x_bounded
        
        # --- CONSERVATION STEP 3: Global drift correction ---
        x_next -= (np.sum(x_next) - total_conserved_quantity) / L

        # Conditionally clip spin datasets ONLY; energy densities exceed [-1, 1]
        if not is_energy_dataset:
            x_next = np.clip(x_next, -1.0, 1.0)
            
        sigmaz_pred_full[i + 1] = x_next
        history_buffer.append(x_next)

    # Combine exact ground truth and forecast
    sigmaz_combined = np.copy(sigmaz_coarse)
    sigmaz_combined[cutoff_idx + 1 :, :] = sigmaz_pred_full[cutoff_idx + 1 :, :]

    # 5. Compute Extrapolation Error
    future_indices = np.where(times > t_short)[0]
    diff = np.abs(sigmaz_coarse - sigmaz_combined)
    mae_future = float(np.mean(diff[future_indices, :])) if len(future_indices) > 0 else 0.0

    # 6. Save Data Matrix
    os.makedirs(output_dir_data, exist_ok=True)
    data_matrix = np.column_stack([times, sigmaz_combined])
    site_headers = [f"site_{j}" for j in range(L)]
    header_str = "time " + " ".join(site_headers)
    output_txt_file = f"{output_dir_data}/{prefix}_{t_short:.1f}_ngrc_predicted_sigmaz.txt"
    np.savetxt(output_txt_file, data_matrix, fmt="%.8e", header=header_str, comments="# ")

    # 7. Plotting
    fig = None
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        title_quantity = r"\langle h_j(t) \rangle" if is_energy_dataset else r"\langle \sigma^z_j(t) \rangle"

        # Ground Truth
        im1 = axes[0].imshow(sigmaz_coarse.T, aspect="auto", extent=[times[0], times[-1], 0, L], origin="lower", cmap="coolwarm")
        axes[0].axvline(x=t_short, color="black", linestyle="--", linewidth=2, label=r"Forecast Start ($t_{\mathrm{short}}$)")
        axes[0].set_title(f"True TEBD Dynamics ${title_quantity}$")
        axes[0].set_xlabel("Time $t$")
        axes[0].set_ylabel("Site Index $j$")
        axes[0].legend(loc="upper right")
        fig.colorbar(im1, ax=axes[0])

        # Combined Forecast
        im2 = axes[1].imshow(sigmaz_combined.T, aspect="auto", extent=[times[0], times[-1], 0, L], origin="lower", cmap="coolwarm")
        axes[1].axvline(x=t_short, color="black", linestyle="--", linewidth=2, label=r"Forecast Start ($t_{\mathrm{short}}$)")
        axes[1].set_title(r"Exact ($t \leq t_{\mathrm{short}}$) + NG-RC ($t > t_{\mathrm{short}}$)")
        axes[1].set_xlabel("Time $t$")
        axes[1].set_ylabel("Site Index $j$")
        axes[1].legend(loc="upper right")
        fig.colorbar(im2, ax=axes[1])

        # Absolute Error
        im3 = axes[2].imshow(diff.T, aspect="auto", extent=[times[0], times[-1], 0, L], origin="lower", cmap="viridis", vmin=0, vmax=np.max(diff))
        axes[2].axvline(x=t_short, color="white", linestyle="--", linewidth=2, label=r"Forecast Start ($t_{\mathrm{short}}$)")
        axes[2].set_title(r"Absolute Difference $|\mathrm{True} - \mathrm{NG\text{-}RC}|$")
        axes[2].set_xlabel("Time $t$")
        axes[2].set_ylabel("Site Index $j$")
        axes[2].legend(loc="upper right")
        fig.colorbar(im3, ax=axes[2])

        dataset_type_label = "Energy" if is_energy_dataset else "Spin XXZ"
        fig.suptitle(f"Dataset: '{prefix}' ({dataset_type_label} Conserved NG-RC)\nExtrapolation MAE: {mae_future:.6f}", fontsize=13, fontweight="bold", y=1.04)
        plt.tight_layout()
        os.makedirs(output_dir_plots, exist_ok=True)
        output_pdf_file = f"{output_dir_plots}/{prefix}_{t_short:.1f}_ngrc_extrapolation_comparison.pdf"
        plt.savefig(output_pdf_file, format="pdf", bbox_inches="tight")
        plt.clf()
        plt.close()

    return sigmaz_combined, fig, mae_future


# ==============================================================================
# MULTI-DATASET BENCHMARKING EXECUTION
# ==============================================================================
if __name__ == "__main__":
    t_start = 1.0
    #t_short = 10.0
    t_shorts = [3.0, 5.0, 10.0, 20.0]
    
    datasets = [
        "og_data_xxz_2/xxz_delta_0p00_step_sigmaz_data.txt",
        "og_data_xxz_2/xxz_delta_0p50_step_sigmaz_data.txt",
        "og_data_xxz_2/xxz_delta_1p00_step_sigmaz_data.txt",
        "og_data_xxz_2/xxz_delta_1p50_step_sigmaz_data.txt",
        "og_data_xxz_2/xxz_delta_2p00_step_sigmaz_data.txt",
        "og_data_xxz_2/xxz_delta_2p50_step_sigmaz_data.txt"
    ]
    
    for path in datasets:
        for t_short in t_shorts:
            if os.path.exists(path):
                print(f"Processing: {path} with t_short={t_short}...")
                _,_, mae = run_ngrc_boundary(path, t_short=t_short, t_start=t_start, plot=True)
                print(f"Predicted with MAE: {mae}")
            else:
                print(f"Skipping {path} (file not found).")