"""
synthetic_generation.py
-----------------------
Generates synthetic survival datasets to study the dependency between
event times (T) and censoring times (C) under controlled conditions.
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import rankdata, kendalltau
from copulas.bivariate import Clayton


# --- AFT Weibull generation functions --------------------------------------

def generate_T(u, X, beta_T, rho_T):
    lam_T = np.exp(X @ beta_T)
    return (-np.log(u) / lam_T)**(1.0 / rho_T)

def generate_C(v, X, beta_C, rho_C, s=1.0):
    lam_C = np.exp(X @ beta_C)
    return (-np.log(v) / (s * lam_C))**(1.0 / rho_C)


# --- Calibration of censoring rate -----------------------------------------

def calibrate_censoring_scale(v, X, beta_C, rho_C, T, target=0.50, tol=0.02,
                              max_iter=40, s_min=1e-6, s_max=1e6):
    s_lo, s_hi = s_min, s_max
    for _ in range(max_iter):
        s_mid = np.sqrt(s_lo * s_hi)
        C_mid = generate_C(v, X, beta_C, rho_C, s_mid)
        censor_rate = np.mean(T > C_mid)
        if abs(censor_rate - target) <= tol:
            return s_mid, censor_rate
        if censor_rate > target:
            s_hi = s_mid
        else:
            s_lo = s_mid
    return s_mid, np.mean(T > generate_C(v, X, beta_C, rho_C, s_mid))


# --- Main generation routine ----------------------------------------------

def generate_dataset(n, d, beta_T, rho_T, beta_C, rho_C,
                     target_censor=0.50, tol=0.02,
                     output_dir="data/synthetic", trial_idx=1):
    os.makedirs(output_dir, exist_ok=True)
    X = np.random.rand(n, d)
    u = np.random.rand(n)
    v = np.random.rand(n)

    T = generate_T(u, X, beta_T, rho_T)
    s_star, _ = calibrate_censoring_scale(v, X, beta_C, rho_C, T, target=target_censor, tol=tol)
    C = generate_C(v, X, beta_C, rho_C, s_star)

    Y = np.minimum(T, C)
    delta = (T <= C).astype(int)

    u_rank, v_rank = rankdata(T)/(n+1), rankdata(C)/(n+1)
    cop = Clayton()
    cop.fit(np.column_stack([u_rank, v_rank]))
    tau_emp, _ = kendalltau(T, C)

    df_full = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(d)])
    df_full["T"], df_full["C"], df_full["time"], df_full["event"] = T, C, Y, delta
    df_full["indice_indiv"] = np.arange(n)
    df_full.to_csv(f"{output_dir}/df_T_C_trial{trial_idx}.csv", index=False)

    df_for_simul = df_full[["indice_indiv"] + [f"x{i+1}" for i in range(d)] + ["time", "event"]]
    df_for_simul.to_csv(f"{output_dir}/df_for_simul_trial{trial_idx}.csv", index=False)

    print(f"[OK] Trial {trial_idx} | tau_emp={tau_emp:.3f} | Censoring={np.mean(T>C)*100:.2f}%")
    return df_full


# --- Entry point -----------------------------------------------------------

if __name__ == "__main__":
    n, d = 10_000, 2
    beta_T = np.array([3.0, 3.0])
    beta_C = np.array([3.0, 3.0])
    rho_T, rho_C = 10.0, 10.0

    for i in range(1, 7):
        generate_dataset(n, d, beta_T, rho_T, beta_C, rho_C, trial_idx=i)
        beta_T *= 1.5
        beta_C *= 1.5
