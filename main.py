import os
import pandas as pd
from simulate_censoring_pipeline import process_and_save_censored_datasets
from synthetic_generation import generate_dataset  # 🔹 import du module de génération synthétique

# === 1) PATH SETUP =========================================================
RESULTS_DIR = "results"
DATA_DIR = "data"
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

# === 2) LOAD REAL DATASETS =================================================
datasets = {
    "pbc": (pd.read_csv(os.path.join(DATA_DIR, "pbc_finale.csv")), os.path.join(RESULTS_DIR, "pbc")),
    "metabric": (pd.read_csv(os.path.join(DATA_DIR, "metabric_finale.csv")), os.path.join(RESULTS_DIR, "metabric")),
}

# Create subdirectories for each dataset
for _, (_, path) in datasets.items():
    os.makedirs(path, exist_ok=True)

# === 3) OPTIONAL: GENERATE SYNTHETIC DATASETS ==============================
def generate_synthetic_trials():
    """
    Generate six synthetic datasets to analyze the dependency between
    event times (T) and censoring times (C) under controlled conditions.
    """
    n, d = 10_000, 2
    beta_T = np.array([3.0, 3.0])
    beta_C = np.array([3.0, 3.0])
    rho_T, rho_C = 10.0, 10.0

    print("Generating synthetic datasets...")
    for i in range(1, 7):
        generate_dataset(
            n=n, d=d,
            beta_T=beta_T, rho_T=rho_T,
            beta_C=beta_C, rho_C=rho_C,
            output_dir=SYNTHETIC_DIR,
            trial_idx=i
        )
        # Gradually increase dependency strength
        beta_T *= 1.5
        beta_C *= 1.5
    print(" Synthetic datasets saved under:", SYNTHETIC_DIR)


# === 4) MAIN EXECUTION =====================================================
if __name__ == "__main__":
    # --- Step 1: Run censoring simulation on real datasets
    process_and_save_censored_datasets(
        datasets,
        desired_censor_rates=[0.1, 0.3, 0.5, 0.7, 0.9],
        random_state=42
    )

    # --- Step 2 (Optional): Generate synthetic datasets for dependency study
    from synthetic_generation import np  # import np if synthetic_generation not imported globally
    generate_synthetic_trials()
