import pandas as pd
import numpy as np
import pickle
import os

try:
    from ctgan import CTGAN
except ImportError:
    print("Error: The 'ctgan' library is not installed.")
    print("Please install it by running: pip install ctgan")
    exit(1)

def train_and_generate():
    input_file = 'ctgan_training_table.csv'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run prepare_ctgan_data.py first.")
        return

    print("Loading prepared training table...")
    df = pd.read_csv(input_file)

    # 1. Handle any missing values in the outcome (CTGAN needs clean data)
    if df.isnull().values.any():
        print(f"Dropping {df.isnull().any(axis=1).sum()} rows with missing values...")
        df = df.dropna().reset_index(drop=True)

    # 2. Identify Discrete vs Continuous Columns
    # Age and all PCA components are continuous. 
    # Everything else (One-hot demographic columns, Treatment 'T', Outcome 'Y_cat') is discrete.
    continuous_cols = ['age'] + [c for c in df.columns if '_pca_' in c]
    discrete_cols = [c for c in df.columns if c not in continuous_cols]

    print(f"Identified {len(continuous_cols)} continuous columns.")
    print(f"Identified {len(discrete_cols)} discrete columns.")

    # 3. Initialize CTGAN Synthesizer
    # The paper uses default hyperparameters, but we set epochs=100 for a solid baseline
    # You can increase epochs to 300 for higher fidelity if you have more compute time.
    epochs = 100
    print(f"\nInitializing CTGAN (Epochs: {epochs})...")
    ctgan = CTGAN(epochs=epochs, verbose=True)

    # 4. Train the Model
    print("Training the generative model. This may take a few minutes...")
    ctgan.fit(df, discrete_cols)

    # Save the trained generator (G)
    model_path = 'ctgan_generator.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(ctgan, f)
    print(f"\nSaved trained generator to: {model_path}")

    # 5. Generate Synthetic Superpopulation (N = 20,000)
    # As per Section 5.2: "we generate the synthetic population of sample size N = 20,000"
    N = 20000
    print(f"Generating {N} synthetic participants...")
    synthetic_data = ctgan.sample(N)

    # Save the synthetic data
    output_path = 'synthetic_population.csv'
    synthetic_data.to_csv(output_path, index=False)
    print(f"Success! Synthetic superpopulation saved to: {output_path}")

if __name__ == "__main__":
    train_and_generate()
