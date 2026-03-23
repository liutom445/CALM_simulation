import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import joblib
import os

def prepare_unified_training_table(df):
    """
    Prepares the BRIGHTEN dataset for CTGAN by processing structured data
    and compressing unstructured text embeddings to 32 dimensions via PCA.
    """
    print(f"Total participants for processing: {len(df)}")
    
    # ---------------------------------------------------------
    # 1. Define Column Types (Adjusted to match your actual columns)
    # ---------------------------------------------------------
    numeric_cols = ['age']
    # 'sex' renamed to 'gender' based on reproduce_similarity.py's columns
    categorical_cols = ['gender', 'education', 'working', 'marital_status', 'race', 'device', 'heard_about_us']
    # Text columns renamed to match reproduce_similarity.py's columns
    text_cols = ['reason_to_enroll', 'app_satisfaction_feedback']
    
    processed_features = []
    feature_names = []

    # ---------------------------------------------------------
    # 2. Process Structured Numeric Data
    #    "Median-imputed and standardized"
    # ---------------------------------------------------------
    print("Processing numeric covariates...")
    if numeric_cols:
        num_imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        # Ensure numeric columns are actually numeric
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        num_data = num_imputer.fit_transform(df[numeric_cols])
        num_data = scaler.fit_transform(num_data)
        
        processed_features.append(num_data)
        feature_names.extend(numeric_cols)

    # ---------------------------------------------------------
    # 3. Process Structured Categorical Data
    #    "Categorical covariates are one-hot encoded"
    # ---------------------------------------------------------
    print("Processing categorical covariates...")
    if categorical_cols:
        # Fill missing categorical with a placeholder or mode before encoding
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_data = cat_imputer.fit_transform(df[categorical_cols].astype(str))
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(cat_data)
        
        processed_features.append(cat_encoded)
        feature_names.extend(encoder.get_feature_names_out(categorical_cols))

    # ---------------------------------------------------------
    # 4. Process Unstructured Text Data
    #    "Embedded using all-MiniLM-L6-v2, then compressed via PCA to 32 components"
    # ---------------------------------------------------------
    print("Loading Sentence Transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # We will save the PCA models because we need them later to 
    # project generated/paraphrased text back into the same 32D space!
    pca_models = {}

    for col in text_cols:
        print(f"Embedding and compressing text column: {col}...")
        
        # "Missing entries are mapped to the string 'No response'"
        texts = df[col].fillna("No response").astype(str).tolist()
        
        # Get 384-dimensional dense embeddings
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Compress to 32 dimensions using PCA
        pca = PCA(n_components=32, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)
        
        # Save the PCA model for Phase 3 (Text Recovery)
        pca_models[col] = pca
        
        processed_features.append(embeddings_pca)
        feature_names.extend([f"{col}_pca_{i}" for i in range(32)])

    # ---------------------------------------------------------
    # 5. Concatenate into a Unified Training Table
    # ---------------------------------------------------------
    print("Concatenating into a unified training table...")
    final_matrix = np.hstack(processed_features)
    training_table = pd.DataFrame(final_matrix, columns=feature_names)
    
    # Append the outcome variable (for Causal Forest later) and treatment assignment
    if 'study_arm' in df.columns:
        # Mapping iPST to 1, others to 0 based on study description
        training_table['T'] = df['study_arm'].apply(lambda x: 1 if x == 'iPST' else 0)
    
    if 'outcome_rank' in df.columns:
        # reproduce_similarity.py already has outcome_rank (1 to 5)
        training_table['Y_cat'] = df['outcome_rank']
    elif 'sum_phq9' in df.columns:
        # Categorize into 5 levels and numerically encode (1 to 5)
        def categorize_phq9(score):
            if pd.isna(score): return np.nan
            if score <= 4: return 1
            if score <= 9: return 2
            if score <= 14: return 3
            if score <= 19: return 4
            return 5
        training_table['Y_cat'] = df['sum_phq9'].apply(categorize_phq9)

    # Save the PCA models to disk for later use
    joblib.dump(pca_models, 'text_pca_models.pkl')
    
    return training_table

if __name__ == "__main__":
    # 1. Import your existing data loading logic from your other script
    from reproduce_similarity import load_data
    
    # 2. Load the original dataset
    print("Loading original BRIGHTEN data...")
    data = load_data() 
    
    # 3. Process the dataset using our new function
    training_table = prepare_unified_training_table(data)
    
    # 4. Save the fully processed, numerical matrix to a CSV for the CTGAN
    output_path = 'ctgan_training_table.csv'
    training_table.to_csv(output_path, index=False)
    
    print(f"Success! Saved unified training table with shape: {training_table.shape}")
    print(f"Saved to: {output_path}")
    print("PCA models saved to: text_pca_models.pkl")
