import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from reproduce_similarity import load_data

def recover_text():
    # 1. Load Original Text (The "Phrase Bank")
    print("Loading original BRIGHTEN data to build the Phrase Bank...")
    original_df = load_data()
    
    # 2. Load Synthetic Data
    print("Loading synthetic population for text recovery...")
    df_syn = pd.read_csv('synthetic_population_with_outcomes.csv')
    
    # 3. Load PCA Models
    pca_models = joblib.load('text_pca_models.pkl')
    
    # Text columns were: ['reason_to_enroll', 'app_satisfaction_feedback']
    # We will recover each one separately
    
    for col in pca_models.keys():
        print(f"\nRecovering text for column: {col}...")
        
        # a. Get unique original texts and their PCA representations (The Search Space)
        phrase_bank = original_df[col].fillna("No response").astype(str).unique()
        
        # We need to project these phrases into the 32D PCA space
        # (Using the same method as in prepare_ctgan_data.py)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"  Encoding {len(phrase_bank)} unique phrases from bank...")
        phrase_embeddings = model.encode(phrase_bank.tolist(), show_progress_bar=True)
        
        pca = pca_models[col]
        phrase_pca = pca.transform(phrase_embeddings)
        
        # b. Build Nearest Neighbors model on the PCA space
        # We use k=1 as specified on page 23: "we perform top-1 nearest neighbor retrieval"
        nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn.fit(phrase_pca)
        
        # c. Get synthetic PCA vectors
        pca_cols = [c for c in df_syn.columns if c.startswith(f"{col}_pca_")]
        syn_pca_vecs = df_syn[pca_cols].values
        
        # d. Retrieve best matches
        print(f"  Finding best matches for 20,000 synthetic rows...")
        distances, indices = nn.kneighbors(syn_pca_vecs)
        
        # e. Map indices back to original text
        recovered_texts = [phrase_bank[idx[0]] for idx in indices]
        
        # Save recovered text to dataframe
        df_syn[f"{col}_recovered"] = recovered_texts

    # 4. Final Cleanup
    # Map Y_cat_syn back to labels for easier reading
    phq_labels = ["Minimal", "Mild", "Moderate", "Moderately severe", "Severe"]
    df_syn['phq9_label_recovered'] = df_syn['Y_cat_syn'].apply(lambda x: phq_labels[int(x)-1])

    # 5. Save Final Synthetic Superpopulation
    # We only keep a selection of columns for the final usable file
    # including basic covariates and the recovered text
    final_cols = ['T', 'age', 'gender_Male', 'gender_Female', 'education_Graduate Degree', 'education_University', 'education_High School',
                  'reason_to_enroll_recovered', 'app_satisfaction_feedback_recovered', 'Y_cat_syn', 'phq9_label_recovered']
    
    # (Just taking a subset of demographic ones for demonstration, 
    # but we'll keep all for the real file)
    output_path = 'final_synthetic_superpopulation.csv'
    df_syn.to_csv(output_path, index=False)
    print(f"\nSuccess! Final synthetic superpopulation saved to: {output_path}")
    print("Example rows:")
    print(df_syn[['reason_to_enroll_recovered', 'phq9_label_recovered']].head(5))

if __name__ == "__main__":
    recover_text()
