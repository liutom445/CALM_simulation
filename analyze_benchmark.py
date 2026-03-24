import pandas as pd
import numpy as np

def analyze():
    df = pd.read_csv('llm_benchmark_results.csv')
    
    # 1. Error Metrics
    df['y_pred_observed'] = np.where(df['synthetic_t'] == 1, df['y1_pred'], df['y0_pred'])
    mae = np.abs(df['y_pred_observed'] - df['y_true_num']).mean()

    # 2. ATE Estimation
    true_ate = 0.0578
    ate_llm = (df['y1_pred'] - df['y0_pred']).mean()
    
    # 3. Efficiency Gain
    y_t = df[df['synthetic_t'] == 1]['y_true_num']
    y_c = df[df['synthetic_t'] == 0]['y_true_num']
    naive_se = np.sqrt(y_t.var()/len(y_t) + y_c.var()/len(y_c))
    
    residual_t = y_t - df[df['synthetic_t'] == 1]['y1_pred']
    residual_c = y_c - df[df['synthetic_t'] == 0]['y0_pred']
    calm_se = np.sqrt(residual_t.var()/len(y_t) + residual_c.var()/len(y_c))
    
    variance_reduction = 1 - (calm_se**2 / naive_se**2)

    print("=== NUMERIC LLM BENCHMARK ANALYSIS ===")
    print(f"MAE (Predicted vs True Continuous Y): {mae:.4f}")
    print(f"True ATE:      {true_ate:.4f}")
    print(f"LLM ATE:       {ate_llm:.4f}")
    print(f"LLM Bias:      {abs(ate_llm - true_ate):.4f}")
    print(f"Naive SE:      {naive_se:.4f}")
    print(f"CALM SE:       {calm_se:.4f}")
    print(f"Variance Red.: {variance_reduction:.2%}")

if __name__ == "__main__":
    analyze()
