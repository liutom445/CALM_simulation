# Step 2.2: Fit Causal Forests (using grf) to learn conditional means and variances
library(grf)

# 1. Load the embedded BRIGHTEN dataset
df <- read.csv("ctgan_training_table.csv")
print(paste("Loaded data with", nrow(df), "participants and", ncol(df), "columns."))

# 2. Extract Features (W), Treatment (T), and Outcome (Y)
# Features are all columns except T and Y_cat
W_cols <- setdiff(names(df), c("T", "Y_cat"))
W <- as.matrix(df[, W_cols])
T_vec <- df$T
Y <- df$Y_cat

# 3. Fit Models for Treatment (T=1) and Control (T=0)
# We need to estimate mu_t(W) and sigma_t^2(W) for t in {0, 1}
# The paper suggests: mu_t(W) = E(Y | W, T=t) and sigma_t^2(W) = E((Y - mu_t(W))^2)

groups <- list(control = 0, treatment = 1)
mu_models <- list()
sigma_models <- list()

for (name in names(groups)) {
  t_val <- groups[[name]]
  idx <- which(T_vec == t_val)
  
  print(paste("Fitting models for", name, "group (n =", length(idx), ")..."))
  
  # a. Estimate mu_t(W) using regression_forest
  # Using default hyperparameters as specified in the paper
  mu_forest <- regression_forest(W[idx, ], Y[idx])
  mu_models[[name]] <- mu_forest
  
  # b. Estimate sigma_t^2(W)
  # First, calculate residuals from the training data
  mu_hat <- predict(mu_forest)$predictions
  residuals_sq <- (Y[idx] - mu_hat)^2
  
  # Then fit another forest to the squared residuals
  sigma_sq_forest <- regression_forest(W[idx, ], residuals_sq)
  sigma_models[[name]] <- sigma_sq_forest
}

# 4. Save Models
saveRDS(mu_models, "mu_t_models.rds")
saveRDS(sigma_models, "sigma_t_models.rds")
print("Causal forest models saved: mu_t_models.rds, sigma_t_models.rds")

# 5. Diagnostic: Check Average Treatment Effect (ATE) prediction vs original data
# ATE in original data (simplified)
ate_orig <- mean(Y[T_vec == 1]) - mean(Y[T_vec == 0])
print(paste("Observed ATE in training data:", round(ate_orig, 4)))

# Check average predicted outcomes
avg_mu_1 <- mean(predict(mu_models$treatment, W)$predictions)
avg_mu_0 <- mean(predict(mu_models$control, W)$predictions)
print(paste("Predicted ATE over all participants:", round(avg_mu_1 - avg_mu_0, 4)))
