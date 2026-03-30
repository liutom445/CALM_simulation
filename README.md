# CALM Simulation

This repository implements a semi-synthetic benchmarking pipeline for PHQ-9 prediction in the BRIGHTEN study. It combines synthetic baseline-covariate generation, retrieval-based recovery of text responses, and initial LLM benchmarking under treated and control settings. The goal is to assess whether unstructured text can improve prediction relative to conventional structured covariates alone.

## Rationale

Our goal is to leverage unstructured text data in the BRIGHTEN study to improve the efficiency of prediction beyond traditional structured tabular covariates.

From the code in [reproduce_similarity.py](reproduce_similarity.py), we see that unstructured baseline text contains prognostic signal for later depression outcomes in addition to standard demographic and behavioral variables. This analysis is intended only as an **exploratory screening step**, not as a formal inferential procedure. The outcome used here is each participant’s **last observed PHQ-9 severity category**, which for most successfully followed participants corresponds to the **Week 12 endpoint**.

For unstructured text variables, similarity is quantified by cosine similarity on the embedding scale. For structured tabular covariates, we use simple numeric representations such as one-hot encoding for categorical variables and standardized values for continuous variables, then compare them to the outcome labels in the same exploratory spirit. These summaries are only meant to provide a rough descriptive sense of which variables appear most aligned with later depression outcomes.

Empirically, the unstructured variables are much more strongly aligned with depression outcomes than the standard structured covariates. In particular, the two text variables — **reason for enrollment** and **app satisfaction feedback** — show the strongest similarity scores, substantially exceeding the demographic variables.

| Data Group | Variable | Cosine Similarity | Relevance |
|---|---|---:|---|
| **Unstructured Text** | App Satisfaction Feedback | 0.5708 | Highest |
| **Unstructured Text** | Reason for Enrollment | 0.4907 | Very High |
| Structured (Behavioral) | Device Type | 0.2352 | Moderate |
| Structured (Demographic) | Working Status | 0.2144 | Low |
| Structured (Demographic) | Sex | 0.2086 | Low |
| Structured (Demographic) | Marital Status | 0.1896 | Low |
| Structured (Demographic) | Education Level | 0.1524 | Low |
| Structured (Behavioral) | Heard About Us | 0.1229 | Low |
| Structured (Demographic) | Race | 0.1093 | Low |
| Structured (Demographic) | Age | 0.0148 | Minimal |

This motivates the main question of our simulation study:

> Can LLM-based representations exploit the rich, high-signal unstructured data in a way that improves efficiency relative to analyses based only on conventional structured covariates?

In other words, if the text-derived representations carry information about the outcome that is only weakly reflected in the structured covariates, then methods that use these representations should have the potential to recover prognostic structure that standard tabular approaches miss.

## Data Preprocessing

Following the study's design, the preprocessing of the original BRIGHTEN covariates is implemented in [prepare_ctgan_data.py](prepare_ctgan_data.py). Structured numeric covariates are median-imputed and standardized, while categorical covariates are converted into one-hot encoded indicator variables. To handle unstructured survey responses, the study maps missing entries to the string “No response” and generates dense text embeddings via the pretrained [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) transformer. This pipeline represents variable-length text by fixed-dimensional embeddings using a standard machine learning normalization step, making the resulting features comparable across observations.

One notable preprocessing step is that the 384-dimensional dense embeddings are compressed to 32 dimensions using PCA. Using the prepared numerical matrix, we then train a [Conditional Generative Adversarial Network](https://github.com/sdv-dev/CTGAN) via [train_ctgan.py](train_ctgan.py). **CTGAN generates synthetic baseline covariates only; it does not generate outcomes.**

To generate realistic outcomes for our synthetic population, we use the original data to learn the relationship between covariates and clinical outcomes.

- **Modeling:** Using `fit_causal_forests.R`, we fit Generalized Random Forests ([`grf`](https://grf-labs.github.io/grf/)) to the original data to estimate the conditional mean `mu_t(W)` and conditional variance `sigma_t^2(W)` under both treatment and control.
- **Synthesis:** In `generate_synthetic_outcomes.R`, we apply these fitted models to 20,000 synthetic participants to generate potential outcomes `Y(0)` and `Y(1)`. In other words, outcomes are simulated from the fitted conditional mean and variance models evaluated on the synthetic covariates, and treatment is then assigned independently as `T ~ Bernoulli(0.5)`.

The synthetic participants initially exist only as 32-dimensional PCA vectors. To recover natural language, `recover_text.py` uses a **top-1 nearest-neighbor retrieval** procedure. We first construct a phrase bank from the original BRIGHTEN responses and project those responses into the same PCA space. For each synthetic row, we then retrieve the original text whose embedding is closest to the synthetic PCA vector in cosine distance. **Recovered text is nearest-neighbor retrieval from original responses, not newly generated text.**

## Initial Benchmarking With [GPT 5.4 Mini](https://openai.com/index/introducing-gpt-5-4-mini-and-nano/)

We first conduct an initial benchmark using the LLM as a **direct outcome predictor** on a random benchmark sample of **N=200** participants drawn from the final synthetic superpopulation. For each sampled participant, the model is given baseline covariates, recovered enrollment text, and a specified treatment assignment, together with a small few-shot set of labeled examples, and is then asked to predict the participant’s follow-up PHQ-9 severity category. This is done twice per participant, once under active treatment and once under control, so that predicted potential outcomes can be formed and compared.

The prediction target is the participant’s **follow-up PHQ-9 severity category** under the specified treatment condition. The initial benchmark is evaluated primarily by **prediction performance on this ordinal outcome**, with downstream comparison of predicted treated and control outcomes to assess how well the benchmark recovers potential-outcome structure.

Drawing from existing literature such as [TabLLM](https://github.com/clinicalml/TabLLM), [Unipredict](https://arxiv.org/pdf/2310.03266), and another relevant [survey](https://github.com/tanfiona/LLM-on-Tabular-Data-Prediction-Table-Understanding-Data-Generation?tab=readme-ov-file), we use the following prompting strategy for LLMs in tabular data: each participant’s baseline covariates are rendered into a short natural-language profile, combined with a task instruction and a small set of labeled examples, and passed to an LLM for ordinal outcome prediction.
