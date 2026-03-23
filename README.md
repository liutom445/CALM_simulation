## Rationale

Our goal is to leverage unstructured text data in the BRIGHTEN study to improve the efficiency of prediction beyond traditional structured tabular covariates.

From the code in [reproduce_similarity.py](reproduce_similarity.py), we see that unstructured baseline text contains prognostic signal for later depression outcomes in addition to standard demographic and behavioral variables. In particular, for each participant, we represent an unstructured text response `u` by a Sentence-Transformer embedding `v_u in R^384`, and we represent the PHQ-9 outcome category `o` by its corresponding embedding `v_o in R^384`. Our code uses the final assessment available for each person, which for the majority of successful participants is the Week 12 endpoint.

For unstructured text variables, similarity is quantified by cosine similarity on the embedding scale:

`sim(u, o) = (v_u^T v_o) / (||v_u|| ||v_o||)`

In the current implementation, this is computed at the variable level by taking the mean embedding across unique responses and comparing it to the mean embedding across unique outcome labels.

For structured tabular covariates, the representation is different:
- categorical variables are mapped to one-hot vectors;
- continuous variables are centered and scaled.

Thus, if a structured covariate is represented by `v_s` and the outcome representation by `v_o`, the same generic cosine form is used:

`sim(s, o) = (v_s^T v_o) / (||v_s|| ||v_o||)`

but now `v_s` comes from one-hot coding or standardized numeric values rather than a language-model embedding.

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

In other words, if the text-derived vectors `v_u` carry information about the outcome that is only weakly reflected in the structured covariates, then methods that use these embeddings should have the potential to recover prognostic structure that standard tabular approaches miss.

## Data Preprocessing

Following the study's design, the preprocessing of the original BRIGHTEN covariates is implemented in [prepare_ctgan_data.py](prepare_ctgan_data.py). Structured numeric covariates are median-imputed and standardized, while categorical covariates are converted into one-hot encoded indicator variables. To handle unstructured survey responses, the study maps missing entries to the string “No response” and generates dense text embeddings via the pretrained [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) transformer. This pipeline represents variable-length text by fixed-dimensional embeddings using a standard machine learning normalization step, making the resulting features comparable across observations. One notable preprocessing step is that, the 384-dimensional dense embeddings is compress to 32 dimensions using PCA. 

Using the Generated data, we further fit causal forests to learn the conditional mean and variance using [rgf](https://cran.r-project.org/web/packages/RGF/index.html).  


