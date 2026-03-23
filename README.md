# CALM_simulation
Simulation studies of the paper https://arxiv.org/abs/2510.05545

## Rationale

Our goal is to leverage unstructured text data in BRIGHTEN study to increase the efficiency of tabular, structured data. From the code [reproduce_similarity.py](reproduce_similarity.py),  we can see that unstructured data contains prognostic "signal" for depression outcomes in addition to traditional structured demographics. Specifically,  We take the raw text (e.g., "I want to improve my mood") and turn it into a 384-dimensional vector using a Sentence-Transformer (denoted as $v_u$ ), as well as the outcome (categorized as ordinal, post-transformation vector denoted as $v_o$). 

The reason to enroll and app satisfaction feedback scores showed a max correlation of 0.16–0.21. 

### Pre-Treatment Covariates vs. PHQ-9 Outcomes (Cosine Similarity)

| Data Group | Variable | Cosine Similarity | Relevance |
|---|---|---:|---|
| **Unstructured Text** | App Satisfaction Feedback | 0.5708 | Highest |
| **Unstructured Text** | Reason for Enrollment | 0.4907 | Very High  |
| Structured (Behavioral) | Device Type | 0.2352 | Moderate |
| Structured (Demographic) | Working Status | 0.2144 | Low |
| Structured (Demographic) | Sex | 0.2086 | Low |
| Structured (Demographic) | Marital Status | 0.1896 | Low |
| Structured (Demographic) | Education Level | 0.1524 | Low |
| Structured (Behavioral) | Heard About Us | 0.1229 | Low |
| Structured (Demographic) | Race | 0.1093 | Low |
| Structured (Demographic) | Age | 0.0148 | Minimal |


This motivates us to further study: can LLMs leverage this rich, high-signal unstructured data
