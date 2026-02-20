# 🫀 Heart Disease Prediction — End-to-End ML Pipeline

## 📌 Project Overview

This project aims to build a production-grade machine learning system to predict the presence of heart disease using structured clinical data. The dataset consists of **630,000 patient records** with 15 clinical attributes including demographic information, stress-test results, imaging findings, and laboratory values.

Rather than treating this as a generic classification task, the problem was approached as a **medical risk prediction system**, where:

* False negatives are clinically dangerous.
* Probability calibration matters.
* Threshold selection is more important than raw accuracy.
* Model decisions must be explainable.

The goal was not simply to maximize ROC-AUC, but to understand **where predictive signal comes from**, how stable it is, and how models behave under medical constraints.

---

# 📊 Phase 1 — Exploratory Data Analysis (EDA)

The first phase focused on understanding the structure, quality, and signal strength of the dataset.

## Data Integrity Checks

* No missing values.
* No duplicate records.
* All features within plausible medical ranges.
* Class distribution: ~44.8% positive (mild imbalance).

A physiological sanity check was also performed:

> Max HR was compared against (220 − Age) to detect implausible values.

Approximately 26% exceeded the theoretical estimate, but remained within plausible stress-test ranges. No removal was performed at this stage.

---

## Numeric Feature Analysis

Each continuous feature was evaluated using:

* Distribution plots
* Group-wise summary statistics
* Welch’s t-test
* Cohen’s d (effect size)
* Standalone ROC-AUC

Key findings:

* **Max HR** and **ST Depression** showed very large effect sizes (≈ 0.95).
* Age showed moderate separation.
* Cholesterol showed small but measurable separation.
* BP showed negligible effect (AUC ≈ 0.50).

Importantly, statistical significance alone was not used as a decision criterion, since large sample size guarantees extremely small p-values. Effect size and AUC were prioritized instead.

---

## Categorical Feature Analysis

Categorical features were evaluated using:

* Disease rate per category
* Chi-square test
* Cramér’s V (association strength)
* Standalone ROC-AUC

Strong predictors identified:

* **Thallium**
* **Chest Pain Type**
* **Number of Vessels (Fluoroscopy)**
* **Exercise Angina**
* **Slope of ST**

Weak predictor:

* FBS over 120 (AUC ≈ 0.51)

The dataset appeared structurally dominated by stress-test and imaging variables.

---

## Interaction Analysis

A critical insight emerged from analyzing:

> Thallium × Number of Vessels

Certain combinations (e.g., Thallium = 7 and Vessels ≥ 2) resulted in ~97% disease probability.

This revealed:

* Strong rule-like segments.
* Multiplicative interaction patterns.
* High structural separability.

This insight influenced later model selection decisions.

---

# 🧪 Phase 2 — Minimal Feature Hypothesis Testing

Instead of immediately training models on all features, a controlled experiment was conducted:

> How much predictive power exists in the strongest few features alone?

Selected features:

* Thallium
* Max HR
* Chest Pain Type

Logistic Regression AUC: **0.916**

Adding:

* Number of Vessels

Logistic Regression AUC: **0.930**

This demonstrated:

* The dataset contains concentrated predictive signal.
* Only 4 features capture most discrimination.
* The problem is strongly separable.

---

# 🌳 Phase 3 — Model Class Comparison (4 Features Only)

To test whether nonlinear models meaningfully improve performance, three models were compared using 5-fold stratified cross-validation:

* Logistic Regression
* Decision Tree
* Random Forest

Results:

| Model               | CV AUC |
| ------------------- | ------ |
| Logistic Regression | 0.9296 |
| Decision Tree       | 0.9317 |
| Random Forest       | 0.9321 |

Interpretation:

* Tree-based models provided only marginal improvement.
* Additive linear structure dominates.
* Model complexity contributes little compared to feature strength.

This prevented unnecessary algorithm escalation.

---

# 🔍 Phase 4 — Full Feature Expansion

After validating that 4 features achieved strong performance, remaining features were incorporated:

* ST Depression
* Exercise Angina
* Slope of ST
* Sex
* EKG Results
* Age
* Cholesterol

A full preprocessing pipeline was built using:

* ColumnTransformer
* OneHotEncoder for nominal variables
* Proper handling of ordinal features
* Stratified cross-validation
* Leakage prevention through pipelines

Performance gains were incremental, confirming that most signal was already captured by the top features.

---

# ⚙️ Phase 5 — Feature Engineering

Based on EDA findings, additional transformations were explored:

* Explicit interaction terms (e.g., Thallium × Vessels)
* Binary indicator for asymptomatic chest pain (Type 4)
* Alternative encodings for ordinal features

Feature engineering produced marginal improvements, reinforcing the conclusion that dataset separability is intrinsic rather than model-dependent.

---

# 🚀 Phase 6 — Advanced Models & Tuning

Multiple models were evaluated:

* Logistic Regression (regularization tuning)
* Random Forest
* LightGBM
* XGBoost
* CatBoost

Early stopping and cross-validation were used to prevent overfitting.

Key observation:

Boosting models improved performance slightly but did not dramatically outperform logistic regression. This confirmed earlier findings that the dataset is largely additive and structurally clean.

---

# 📏 Phase 7 — Calibration & Threshold Optimization

Since this is a medical prediction task, evaluation extended beyond ROC-AUC.

## Calibration Analysis

* Calibration curves plotted.
* Brier scores computed.
* Logistic regression showed strong calibration.
* Tree-based models required post-calibration.

## Threshold Selection

Given clinical priorities (minimizing false negatives):

* Precision–Recall curves analyzed.
* Alternative thresholds evaluated.
* Trade-offs between sensitivity and specificity explored.

This phase shifted focus from “best model” to “best clinical operating point.”

---

# 🧠 Final Insights

1. The dataset is strongly driven by stress-test and imaging features.
2. A small subset of variables explains most variance.
3. Model complexity provides diminishing returns.
4. Logistic regression performs competitively with ensembles.
5. Calibration and threshold selection matter more than architecture tuning.
6. Feature quality outweighs algorithm choice in this dataset.

---

# 📦 Production Readiness

The final system includes:

* Fully encapsulated preprocessing pipeline
* Cross-validation framework
* Feature grouping strategy
* Hyperparameter tuning
* Probability calibration
* Threshold optimization
* Model serialization (Joblib)
* Deployment readiness considerations

---

# 📌 Conclusion

This project demonstrates disciplined ML development:

* Statistical validation before modeling.
* Controlled hypothesis testing with minimal features.
* Model comparison grounded in structural insight.
* Avoidance of unnecessary complexity.
* Emphasis on interpretability and calibration.
