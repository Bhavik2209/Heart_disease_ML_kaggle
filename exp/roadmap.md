# 📌 Heart Disease Prediction Project Roadmap

## Phase 1 — Problem Definition & Clinical Understanding

### 1.1 Business Objective

* Build a **binary classification model** to predict presence/absence of heart disease.
* Optimize for **high recall** (minimize false negatives) due to clinical risk.

### 1.2 Clinical Context Understanding

* Interpret medical features:

  * ST depression → ischemia indicator
  * Thallium test → myocardial perfusion defect
  * Chest pain types → angina classification
  * Number of vessels → coronary blockage severity

**Why this matters:**
Without clinical understanding, feature engineering and interpretation become shallow and potentially misleading.

### 1.3 Success Criteria Definition

* Primary metric: Recall / PR-AUC
* Secondary metrics: ROC-AUC, F1-score
* Business constraint: Acceptable false positive rate

### 1.4 Risk Identification

* Class imbalance
* Data leakage
* Biased dataset (gender/age skew)

---

## Phase 2 — Data Acquisition & Inspection

### 2.1 Data Loading Strategy

* Efficient loading (Pandas initially; Polars/Dask if scaling required).
* Validate dataset size and schema.

### 2.2 Initial Data Audit

* Shape, data types
* Duplicate detection (especially `id`)
* Cardinality check of categorical features
* Constant/near-constant columns

### 2.3 Target Variable Inspection

* Encode target (object → binary)
* Analyze class distribution
* Quantify imbalance %

**Why this matters:**
Imbalance determines modeling and evaluation strategy.

---

## Phase 3 — Data Quality & Cleaning

### 3.1 Missing Value Analysis

* Explicit nulls
* Hidden nulls (e.g., Cholesterol = 0)
* Missingness pattern analysis

### 3.2 Outlier Detection

* Statistical: IQR, Z-score
* Domain-based filtering (e.g., Max HR > 220)

### 3.3 Imputation Strategy

* Mean/Median for skewed distributions
* KNN Imputer for structured missingness
* Iterative Imputer (MICE) for complex relationships

### 3.4 Drop vs Impute Decision Framework

* Impact on sample size
* Clinical relevance
* Risk of bias introduction

---

## Phase 4 — Exploratory Data Analysis (EDA)

### 4.1 Univariate Analysis

* Distribution plots
* Skewness detection
* Feature variance inspection

### 4.2 Bivariate Analysis

* Feature vs Target relationships
* Statistical tests (Chi-square, t-test, ANOVA)

### 4.3 Multivariate Analysis

* Correlation heatmap
* Feature interaction detection
* Multicollinearity assessment (VIF)

### 4.4 Clinical Insight Extraction

* Risk patterns across age groups
* Chest pain type vs heart disease rate
* Vessel count impact

---

## Phase 5 — Feature Engineering

### 5.1 Encoding

* Target encoding (binary mapping)
* Ordinal vs Nominal encoding decision

### 5.2 Feature Transformation

* Age bins
* BP categories
* Cholesterol risk buckets

### 5.3 Interaction Features

* Age × Max HR
* BP × Cholesterol

### 5.4 Scaling Decision

* StandardScaler (linear models)
* RobustScaler (outliers present)
* No scaling (tree-based models)

### 5.5 Feature Selection

* Correlation-based pruning
* Recursive Feature Elimination
* Model-based importance

---

## Phase 6 — Data Splitting & Validation Strategy

### 6.1 Train-Test Split

* Stratified split to preserve imbalance ratio

### 6.2 Cross-Validation

* Stratified K-Fold
* Repeated K-Fold (if variance high)

### 6.3 Leakage Prevention

* All preprocessing inside pipeline
* No resampling before split

---

## Phase 7 — Baseline Modeling

### 7.1 Dummy Classifier

* Establish baseline accuracy and recall

### 7.2 Logistic Regression

* Interpretability
* Regularization (L1/L2)

---

## Phase 8 — Advanced Modeling

### 8.1 Tree-Based Models

* Decision Tree
* Random Forest
* Gradient Boosting (XGBoost, LightGBM, CatBoost)

### 8.2 Distance-Based / Margin-Based

* KNN (scalability caution)
* SVM (kernel trick, expensive at 630K rows)

### 8.3 Neural Networks

* MLP classifier (baseline deep learning)

### 8.4 Bias-Variance Analysis

* Overfitting detection
* Learning curves

---

## Phase 9 — Model Evaluation

### 9.1 Core Metrics

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC

### 9.2 Confusion Matrix Analysis

* Clinical risk evaluation
* False Negative cost emphasis

### 9.3 Threshold Optimization

* Custom probability threshold
* Youden’s J statistic
* Business-cost-based tuning

---

## Phase 10 — Hyperparameter Optimization

### 10.1 GridSearchCV

* Exhaustive search (small parameter space)

### 10.2 RandomizedSearchCV

* Faster exploration

### 10.3 Bayesian Optimization (Optuna)

* Efficient large search spaces

### 10.4 Early Stopping

* Required for boosting models

---

## Phase 11 — Model Interpretability

### 11.1 Global Interpretation

* Feature Importance
* Permutation Importance

### 11.2 SHAP Analysis

* Summary plot
* Individual prediction explanation

### 11.3 Bias Audit

* Performance by gender/age subgroup
* Fairness metrics

---

## Phase 12 — Model Finalization & Saving

### 12.1 Final Model Selection

* Based on recall + PR-AUC + stability

### 12.2 Serialization

* Joblib (preferred for sklearn)
* Versioning strategy

### 12.3 Prediction Interface

* Input validation
* Probability output
* Threshold-based classification

---

## Phase 13 — Deployment Preparation

### 13.1 API Layer

* FastAPI service

### 13.2 Containerization

* Docker

### 13.3 Experiment Tracking

* MLflow

### 13.4 Data Versioning

* DVC

---

## Phase 14 — Monitoring & Maintenance

### 14.1 Data Drift Monitoring

* Population stability index (PSI)

### 14.2 Model Drift Monitoring

* Performance tracking over time

### 14.3 Periodic Retraining Plan

---

## Phase 15 — Documentation & Reporting

### 15.1 Technical Report

* Dataset description
* Modeling decisions
* Evaluation results

### 15.2 Clinical Interpretation

* Key predictive factors
* Limitations

### 15.3 Model Card

* Intended use
* Limitations
* Ethical considerations

---

# 🔎 High-Level Timeline View

1. Problem Framing
2. Data Audit
3. Cleaning
4. EDA
5. Feature Engineering
6. Splitting & Validation
7. Baseline Models
8. Advanced Models
9. Evaluation & Thresholding
10. Tuning
11. Interpretability
12. Deployment Prep
13. Monitoring

---
