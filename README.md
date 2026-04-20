# Credit Card Fraud Detection — Data Mining

A machine learning pipeline for real-time detection of fraudulent banking transactions, combining supervised classification, unsupervised clustering, and association rule mining.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scientific Rationale](#2-scientific-rationale)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Models & Results](#5-models--results)
6. [Clustering Analysis](#6-clustering-analysis)
7. [Association Rules](#7-association-rules)
8. [Operational ML Pipeline](#8-operational-ml-pipeline)
9. [Project Structure](#9-project-structure)
10. [Installation & Usage](#10-installation--usage)
11. [KDD Validation Checklist](#11-kdd-validation-checklist)
12. [Roadmap](#12-roadmap)
13. [Authors](#13-authors)

---

## 1. Project Overview

Credit card fraud represents a critical challenge for financial institutions. The fraud rate in transaction datasets is typically below 0.2%, creating a severe class imbalance that invalidates naive accuracy-based evaluation and demands specialized modeling strategies.

This project implements a complete Data Mining pipeline — from raw data ingestion to a decision-ready inference layer — applying:

- **Supervised learning** (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- **Unsupervised learning** (K-Means clustering, Isolation Forest)
- **Frequent pattern mining** (Apriori — association rules)

The system is designed to operate in near real-time (< 100 ms inference) and comply with financial regulation frameworks (PSD2, KYC, AML).

---

## 2. Scientific Rationale

### Why Recall, not Accuracy

In heavily imbalanced datasets, accuracy is a misleading metric. A trivial classifier that always predicts "non-fraudulent" achieves:

```
Accuracy = 99.8%   |   Fraud detected = 0
```

This classifier is operationally worthless. The correct primary metric is **Recall (Sensitivity)**:

```
Recall = True Positives / (True Positives + False Negatives)
       = Detected frauds / Total actual frauds
```

**Business justification:**

| Error type | Consequence | Cost |
|---|---|---|
| False Negative (missed fraud) | Direct financial loss, client harm | High |
| False Positive (false alert) | Minor friction (OTP, manual review) | Low |

A high-Recall model absorbs a controlled number of false positives in exchange for minimizing undetected fraud — which is the operationally correct trade-off.

### Evaluation Metrics Used

- **Recall** — primary metric (fraud detection rate)
- **Precision** — controls alert fatigue
- **F1-Score** — harmonic balance of both
- **ROC-AUC** — global discriminative power across all decision thresholds
- **Confusion Matrix** — raw breakdown of TP / TN / FP / FN

---

## 3. Dataset

The dataset used is the standard benchmark for credit card fraud detection research.

| Property | Value |
|---|---|
| Transactions | 284,807 |
| Fraudulent transactions | 492 (0.172%) |
| Features | 30 (V1–V28 PCA-transformed, Time, Amount) |
| Target variable | `Class` (0 = legitimate, 1 = fraud) |

**Key preprocessing steps:**

```python
from sklearn.preprocessing import StandardScaler

# Scale Amount and Time (not PCA-transformed)
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time']   = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
```

**Class imbalance handling:**

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
```

---

## 4. Methodology

The project follows the **KDD (Knowledge Discovery in Databases)** process:

```
Step 1  Data Selection
        Raw transaction logs, feature extraction (amount, time, device, location)

Step 2  Preprocessing
        Null handling, scaling, SMOTE resampling, feature engineering

Step 3  Transformation
        PCA already applied upstream (V1-V28)
        Cluster labels added as engineered feature

Step 4  Data Mining
        Supervised:    Logistic Regression, Decision Tree, Random Forest, XGBoost
        Unsupervised:  K-Means, Isolation Forest
        Pattern:       Apriori association rules

Step 5  Evaluation
        Recall, Precision, F1, ROC-AUC, Confusion Matrix

Step 6  Interpretation
        Business rule generation, pipeline design, deployment specification
```

---

## 5. Models & Results

### Comparative Performance

| Model | Recall | Precision | ROC-AUC | Notes |
|---|---|---|---|---|
| Logistic Regression | Moderate | Moderate | Good | Baseline; limited on non-linear patterns |
| Decision Tree | High (unstable) | Moderate | Variable | Prone to overfitting; poor generalization |
| Random Forest | Very high | Good | Excellent | Best overall compromise |
| XGBoost | Very high | Very good | Excellent | Best after hyperparameter tuning |
| Isolation Forest | Low | n/a | Moderate | Unsupervised; useful as anomaly pre-filter |
| K-Means | Not applicable | n/a | Low | Behavioral segmentation, not direct detection |

### Random Forest — Recommended for Production

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

**Why Random Forest:**

- Robust to class imbalance via `class_weight='balanced'`
- High Recall without collapsing Precision
- Native feature importance for model explainability
- No assumption on feature distributions

### XGBoost — Recommended after Tuning

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    use_label_encoder=False,
    eval_metric='aucpr',
    random_state=42
)
xgb.fit(X_train, y_train)
```

`scale_pos_weight` explicitly compensates for class imbalance without SMOTE.

---

## 6. Clustering Analysis

K-Means clustering is not a fraud detector in isolation — it is a **behavioral profiling tool** that improves the supervised pipeline.

### Behavioral Segments Identified

| Cluster | Behavioral Profile | Fraud Risk |
|---|---|---|
| 0 | Frequent transactions, low amounts | Low |
| 1 | Irregular transactions, high amounts | Elevated |
| 2 | Heavy online usage | Medium |
| 3 | Nocturnal transactions, irregular patterns | High |

### Usage in the ML Pipeline

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Cluster label used as engineered feature in supervised model
X_with_cluster = pd.concat([X, df['cluster']], axis=1)
```

Cluster labels function as a compressed behavioral signal, improving the discriminative power of downstream classifiers.

### Outlier Pre-filtering

Transactions that fall in low-density cluster regions are flagged as anomaly candidates before supervised scoring — reducing the inference load on the main model.

---

## 7. Association Rules

Apriori mining extracts frequent co-occurrence patterns from discretized transaction features, producing human-readable antiffraud rules.

### Example Rules Extracted

| Antecedent | Consequent | Interpretation |
|---|---|---|
| Nocturnal transaction + high amount | Fraud risk elevated | Temporal + amount signal |
| Inter-transaction delay < 10 sec | Suspicious burst | Automated attack pattern |
| Country != usual country | Alert | Geographic anomaly |
| 3+ consecutive failed attempts | Abnormal behavior | Brute-force or card testing |

### Implementation

```python
from mlxtend.frequent_patterns import apriori, association_rules

# Discretize and binarize transaction features
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

rules_sorted = rules.sort_values('lift', ascending=False)
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
```

### Operational Value

- Direct input for rule-based antifraude engines (alongside the ML model)
- Triggers for 3D Secure / OTP challenges on suspicious patterns
- Audit trail: rules are fully explainable and traceable for compliance

---

## 8. Operational ML Pipeline

### End-to-End Architecture

```
Step 1  Data Ingestion
        Sources: internal API (amount, merchant, timestamp,
                 client_id, device_id, IP, transaction history)

Step 2  Preprocessing (< 10 ms)
        - StandardScaler on Amount / Time
        - Feature engineering (velocity, geo-delta, cluster label)
        - Enrichment from historical profile

Step 3  Inference (< 100 ms)
        - Random Forest / XGBoost
        - Output: fraud probability score [0.0, 1.0]

Step 4  Decision Layer

        Score < 0.20         Transaction approved
        Score 0.20 -- 0.50   Challenge (OTP / 3D Secure)
        Score > 0.50         Blocked + compliance alert

Step 5  Monitoring & Retraining
        - Weekly / monthly retraining on labeled feedback
        - Drift detection (feature distribution shift)
        - A/B testing between model versions
        - Dashboard: Recall trend, prevented losses

Step 6  Audit & Compliance
        - Full decision logging (prediction, score, features used)
        - SHAP-based prediction explanation on flagged transactions
        - Regulatory alignment: PSD2, KYC, AML
```

### SHAP Explainability

```python
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Visualize feature contribution for a flagged transaction
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_test.iloc[0]
)
```

---

## 9. Project Structure

```
fraud-detection/
├── data/
│   ├── creditcard.csv              # Raw dataset (not committed — see note below)
│   └── processed/                  # Preprocessed train/test splits
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb      # Scaling, SMOTE, feature engineering
│   ├── 03_Supervised_Models.ipynb  # LR, DT, RF, XGBoost training & eval
│   ├── 04_Unsupervised.ipynb       # K-Means, Isolation Forest
│   └── 05_Association_Rules.ipynb  # Apriori mining
├── src/
│   ├── preprocessing.py            # Scaling, SMOTE pipeline
│   ├── feature_engineering.py      # Cluster labels, velocity features
│   ├── train.py                    # Model training entrypoint
│   ├── evaluate.py                 # Metrics, confusion matrix, ROC curve
│   ├── predict.py                  # Inference on new transactions
│   └── rules.py                    # Apriori pipeline
├── models/
│   └── random_forest_v1.pkl        # Serialized production model
├── reports/
│   └── final_report.md             # Business interpretation & conclusions
├── requirements.txt
└── README.md
```

> The dataset `creditcard.csv` is not included in this repository.
> Download it from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.

---

## 10. Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

```
pandas==2.0.3
numpy==1.25.2
scikit-learn==1.3.0
imbalanced-learn==0.11.0
xgboost==1.7.6
mlxtend==0.22.0
shap==0.42.1
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
```

### Run the full pipeline

```bash
# 1. Preprocess data
python src/preprocessing.py --input data/creditcard.csv --output data/processed/

# 2. Train models
python src/train.py --model random_forest --output models/

# 3. Evaluate
python src/evaluate.py --model models/random_forest_v1.pkl --data data/processed/X_test.csv

# 4. Predict on new transactions
python src/predict.py --model models/random_forest_v1.pkl --input data/new_transactions.csv
```

### Run notebooks sequentially

```bash
jupyter notebook notebooks/
```

Execute notebooks `01` through `05` in order.

---

## 11. KDD Validation Checklist

| KDD Step | Description | Status |
|---|---|---|
| Selection | Objective definition, dataset identification | Done |
| Preprocessing | Null handling, scaling, SMOTE resampling | Done |
| Transformation | Feature engineering, cluster enrichment | Done |
| Data Mining | Supervised + unsupervised + pattern mining | Done |
| Evaluation | Recall, Precision, F1, ROC-AUC, business interpretation | Done |

All KDD phases are complete. The project is ready for notebook finalization and deployment specification.

---

## 12. Roadmap

**Short term**
- Hyperparameter tuning (GridSearchCV / Optuna) for XGBoost
- Full SHAP integration for every flagged transaction
- REST API wrapper for real-time inference (FastAPI)

**Medium term**
- Drift detection module (evidently.ai or custom KS-test on feature distributions)
- Automated retraining trigger on Recall degradation
- Grafana dashboard for operational monitoring (Recall trend, blocked transaction volume)

**Long term**
- Graph-based fraud detection (transaction network analysis — GNN)
- Federated learning for multi-institution collaborative detection
- LLM-assisted anomaly narrative generation for compliance reports

---

## 13. Authors

**Data Mining Project — Credit Card Fraud Detection**

Developed as part of an academic Data Mining curriculum, applying the full KDD methodology to a real-world, publicly available financial dataset.

---

*For questions regarding the methodology or results, refer to `reports/final_report.md` or open an issue.*
