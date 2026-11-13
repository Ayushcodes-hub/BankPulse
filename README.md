# BankPulse | Bank Term-Deposit Subscription Prediction

<p align="center">
  <strong>Predicting customer subscription behavior using machine learning.</strong>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/Ayushcodes-hub/BankPulse/blob/main/notebooks/BankPulse.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Google Colab">
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Status-Completed-2ea44f" alt="Project status">
</p>

---

## Overview

**BankPulse** is an end-to-end machine learning project designed to predict whether a bank customer is likely to subscribe to a term deposit following a direct marketing campaign.

Using customer demographics, financial information, and previous campaign interactions, the project develops classification models that estimate the probability of customer subscription.

The implementation covers the complete machine learning lifecycle, including data acquisition, exploratory data analysis, feature engineering, preprocessing, model development, performance comparison, decision-threshold optimization, hyperparameter tuning, model evaluation, and prediction.

The project places particular emphasis on **class imbalance, data leakage prevention, validation-based threshold selection, and reproducible machine learning pipelines.**

**Objective:** Develop a reliable classification workflow that can help identify potential subscribers and support more informed marketing decisions.

## Problem Statement

Direct marketing campaigns require banks to contact customers to promote financial products such as term deposits. However, contacting every customer without considering their likelihood of subscribing can lead to inefficient allocation of marketing resources.

The objective is to build a binary classification model that uses historical customer and campaign information to predict whether a customer will subscribe to a term deposit.

The model predicts one of two outcomes:

- **Yes (1):** Customer subscribes to the term deposit.
- **No (0):** Customer does not subscribe.

Rather than relying on accuracy alone, the project evaluates how effectively the models identify the minority class of actual subscribers.

## Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The project uses the `bank-full.csv` dataset, which contains information collected from a Portuguese bank's direct marketing campaigns.

The dataset is downloaded programmatically from the UCI repository when the notebook executes.

| Dataset characteristic | Value |
|---|---:|
| Total observations | 45,211 |
| Original columns | 17 |
| Input variables | 16 |
| Target variable | 1 |
| Classification type | Binary |
| Target column | `y` |

### Feature categories

| Category | Variables |
|---|---|
| Demographic | Age, job, marital status, education |
| Financial | Balance, default, housing loan, personal loan |
| Campaign | Contact, day, month, campaign |
| Previous interactions | Pdays, previous, poutcome |
| Target | Subscription (`y`) |

### Class distribution

The target variable is significantly imbalanced.

| Subscription outcome | Observations | Distribution |
|---|---:|---:|
| No | 39,922 | 88.30% |
| Yes | 5,289 | 11.70% |
| **Total** | **45,211** | **100%** |

This imbalance makes it important to evaluate precision, recall, F1-score, and ROC-AUC alongside accuracy.

---

## Methodology

The project follows a structured machine learning pipeline.

### 1. Data Acquisition and Exploration

The dataset is downloaded from the UCI repository using Python's `requests` and `zipfile` libraries.

Pandas is used to load and inspect the dataset.

Exploratory data analysis examines:

- Target-class distribution.
- Customer age and subscription behavior.
- Subscription patterns across job categories.
- Account balance distributions.
- Relationships between customer characteristics and subscription outcomes.

**Visualization:** Matplotlib and Seaborn.

### 2. Feature Engineering

Feature engineering prepares the raw dataset for model training.

The following transformations are performed:

- Convert the target variable into numerical binary values: `yes = 1`, `no = 0`.
- Create `was_previously_contacted` using the `pdays` variable.
- Replace the `-1` sentinel value in `pdays` with a missing value.
- Separate input features from the target variable.
- Exclude the original target columns from model inputs.

**Preventing target leakage**

The `duration` feature is deliberately excluded from the model.

Call duration is only known after a marketing call has taken place. Including it in a model intended to identify customers before contacting them would introduce target leakage and produce an unrealistic prediction scenario.

### 3. Data Splitting

The dataset is divided into training, validation, and testing subsets using stratified sampling.

| Subset | Proportion | Records |
|---|---:|---:|
| Training | 70% | 31,647 |
| Validation | 15% | 6,782 |
| Testing | 15% | 6,782 |

A fixed random state of `42` ensures reproducibility.

- **Training set:** Fits the machine learning models.
- **Validation set:** Supports model comparison and decision-threshold selection.
- **Test set:** Evaluates final model performance on unseen observations.

Stratification preserves the distribution of subscription and non-subscription classes across the splits.

### 4. Data Preprocessing

A Scikit-learn `ColumnTransformer` combines separate preprocessing pipelines for numerical and categorical features.

**Numerical preprocessing**

- `SimpleImputer(strategy="median")` handles missing numerical values.
- `StandardScaler` standardizes numerical features.

**Categorical preprocessing**

- `SimpleImputer(strategy="most_frequent")` handles missing categorical values.
- `OneHotEncoder(handle_unknown="ignore")` converts categorical variables into numerical representations while handling previously unseen categories.

The preprocessing transformations are integrated into Scikit-learn `Pipeline` objects with the classifiers.

This ensures that preprocessing is fitted using training data and consistently applied during validation, testing, and inference.

### 5. Model Development

Two supervised classification algorithms are implemented.

**Logistic Regression**

A linear classification algorithm used as a baseline model.

Configuration:
- Maximum iterations: 2,000.
- Class weighting: `balanced`.
- Random state: 42.

It estimates the probability of subscription based on a weighted combination of input features.

**Random Forest Classifier**

An ensemble learning algorithm that combines predictions from multiple decision trees.

Initial configuration:
- Number of estimators: 300.
- Maximum depth: 15.
- Minimum samples split: 10.
- Minimum samples per leaf: 4.
- Maximum features: `sqrt`.
- Class weighting: `balanced`.

Random Forest is used to capture nonlinear relationships and interactions between customer characteristics.

### 6. Model Comparison

Both initial models are evaluated on the validation set using a classification threshold of 0.50.

| Evaluation metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.7569 | 0.8453 |
| Precision | 0.2693 | 0.3913 |
| Recall | 0.6285 | 0.5781 |
| F1-score | 0.3770 | 0.4667 |
| ROC-AUC | 0.7720 | 0.7989 |

**Observation:** Random Forest performs better in validation accuracy, precision, F1-score, and ROC-AUC, while Logistic Regression achieves higher recall at the default threshold.

This comparison illustrates the trade-off between identifying more potential subscribers and reducing false-positive predictions.

### 7. Decision Threshold Optimization

A classification model generates probabilities, which are converted into class predictions using a decision threshold.

The default threshold is 0.50, but this is not necessarily optimal for an imbalanced dataset.

The project evaluates thresholds from 0.10 to 0.90 in increments of 0.01.

The threshold maximizing validation F1-score is selected.

For the initial Random Forest:

| Result | Value |
|---|---:|
| Selected threshold | 0.54 |
| Validation F1-score | 0.4816 |

Precision, recall, and F1-score are also visualized across different thresholds to demonstrate how classification decisions affect model performance.

### 8. Hyperparameter Optimization

The Random Forest model is further optimized using `RandomizedSearchCV`.

Search configuration:

| Parameter | Value |
|---|---:|
| Number of sampled combinations | 12 |
| Cross-validation folds | 3 |
| Optimization metric | F1-score |
| Random state | 42 |

The search explores:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `class_weight`

**Best hyperparameters obtained:**

| Hyperparameter | Selected value |
|---|---|
| `n_estimators` | 150 |
| `max_depth` | None |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 4 |
| `max_features` | `log2` |
| `class_weight` | `balanced_subsample` |

Best cross-validation F1-score: **0.4638**

The hyperparameter search is fitted using training data. The validation set is subsequently used for threshold selection.

### 9. Final Model Evaluation

The tuned Random Forest is evaluated on the held-out test set using a decision threshold selected on validation data.

**Final test performance:**

| Metric | Score |
|---|---:|
| Accuracy | 0.87 |
| Precision | 0.46 |
| Recall | 0.49 |
| F1-score | 0.47 |
| ROC-AUC | 0.7904 |

The notebook additionally generates:

- Confusion matrix.
- Classification report.
- Receiver Operating Characteristic (ROC) curve.
- Precision-Recall curve.

These visualizations help examine the model's classification behavior and its ability to distinguish between subscribers and non-subscribers.

### 10. Feature Importance

Permutation feature importance is used to estimate the contribution of individual input features to model performance.

The leading features in the validation analysis include:

1. Month
2. Contact method
3. Housing loan
4. Previous campaign outcome
5. Contact day
6. Age
7. Personal loan

The notebook also visualizes the top 15 features.

Permutation importance measures the effect of disrupting a feature on model performance; it does not establish a causal relationship.

### 11. Model Persistence and Prediction

The tuned Random Forest pipeline is saved using Joblib.

The saved model artifact contains:

- Trained machine learning pipeline.
- Selected decision threshold.
- Expected feature-column names.

The notebook demonstrates loading the saved model and generating predictions for customer records.

A reusable `predict_customer()` function is implemented to return:

- Estimated subscription probability.
- Predicted subscription class.
- Human-readable `Likely` or `Unlikely` outcome.

The model is saved locally in the notebook runtime under:

`bankpulse_models/bankpulse_random_forest.pkl`

---

## Results Summary

BankPulse demonstrates a complete classification workflow, from raw customer data to a reusable trained model.

| Component | Implementation |
|---|---|
| Data source | UCI Bank Marketing |
| Observations | 45,211 |
| Preprocessing | Imputation, scaling, one-hot encoding |
| Feature engineering | Previous-contact indicator |
| Leakage prevention | Exclusion of call duration |
| Baseline algorithms | Logistic Regression, Random Forest |
| Model selection | Validation comparison |
| Optimization | RandomizedSearchCV |
| Threshold selection | Validation F1-score |
| Final evaluation | Held-out test dataset |
| Model persistence | Joblib |
| Prediction | Reusable inference function |

## Repository Structure

```text
BankPulse/
│
├── notebooks/
│   └── BankPulse.ipynb
│
├── LICENSE
│
└── README.md
```

The trained model artifact is generated when the notebook is executed. It is not currently committed as a repository file.

## Installation and Execution

### Option 1: Google Colab

The easiest way to reproduce the project is through Google Colab.

1. Open the notebook using the badge at the top of this README.
2. Select **Runtime → Run all**.
3. The notebook installs the required dependencies.
4. The dataset is downloaded automatically from UCI.
5. Execute the notebook sequentially to reproduce the complete workflow.

No manual dataset download is required.

### Option 2: Local Environment

Clone the repository:

```bash
git clone https://github.com/Ayushcodes-hub/BankPulse.git
cd BankPulse
```

Install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib requests
```

Launch Jupyter Notebook:

```bash
jupyter notebook notebooks/BankPulse.ipynb
```

Run the notebook cells sequentially.

**Requirements:** Python 3.x, the listed Python packages, and an internet connection for dataset acquisition.

## Technology Stack

| Technology | Role |
|---|---|
| Python | Core development |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, classification, optimization, evaluation |
| Joblib | Model serialization |
| Requests | Dataset download |
| Zipfile | Archive extraction |
| Google Colab | Development and execution environment |

## Key Concepts Demonstrated

- Supervised machine learning
- Binary classification
- Exploratory data analysis
- Feature engineering
- Data preprocessing
- Missing-value imputation
- Feature scaling
- One-hot encoding
- Data leakage prevention
- Stratified sampling
- Class imbalance
- Logistic Regression
- Random Forest and ensemble learning
- Probability estimation
- Decision-threshold optimization
- Cross-validation
- Hyperparameter optimization
- Classification metrics
- ROC and Precision-Recall analysis
- Permutation feature importance
- Model persistence and inference

## Limitations and Future Improvements

### Current limitations

- The dataset represents a historical bank-marketing context and may not reflect current customer behavior.
- The target class is imbalanced, making accuracy alone an insufficient measure of performance.
- Test-set performance does not establish real-world campaign effectiveness or financial return.
- Feature importance does not imply causality.
- The trained model has not been validated through a live marketing deployment.

### Future improvements

- Probability calibration to improve the reliability of predicted probabilities.
- Precision-Recall AUC analysis.
- Business-cost-based decision-threshold optimization.
- Time-based validation to evaluate changing campaign behavior.
- Model drift monitoring.
- Fairness and subgroup performance evaluation.
- Deployment through a FastAPI prediction service.
- Interactive customer prediction dashboard.

## References

1. Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, 62, 22–31.
2. [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

<p align="center">
  <strong>BankPulse</strong><br>
  End-to-end machine learning project by <a href="https://github.com/Ayushcodes-hub">Ayush Aryaman</a>.
</p>
