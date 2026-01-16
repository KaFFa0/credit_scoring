# Credit Scoring ML Project
The objective of the project is to predict whether a borrower will experience a serious delinquency (90+ days past due)

Production-ready machine learning project focused on credit risk modeling, combining strong predictive performance with deep model interpretability.
The project demonstrates best practices in preprocessing pipelines, hyperparameter optimization, and SHAP-based analysis of nonlinear effects and feature interactions.

## Data Overview

`Target`: `SeriousDlqin2yrs` (class ratio ~93/7)

| Feature                             | Description                                         |
|-------------------------------------|-----------------------------------------------------|
| **RevolvingUtilizationOfUnsecuredLines**| Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits                                |
| **age**                                 | Age of borrower in years   |
| **NumberOfTime30-59DaysPastDueNotWorse**| Number of times borrower has been 30-59 days past due but no worse in the last 2 years. |
| **DebtRatio**                           | Monthly debt payments, alimony,living costs divided by monthy gross income |
| **MonthlyIncome**                       | Monthly Income |
| **NumberOfOpenCreditLinesAndLoans**     | Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)    |
| **NumberOfTimes90DaysLate**             | Number of times borrower has been 90 days or more past due. |
| **NumberRealEstateLoansOrLines**        | Number of mortgage and real estate loans including home equity lines of credit |
| **NumberOfTime60-89DaysPastDueNotWorse**| Number of times borrower has been 60-89 days past due but no worse in the last 2 years.     |
| **NumberOfDependents**                  | Number of dependents in family excluding themselves (spouse, children etc.) |

## Project structure

```
credit_scoring/
├── Data/
│   ├── cs-training.csv
|   └── cs-test.csv
├── models/
│   ├── bayesian_mi.joblib
|   ├── bayesian_nd.joblib
|   ├── scaler.joblib
│   └── best_lgbm.joblib
├── src/
│   ├── preprocessing.py              # preprocessing pipeline
│   └── features.py                   # polynomial features (for LogisticRegression)
├── notebooks/
|   ├── EDA.ipynb                     
|   ├── Preprocessing.ipynb
│   └── Training.ipynb                # training, tuning, SHAP analysis
└── README.md
```

## Technologies & Approaches

| Component                   | Technology / Approach                                            |
| --------------------------- | ---------------------------------------------------------------- |
| **Preprocessing**           | Custom sklearn-compatible preprocessing pipeline                 |
| **Imputation**              | Model-based imputation using BayesianRidge regression            |
| **Feature Engineering**     | Aggregation of correlating features into `PastDueAggregated`     |
| **Outlier Handling**        | Winsorization (upper percentile clipping)                        |
| **Distribution Correction** | Log-transformations for heavily right-skewed features            |
| **Scaling**                 | MinMax scaling of features                                       |
| **Baseline Model**          | Logistic Regression with regularization and polynomial features  |
| **Nonlinear Models**        | Random Forest, LightGBM                                          |
| **Hyper-parameter Tuning**  | `Optuna` hyperparameter optimization framework                   |
| **Evaluation**              | ROC-AUC as primary metric due to strong class imbalance                                        |
| **Interpretability**        | feature importances, SHAP (summary, dependence, force plots)     |
| **Interaction Analysis**    | SHAP dependence plots for nonlinear interactions                 |
| **Model Selection**         | Performance–interpretability trade-off                           |

## Results

| Model               | ROC-AUC (validation) |
| ------------------- | -------------------- |
| Logistic Regression | **0.8583**               |
| Random Forest       | **0.8593**               |
| **LightGBM**        | **0.8633**           |

### LightGBM achieves the best overall performance while maintaining strong interpretability through SHAP analysis.


## Interpretability

| Model               | Interpretation |
| ------------------- | -------------------- |
| Logistic Regression | Coefficients               |
| Random Forest       | Feature importances                |
| LightGBM            | SHAP summary/dependence/force plots           |

## Key Insights

- The presence of loan delinquencies is the strongest risk factor for default. Meanwhile, the presence/absence of delinquencies changes the interpretation of debt
- The high level of use of unsecured credit lines significantly increases the probability of default.
- The debt ratio shows strong non-linear behavior and interacts with delinquency rates.
- In the absence of past delinquencies, a high debt ratio in itself does not necessarily mean an increased risk.

