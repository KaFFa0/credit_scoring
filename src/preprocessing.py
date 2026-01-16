import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

class MonthlyIncomeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, model_path):
        self.model_path = model_path
        self.target_col = 'MonthlyIncome'
        self.predictor_cols = [
            'DebtRatio',
            'RevolvingUtilizationOfUnsecuredLines',
            'NumberRealEstateLoansOrLines'
        ]

    def fit(self, X, y=None):
        self.model_ = joblib.load(self.model_path)
        return self

    def transform(self, X):
        X = X.copy()
        mask = X[self.target_col].isna()
        if mask.any():
            preds = self.model_.predict(X.loc[mask, self.predictor_cols])
            X.loc[mask, self.target_col] = np.maximum(preds, 0)
        return X


class NumberOfDependentsImputer(BaseEstimator, TransformerMixin):
    def __init__(self, model_path):
        self.model_path = model_path
        self.target_col = 'NumberOfDependents'
        self.predictor_cols = [
            'MonthlyIncome',
            'DebtRatio',
            'age',
            'RevolvingUtilizationOfUnsecuredLines',
            'NumberOfOpenCreditLinesAndLoans'
        ]

    def fit(self, X, y=None):
        self.model_ = joblib.load(self.model_path)
        return self

    def transform(self, X):
        X = X.copy()
        mask = X[self.target_col].isna()
        if mask.any():
            preds = self.model_.predict(X.loc[mask, self.predictor_cols])
            X.loc[mask, self.target_col] = np.round(np.maximum(preds, 0))
        return X

class PastDueAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['PastDueAggregated'] = (
            X['NumberOfTime30-59DaysPastDueNotWorse'] +
            X['NumberOfTime60-89DaysPastDueNotWorse'] +
            X['NumberOfTimes90DaysLate']
        )

        return X.drop([
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ], axis=1)

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, q=0.95):
        self.q = q

    def fit(self, X, y=None):
        self.upper_bounds_ = X.quantile(self.q)
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].clip(0, self.upper_bounds_[col])
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = np.log1p(X[col])
        return X

def make_preprocess_pipeline():
    return Pipeline([
        ('mi_imputer', MonthlyIncomeImputer('../models/bayesian_mi.joblib')),
        ('nd_imputer', NumberOfDependentsImputer('../models/bayesian_nd.joblib')),
        ('pastdue', PastDueAggregator()),
        ('winsor', Winsorizer(q=0.95)),
        ('log', LogTransformer(cols=['DebtRatio', 'PastDueAggregated'])),
        ('scaler', MinMaxScaler())
    ])
