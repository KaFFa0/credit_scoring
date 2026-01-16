from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def make_feature_pipeline():
    return Pipeline([
        ('poly', PolynomialFeatures(
            degree=2,
            interaction_only=True,
            include_bias=False
        ))
    ])
