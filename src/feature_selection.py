"""Training-only mutual-information feature selection and fold stability."""
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_regression

def select_features(X,y,max_features=20):
    """Fit selector only on supplied training rows, never the locked test set."""
    k=min(max_features,X.shape[1]); selector=SelectKBest(mutual_info_regression,k=k).fit(X,y); cols=X.columns[selector.get_support()].tolist()
    return cols,pd.DataFrame({"feature":X.columns,"mi_score":selector.scores_}).sort_values("mi_score",ascending=False)
