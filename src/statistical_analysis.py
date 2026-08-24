"""Descriptive and non-causal exploratory multivariate analyses."""
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def descriptive_analysis(df,target):
    """Compute descriptive statistics and EDA correlation/dependency tables."""
    n=df.select_dtypes(include="number"); d=pd.DataFrame({"count":n.count(),"missing":n.isna().sum(),"mean":n.mean(),"median":n.median(),"std":n.std(),"variance":n.var(),"min":n.min(),"max":n.max(),"Q1":n.quantile(.25),"Q3":n.quantile(.75),"IQR":n.quantile(.75)-n.quantile(.25),"skewness":n.skew(),"kurtosis":n.kurt()})
    mi=pd.DataFrame(columns=["feature","mutual_information"])
    if target in n:
        x=n.drop(columns=target).dropna(); aligned=n.loc[x.index,target]
        if len(x) and x.shape[1]: mi=pd.DataFrame({"feature":x.columns,"mutual_information":mutual_info_regression(x,aligned)}).sort_values("mutual_information",ascending=False)
    return d,n.corr("pearson"),n.corr("spearman"),mi
