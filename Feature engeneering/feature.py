
#%%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_excel("concrete.xls")
df.columns = df.columns.str.strip()
df.head()

X = df.copy()
y = X.pop("CompressiveStrength")

baseline = RandomForestRegressor(criterion="absolute_error",random_state=0)
baseline_score = cross_val_score(
    baseline, X,y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()
print(f"MAE Baseline Score: {baseline_score:.4}")

X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRatio"] = X["Water"] / X["Cement"]
model = RandomForestRegressor(criterion="absolute_error", random_state=0)
score = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
score = -1 * score.mean()

print(f"MAE Score with Ratio Features: {score:.4}")

#Matual Information
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Source - https://stackoverflow.com/a/78339240
# Posted by Alejandro Sa
# Retrieved 2026-10-31, License - CC BY-SA 4.0

plt.style.use("seaborn-v0_8-whitegrid")


df = pd.read_csv("autos.csv")
df.head()
X = df.copy()
y = X.pop('price')

for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

discrete_features = X.dtypes == int

from sklearn.feature_selection import mutual_info_regression
def make_mi_scores(X,y, discerte_features):
    mi_scores = mutual_info_regression(X,y,discrete_features=discerte_features)
    mi_scores = pd.Series(mi_scores, name="Mi Scores", index = X.columns)
    

# %%
