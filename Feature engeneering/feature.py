
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


df = pd.read_csv("autos.csv", na_values="?")
df = df.dropna()
df.columns = df.columns.str.replace("-", "_")
X = df.copy()
y = X.pop("price")

for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

discrete_features = X.dtypes == int

from sklearn.feature_selection import mutual_info_regression
def make_mi_scores(X,y, discerte_features):
    mi_scores = mutual_info_regression(X,y,discrete_features=discerte_features)
    mi_scores = pd.Series(mi_scores, name="Mi Scores", index = X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X,y,discrete_features)
mi_scores[::3]

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width,scores)
    plt.yticks(width,ticks)
    plt.title("Mutual Information Scores")

plt.figure(dpi=100,figsize=(8,5))
plot_mi_scores(mi_scores)

sns.relplot(x="curb_weight",y="price",data=df)
sns.lmplot(x="horsepower",y="price",hue="fuel_type",data=df)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

#accidents = pd.read_csv("accidents.csv")
autos = pd.read_csv("autos.csv", na_values="?")
autos = autos.dropna()
autos.columns = autos.columns.str.replace("-", "_")
concrete = pd.read_excel("concrete.xls")
customer = pd.read_csv("customer.csv")

cylinder_map = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "eight": 8,
    "twelve": 12
}

autos["num_of_cylinders"] = autos["num_of_cylinders"].replace(cylinder_map)
autos["stroke_ratio"] = autos.stroke / autos.bore
autos[["stroke","bore","stroke_ratio"]].head()
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)

# accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)

# # Plot a comparison
# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
# sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);


roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
#accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

#accidents[roadway_features + ["RoadwayFeatures"]].head(10)


components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

concrete[components + ["Components"]].head(10)

customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)

customer[["Policy", "Type", "Level"]].head(10)

autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
autos[["make", "body_style", "make_and_style"]].head()

customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)

customer[["State", "Income", "AverageIncome"]].head(10)

customer["StateFreq"] = (
    customer.groupby("State")
    ["State"]
    .transform("count")
    / customer.State.count()
)

customer[["State", "StateFreq"]].head(10)

# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)

df_valid[["Coverage", "AverageClaim"]].head(10)

from sklearn.cluster import KMeans

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

df = pd.read_csv("/housing.csv")
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
X.head()
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")
sns.regplot(
    x="Longitude",y="Latitude", hue="Cluster", data=X, height=6
)


# %%
#feladat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


# Load data
df = pd.read_csv("ames.csv")


# Utility functions from Tutorial
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

X = df.copy()
y = X.pop('SalePrice')

mi_scores = make_mi_scores(X, y)

print(mi_scores.head(20))
# print(mi_scores.tail(20))  # uncomment to see bottom 20

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(20))
sns.catplot(x="BldgType", y="SalePrice", data=df, kind="boxen");

feature = "GrLivArea"

sns.lmplot(
    x=feature, y="SalePrice", hue="BldgType", col="BldgType",
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);
# YOUR CODE HERE: 
feature = "MoSold"

sns.lmplot(
    x=feature, y="SalePrice", hue="BldgType", col="BldgType",
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);

mi_scores.head(10)

from xgboost import XGBRegressor
def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# Prepare data
df = pd.read_csv("ames.csv")
X = df.copy()
y = X.pop("SalePrice")
X_1 = pd.DataFrame()  # dataframe to hold new features

X_1["LivLotRatio"] = df["GrLivArea"] / df["LotArea"]
X_1["Spaciousness"] = (df["FirstFlrSF"] + df["SecondFlrSF"]) / df["TotRmsAbvGrd"]
X_1["TotalOutsideSF"] = df["WoodDeckSF"] + df["OpenPorchSF"] + df["EnclosedPorch"] + df["Threeseasonporch"] + df["ScreenPorch"]

X_2 = pd.get_dummies(df.BldgType,prefix="Bldg")
X_2 = X_2.mul(df.GrLivArea, axis =0)

X_3 = pd.DataFrame()
X_3["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)

df.MSSubClass.unique()

X_4 = pd.DataFrame()
X_4["MSClass"] = df.MSSubClass.str.split("_",n=1, expand=True)[0]

X_5 = pd.DataFrame()
X_5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
X_new = X.join([X_1, X_2, X_3, X_4, X_5])
score_dataset(X_new, y)

# %%
