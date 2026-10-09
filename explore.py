# %%
import pandas as pd
home_data = pd.read_csv('train.csv')
home_data.describe()

home_data.columns
y = home_data.SalePrice

feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = home_data[feature_names]

print(X.describe())
print(X.head())

from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state=6)
iowa_model.fit(X,y)

predictions = iowa_model.predict(X)
print(predictions)
# %%
print(house_data.describe())

avg_lot_size = round(house_data.LotArea.mean(),0)
newest_home_age = 2026 - house_data.YearBuilt.max()
# %%
import pandas as pd
melbourne_data = pd.read_csv('melb_data.csv')

melbourne_data.describe()
melbourne_data.columns
melbourne_data.dropna(axis=0)

y = melbourne_data.Price
melbourne_features = ['Rooms','Bathroom','Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]
x.describe()
x.head()

from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(x,y)

print("Making predictions for the following 5 houses: ")
print(x.head())
print("The predictions are  ")
print(melbourne_model.predict(x.head()))
# %%
