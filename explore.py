# %%

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


from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(x)
mean_absolute_error(y,predicted_home_prices)

from sklearn.model_selection import train_test_split
train_x,val_x, train_y,val_y = train_test_split(x,y, random_state=0)
melbourne_model1 = DecisionTreeRegressor()
melbourne_model1.fit(train_x,train_y)
val_predictions = melbourne_model1.predict(val_x)
print(mean_absolute_error(val_y,val_predictions))

def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y,preds_val)
    return(mae)
for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes, train_x,val_x,train_y, val_y)
    print("My leaf nodes: %d \t\t Mean Absolut Error:  %d" %(max_leaf_nodes,my_mae))
# %%
import pandas as pd
home_data = pd.read_csv('train.csv')
home_data.describe()

print(home_data.describe())

avg_lot_size = round(home_data.LotArea.mean(),0)
newest_home_age = 2026 - home_data.YearBuilt.max()
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

from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)

iowa_model1 = DecisionTreeRegressor(random_state=1)
iowa_model1.fit(train_X,train_y)
val_predictions = iowa_model.predict(val_X)

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y,val_predictions)

print(val_mae)