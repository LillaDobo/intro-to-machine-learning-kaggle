# %%
# megértést segítő feladatok
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

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_x, train_y)
melb_preds_forest = forest_model.predict(val_x)
print(mean_absolute_error(val_y,melb_preds_forest))
# %%
# gyakorló feladatok
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

def get_mea(max_leaf_nodes,train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_X,train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y,preds_val)
    return(mae)

candidate_max_leaf_nodes = [5,25,50,100,250,500]
results = []
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    results.append((my_mae,max_leaf_nodes))

best_tree_size = min(results)[1]

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)
final_model.fit(X,y)
print(final_model)

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X,train_y)
rf_pred = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y,rf_pred)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


