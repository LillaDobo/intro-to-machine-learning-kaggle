# %%
import pandas as pd
melbourne_data = pd.read_csv('melb_data.csv')

melbourne_data.describe()
# %%
house_data = pd.read_csv('train.csv')
house_data.describe()
# %%
print(house_data.describe())

avg_lot_size = round(house_data.LotArea.mean(),0)
newest_home_age = 2026 - house_data.YearBuilt.max()
# %%