import pandas as pd
import os

from sklearn.tree import DecisionTreeRegressor

# save filepath to variable for easier access
melbourne_file_path = os.getcwd() + '/data/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

#choose the price as the prediction target
y = melbourne_data.Price

#choose the features which help to determine the house price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

#print all the basic informations - mentioned earlier in basicDataExploration
print(X.describe())
#print the first rows of the data - visually checking the data is an important job
print(X.head())

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

#making first simple predictions
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

print (melbourne_data.Price)

# Model equals exact -> overfitting. 