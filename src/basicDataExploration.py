import pandas as pd
import os

# save filepath to variable for easier access
melbourne_file_path = os.getcwd() + '/data/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
print(melbourne_data.describe())
#print the amount of rows and the coloumns of the data
print(melbourne_data.shape)
#print the coloumns of the data
print(melbourne_data.columns)
#print the first 5 rows, easy way to figure out, if the data looks like expected
print(melbourne_data.head())
#explore missing data
print(melbourne_data.isnull().sum())
#drops missing values (na = not available..)
melbourne_data = melbourne_data.dropna(axis=0)
