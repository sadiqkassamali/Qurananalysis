
# Load CSV (using python)
import csv
import numpy as nu
import pandas as pd

d=pd.read_csv('data\quran.csv')
if (len(d) > 0) :
    print("print total rows count: ")
    print(len(d))
    #print("print total data before NaN removal: ")
    #print(d)
    #d.fillna('', inplace=True)
    # print("print total data after NaN removal: " )
    #print(d)
    dropped_column = d.drop(d.columns[-1],axis=1)  # Drop last column
    print("print total data After last column removal: ")
    #print(dropped_column)
    #print(dropped_column.head()) #if you want to view first n-rows
    #print(dropped_column.tail())  # if you want to view last  n-rows
    #print(dropped_column[20:30 + 1])  #slice and dice of rows from 20 : 30
    #print(dropped_column.describe()) #Statistics of your dataframe if value is int
print(dropped_column)