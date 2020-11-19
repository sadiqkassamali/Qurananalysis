
# Load CSV (using python)
import csv
import numpy as nu
import pandas as pd

d=pd.read_csv('data\quran.csv')
if (len(d) > 0) :
    print("print total rows count: ")
    print(len(d))
    print("print total data before NaN removal: ")
    print(d)
    d.fillna('', inplace=True)
    print("print total data after NaN removal: " )
    print(d)
    #print(d.head()) #if you want to view first n-rows
    #print(d.tail())  # if you want to view last  n-rows
    #print(d[20:30 + 1])  #slice and dice of rows from 20 : 30
    #print(d.describe()) #Statistics of your dataframe if value is int
