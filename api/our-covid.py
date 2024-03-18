import numpy as np #Numerical Manipulation
import pandas as pd #Dataframe Creation, Feature Engineering
import math #Additional Numerical Manipulation
from sklearn.linear_model import LinearRegression #Linear Regression
import os 
import opendatasets as od

#Plotting and Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

dataset = 'https://www.kaggle.com/code/jaysoretto/covid-19-analytics-modeling-visuals'

od.download(dataset)

{"username":"katelync05","key":"d4d4c8303a73e5accb7146b97d8503fe"}