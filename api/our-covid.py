import numpy as np #Numerical Manipulation
import pandas as pd #Dataframe Creation, Feature Engineering
import math #Additional Numerical Manipulation
from sklearn.linear_model import LinearRegression #Linear Regression

#Plotting and Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/kaggle/input/covid19-global-dataset/worldometer_coronavirus_daily_data.csv')
data_summary = pd.read_csv('/kaggle/input/covid19-global-dataset/worldometer_coronavirus_summary_data.csv')

data.describe()
data_summary.describe()
data.drop(['date','country'],axis=1).corr()
data_summary.drop(['country','continent'],axis=1).corr()

ad_metrics = pd.DataFrame()

ad_metrics["country"] = data_summary["country"]
ad_metrics["non_critical_cases"] = data_summary['total_confirmed'] - data_summary['serious_or_critical']
ad_metrics["test_confirmed_ratio"] = data_summary["total_tests"] / data_summary['total_confirmed']
ad_metrics['survival_ratio'] = data_summary['total_recovered'] / data_summary['total_deaths']

ad_metrics = ad_metrics.set_index('country')
ad_metrics = ad_metrics.replace({np.nan:0})

ad_metrics

#Creating the global dataframe
mask = data['country'] != 'USA'
global_data = data[mask]

#Creating the us dataframe
mask = data['country'] == 'USA' 
us_data = data[mask]

#Reindexing for the US dataset
index = np.arange(0,len(us_data),1)
us_data = us_data.set_index(index)

#Reindexing for the global dataset
index = np.arange(0,len(global_data),1)
global_data = global_data.set_index(index)

global_data['active_cases'] = global_data['active_cases'].fillna(0)

#Using linear regression to create a trend line of active cases
X = np.arange(0,len(us_data),1).reshape(-1,1)
y = us_data['active_cases']
model = LinearRegression().fit(X,y)
y_pred = model.predict(X)

#Creating the plot
plt.figure(figsize=(8,5))
#plt.plot(us_data['cumulative_total_cases'],label="Cumulative Total Cases") 
#plt.plot(us_data['daily_new_cases'],label="Daily New Cases")
plt.plot(us_data['active_cases'],label="Active Cases", color='steelblue')
plt.plot(y_pred,label="Active Cases (Trendline)",color='steelblue',linestyle='dashed')
plt.plot(us_data['cumulative_total_deaths'],label="Deaths", color='black')
plt.xlabel('Days')
plt.ylabel('Cases (in millions)')
plt.title("US Covid Data [2/15/2020 - 5/14/2022]")
plt.axvline(x=300,color='r',label='Vaccine Released',linestyle='dashed')
plt.axvline(x=583,color='y',label='Booster #1',linestyle='dashed')
plt.axvline(x=765,color='g',label='Booster #2',linestyle='dashed')
plt.legend()
plt.show()

#mask = data_summary['country'] != "USA"
#heatmap = data_summary[mask]
heatmap = data_summary
heatmap = heatmap.set_index('country')
heatmap = heatmap.sort_values('population',ascending=False)[0:9]
heatmap = heatmap.drop(['continent','population','total_tests'],axis=1)
heatmap = heatmap * .000001
plt.xlabel('Category')
plt.ylabel('Country')
plt.title('Global Heatmap [Scaled by 1e^6, descending by pop.]')
sns.heatmap(heatmap, cmap='YlGnBu',annot=True,linewidths=2)