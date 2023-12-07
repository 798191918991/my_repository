#!/usr/bin/env python
# coding: utf-8

# Name:  Vijaya.Rachakonda
# 
# Course:   Online Session-CCE IIT M-DS-AI-BC=223010 B2
#     
# Project Name:  Capstone Project (Walmart)

# ![image-3.png](attachment:image-3.png)![image-2.png](attachment:image-2.png)

# # A.Problem Statement 1:
# A retail store that has multiple outlets across the country are facing issues in managing the
# inventory - to match the demand with respect to supply.
# 

# # Dataset Information:
# 
# The walmart.csv contains 6435 rows and 8 columns.
# 
# Feature Name ----------- Description
# 
# Store ---------- Store number
# 
# Date ------------ Week of Sales
# 
# Weekly_Sales ----------- Sales for the given store in that week
# 
# Holiday_Flag ----------If it is a holiday week
# 
# Temperature ---------------- Temperature on the day of the sale
# 
# Fuel_Price --------------- Cost of the fuel in the region
# 
# CPI --------------------- Consumer Price Index
# 
# Unemployment ----------------- Unemployment Rate

# # 1. You are provided with the weekly sales data for their various outlets. Use statistical analysis, EDA, outlier analysis, and handle the missing values to come up with various insights that can give them a clear perspective on the following:
# 
#        a. If the weekly sales are affected by the unemployment rate, if yes -which stores are suffering the most?
#        
#        b. If the weekly sales show a seasonal trend, when and what could be the reason?
# 
#        c. Does temperature affect the weekly sales in any manner?
#         
#        d. How is the Consumer Price index affecting the weekly sales of various stores?
#         
#        e. Top performing stores according to the historical data.
# 
#        f.The worst performing store, and how significant is the difference between the highest and lowest performing stores.
# 

# # 2. Use predictive modeling techniques to forecast the sales for each store for the next 12weeks.

# # B. Project Objective:
# Project Objective: The objective of this project is to provide useful insights through data analysis and develop predictive models to forecast sales for the retail store over a specific timeframe (X number of months/years).

# *.Problem Statement ---What will we solve?
# 
# We will solve the problem of predicitng future demand for sales of different Walmart stores, based on the provided data by the organsation. We will need to identify what data is helping the models, vs causing worse predicitons and adjust the data accordingly.

# #*. How we will solve it?
# 
# We will use the data provided by the organsation, clean it up so we can use it in our models. Then create demand predictions
# 
# The intended solution will be based on the test.csv dataset for our demand predictions, where our machine learning algortihms will give use results which have taken the different features from the other datasets to give us good predictions of future demand.
# 

# Model Evaluation and Techniques:
# 
#         Metrics:
#             In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in our project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain.
# 
#         Ways to evalute the different models will be the following statistical measures of how accuracy and robustness; Accuracy: Mean Absolute Error (MAE) — Measure the mean absolute errors. The smaller the number is (closer to 0), the better.
# 
#         Mean Squared Error (MSE) — 
#             It measures the average of the squares of the errors. The smaller the number is (closer to 0), the better.
# 
#         Root Mean Squared Error (RMSE) — 
#             the square root of the variance, known as the standard error. The smaller the number is (closer to 0), the better.
# 
#         Robustness: Model robustness refers to the degree that a model's performance changes when using new data versus training data.
# 
#         R-square (R²) — 
#             It is for model robustness. The smaller the number is (closer to 0), the better.

# # Importing neccesary libraries:

# In[265]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib import dates
from datetime import datetime
import sklearn


# # Dataset Walmart.csv Reading:

# In[3]:


walmart=pd.read_csv('walmart dataset.csv')
walmart.head(2)


# In[171]:


walmart.info()


# In[170]:


walmart['Holiday_Flag'].nunique()


# # c. Data Description:
#     The features of the dataset was provided by Walmart :

# # Store:
# The store number — This feature will be used for linking each other feature to the parent store to see if any linkage is find such as size etc.

# # Date:
# the week — The date feature keeps the data in alignment so we compare the correct datapoints to eachother. However, we must note that it is very common that dates are in a string format, which was the case here, where we would change the data type into datetype format (Python Software Foundation, 2002) to handle the complex nature of dates and time.

# # Temperature:
# average temperature in the region — The reason for including the temperature in this analysis is to see if perhaps we can find a postitive or negative relationship between how much people buy depending on how warm it is that week. For example it could be that on extremely hot weeks, people only buy a limited amount of products which relate to simply making handle the warmth itself. Thereby only buying ice cream. However, with the Data Scientist, Machine Learning Engineer or good old fashioned satatistician approach: we never assume, we test and find statistical significant measurments that indicate as such before any assumptions are made.

# # Fuel_Price:
# cost of fuel in the region — This feature of cost of fuel in the region will be tested in a similar sense as the “temperature” feature meentioned above. For example it could be the theoretical case that people don’t go to walmart if the fuel price is getting to high as for some the transportation cost might be more. In a sense, this seems unlikely as it seems like walmart is a store you can travel longer distances as you’ll still “save” money with the overall low prices they provide on their products. Still, this needs to be tested to verify our hypothesis.

# # CPI, the consumer price index —
# The CPI measures the average price trend for the entire private domestic consumption based on prices consumers actually pay. This can potentially be used for identifying trends to assist in the overall predicitons, but we must test this.

# # Unemployment: the unemployment rate —
# The idea of Walmart including the unemployment rate is a nice idea. It could be the case that with fewer people unemployed, the more money they have, therefore more money can be spent on walmarts products. But it could also be the case that with a high unemployment, people tends to go to target as they have a lower pricing strategy than their competitiors where people will buy their essentials at Walmart instead of a more expenisve local stors.

# # d. Data Pre-processing Steps and Inspiration:
# Outline the steps taken to preprocess the data, including cleaning, handling missing values, feature engineering, and any other necessary transformations. Describe the factors that influenced the choice of preprocessing techniques.

# # Task-1:

# 1. You are provided with the weekly sales data for their various outlets. Use statistical
# analysis, EDA, outlier analysis, and handle the missing values to come up with various
# insights that can give them a clear perspective on the following:

# # Statistical Analysis:

# In[4]:


walmart.describe(include='all')


# In[5]:


walmart.describe()


# # EDA:

# In[6]:


from pandas_profiling import ProfileReport
profile = ProfileReport(walmart)
profile


# # Preprocessing of Data:

# # Handle the missing values:
# 
# there were Zero null values.

# In[7]:


walmart.isna().sum()


# # Outlier analysis:

# In[8]:


fig,axis=plt.subplots(4,figsize=(12,7))
S1=walmart[['Temperature','Fuel_Price','CPI','Unemployment']]

for i,column in enumerate(S1):
    sns.boxplot(walmart[column],ax=axis[i], palette='hls',dodge=True,
    fliersize=5,
    linewidth=None,orient='h',
    saturation=0.75,
    width=0.8)
    
import warnings
warnings.filterwarnings('ignore')


# In[9]:


for i in walmart.columns:
    if(walmart[i].dtypes=='int64' or walmart[i].dtypes=='float64'):
        plt.boxplot(walmart[i],showmeans=True)
        plt.xlabel(i)
        plt.ylabel('count')
        plt.show()


# # Dropping the outliers:

# In[10]:


Q1 = walmart.Weekly_Sales.quantile(0.25)
Q3 = walmart.Weekly_Sales.quantile(0.75)
IQR = Q3 - Q1
walmart = walmart[(walmart.Weekly_Sales >= Q1 - 1.5*IQR) & (walmart.Weekly_Sales <= Q3 + 1.5*IQR)]

Q1 = walmart.Holiday_Flag.quantile(0.25)
Q3 = walmart.Holiday_Flag.quantile(0.75)
IQR = Q3 - Q1
walmart = walmart[(walmart.Holiday_Flag >= Q1 - 1.5*IQR) & (walmart.Holiday_Flag <= Q3 + 1.5*IQR)]

Q1 = walmart.Temperature.quantile(0.25)
Q3 = walmart.Temperature.quantile(0.75)
IQR = Q3 - Q1
walmart = walmart[(walmart.Temperature >= Q1 - 1.5*IQR) & (walmart.Temperature <= Q3 + 1.5*IQR)]

Q1 = walmart.CPI.quantile(0.25)
Q3 = walmart.CPI.quantile(0.75)
IQR = Q3 - Q1
walmart = walmart[(walmart.CPI >= Q1 - 1.5*IQR) & (walmart.CPI <= Q3 + 1.5*IQR)]

Q1 = walmart.Unemployment.quantile(0.25)
Q3 = walmart.Unemployment.quantile(0.75)
IQR = Q3 - Q1
walmart = walmart[(walmart.Unemployment >= Q1 - 1.5*IQR) & (walmart.Unemployment <= Q3 + 1.5*IQR)]


# # Checking for outliers: After droping them:

# In[11]:


for i in walmart.columns:
    if(walmart[i].dtypes=='int64' or walmart[i].dtypes=='float64'):
        plt.boxplot(walmart[i],showmeans=True)
        plt.xlabel(i)
        plt.ylabel('count')
        plt.show()


# In[12]:


fig,axis=plt.subplots(4,figsize=(12,7))
S1=walmart[['Temperature','Fuel_Price','CPI','Unemployment']]

for i,column in enumerate(S1):
    ax=sns.violinplot(walmart[column],ax=axis[i],title=True, palette='hls',dodge=True,
    fliersize=5,
    linewidth=None,orient='h',
    saturation=0.75,
    width=0.8)
    #ax.set_xticklabels(walmart[column])
    
import warnings
warnings.filterwarnings('ignore')


# # Cheking Corelation between Atributes in the walmart data:

# In[264]:


plt.figure(figsize = (18,12))
sns.heatmap(walmart.corr(), annot = True)


# # 1. You are provided with the weekly sales data for their various outlets. Use statistical analysis, EDA, outlier analysis, and handle the missing values to come up with various insights that can give them a clear perspective on the following:
#    
#     a. If the weekly sales are affected by the unemployment rate, if yes - which stores are suffering the most?
#     
#     b. If the weekly sales show a seasonal trend, when and what could be the reason?
# 
#     c. Does temperature affect the weekly sales in any manner?
# 
#     d. How is the Consumer Price index affecting the weekly sales of various stores?
# 
#     e. Top performing stores according to the historical data.
# 
#     f. The worst performing store, and how significant is the difference between the highest and lowest performing stores

# # changing the Date formate:

# In[13]:


walmart['Date'] = pd.to_datetime(walmart['Date'], dayfirst=True)
walmart.head(2)


# In[14]:


#Creating year,month and days columns:
walmart['Days']=pd.DatetimeIndex(walmart['Date']).day
walmart['month']=pd.DatetimeIndex(walmart['Date']).month
walmart['year']=pd.DatetimeIndex(walmart['Date']).year
walmart.columns


# In[15]:


walmart.info()


# # changing the data type to int type

# In[16]:


walmart['Fuel_Price']=walmart['Fuel_Price'].astype(int)
walmart['Days']=walmart['Days'].astype(int)
walmart['month']=walmart['month'].astype(int)
walmart['year']=walmart['year'].astype(int)
walmart.info()


# # Distibution of Attributes:

# In[17]:


#histograms distibution of the data with respect to the features.
plt.figure(figsize=(20,15))
walmart.hist()
plt.show()


# In[18]:


#Frequency plot:
plt.figure(figsize=(20,15))
walmart.plot(kind='hist',subplots=True,grid=True,layout=(6,2),fontsize=3)
plt.title('Fequency of the Attributes',fontsize=5)
plt.show()


# In[19]:


fig, axes = plt.subplots(1, 4, figsize=(28,8))

fig.suptitle('Variation of Weekly_Sales')

sns.barplot(ax=axes[0], data=walmart, y='Store', x='Weekly_Sales')
sns.barplot(ax=axes[1], data=walmart, y='CPI', x='Weekly_Sales')

sns.barplot(ax=axes[2], data=walmart, y='Unemployment', x='Weekly_Sales')
sns.barplot(ax=axes[3], data=walmart, y='Temperature', x='Weekly_Sales')


# In[30]:


walmart['Store'].nunique()


# # No.of Unique Values in walmart data:

# In[31]:


print(f"there are {walmart.shape[0]} rows")

print(f"the date is from {min(walmart.Date)} to {max(walmart.Date)}")

for name in walmart.columns:
    print(f"column {name} has {len(set(walmart.loc[:,name]))} unique values")


# # Which store has max Sales?

# In[36]:


total_sales_y=walmart.groupby(['Store','year'])['Weekly_Sales'].sum().sort_values()
print(total_sales_y.head(22))
plt.figure(figsize=(20,5))
sns.barplot(data=walmart,x='Store',y='Weekly_Sales',hue='year')


# # MAX WEEKELY_SALES:
#     
#     STORE - 14      IN  2010
#     
#     STORE - 20      IN  2011
#     
#     STORE - 4      IN   2012
#     
# MIN WEEKLY_SALES:
# 
#     STORE - 33     IN     2010,2011, AND IN 2012.
#     
# There was No Weekely Sales Detailes in the walmart dataset regarding The STORES-12,28,38   in  the year 2010 and 2011

# In[37]:


total_sales=walmart.groupby('Store')['Weekly_Sales'].sum().sort_values()
print('max=',total_sales.max())
print('min  =',total_sales.min())
print('Average=',total_sales.mean())

plt.figure(figsize=(25,7))
total_sales.plot(kind='bar')
plt.xticks(rotation=90,fontsize=25)
plt.ticklabel_format(useOffset=False, style='plain', axis='y')
plt.title('Total sales for each store',fontsize=44,color='red')
plt.xlabel('Store',fontsize=35,color='green')
plt.ylabel('Total Sales',fontsize=35,color='green')


# Store-20 having the highest Sales among all Stores.
# 
# Store-38 having the lowest Sales among all Stores.

# In[38]:


Store20=walmart[walmart['Store']==20]
Sales20_sort=Store20.sort_values('Weekly_Sales')
plt.figure(figsize=(25,5))
Sales20_sort['Weekly_Sales'].plot(kind='bar')

Store20_m=Store20.groupby(['month','Store'])['Weekly_Sales'].agg('sum')


# In[39]:


Store38=walmart[walmart['Store']==38]
Store38.shape
Sales38_sort=Store38.sort_values('Weekly_Sales')

Sales38_sort['Weekly_Sales'].plot(kind='bar')


# In[40]:


fig, axes = plt.subplots(1, 4, figsize=(28,8))

fig.suptitle('Variation of Weekly_Sales in Store-20')

sns.barplot(ax=axes[0], data=Store20, x='Store', y='Weekly_Sales')
sns.barplot(ax=axes[1], data=Store20, x='year', y='Weekly_Sales')

sns.barplot(ax=axes[2], data=Store20, x='month', y='Weekly_Sales')
sns.barplot(ax=axes[3], data=Store20, x='Days', y='Weekly_Sales')


# In[41]:


fig, axes = plt.subplots(1, 4, figsize=(28,8))

fig.suptitle('Variation of Weekly_Sales in Store-38')

sns.barplot(ax=axes[0], data=Store38, x='Store', y='Weekly_Sales')
sns.barplot(ax=axes[1], data=Store38, x='year', y='Weekly_Sales')

sns.barplot(ax=axes[2], data=Store38, x='month', y='Weekly_Sales')
sns.barplot(ax=axes[3], data=Store38, x='Days', y='Weekly_Sales')


# In[42]:


pivot_table =walmart.pivot_table(values=['Weekly_Sales','CPI','Unemployment','Temperature'],
                                  index=['Store','year','month','Days'],                                  
                                  aggfunc=[np.mean,np.sum,min])
pivot_table.head()


# In[43]:


plt.figure(figsize=(25,7))
total_sales_y.plot(kind='bar')


# In[44]:


pivot_table1 =walmart.pivot_table(index='year',values='Store',aggfunc=np.count_nonzero)
print(pivot_table1)



# In[69]:


yearlySales=walmart['year'].value_counts().sort_values()
yearlySales.plot(kind='bar')


# In[70]:


monthlySales=walmart['month'].value_counts().sort_values()
monthlySales.plot(kind='bar')


# In[71]:


walmart['monthlySales']=monthlySales
M_S_Store=walmart.groupby('Store')['monthlySales'].value_counts()
M_S_Store


# In[72]:


DaysSales=walmart['Days'].value_counts().sort_values()
DaysSales.plot(kind='bar')


# In[73]:


# The distribution of the data
col=walmart.columns
for i in col:
    if(walmart[i].dtypes=='int64' or walmart[i].dtypes=='float64'):
        plt.hist(walmart[i])
        plt.xlabel(i)
        plt.ylabel('count')
        plt.show()


# In[74]:


Store=walmart.groupby(['Store','year','month','Days'])['Weekly_Sales'].agg('sum')
Store.plot(kind='hist')
plt.xlabel('Weekly_Sales')
print(Store)


# # Which store has maximum standard deviation? i.e. the sales vary a lot. Also, find out the coefficient of mean to standard deviation.

# In[75]:


walmart_Store_means = pd.DataFrame(walmart.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False))
print('walmart_Stores_means',walmart_Store_means.head(5))
walmart_Store_std = pd.DataFrame(walmart.groupby('Store')['Weekly_Sales'].std().sort_values(ascending=False))
print('walmart_Stores_stds',walmart_Store_std.head(5))
walmart_Store_max = pd.DataFrame(walmart.groupby('Store')['Weekly_Sales'].max().sort_values(ascending=False))
walmart_Store_min = pd.DataFrame(walmart.groupby('Store')['Weekly_Sales'].min().sort_values(ascending=False))
print('walmart_Stores_max',walmart_Store_max.head(5))
print('walmart_Stores_min',walmart_Store_min.head(5))


# In[76]:


walmart_data_std = pd.DataFrame(walmart.groupby('Store')['Weekly_Sales'].std().sort_values(ascending=False))
walmart_data_std.head(1).index[0] , walmart_data_std.head(1).Weekly_Sales[walmart_data_std.head(1).index[0]]


# In[77]:


coeficient=walmart_Store_std/walmart_Store_means
coeficient = coeficient.rename(columns={'Weekly_Sales':'Coefficient of mean to std'})
coeficient=pd.DataFrame(coeficient).sort_values(by='Coefficient of mean to std',ascending=False)
print('max=',coeficient.max())
print('min  =',coeficient.min())
print('Average=',coeficient.mean())
print('std=',coeficient.std())
coeficient


# In[78]:


sns.distplot(walmart[walmart['Store'] == walmart_data_std.head(1).index[0]]['Weekly_Sales'])
plt.title('The Sales Distribution of Store No.'+ str(walmart_data_std.head(1).index[0]))
import warnings
warnings.filterwarnings('ignore')


# In[79]:


#plt.line(walmart['Weekly_Sales'],color='green',edgecolor='red')
plt.hist(walmart['Weekly_Sales'],bins=30,color='green',edgecolor='red')

plt.xticks(rotation=90,fontsize=15)
plt.ticklabel_format(useOffset=False, style='plain', axis='y')
plt.title('Distribution of Sales',fontsize=34,color='red')
plt.xlabel('Weekly_Sales',fontsize=25,color='green')
plt.ylabel('Freequency',fontsize=25,color='green')


# # Provide a monthly and semester view of sales in units and give insights. Year2010:

# In[80]:


plt.figure(figsize=(15,7))
plt.scatter(walmart[walmart.year==2010]['month'],walmart[walmart.year==2010]['Weekly_Sales'])
plt.xlabel("Months",fontsize=25,color='g')
plt.ylabel("Weekly Sales",fontsize=25,color='g')
plt.title("Monthly view of sales in 2010",fontsize=35,color='g')
plt.show()
plt.figure(figsize=(15,7))
plt.bar(walmart[walmart.year==2010]['month'],walmart[walmart.year==2010]['Weekly_Sales'])
plt.xlabel("Months",fontsize=25,color='g')
plt.ylabel("Weekly Sales",fontsize=25,color='g')
plt.title("Monthly view of sales in 2010",fontsize=35,color='g')
plt.show()


# # Year-2011:

# In[81]:


plt.figure(figsize=(15,7))
plt.scatter(walmart[walmart.year==2011]['month'],walmart[walmart.year==2011]['Weekly_Sales'])
plt.xlabel("Months",fontsize=25,color='g')
plt.ylabel("Weekly Sales",fontsize=25,color='g')
plt.title("Monthly view of sales in 2011",fontsize=35,color='g')
plt.show()
plt.figure(figsize=(15,7))
plt.bar(walmart[walmart.year==2011]['month'],walmart[walmart.year==2011]['Weekly_Sales'])
plt.xlabel("Months",fontsize=25,color='g')
plt.ylabel("Weekly Sales",fontsize=25,color='g')
plt.title("Monthly view of sales in 2011",fontsize=35,color='g')
plt.show()


# # Year-2012:

# In[82]:


plt.figure(figsize=(15,7))
plt.scatter(walmart[walmart.year==2012]['month'],walmart[walmart.year==2012]['Weekly_Sales'])
plt.xlabel("Months",fontsize=25,color='g')
plt.ylabel("Weekly Sales",fontsize=25,color='g')
plt.title("Monthly view of sales in 2012",fontsize=35,color='g')
plt.show()
plt.figure(figsize=(15,7))
plt.bar(walmart[walmart.year==2012]['month'],walmart[walmart.year==2012]['Weekly_Sales'])
plt.xlabel("Months",fontsize=25,color='g')
plt.ylabel("Weekly Sales",fontsize=25,color='g')
plt.title("Monthly view of sales in 2012",fontsize=35,color='g')
plt.show()


# In[83]:


walmart


# # a. If the weekly sales are affected by the unemployment rate, if yes - which stores are suffering the most?

# In[84]:


plt.figure(figsize=(16,8))
sns.scatterplot(x=walmart["Unemployment"],y=walmart["Weekly_Sales"])
plt.tight_layout()


# In[85]:


unemData=walmart[['Store','Unemployment','Weekly_Sales']]
unemData.set_index(unemData['Store'])

x=unemData['Weekly_Sales'].sort_values()
y=unemData['Unemployment']


plt.plot(x,y)
plt.xlabel('Weekly_Sales')
plt.ylabel('Unemployment')

a=unemData['Store']
b=unemData['Unemployment']
c=unemData['Weekly_Sales']
plt.subplots(1,1)
plt.plot(a,b)
plt.xlabel('Store')
plt.ylabel('Unemployment')
plt.grid(True)


plt.subplots(1,1)
plt.plot(a,c)
plt.xlabel('Store')
plt.ylabel('Weekly_Sales')


# In[86]:


walmart[['Unemployment','Store']].sort_values('Unemployment').max()


# In[87]:


unemData.groupby(['Store','Unemployment'])['Unemployment'].agg('count')


# In[88]:


unemData.groupby(['Store','Unemployment'])['Unemployment'].agg('count').max()


# In[89]:


Day_df = walmart.groupby("Days")["Weekly_Sales"].mean().to_frame().reset_index()
Month_df = walmart.groupby("month")["Weekly_Sales"].mean().to_frame().reset_index()
Year_df = walmart.groupby("year")["Weekly_Sales"].mean().to_frame().reset_index()
fig , ax = plt.subplots(3,1,figsize = (18,12))
ax1 = sns.lineplot(x=Day_df["Days"],y=Day_df["Weekly_Sales"],linewidth=2.5,ax=ax[0])
ax1.set_ylabel("Mean of Weekly Sales")
ax2 = sns.lineplot(x=Month_df["month"],y=Month_df["Weekly_Sales"],linewidth=2.5,ax=ax[1],color = "r")
ax2.set_ylabel("Mean of Weekly Sales")
ax3 = sns.lineplot(x=Year_df["year"],y=Year_df["Weekly_Sales"],linewidth=2.5,ax=ax[2],color="g")
ax3.set_ylabel("Mean of Weekly Sales")


# In[90]:


walmart[['Store','year','Weekly_Sales']].sort_values(by='Weekly_Sales',ascending=False)


# # Store 19 is Having Higher Weekly Sales.

# In[91]:


unempSales=walmart[['Store','Unemployment','Weekly_Sales']]
y=unempSales['Unemployment']
x=unempSales['Store']
plt.figure(figsize=(10,4))
plt.plot(x,y)
#plt.xticks(GrowthRate2012.index,rotation=90,fontsize=10)
plt.xticks(walmart['Store'].values,rotation=90,fontsize=10)
plt.grid(True)
plt.xlabel('Store Number',color='blue',fontsize=15)
plt.ylabel('Unemployment rate',color='blue',fontsize=15)
plt.title('Unemployment Rate affecting the weekly sales of various stores',color='red',fontsize=15)


#  Store Number 43,38 ,33,29,28,12  are having Higher Unemployment Rate >= 10 because of this effect its Weekly Sales were less.
#  
#  Store 4 , 9, 23 , 40   are having lesser Unemployment rate  <=5. I Might  be Cause  Higher Weekly Sales.

# # Spreading of Weekly Sales and Unemployment in three years form 2010 to 2012:

# In[93]:


x=walmart['Date']
y1=walmart['Weekly_Sales']
y2=walmart['Unemployment']
plt.scatter(x,y1)
plt.scatter(x,y2)
plt.legend(['y1','y2'],loc='best')
plt.xlabel('Date')
plt.xlabel('Weekly_Sales')


# In[94]:


walmart[['Store','Weekly_Sales','Unemployment']].sort_values(by='Unemployment')


# In[234]:


plt.subplots(1,1)
y1=walmart['Unemployment'].sort_values()
y2=walmart['Weekly_Sales'].sort_values()
y3=walmart['Store']

print(y1)
plt.plot(y1,y2)
plt.ylabel('Unemployment Rate')
plt.xlabel('Store')
plt.subplots(1,1)
plt.plot(y3,y2)
plt.ylabel('Weekly_Sales')
plt.xlabel('Store')


# In[96]:


y1=walmart['Temperature'].sort_values()
y2=walmart['Weekly_Sales'].sort_values()
print(y1)
plt.scatter(y1,y2)
plt.xlabel('Temperature')
plt.ylabel('Weekly_Sales')


# In[97]:


walmart.groupby('Unemployment')['Weekly_Sales'].value_counts()


# In[98]:


walmart.groupby('Unemployment')['Weekly_Sales'].agg('count')


# In[221]:


Sales_Unemploy=walmart.groupby(['Unemployment','Store'])['Weekly_Sales'].agg('count')
Sales_Unemploy


# In[100]:


walmart.groupby(['Unemployment','Store'])['Weekly_Sales'].agg(['count','sum'])


# # b. If the weekly sales show a seasonal trend, when and what could be the reason?

# In[101]:


plt.plot(walmart['Weekly_Sales'])
plt.plot(walmart['Date'],walmart['Weekly_Sales'])
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel('Weekly_Sales')


# In[102]:


#plt.plot(walmart['Weekly_Sales'])
plt.plot(walmart['year'].sort_values(),walmart['Weekly_Sales'])
plt.xlabel('year')
plt.xticks(rotation=90)
plt.ylabel('Weekly_Sales')


# In[103]:


plt.subplots(1,1)
y1=walmart['year'].sort_values()
y2=walmart['Weekly_Sales'].sort_values()
y3=walmart['Store']

print(y1)
plt.plot(y1,y2)
plt.xlabel('year')
plt.ylabel('Weekly_Sales')

plt.subplots(1,1)
plt.plot(y3,y2)
plt.xlabel('Store No.')
plt.ylabel('Weekly_Sales')


# # 1) Which store/s has good quarterly growth rate in ’2010 ?

# In[104]:


Q1_sales10=walmart[(walmart['Date']<='2010-03-31')&(walmart['Date']>='2010-01-01')].groupby('Store')['Weekly_Sales'].sum()
Q2_sales10=walmart[(walmart['Date']>='2010-04-01')&(walmart['Date']<='2010-06-30')].groupby('Store')['Weekly_Sales'].sum()
Q3_sales10=walmart[(walmart['Date']>='2010-07-01')&(walmart['Date']<='2010-09-30')].groupby('Store')['Weekly_Sales'].sum()
Q4_sales10=walmart[(walmart['Date']>='2010-10-01')&(walmart['Date']<='2010-12-31')].groupby('Store')['Weekly_Sales'].sum()

Q_Sales10={'Q1_sales':Q1_sales10,'Q2_sales':Q2_sales10,'Q3_sales':Q3_sales10,'Q4_sales':Q4_sales10}
Q_Sales10=pd.DataFrame(Q_Sales10)
Q_Sales10.head(2)
Q_Sales10.index

plt.figure(figsize=(12,5))
plt.plot(Q_Sales10, linewidth=2, markersize=8,marker='*')
plt.legend(Q_Sales10.columns.to_list())
plt.grid(True)
plt.ylabel('Sales in 2012')
plt.xlabel('Srote Number/code')
plt.xticks(Q_Sales10.index,rotation=90)
plt.title('stores and their quarterly growth rate in ’2012')


# In[105]:


Growth_rateQ2 =((Q2_sales10-Q1_sales10)/Q1_sales10*100)/3
Growth_rateQ2
Growth_rateQ3 =((Q3_sales10-Q2_sales10)/Q2_sales10*100)/3
Growth_rateQ3
Growth_rateQ4 =((Q4_sales10-Q3_sales10)/Q3_sales10*100)/3
Growth_rateQ4
GrowthRate2010={'Growth_rateQ2':Growth_rateQ2,'Growth_rateQ3':Growth_rateQ3,'Growth_rateQ4':Growth_rateQ4}
GrowthRate2010=pd.DataFrame(GrowthRate2010)
print(GrowthRate2010.head())

plt.figure(figsize=(12,5))
plt.plot(GrowthRate2010, linewidth=2, markersize=8,marker='*')
plt.legend(GrowthRate2010.columns.to_list())
plt.grid(True)
plt.ylabel('GrowthRate in 2010')
plt.xlabel('Srote Number/code')
plt.xticks(Q_Sales10.index,rotation=90)
plt.title('GrowthRate2010')

plt.figure(figsize=(12,5))                            #Bar graph
GrowthRate2010.plot(kind='bar')
plt.grid(True)
plt.ylabel('Growth Rate in 2010')
plt.xlabel('Srote Number/code')
plt.xticks(GrowthRate2010.index,rotation=90,fontsize=7)
plt.title('stores and their quarterly growth rate in ’2010')



# In[106]:


#Graph  for Q2 and Q3
Growth_rateQ2.plot(ax=Growth_rateQ3.plot(kind='bar'),kind='bar',color='red',alpha=0.2,legend=True)
plt.legend(['Q2_G_rate','Q3_G_rate'])
plt.ylabel('Growth Rate')
plt.grid(True)
plt.title('stores and their quarterly growth rate in ’2010'' of Q2,Q3')
print('min-G_rate-Q3-',Growth_rateQ3.min())
print('max-G_rate-Q3',Growth_rateQ3.max())

print(Growth_rateQ3.sort_values())



# In[107]:


plt.figure(figsize=(25,10))
Growth_rateQ2.sort_values().plot(kind='bar')
plt.title('Growth rate of Quarter2 in2010',color='r',fontsize=32)
plt.ylabel('Growth rate',color='g',fontsize=22)

print('max=',Growth_rateQ2.max())
print('min  =',Growth_rateQ2.min())
print('Average=',Growth_rateQ2.mean())
print('std=',Growth_rateQ2.std())
Growth_rateQ2.head()

print('Here, store-27 and 41 were  which has performed better in the 2nd quarter as compared to the 3rd and 4th quarter than all othe stores.')
print('Here, store-7 which has performed poor in the 2nd quarter as compared to the 3rd and 4th quarter than all othe stores.')

plt.figure(figsize=(25,10))
Growth_rateQ3.sort_values().plot(kind='bar')
plt.title('Growth rate of Quarter3 in 2010',color='r',fontsize=32)
plt.ylabel('Growth rate',color='g',fontsize=22)

print('max=',Growth_rateQ3.max())
print('min  =',Growth_rateQ3.min())
print('Average=',Growth_rateQ3.mean())
print('std=',Growth_rateQ3.std())
Growth_rateQ3.head()


plt.figure(figsize=(25,10))
Growth_rateQ4.sort_values().plot(kind='bar')
plt.title('Growth rate of Quarter4 in2010',color='r',fontsize=32)
plt.ylabel('Growth rate',color='g',fontsize=22)

print('max=',Growth_rateQ4.max())
print('min  =',Growth_rateQ4.min())
print('Average=',Growth_rateQ4.mean())
print('std=',Growth_rateQ4.std())
Growth_rateQ4.head()


# # 2) Which store/s has good quarterly growth rate in ’2011 ?:

# In[108]:


Q1_sales11=walmart[(walmart['Date']<='2011-03-31')&(walmart['Date']>='2011-01-01')].groupby('Store')['Weekly_Sales'].sum()
Q2_sales11=walmart[(walmart['Date']>='2011-04-01')&(walmart['Date']<='2011-06-30')].groupby('Store')['Weekly_Sales'].sum()
Q3_sales11=walmart[(walmart['Date']>='2011-07-01')&(walmart['Date']<='2011-09-30')].groupby('Store')['Weekly_Sales'].sum()
Q4_sales11=walmart[(walmart['Date']>='2011-10-01')&(walmart['Date']<='2011-12-31')].groupby('Store')['Weekly_Sales'].sum()

Q_Sales11={'Q1_sales':Q1_sales11,'Q2_sales':Q2_sales11,'Q3_sales':Q3_sales11,'Q4_sales':Q4_sales11}
Q_Sales11=pd.DataFrame(Q_Sales11)
Q_Sales11.head(2)
Q_Sales11.index

plt.figure(figsize=(12,5))
plt.plot(Q_Sales11, linewidth=2, markersize=8,marker='*')
plt.legend(Q_Sales11.columns.to_list())
plt.grid(True)
plt.ylabel('Sales in 2011')
plt.xlabel('Srote Number/code')
plt.xticks(Q_Sales11.index,rotation=90)
plt.title('stores and their quarterly growth rate in ’2011')


# From the above inference Quartile3 and Quartile2 Sales are having heigher rating in Sales. in Store 4, 20 ,14 and 13.
# 
# From the above inference Quartile3 and Quartile2 Sales are having lesser rating in Sales. in Stores 3,5,33,36,44..etc. in  the year 2011.
# 
# And there is no Sales information in Store 12,28,38.

# In[109]:


plt.figure(figsize=(12,5))
Q2_sales11.plot(ax=Q3_sales11.plot(kind='bar'),kind='bar',color='g',alpha=0.2,legend=True)
plt.legend(['Q2_sales11','Q3_sales11'])
plt.grid(True)
plt.xlabel('Stores number',color='g',fontsize=12)
plt.ylabel('Weekly Sales',color='g',fontsize=12)
plt.title('Q3,Q2 Sales Vs Stores in 2011',color='r',fontsize=18)
print('Store Number 4  is having More Quaterly growth among the all then stores.')


# # Calculating Growth rate of Stores in '2011
# To calculate the growth rate, take the current value and subtract that from the previous value. Next, divide this difference by the previous value and multiply by 100 to get a percentage representation of the rate of growth.

# In[110]:


Growth_rateQ2 =((Q2_sales11-Q1_sales11)/Q1_sales11*100)/3
Growth_rateQ2
Growth_rateQ3 =((Q3_sales11-Q2_sales11)/Q2_sales11*100)/3
Growth_rateQ3
Growth_rateQ4 =((Q4_sales11-Q3_sales11)/Q3_sales11*100)/3
Growth_rateQ4
GrowthRate2011={'Growth_rateQ2':Growth_rateQ2,'Growth_rateQ3':Growth_rateQ3,'Growth_rateQ4':Growth_rateQ4}
GrowthRate2011=pd.DataFrame(GrowthRate2011)
print(GrowthRate2011.head())

plt.figure(figsize=(12,5))
plt.plot(GrowthRate2011, linewidth=2, markersize=8,marker='*')
plt.legend(GrowthRate2011.columns.to_list())
plt.grid(True)
plt.ylabel('GrowthRate in 2011')
plt.xlabel('Srote Number/code')
plt.xticks(GrowthRate2011.index,rotation=90)
plt.title('GrowthRate2011')

#Bar graph
plt.figure(figsize=(12,5))
GrowthRate2011.plot(kind='bar')
plt.grid(True)
plt.ylabel('Growth Rate in 2011')
plt.xlabel('Srote Number/code')
plt.xticks(GrowthRate2011.index,rotation=90,fontsize=7)
plt.title('stores and their quarterly growth rate in ’2011')


# In[111]:


Growth_rateQ2.plot(ax=Growth_rateQ3.plot(kind='bar'),kind='bar',color='red',alpha=0.2,legend=True)
plt.legend(['Q2_G_rate','Q3_G_rate'])
plt.ylabel('Growth Rate')
plt.grid(True)
plt.title('stores and their quarterly growth rate in ’2011'' of Q2,Q3')
print('min-G_rate-Q3-',Growth_rateQ3.min())
print('max-G_rate-Q3',Growth_rateQ3.max())

print(Growth_rateQ3.sort_values().head(5))


# In[112]:


plt.figure(figsize=(25,10))
Growth_rateQ2.sort_values().plot(kind='bar')
plt.title('Growth rate of Quarter2 in 2011',color='r',fontsize=32)
plt.ylabel('Growth rate',color='g',fontsize=22)

print('max=',Growth_rateQ2.max())
print('min  =',Growth_rateQ2.min())
print('Average=',Growth_rateQ2.mean())
print('std=',Growth_rateQ2.std())
Growth_rateQ2.head()


print('Here, store-23 which has performed better in the 2nd quarter as compared to the 3rd and 4th quarter than all othe stores.')

plt.figure(figsize=(25,10))
Growth_rateQ3.sort_values().plot(kind='bar')
plt.title('Growth rate of Quarter3 in2011',color='r',fontsize=32)
plt.ylabel('Growth rate',color='g',fontsize=22)

print('max=',Growth_rateQ3.max())
print('min  =',Growth_rateQ3.min())
print('Average=',Growth_rateQ3.mean())
print('std=',Growth_rateQ3.std())
Growth_rateQ3.head()

plt.figure(figsize=(25,10))
Growth_rateQ4.sort_values().plot(kind='bar')
plt.title('Growth rate of Quarter4 in 2011',color='r',fontsize=32)
plt.ylabel('Growth rate',color='g',fontsize=22)

print('max=',coeficient.max())
print('min  =',coeficient.min())
print('Average=',coeficient.mean())
print('std=',coeficient.std())
coeficient.head()


# # 3) Which store/s has good quarterly growth rate in ’2012 ?

# In[113]:


Q1_sales12=walmart[(walmart['Date']<='2012-03-31')&(walmart['Date']>='2012-01-01')].groupby('Store')['Weekly_Sales'].sum()
Q2_sales12=walmart[(walmart['Date']>='2012-04-01')&(walmart['Date']<='2012-06-30')].groupby('Store')['Weekly_Sales'].sum()
Q3_sales12=walmart[(walmart['Date']>='2012-07-01')&(walmart['Date']<='2012-09-30')].groupby('Store')['Weekly_Sales'].sum()
Q4_sales12=walmart[(walmart['Date']>='2012-10-01')&(walmart['Date']<='2012-12-31')].groupby('Store')['Weekly_Sales'].sum()

Q_Sales12={'Q1_sales':Q1_sales12,'Q2_sales':Q2_sales12,'Q3_sales':Q3_sales12,'Q4_sales':Q4_sales12}
Q_Sales12=pd.DataFrame(Q_Sales12)
Q_Sales12.head(2)
Q_Sales12.index

plt.figure(figsize=(12,5))
plt.plot(Q_Sales12, linewidth=2, markersize=8,marker='*')
plt.legend(Q_Sales12.columns.to_list())
plt.grid(True)
plt.ylabel('Sales in 2012')
plt.xlabel('Srote Number/code')
plt.xticks(Q_Sales12.index,rotation=90)
plt.title('stores and their quarterly growth rate in ’2012')


# From the above inference Quartile3 and Quartile2 Sales are having heigher rating in Sales. in Store 4, 20 and 13
# 
# From the above inference Quartile3 and Quartile2 Sales are having lesser rating in Sales. in Stores3,5,30, 33,36,38,40..etc. in  the year 2012.
# 
# And there is no Sales information in Store 23,40

# In[114]:


plt.figure(figsize=(12,5))
Q2_sales12.plot(ax=Q3_sales12.plot(kind='bar'),kind='bar',color='g',alpha=0.2,legend=True)
plt.legend(['Q2_2012','Q3_2012'])
plt.grid(True)
plt.xlabel('Stores number',color='g',fontsize=12)
plt.ylabel('Weekly Sales',color='g',fontsize=12)
plt.title('Q3,Q2 Sales Vs Stores in 2012',color='r',fontsize=18)
print('Store Number 4  is having More  Quaterly growth among the all then stores. in Q3\n','Store Number 20  is having More  Quaterly growth among the all then stores. in both Q2 and Q3')


# # Calculating Growth rate of Stores in '2012
# To calculate the growth rate, take the current value and subtract that from the previous value. Next, divide this difference by the previous value and multiply by 100 to get a percentage representation of the rate of growth.

# In[115]:


Growth_rateQ2 =((Q2_sales12-Q1_sales12)/Q1_sales12*100)/3
Growth_rateQ2
Growth_rateQ3 =((Q3_sales12-Q2_sales12)/Q2_sales12*100)/3
Growth_rateQ3
Growth_rateQ4 =((Q4_sales12-Q3_sales12)/Q3_sales12*100)/3
Growth_rateQ4
GrowthRate2012={'Growth_rateQ2':Growth_rateQ2,'Growth_rateQ3':Growth_rateQ3,'Growth_rateQ4':Growth_rateQ4}
GrowthRate2012=pd.DataFrame(GrowthRate2012)
print(GrowthRate2012.head())

plt.figure(figsize=(12,5))
plt.plot(GrowthRate2012, linewidth=2, markersize=8,marker='*')
plt.legend(GrowthRate2012.columns.to_list())
plt.grid(True)
plt.ylabel('GrowthRate in 2012')
plt.xlabel('Srote Number/code')
plt.xticks(GrowthRate2012.index,rotation=90)
plt.title('GrowthRate2012')

print('[There is No information about  growth rate in Stores12, 23, 28, 38 and 40 .]')

plt.figure(figsize=(12,5))
GrowthRate2012.plot(kind='bar')
plt.grid(True)
plt.ylabel('Growth Rate in 2012')
plt.xlabel('Srote Number/code')
plt.xticks(GrowthRate2012.index,rotation=90,fontsize=7)
plt.title('stores and their quarterly growth rate in ’2012')


# In[116]:


Growth_rateQ2.plot(ax=Growth_rateQ3.plot(kind='bar'),kind='bar',color='red',alpha=0.2,legend=True)
plt.legend(['Q2_G_rate','Q3_G_rate'])
plt.ylabel('Growth Rate')
plt.grid(True)
plt.title('stores and their quarterly growth rate in ’2012'' of Q2,Q3')
print('min-G_rate-Q3-',Growth_rateQ3.min())
print('max-G_rate-Q3',Growth_rateQ3.max())

print(Growth_rateQ3.sort_values().head(5))


# In[117]:


plt.figure(figsize=(25,10))
Growth_rateQ2.sort_values().plot(kind='bar')
plt.title('Growth rate of Quarter2 in 2012',color='r',fontsize=32)
plt.ylabel('Growth rate',color='g',fontsize=22)

print('max=',Growth_rateQ2.max())
print('min  =',Growth_rateQ2.min())
print('Average=',Growth_rateQ2.mean())
print('std=',Growth_rateQ2.std())
Growth_rateQ2.head()

print('Here, store-23 which has performed better in the 2nd quarter as compared to the 3rd and 4th quarter than all othe stores.')

plt.figure(figsize=(25,10))
Growth_rateQ3.sort_values().plot(kind='bar')
plt.title('Growth rate of Quarter3 in 2012',color='r',fontsize=32)
plt.ylabel('Growth rate',color='g',fontsize=22)

print('max=',Growth_rateQ3.max())
print('min  =',Growth_rateQ3.min())
print('Average=',Growth_rateQ3.mean())
print('std=',Growth_rateQ3.std())
Growth_rateQ3.head()

plt.figure(figsize=(25,10))
Growth_rateQ4.sort_values().plot(kind='bar')
plt.title('Growth rate of Quarter4 in 2012',color='r',fontsize=32)
plt.ylabel('Growth rate',color='g',fontsize=22)

print('max=',coeficient.max())
print('min  =',coeficient.min())
print('Average=',coeficient.mean())
print('std=',coeficient.std())
coeficient.head()


# In[118]:


walmart['monthlySales']=monthlySales
M_S_Store=walmart.groupby('Store')['monthlySales'].value_counts()
M_S_Store


# In[119]:


sns.relplot(
    data=walmart, legend='auto',
    kind='scatter',
    x="Weekly_Sales", y="Store", col=None,palette='hls',
    hue="year", style="Holiday_Flag",
    facet_kws=dict(sharex=True),
)

sns.relplot(
    data=walmart, legend='auto',
    kind='scatter',
    x="Weekly_Sales", y="Store", col=None,palette='hls',
    hue="month", style="Holiday_Flag",
    facet_kws=dict(sharex=False),
)


# In[120]:


w2010=walmart[(walmart['Date']<='2010-12-31')&(walmart['Date']>='2010-01-01')].groupby(['Store','year'])['Weekly_Sales'].sum()
w2011=walmart[(walmart['Date']<='2011-12-31')&(walmart['Date']>='2011-01-01')].groupby(['Store','year'])['Weekly_Sales'].sum()
w2012=walmart[(walmart['Date']<='2012-12-31')&(walmart['Date']>='2012-01-01')].groupby(['Store','year'])['Weekly_Sales'].sum()
x=walmart['Store']


# In[121]:


plt.figure(figsize=(10,4))
w2010.plot(kind='line')
w2011.plot(kind='line')
w2012.plot(kind='line')
plt.xticks(GrowthRate2012.index,rotation=90,fontsize=10)
plt.legend([2010,2011,2012],loc='best')
plt.grid(True)


#     In 2011- Store No 4>13>19 Re the top three Store which are having highest Sales Athan  2010, 2012 years.
#  
#     In 2010   -Store 19 is having the highest Sales than  all stores in 2012.
#     
#     In 2012  Store Numbers 2 and 20  having highest Sales  among all the stores in 2012.

# In[122]:


plt.figure(figsize=(12,5))
plt.plot(Q_Sales11, linewidth=2, markersize=8,marker='*')
plt.legend(Q_Sales11.columns.to_list())
plt.grid(True)
plt.ylabel('Sales in 2011')
plt.xlabel('Srote Number/code')
plt.xticks(Q_Sales11.index,rotation=90)
plt.title('stores and their quarterly growth rate in ’2011')


# In[123]:


plt.subplots(1,1)
y1=walmart['month'].sort_values()
y2=walmart['Weekly_Sales'].sort_values()
y3=walmart['Store']

print(y1)
plt.plot(y1,y2)
plt.xlabel('month')
plt.ylabel('Weekly_Sales')

plt.subplots(1,1)
plt.plot(y3,y2)
plt.xlabel('Store No.')
plt.ylabel('Weekly_Sales')


# In[124]:


plt.subplots(1,1)
y1=walmart['Days'].sort_values()
y2=walmart['Weekly_Sales'].sort_values()
y3=walmart['Store']

print(y1)
plt.plot(y1,y2)
plt.xlabel('Days of month')
plt.ylabel('Weekly_Sales')

plt.subplots(1,1)
plt.plot(y3,y2)
plt.xlabel('Store No.')
plt.ylabel('Weekly_Sales')


# # c.Does temperature affect the weekly sales in any manner?

# In[219]:


print(walmart['Temperature'].value_counts())
print(walmart['Temperature'].sort_values())


# In[125]:


plt.figure(figsize=(16,8))
sns.scatterplot(x=walmart["Temperature"],y=walmart["Weekly_Sales"])
plt.tight_layout()


# In[126]:


tempSales=walmart[['Store','Temperature','Weekly_Sales']]
y=tempSales['Temperature']
x=tempSales['Store']
plt.figure(figsize=(10,4))
plt.plot(x,y)
plt.xticks(GrowthRate2012.index,rotation=90,fontsize=10)
plt.grid(True)
plt.xlabel('Store Number',color='blue',fontsize=15)
plt.ylabel('Temperature',color='blue',fontsize=15)
plt.title('Temperature affecting the weekly sales of various stores',color='red',fontsize=15)


# In[127]:


Sales_temp=walmart[['Store','Weekly_Sales','Temperature']].sort_values(by='Temperature')
print(Sales_temp)
Sales_Temp=walmart[['Store','Weekly_Sales','Temperature']].value_counts()
print(Sales_Temp)


# Store    Weekly_Sales     Temperature
# 40       775910.43         9.51  (Lower Temperature)
# 33       280937.84         100.14  (Higher Temperature)

# # d. How is the Consumer Price index affecting the weekly sales of various stores?

# In[128]:


x=walmart["CPI"]
y=walmart["Weekly_Sales"]
plt.plot(x,y)


# In[129]:


plt.figure(figsize=(16,8))
sns.scatterplot(x=walmart["CPI"],y=walmart["Weekly_Sales"])
plt.tight_layout()


# In[130]:


cpiSales=walmart[['Store','CPI','Weekly_Sales']]
y=cpiSales['CPI']
x=cpiSales['Store']
plt.figure(figsize=(10,4))
plt.plot(x,y)
plt.xticks(GrowthRate2012.index,rotation=90,fontsize=10)
plt.grid(True)
plt.xlabel('Store Number',color='blue',fontsize=15)
plt.ylabel('CPI',color='blue',fontsize=15)
plt.title('CPI affecting the weekly sales of various stores',color='red',fontsize=15)


# In[183]:


print(cpiSales.sort_values(by='CPI'))
cpiSales.sort_values(by='CPI').plot()


# Max   CPI=126.064000     583079.97(weekly sales)
# Min   CPI=227.232807     549731.49(weekly sales)
# 

# In[195]:


print(walmart[walmart['CPI']==227.232807].count())
walmart[walmart['CPI']==227.232807]


# In[196]:


print(walmart[walmart['CPI']==126.064000].count())
walmart[walmart['CPI']==126.064000]


# # There was No effect of CPI on the weekly sales of various stores.

# # Which store has maximum standard deviation? i.e. the sales vary a lot. Also, find out the coefficient of mean to standard deviation.

# In[131]:


walmart_Store_Sales_mean = pd.DataFrame(walmart.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False))
print('walmart_Stores_means',walmart_Store_Sales_mean.head(5))
walmart_Store_Sales_std = pd.DataFrame(walmart.groupby('Store')['Weekly_Sales'].std().sort_values(ascending=False))
print('walmart_Stores_stds',walmart_Store_Sales_std.head(5))
walmart_Store_Sales_max = pd.DataFrame(walmart.groupby('Store')['Weekly_Sales'].max().sort_values(ascending=False))
walmart_Store_Sales_min = pd.DataFrame(walmart.groupby('Store')['Weekly_Sales'].min().sort_values(ascending=False))
print('walmart_Stores_max',walmart_Store_Sales_max.head(5))
print('walmart_Stores_min',walmart_Store_Sales_min.head(5))


# In[132]:


walmart_data_std = pd.DataFrame(walmart.groupby('Store')['Weekly_Sales'].std().sort_values(ascending=False))
walmart_data_std.head(1).index[0] , walmart_data_std.head(1).Weekly_Sales[walmart_data_std.head(1).index[0]]


# In[133]:


coeficient=walmart_Store_std/walmart_Store_means
coeficient = coeficient.rename(columns={'Weekly_Sales':'Coefficient of mean to std'})
coeficient=pd.DataFrame(coeficient).sort_values(by='Coefficient of mean to std',ascending=False)
print('max=',coeficient.max())
print('min  =',coeficient.min())
print('Average=',coeficient.mean())
print('std=',coeficient.std())
coeficient.head()


# In[134]:


sns.distplot(walmart[walmart['Store'] == walmart_data_std.head(1).index[0]]['Weekly_Sales'])
plt.title('The Sales Distribution of Store No.'+ str(walmart_data_std.head(1).index[0]))
import warnings
warnings.filterwarnings('ignore')


# # e. Top performing stores according to the historical data.

# In[136]:


walmart[['Store','year','Weekly_Sales']].sort_values(by='Weekly_Sales',ascending=False)


# # Top performing stores according to the historical data. is  Store-19	 in 2010	 with 2678206.42 Weekly Sales.

# # f. The worst performing store, and how significant is the difference between the highest and lowest performing stores.
# 

# In[137]:


per=walmart[['Store','year','Weekly_Sales']].sort_values(by='Weekly_Sales',ascending=True)
per=pd.DataFrame(per)
print('Store Number _33 is the Worst performing store among all stores')
per


# # Worst performing stores according to the historical data. is Store-33 in 2010 with 209986.25 Weekly Sales.

# In[138]:


2678206.42-209986.25


# In[139]:


differnce=walmart[['Weekly_Sales','Store']].sort_values(by='Weekly_Sales',ascending=False).max()-walmart[['Weekly_Sales','Store']].sort_values(by='Weekly_Sales',ascending=False).min()
differnce


# # The Significant  difference between the highest and lowest performing stores is nearly equal to the Total Weekly Sales of the Stroe 44.

# # Insights Form Task-1:
# A.
# 1. Store Number 43,38 ,33,29,28,12 are having Higher Unemployment Rate >=10 because of this effect its Weekly Sales were less.
# 2. Store 4 , 9, 23 , 40 are having lesser Unemployment rate <=5. I Might be Cause Higher Weekly Sales.
# 3. Store 19 is Having Higher Weekly Sales.
# 
# 
# B.
# 1. 4th month of the year has Max Sales and 1st month of the year is has min Sales. 
# 2. Monthly Higher Sales:
#     In 2010 and in 2011 -- December month
#     But in 2012 --- June month
#     Here, overall monthly sales are higher in the month of December while the yearly sales in the year 2011 are the highest.
# 3. Monthly Higher Orders:
#         In 2010 --Aplril,July,October and December months
#         in 2011 -- April,July,September and  December months
#         But in 2012 --- March, june  and August months
#     Monthly Lesser Orders:
#         In 2010 --January,February,March,May,June,August,September and November months
#         in 2011 -- February,March,May,June,August ,October and November months
#         But in 2012 ---January,February,April, May.July,September  and October months
# 3.  Continuous 4 days of the  every week is  having the Heigher sales in a month. 
# 
# 4.  In 2011- Store No 4>13>19 Re the top three Store which are having highest Sales than  2010, 2012 years.
# 
# 5.  In 2010   -Store 19 is having the highest Sales than  all stores in 2012.
# 
# 6.  In 2012  Store Numbers 2 and 20  having highest Sales  among all the stores in 2012.
# 
# C.
#      Store    Weekly_Sales     Temperature
#         40       775910.43         9.51  (Lower Temperature)
#         33       280937.84         100.14  (Higher Temperature)
# 
# D. There was No effect of CPI on the weekly sales of various stores.
# 
# E. Top performing stores according to the historical data. is Store-19 in 2010 with 2678206.42 Weekly Sales.
# 
# F.  Worst performing stores according to the historical data. is Store-33 in 2010 with 209986.25 Weekly Sales.
# 
# The Significant difference between the highest and lowest performing stores is nearly equal to the Total Weekly Sales of the Stroe 44.

# # Task  -2:

# # 2. Use predictive modeling techniques to forecast the sales for each store for the next 12weeks.

# # What is forecasting?
# According to the PMBOK guide, “a forecast is an estimate or prediction of conditions and events in the project’s future, based on information and knowledge available at the time of the forecast.”
# 
# We may use forecasts in various situations. For example, in finance, companies use financial forecasting to project employee’s wages or set the annual budget. On the other hand, in stock trading and investing, forecasting is used to predict the future market price and performance.
# 
# forecasting can help business analysts study the impact of certain changes in the working environment (such as adjusting business hours). Another type of forecasting is weather forecasting, which predicts future atmospheric changes for a certain area and time, or changes on the Earth surface, based on meteorological observations.
# 
# In this Project we going to Forcast the futur Weekly sales in next 12 weeks in each Store.
# 
# Top benefits of forecasting Let’s see how forecasting can help your business succeed:
# 
# Forecasting helps in setting goals and plans ahead of time — Analyzing data and statistics helps businesses better evaluate their progress and adapt business operations accordingly. Forecasting helps in allocating a business budget — A forecast will give you estimates about the amount of revenue or income that is expected in a future period. This, in turn, helps companies get insight into where to allocate their budget. Forecasting helps in predicting market changes — Data and projections help companies make better adjustments to their strategies and improve operations in order to meet current market trends. This, in turn, helps them stand out from the competition.
# 
# To sum up, forecasting is an absolute necessity for any business because it helps you:
# 
#        1.Plan for both short and long term future,
#    
#        2.Invest your money wisely,
#    
#        3.Expand into new markets,
#    
#        4.Use real-time data,
#    
#        5.Improve collaboration between team leaders, and most importantly,
#    
#        6.Plan the next steps for your business.
# 
# There are various tools that help businesses get better insight into how operations and processes currently work, and find out what needs to be changed or improved. We will mention a few forecasting tools below

# # 1.Data Preparation:
# Load the 'Walmart.csv' dataset and ensure the 'Date' column is in a datetime format. If not, convert it using the pd.to_datetime function.

# In[235]:


walmart['Date'] = pd.to_datetime(walmart['Date'], dayfirst=True)
walmart['Date']


# In[236]:


walmart['Date'] = pd.to_datetime(walmart['Date'])


# # 2.Time Series Analysis:
# Analyze the time series data to identify patterns, such as trends and seasonality, which can help in selecting appropriate forecasting methods. Plot the sales data over time and check for any obvious patterns.

# In[237]:


plt.figure(figsize=(12, 6))
plt.plot(walmart['Date'], walmart['Weekly_Sales'])
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales Over Time')
plt.show()


# In[238]:


plt.figure(figsize=(12, 6))
plt.plot(walmart['Store'], walmart['Weekly_Sales'])
plt.xlabel('Store')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales Over Store')
plt.show()


# # 3.Data Preprocessing:
# Ensure the time series data is in a suitable format for forecasting. Set the 'Date' column as the index and sort the data in ascending order based on dates.

# In[239]:


walmart = walmart.set_index(walmart['Date']).sort_index(ascending=True)
walmart


# In[240]:


walmart['Date'].unique()


# # 4.Aggregating Sales Data:
# If the data has multiple records per week, such as sales for different stores, you'll need to aggregate the sales at the weekly level. Use the 'resample' method to convert the data to a weekly frequency and sum the sales values.

# In[241]:


walmart_weekly = walmart['Weekly_Sales'].resample('W').sum()
walmart_weekly


# In[242]:


walmart_weekly.plot(kind='line')


# In[243]:


walmart_weekly.describe()


# # 5.Train-Test Split:
# Split the dataset into training and testing sets, keeping the last 12 weeks (3 months) for validation.

# In[244]:


walmart.shape


# In[245]:


train_data = walmart_weekly[:-12]
test_data = walmart_weekly[-12:]
print('train data\n',train_data)
print('\ntrain_data shpe  :',train_data.shape)
print('\ntest data\n',test_data)
print('\ntest_data shape   :',test_data.shape)


# # 6.Model Selection and Training:
# Model Selection and Training: Choose a suitable time series forecasting model, such as SARIMA, Prophet, or an exponential smoothing method. Fit the selected model to the training data.Choose a suitable time series forecasting model, such as SARIMA, Prophet, or an exponential smoothing method. Fit the selected model to the training data.

# # Sarimax Model:

# In[246]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
# Fit the SARIMAX model
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False)
model_fit = model.fit()


# # 7.Forecasting:using Sarimax
# Use the trained model to make forecasts for the next 12 weeks.

# In[247]:


forecast = model_fit.forecast(steps=12)
forecast


# In[248]:


# Generate predictions for the test set
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=True)

# Plot the actual and predicted values
walmart['predict'] = predictions
walmart[['Weekly_Sales', 'predict']].plot()


# In[249]:


error=test_data-predictions
error.plot()
print(error)


# # 8.Evaluation and Visualization:
# Compare the forecasted values with the actual sales data from the test set. Calculate evaluation metrics such as mean absolute error (MAE) or root mean squared error (RMSE) to assess the accuracy of the model.

# In[250]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math 
from math import sqrt
forecasted_values = forecast.values
mae = mean_absolute_error(test_data, forecasted_values)
print("Mean Absolute Error (MAE):", mae)

MSE = mean_squared_error(test_data, forecasted_values)
print("mean squared error (MSE):", MSE)
rmse = sqrt(MSE)
print('Root mean Squred error   :',rmse)


# In[251]:


plt.figure(figsize=(12, 6))
plt.plot(walmart_weekly.index, walmart_weekly, label='Actual Sales')
plt.plot(test_data.index, forecasted_values, label='Forecasted Sales')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('Actual vs. Forecasted Sales')
plt.legend()
plt.show()


# In[252]:


# scatter plot between observed and predicted values of weekly sales from KNN regressor
plt.figure(figsize = (12,6))
sns.scatterplot(x = test_data, y = forecasted_values,markers=True)
sns.lineplot(x = test_data, y = forecasted_values,markers=True)


# # 9.Sales Forecasting: using Arima model
# Use the trained model to make future sales forecasts for each store over the next 12 weeks.

# In[253]:


future_forecast = model_fit.forecast(steps=12)
future_forecast


# In[254]:


from statsmodels.tsa.arima.model import ARIMA


# In[255]:


train_data = walmart_weekly[:-12]
test_data = walmart_weekly[-12:]
print('train data\n',train_data)
print('\ntest data\n',test_data)


# In[256]:


# ARIMA ---> AR + MA +I ---> ARIMA--> 3= AR , I=0 , MA=3
model = ARIMA(train_data, order=(1, 1, 1))
model_fit = model.fit() # training 
model_fit.summary()


# In[257]:


walmart['predict'] = model_fit.predict(start= len(train_data), 
                                    end=len(train_data)+len(test_data)- 1, 
                                    dynamic=True)
walmart[['Weekly_Sales','predict']].plot()


# In[258]:


plt.figure(figsize=(12, 6))
plt.plot(walmart_weekly.index, walmart_weekly, label='Actual Sales')
plt.plot(test_data.index, forecasted_values, label='Forecasted Sales')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('Actual vs. Forecasted Sales')
plt.legend()
plt.show()

# Fit the SARIMAX model
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False)
model_fit = model.fit()

# Generate predictions for the test set
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=True)

# Plot the actual and predicted values
walmart['predict'] = predictions
walmart[['Weekly_Sales', 'predict']].plot()

# Forecast future values
forecast = model_fit.forecast(steps=12)

# Plot the forecasted values
walmart['Weekly_Sales'].plot()
plt.plot(forecast)

plt.show()

# # Preprocessing the Data:
# 
# 1.Ensure that the 'Date' column is in datetime format.
# 
# 2.Group the data by store and date to aggregate sales data for each store on a weekly basis.

# In[259]:


# Group data by store and date to get weekly sales for each store
weekly_sales = walmart.groupby(['Store', pd.Grouper(key='Date', freq='W')])['Weekly_Sales'].sum().reset_index()
weekly_sales


# # Forecasting Weekly Sales for Each Store:
# 
# 1.Iterate over each store and perform the forecast using SARIMAX or any other suitable time series forecasting model.
# 
# 2.Generate forecasts for a specific number of future time steps.

# In[260]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Iterate over each store and perform the forecast
forecasts = []
for store in weekly_sales['Store'].unique():
    store_data = weekly_sales[weekly_sales['Store'] == store]
    
    # Fit SARIMAX model
    model = SARIMAX(store_data['Weekly_Sales'], order=(1, 0, 2), seasonal_order=(1, 0, 2, 12), enforce_stationarity=False)
    model_fit = model.fit()
    
    # Forecast future values
    forecast = model_fit.forecast(steps=12)  # Change the number of steps as needed
    
    # Append the forecast to the list of forecasts
    forecasts.append(forecast)


# # Visualizing the Forecasts:
# 
# Plot the forecasts for each store to visualize the predicted sales trends.

# In[262]:


# Display the forecasts for the current store
print(f"Store {store} - Weekly Sales Forecasts:")
for j, forecast in enumerate(forecasts):
    print(f"Week {j+1}: {forecast}")
print()


# In[263]:


# Loop through all stores
for i, store in enumerate(weekly_sales['Store'].unique()):
    store_data = weekly_sales[weekly_sales['Store'] == store]
    
    # Get the corresponding forecasts for the current store
    store_forecasts = forecasts[i]
    
    # Align the dimensions of the Date and forecasts arrays
    date_range = store_data['Date'].iloc[-len(store_forecasts):].reset_index(drop=True)  # Match the length of forecasts
    
    plt.plot(date_range, store_data['Weekly_Sales'].values[-len(store_forecasts):], label=f"Store {store} - Actual")
    plt.plot(date_range, store_forecasts, label=f"Store {store} - Forecast")
    plt.xlabel('Date', rotation=0)
    plt.ylabel('Weekly Sales')
    plt.title(f"Weekly Sales Forecast - Store {store}")
    plt.legend()
    
    plt.show()
    
    # Display the forecasts for the current store
    print(f"Store {store} - Weekly Sales Forecasts:")
    for j, forecast in enumerate(store_forecasts):
        print(f"Week {j+1}: {forecast}")
    print()


# # h. Model Evaluation and Techniques:
# When evaluating a model for a Walmart project, there are several techniques and metrics that can be used to assess its performance and effectiveness. Here are some commonly used techniques and evaluation metrics:
# 
# Evaluation Metrics:I have Choosen appropriate evaluation metrics based on the nature of the problem you are trying to solve. Some common metrics for model evaluation include: Mean Absolute Error (MAE): The average absolute difference between the predicted and actual values. Mean Squared Error (MSE): The average squared difference between the predicted and actual values. It emphasizes larger errors more than MAE. Root Mean Squared Error (RMSE): The square root of MSE. It is in the same unit as the target variable and is easier to interpret. Accuracy: The proportion of correctly classified instances.

# i. Inferences from the Same:
# From the Above Used tool and Techniques like preprocess the data, including cleaning, handling missing values, feature engineering, and any other necessary transformations.
# 
# the Weekly Sales Data is Continuous data nad it is Non Stationary data We can get an Inference from the Data set is :
# 
# 1. Clearly, from the above graph, it is visible that the store which has maximum sales is store number 19 and 20 and the store which has minimum sales is the store number 33.
# 2. The store number 37 store which has min coefficient of mean to standard deviation .
# 3. The store number 35 store which has maximum coefficient of mean to standard deviation .
# 4. Store Number 4  is having More Quaterly growth among the all then stores.
# 5. Here, store-23 which has performed better in the 2nd quarter as compared to the 3rd and 4th quarter than all othe stores.
# 6. From the above inference Quartile3 and Quartile2 Sales are having heigher rating in Sales.
# 7.  4th month of the year has Max Sales and 1st month of the year is has min Sales. 
# 8. Monthly Higher Sales:
#     In 2010 and in 2011 -- December month
#     But in 2012 --- June month
#     Here, overall monthly sales are higher in the month of December while the yearly sales in the year 2011 are the highest.
# 9. Monthly Higher Orders:
#         In 2010 --Aplril,July,October and December months
#         in 2011 -- April,July,September and  December months
#         But in 2012 --- March, june  and August months
#     Monthly Lesser Orders:
#         In 2010 --January,February,March,May,June,August,September and November months
#         in 2011 -- February,March,May,June,August ,October and November months
#         But in 2012 ---January,February,April, May.July,September  and October months
# 10.  Continuous 4 days of the  every week is  having the Heigher sales in a month.    
# 11. From the results we observe that most orders are made in the UK and customers from UK spend the highest amount of money in their purchases.
# 
# Summarize the key insights obtained from the predictive models and their implications for inventory management and sales forecasting. Discuss any patterns or trends discovered in the data that can aid decision-making.

# # j.Future Possibilities of the Project:
# Based on the provided two datasets: 'Walmart.csv' and 'OnlineRetail.csv'. Let's explore some future possibilities for each dataset:
# 
# Walmart Dataset:
# 
# *. Forecasting Sales: You can use time series analysis techniques to predict future sales based on historical data. This can help identify trends, seasonality, and other factors that affect Walmart's sales. Models like ARIMA, SARIMA, or Prophet can be applied to generate sales forecasts.
# 
# *. Customer Segmentation: Analyzing customer behavior and segmenting them based on various factors such as demographics, purchase history, or shopping patterns can provide insights for targeted marketing strategies and personalized recommendations.
# 
# *. Inventory Optimization: Utilize data to optimize inventory management by predicting demand for different products, identifying stockouts, and optimizing replenishment strategies.
# 
# *. Pricing Optimization: Analyze pricing data and historical sales to optimize pricing strategies, such as dynamic pricing or pricing elasticity models, to maximize revenue and profitability.
# 
# Online Retail Dataset:
# 
# *.Customer Lifetime Value (CLV) Prediction: Analyze customer behavior, purchase history, and demographic data to estimate the CLV for each customer. This information can help in customer segmentation, personalized marketing campaigns, and customer retention strategies.
# 
# *.Market Basket Analysis: Identify frequently co-occurring items in customers' shopping baskets to understand buying patterns and optimize product recommendations or cross-selling strategies.
# 
# *.Churn Prediction: Utilize historical data to predict customer churn, i.e., identify customers who are likely to stop using the online retail platform. This information can guide targeted retention efforts and proactive customer engagement.
# 
# *.Fraud Detection: Apply anomaly detection or classification techniques to identify fraudulent transactions or suspicious activities, helping to protect the business and enhance security measures.
# 
# These are just a few examples of the future possibilities for each dataset. The actual opportunities and insights depend on the specific characteristics of the data, business goals Highlight potential areas for further improvement or expansion of the project. Discuss possible enhancements to the predictive models, additional data sources, or advanced techniques that could be applied to enhance inventory management and sales forecasting in the future.
# 
# Based on relevant tables, charts, and visualizations throughout the report to support your findings.

# # FUTURE ENHANCEMENT
# For future enhancements in the project using the Walmart dataset and the Online Retail dataset, here are some potential areas to consider:
# 
# 1.Advanced Predictive Modeling: Explore more advanced predictive modeling techniques, such as deep learning models (e.g., neural networks) or ensemble methods, to improve the accuracy and robustness of your predictions. These models can capture complex relationships and patterns in the data, potentially leading to better forecasting or classification results.
# 
# 2.Incorporate External Data: Augment the existing datasets with relevant external data sources. For example, you could include economic indicators, weather data, or demographic information that may impact sales or customer behavior. By incorporating such data, you can potentially improve the accuracy of predictions and gain additional insights.
# 
# 3.Real-Time Analytics: Develop real-time analytics capabilities to monitor and analyze data as it becomes available. This can enable you to detect and respond to emerging trends, anomalies, or customer behavior changes promptly. Real-time analytics can be particularly valuable in dynamic retail environments, allowing you to make timely business decisions.
# 
# 4.Personalization and Recommendation Systems: Implement personalized recommendation systems based on customer preferences, browsing history, and purchase patterns. Utilize techniques like collaborative filtering, content-based filtering, or hybrid approaches to provide targeted product recommendations, thereby enhancing the customer experience and increasing sales.
# 
# 5.Supply Chain Optimization: Explore optimization techniques to enhance supply chain management. This could involve optimizing inventory levels, demand forecasting, logistics, and distribution, as well as integrating with suppliers and improving overall operational efficiency.
# 
# 6.Data Visualization and Dashboards: Develop interactive dashboards and visualizations to provide a comprehensive and intuitive overview of key metrics, trends, and insights. This allows stakeholders to explore the data, gain actionable insights, and make informed decisions based on visualized information.
# 
# 7.Customer Sentiment Analysis: Apply natural language processing (NLP) techniques to analyze customer reviews, feedback, or social media data related to Walmart or the online retail platform. This can provide insights into customer sentiment, opinions, and preferences, enabling you to address any issues, improve customer satisfaction, and adapt your strategies accordingly.
# 
# 8.Integration with Other Systems: Integrate the analytics and insights from the project with other business systems such as CRM (Customer Relationship Management), ERP (Enterprise Resource Planning), or marketing automation tools. This integration can provide a more holistic view of the business operations, enabling effective decision-making and coordinated actions.
# 
# Remember to prioritize enhancements based on business needs, available resources, and the potential impact on improving key performance metrics. Regularly evaluate the project's outcomes, gather feedback, and iterate on the solutions to continuously enhance the effectiveness and value of the project.

# # CONCLUSION
# In conclusion, Wal-Mart is the number one retailer within the USA and it too works in numerous other nations all around the world and is moving into unused nations as a long time pass by. There, are other companies who are continually rising as well and would donate Walmarta extreme competition within the future in case Walmart does not remain to the best of their amusement. In order to do so, the individuals will have to be get it their commerce patterns, the client needs and oversee the assets shrewdly. In this time when the innovations are coming to out to unused levels, Enormous Information is taking over the conventional strategy of overseeing and analyzing information. These advances are always utilized to get it complex datasets in a matter of time with lovely visual representations. 
# 
# The main purpose of this study was to predictWalmart’s sales based on the available historic data and identify whether factors like temperature, unemployment, fuel prices, etc
# affect the weekly sales of particular stores under study.
# 
# Pertaining to the specific factors provided in the study (temperature, unemployment, CPI, and fuel price), it was observed that sales do tend to go up slightly during
# favorable climate conditions as well as when the prices of fuel are adequate. However, it
# is difficult to make a strong claim about this assumption considering the limited scope
# of the training dataset provided as part of this study. By the observations in the exploratory data analysis, sales also tend to be relatively higher when the unemployment level is lower. Additionally, with the dataset provided for this study, there does not seem to be a relationship between sales and the CPI index. Again, it is hard to make a
# substantial claim about these findings without the presence of a larger training dataset
# with additional information available.
# 
# Interaction effects were studied as part of the linear regression model to identify if
# a combination of different factors could influence the weekly sales for Walmart.

# # -------------------------------Completed-------------------------------------

# In[ ]:





# In[ ]:




