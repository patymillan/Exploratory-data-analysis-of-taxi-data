
# coding: utf-8

# In[9]:


# #Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels


from sklearn import linear_model

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import time
import requests
import datetime

import warnings
warnings.filterwarnings('ignore')


# In this notebook, I will explore data on New York City Green Taxi of september 2015. I have focused only on the exploratory analysis of the data

# In[63]:


#Train and Test Datasets
df_train = pd.read_csv("C:/Users/Sreekanth/Desktop/NYC trip data/train.csv")
df_test = pd.read_csv("C:/Users/Sreekanth/Desktop/NYC trip data/test.csv")


# 
# Data Dictionary
# 
# Dataset: train.csv
# 
#     id - a unique identifier for each trip
#     vendor_id - a code indicating the provider associated with the trip record
#     pickup_datetime - date and time when the meter was engaged
#     dropoff_datetime - date and time when the meter was disengaged
#     passenger_count - the number of passengers in the vehicle (driver entered value)
#     pickup_longitude - the longitude where the meter was engaged
#     pickup_latitude - the latitude where the meter was engaged
#     dropoff_longitude - the longitude where the meter was disengaged
#     dropoff_latitude - the latitude where the meter was disengaged
#     store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
#     trip_duration - duration of the trip in seconds
# 
# Dataset: test.csv
# 
#     The train dataset contains 11 columns and the test dataset contains 9 columns. The two additional columns that are present in the train dataset, and not in the test dataset are dropoff_datetime and trip_duration.
# 
# 
# Dataset Overview:
# 
# Training Dataset
# 

# In[64]:


print("Total number of samples in train dataset: ", df_train.shape[0])
print("Number of columns in train dataset: ", df_train.shape[1])


# In[65]:


df_train.head()


# In[69]:


df_train.describe()


# In[70]:


df_train.info()


# Test Dataset

# In[71]:


print("Total number of samples in test dataset: ", df_test.shape[0])
print("Number of columns in test dataset: ", df_test.shape[1])


# In[72]:


df_test.head()


# 
# Missing Values:
# 
# Are there any missing values in the train and test datasets?
# 

# In[73]:


df_train.isnull().sum()


# In[74]:


df_test.isnull().sum()


# Luckily, there are no missing values in the train and test datasets
# 
# Are all the id's in the train and test datasets unique? Is there an overlap, in the observations, between the train and test datasets?

# In[75]:


print("Number of ids in the train dataset: ", len(df_train["id"]))
print("Number of unique ids in the train dataset: ", len(pd.unique(df_train["id"])), "\n")

print("Number of ids in the test dataset: ", len(df_test["id"]))
print("Number of unique ids in the test dataset: ", len(pd.unique(df_test["id"])), "\n")

print("Number of common ids(if any) between the train and test datasets: ", len(set(df_train["id"].values).intersection(set(df_test["id"].values))))


#  Are all the vendor_id's in the train and test datasets unique?
# 
#     vendor_id takes on only two values in both the train and test datasets i.e. 1 and 2 (Hypothesis - This could represent data from two different taxi companies)
# 
# This leads to a set of follow-up questions:
# 
#     If the hypothesis is right and the values in the vendor_id column actually represent the data from two different taxi companies; are the number of observations in the dataset from each of the companies comparable or is there any imbalance?(Both in the train and test datasets)
# 
#     Among the vendor_id's(1 and 2) - what is the distribution in the number of passengers (passenger_count) across the train and test datasets?
# 

# In[76]:


print("Number of vendor_ids in the train dataset: ", len(df_train["vendor_id"]))
print("Number of unique vendor_ids in the train dataset: ", len(pd.unique(df_train["vendor_id"])), "\n")

print("Number of vendor_ids in the test dataset: ", len(df_test["vendor_id"]))
print("Number of unique vendor_ids in the test dataset: ", len(pd.unique(df_test["vendor_id"])), "\n")


# Distribution of the number of passengers across the vendor_id variables 1 and 2, in both the train and test datasets

# In[77]:


fig, ax =plt.subplots(1,2)
sns.countplot(x="passenger_count", data=df_train[df_train["vendor_id"] == 1], ax=ax[0])
sns.countplot(x="passenger_count", data=df_train[df_train["vendor_id"] == 2], ax=ax[1])
fig.tight_layout() 


# In[78]:


fig, ax =plt.subplots(1,2)
sns.countplot(x="passenger_count", data=df_test[df_test["vendor_id"] == 1], ax=ax[0])
sns.countplot(x="passenger_count", data=df_test[df_test["vendor_id"] == 2], ax=ax[1])
fig.tight_layout()


# Distribution of the trip_duration across the train dataset

# In[79]:


#String to Datetime conversion
df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"])
df_train["dropoff_datetime"] = pd.to_datetime(df_train["dropoff_datetime"])
df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"])


# In[80]:


#trip_duration represents the difference between the dropoff_datetime and the pickup_datetime in the train dataset
df_train["trip_duration"].describe()


# In[81]:


#The trip_duration would be a lot more intuitive when the datetime representation is used, rather than the representation with seconds. 
(df_train["dropoff_datetime"] - df_train["pickup_datetime"]).describe()


# It is interesting to see that there happens to be a trip that lasted for over 40 days. Let us plot the trip duration in seconds to view any other possbile outliers.

# In[82]:


plt.figure(figsize=(10,10))
plt.scatter(range(len(df_train["trip_duration"])), np.sort(df_train["trip_duration"]))
plt.xlabel('index')
plt.ylabel('trip_duration in seconds')
plt.show()



# We see that there are four outliers with trip durations of 20 days or more. We remove the outliers from the dataset
# 

# In[84]:


df_train = df_train[df_train["trip_duration"] < 500000]
(df_train["dropoff_datetime"] - df_train["pickup_datetime"]).describe()


#  Exploring the number of trips at each timestamp feature in the train dataset.
#  
#  
# 
# Once the train dataset has been cleaned, based on the outliers in column - trip duration(that consisted of a few trips lasting for 20 days or more), we can now explore the timestamps on a hourly-weekly basis for further exploratory analysis.
# 
# The train dataset contains trips that range from 2016-01-01 to 2016-06-30, i.e. 6 months worth of data

# In[86]:


print("Train dataset start date: ", min(df_train["pickup_datetime"]))
print("Train dataset end date: ", max(df_train["pickup_datetime"]))


# In[88]:


#Conversion to pandas to_datetime has already been performed in section 5.5
#df_train["pickup_datetime"] = pd.to_datetime(df_train['pickup_datetime'])

df_train["pickup_dayofweek"] = df_train.pickup_datetime.dt.dayofweek
df_train["pickup_weekday_name"] = df_train.pickup_datetime.dt.weekday_name
df_train["pickup_hour"] = df_train.pickup_datetime.dt.hour
df_train["pickup_month"] = df_train.pickup_datetime.dt.month

df_train.head()


# 
# 
# Distribution of trips across - months in the six month rage, day of the week and hour in a day.
# 
# We can observe that there are more trips on Friday's and Saturday's, than on any other weekday, and this make sense (TGIF :)); On a 24 hour clock, the number of trips is the highest between 17:00 hrs - 22:00 hrs and reduces post 01:00 hrs; On a six month time range, the number of trips are almost evenly distributed, with none of the months having a surprising spike in the dataset

# In[90]:


plt.figure(figsize=(12,8))
sns.countplot(x="pickup_weekday_name", data=df_train)
plt.show()


# In[91]:


plt.figure(figsize=(12,8))
sns.countplot(x="pickup_hour", data=df_train)
plt.show()



# In[92]:


plt.figure(figsize=(12,8))
sns.countplot(x="pickup_month", data=df_train)
plt.show()


# Exploring the behaviour of trip_duration based on the number of trips for each timestamp feature in the train dataset.
# 
# 
# 
# 
# In order to visualize the trip duration behaviour, it would be important to aggregate the trip duration at each of the timnestamp feature levels. Since there could be outliers in the trip duration variable(and outlier detection has not yet been performed for this variable) median would be a more representative measure, rather than the mean.
# 

# In[93]:


df_train.trip_duration.describe()


# In[94]:


df_train_agg = df_train.groupby('pickup_weekday_name')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.pickup_weekday_name.values, df_train_agg.trip_duration.values)
plt.show()


# In[95]:


df_train.groupby('pickup_weekday_name')['trip_duration'].describe()


# In[97]:


df_train_agg = df_train.groupby('pickup_hour')['trip_duration'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.pickup_hour.values, df_train_agg.trip_duration.values)
plt.show()


# In[98]:


df_train.groupby('pickup_hour')['trip_duration'].describe()


# In[99]:


df_train_agg = df_train.groupby('pickup_month')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.pickup_month.values, df_train_agg.trip_duration.values)
plt.show()


# In[100]:


df_train.groupby('pickup_month')['trip_duration'].describe()


# 
# 
#     Observation at a week-level:
# 
# Trip durations are the most on Thursday's, Wednesday's and Friday's & the least on Sunday's.
# 
#     Observation at an hour-level:
# 
# Trip durations are the most between 11:00 hrs and 16:00 hrs & the least between 04:00 hrs and 07:00 hrs.
# 
#     Observation at a month-level:
# 
# There seems to be a linear increase in the median trip duration from the month of January to the month of June, although the increase is fairly minimal.
# 
