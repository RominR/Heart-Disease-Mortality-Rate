#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 08:04:04 2022

@author: rominrajbhandari
"""

#Importing all the required libraries.

#Standard Libraries
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from numpy.random import randn

#Statistics Libraries
from scipy import stats

#Plotting
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

#Setting directry (if needed)
import os
os.getcwd()
os.chdir('/Users/romin/Library/Mobile Documents/com~apple~CloudDocs/Python/Python_Bootcamp_10-7-2020/Datasets')

#Datetime
import datetime

#Regular Expression
import re

#SimpleImputer
from sklearn.impute import SimpleImputer

#KNNImputer
from sklearn.impute import KNNImputer

#Iterative Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Label Encoding or One Hot Encoding
from sklearn.preprocessing import LabelEncoder

#Grab the data from web (HTTP capabilities)
import requests
import urllib.request #creates a Request object specifying the URL we want.

# We'll also use StringIO to work with the csv file, the DataFrame will require a .read() method. StringIO provides a convenient means of working with text in memory using the file API
from io import StringIO
import io

#if the file to be imported in on zipfile
import zipfile

import xarray as xr # opens up the bytes file as a xarray dataset.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Now, let us read the dataset directly from the web.

'''Heart Disease Mortality Data Among US Adults (35+) by State/Territory and County – 2016-2018'''
#url = https://healthdata.gov/dataset/Heart-Disease-Mortality-Data-Among-US-Adults-35-by/pwn5-iqp5

url = "https://data.cdc.gov/api/views/6x7h-usvx/rows.csv?accessType=DOWNLOAD"


'''
url = https://towardsdatascience.com/an-efficient-way-to-read-data-from-the-web-directly-into-python-a526a0b4f4cb

req = urllib.request.Request(url)

with urllib.request.urlopen(req) as resp:
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zip_file:
        zip_names = zip_file.namelist()
        ds = xr.open_dataset(zip_file.open(zip_names[0]))'''


#Using 'requests' to get the information in text form.
source = requests.get(url).text

#Using 'StringIO' to avoid an IO error with pandas.
heart_disease_mortality_data = StringIO(source)

#Now that we have our data, we can set it as a DataFrame.

#set the key indicators of heart disease data to dataframe.
Hrt_Dsese_Mrt_Data = pd.read_csv(heart_disease_mortality_data)

#let's see the total count and data type of each columns.
Hrt_Dsese_Mrt_Data.info()

#First 5 rows of the dataset.
Hrt_Dsese_Mrt_Data.head()

#Checking first row to see how the data is.
Hrt_Dsese_Mrt_Data.iloc[0]
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Data Prep
#Let's work on missing values.

#There are 32550 non null values out of 59094 total values in 'Data_Value' column.
#There are 26544 non null values out of 59094 total values in 'Data_Value_Footnote_Symbol' and 'Data_Value_Footnote' column.

#Let's see if all the missing values are actually null(NaN) values.
Hrt_Dsese_Mrt_Data[Hrt_Dsese_Mrt_Data['Data_Value'].isnull()]
#There are 26544 null values in this column. That means 32550 non-null values. Therefore, the total missing values is just a null(NaN) value.

Hrt_Dsese_Mrt_Data[Hrt_Dsese_Mrt_Data['Data_Value_Footnote_Symbol'].isnull()]
#There are 32550 null values in this column. That means 26544 non-null values. Therefore, the total missing values is just a null(NaN) value.

Hrt_Dsese_Mrt_Data[Hrt_Dsese_Mrt_Data['Data_Value_Footnote'].isnull()]
#There are 32550 null values in this column. That means 26544 non-null values. Therefore, the total missing values is just a null(NaN) value.

#Let's check if there are any empty observations or special characters in 'not null' values.
Hrt_Dsese_Mrt_Data[(Hrt_Dsese_Mrt_Data['Data_Value_Footnote'] == ' ') | 
                   (Hrt_Dsese_Mrt_Data['Data_Value_Footnote'] == '')][['Year', 'Data_Value_Footnote']]

Hrt_Dsese_Mrt_Data[(Hrt_Dsese_Mrt_Data['Data_Value_Footnote_Symbol'] == ' ')| 
                   (Hrt_Dsese_Mrt_Data['Data_Value_Footnote_Symbol'] == '')][['Year', 'Data_Value_Footnote']]
#Looks like there are no blank characters.

#Now, let's check for other special characters.

Hrt_Dsese_Mrt_Data[(Hrt_Dsese_Mrt_Data['Data_Value_Footnote'].notnull()) & 
                   (Hrt_Dsese_Mrt_Data['Data_Value_Footnote'].str.findall("[^a-zA-Z]")) | 
                   (Hrt_Dsese_Mrt_Data['Data_Value_Footnote'].str.findall("[^a-zA-Z0-9 ]+"))]

Hrt_Dsese_Mrt_Data[(Hrt_Dsese_Mrt_Data['Data_Value_Footnote_Symbol'].notnull()) & 
                   (Hrt_Dsese_Mrt_Data['Data_Value_Footnote_Symbol'].str.findall("[^a-zA-Z]")) | 
                   (Hrt_Dsese_Mrt_Data['Data_Value_Footnote_Symbol'].str.findall("[^a-zA-Z0-9 ]+"))]

'''We couldn't get the desired result because space is also counted as 'special character. 
In the 'Data_Value_Footnote' column, there is space between the value i.e. 'Insufficient Data'.
The code above is taking that character because of which we are not able to make sure if there are any other special character.
So, let's take the space character out.'''

Hrt_Dsese_Mrt_Data['Data_Value_Footnote'].replace({'Insufficient Data' : 'InsufficientData'}, inplace = True)

#Let's us check if the space is removed.
Hrt_Dsese_Mrt_Data['Data_Value_Footnote'].unique() #It's removed.

#Now, let's run those line again. Looks like there are no any other forms of special characters. We are good to go on this columns

#Now, let's check 'Data_Value_Footnote_Symbol' column
Hrt_Dsese_Mrt_Data['Data_Value_Footnote_Symbol'].unique() #There are just two types of values i.e. "nan" and "~". So, the above code should produce 26544 rows.

#At last, let's check if there are any special characters on 'non null' values in 'Data_Value' column
Hrt_Dsese_Mrt_Data[Hrt_Dsese_Mrt_Data['Data_Value'] == ' ']
Hrt_Dsese_Mrt_Data[Hrt_Dsese_Mrt_Data['Data_Value'] == '']
#Looks like there are no any spaces and blank values.

#We don't have to check for other special characters because the dtype for this column is float65 which means all numeric values. If there were any special or alphnumeric characters, the dtype would have been either object or mixed value object.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Now, let us try to fill all the missing data.

#Let's begin with 'Data_Value_Footnote'.
Hrt_Dsese_Mrt_Data.Data_Value_Footnote.unique()

#Since there are just 'nan' and 'InsufficientData', let us replace all the 'nan' value with 'InsufficientData' and also let us change the 'InsufficientData' to its default form.
Hrt_Dsese_Mrt_Data.Data_Value_Footnote.replace({np.nan: 'Insufficient Data', 'InsufficientData' : 'Insufficient Data'}, inplace = True)

#Now, for the 'Data_Value_Footnote_Symbol'
Hrt_Dsese_Mrt_Data.Data_Value_Footnote_Symbol.unique()

#Since there are just 'nan' and '~', let us replace all the 'nan' value with '~'
Hrt_Dsese_Mrt_Data.Data_Value_Footnote_Symbol.replace({np.nan : '~'}, inplace = True)

#There are only 18 values missing in Y_lat and X_lon. We can replace with arbitrary value as less than 5% of the dataset is missing in this column also it is not possible to get the this data via other options like mean and median because these values are latitude and longitude.
Hrt_Dsese_Mrt_Data.Y_lat.fillna(0, inplace = True)
Hrt_Dsese_Mrt_Data.X_lon.fillna(0, inplace = True)

#Hrt_Dsese_Mrt_Data.Y_lat.value_counts()
#Hrt_Dsese_Mrt_Data.X_lon.value_counts()
#checking the info.
Hrt_Dsese_Mrt_Data.info()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Lastly for 'Data Value', we have 26544 null values. We can't just get rid of this column because 44.92% of total data is missing in this column.

#Let's begin dealing with the missing values.

#there are also categorical values in the dataset, for this, we need to use Label Encoding or One Hot Encoding.
#But, first let us remove those columns we won't work with at the moment. This would help us to check the accuracy of data when missing values are excluded or imputed.

#Let's check how the data in each columns are fetched.
for i in Hrt_Dsese_Mrt_Data.columns:
    print(i, Hrt_Dsese_Mrt_Data[i].unique())
    
#Looking at the result from above script, we can now remove those columns which are not necessary at the moment.
Del_Data_Val = Hrt_Dsese_Mrt_Data.drop(columns = ['Year', 'LocationAbbr', 'LocationDesc', 'DataSource', 'Class',
                            'Topic', 'Data_Value_Unit', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote',
                            'StratificationCategory1', 'StratificationCategory2', 'TopicID'], axis = 1) 
#string cannot be passed while checking the accuracy. It should be either excluded or use Label Encoding. Excluding all the object type that are not categorical values.
Del_Data_Val.info()
Del_Data_Val.iloc[0]

#Now,
#from sklearn.preprocessing import LabelEncoder (imported at the very beginning)
le = LabelEncoder()
Del_Data_Val['GeographicLevel'] = le.fit_transform(Del_Data_Val['GeographicLevel'])
Del_Data_Val['Data_Value_Type'] = le.fit_transform(Del_Data_Val['Data_Value_Type'])
Del_Data_Val['Stratification1'] = le.fit_transform(Del_Data_Val['Stratification1'])
Del_Data_Val['Stratification2'] = le.fit_transform(Del_Data_Val['Stratification2'])
Del_Data_Val_New = Del_Data_Val
Del_Data_Val_New.info()
Del_Data_Val.info()

#Splitting the data into x and y.
y = Del_Data_Val['Data_Value_Type']
Del_Data_Val.drop(columns = 'Data_Value_Type', axis = 1, inplace = True)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Method 1: Delete the column with missing data.
del_col_df = Del_Data_Val.dropna(axis = 1)
del_col_df.info()


#Let's check the accuracy

from sklearn import metrics
from sklearn.model_selection import train_test_split #url = https://realpython.com/train-test-split-python-data/
X_train, X_test, y_train, y_test = train_test_split(del_col_df, y, test_size = 0.9)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print(metrics.accuracy_score(pred, y_test))

#we are able to achieve an accuracy of 100%.
#With this method, we do not lose valuable information on that feature, even though we have deleted the column with some null values.
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Lets see what happens if we delete the rows with missing data.

#Method 2: Delete the rows with missing data.
del_row_df = Del_Data_Val_New.dropna(axis = 0)
del_row_df.info()

#Splitting the data into x and y.
y = del_row_df['Data_Value_Type']
del_row_df.drop(columns = 'Data_Value_Type', axis = 1, inplace = True)



#Let's check for the accuracy.
from sklearn import metrics
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(del_row_df, y, test_size = 0.8)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print(metrics.accuracy_score(pred, y_test))

#we are able to achieve an accuracy of 100%. Looks like the column 'DataValue' is not that important like I expected.
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Let's try few more method.

#Method3: Filling the missing values- Imputation.
Fill_Mean_Val = Del_Data_Val.fillna(Del_Data_Val['Data_Value'].mean())
Fill_Mean_Val.info()

#Splitting the data into x and y.
y = Fill_Mean_Val['Data_Value_Type']
Fill_Mean_Val.drop(columns = 'Data_Value_Type', axis = 1, inplace = True)

#Let's check for the accuracy.
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Fill_Mean_Val, y, test_size = 0.8)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print(metrics.accuracy_score(pred, y_test))
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Method4: KNNImputer
#from sklearn.imputer import KNNImputer (imported at the very beginning) #Doesn't work with string values. Only works for numeric variables.
impute_knn = KNNImputer(n_neighbors = 7)
Del_Data_Val= DataFrame(data = impute_knn.fit_transform(Del_Data_Val), columns = Del_Data_Val.columns)
Del_Data_Val.info()
#url = https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e


#Checking the df is not null.
Del_Data_Val.isnull().sum()
Del_Data_Val.isna().sum()

#Checking for the accuracy
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Del_Data_Val, y, test_size= 0.8)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print(metrics.accuracy_score(pred, y_test))

#it looks like the Data_Value column is not that important as it keeps giving us 100% accuracy no matter how we play with the missing data.
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#For now, we will use knnimputer.
#Let's grab the 'Data_Value' column from Del_Data_Val and concatenate into the original df i.e.Hrt_Dsese_Mrt_Data
Hrt_Dsese_Mrt_Data['Data_Value'] = Del_Data_Val['Data_Value']
Hrt_Dsese_Mrt_Data.info()
Hrt_Dsese_Mrt_Data.isnull().sum() #Checking to see if there are any null values left in the dataset.
Hrt_Dsese_Mrt_Data.isna().sum() #Checking to see if there are any null values left in the dataset.
 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Now, let us work on some visualization.


