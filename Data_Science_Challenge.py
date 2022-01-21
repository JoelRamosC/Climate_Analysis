# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 21:34:22 2022

@author: joelr
"""
# Date (DD/MM/YYYY)
# Time (HH.MM.SS)
# PT08.S1 (CO) – Variável de predição
# Non Metanic HydroCarbons Concentration (mg/m^3)
# 4 Benzene Concentration (mg/m^3)
# PT08.S2 (NMHC)
# NOx Concentration (ppb)
# PT08.S3 (NOx)
# 8 NO2 Concentration (mg/m^3)
# PT08.S4 (NO2s)
# PT08.S5 (O3)
# Temperature (C)
# Relative Humidity (%)
# AH Absolute Humidity

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from statistics import mean
import scipy.stats

#header_list = ['Date','Time','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH']
df = pd.read_csv(r'C:\Users\joelr\Desktop\Recruit Project\qualidade_do_ar.csv', sep=';')
print(df.shape)
print(df.info()) 

#number NaN in each column
Number_NAN_column = df.isna().sum()

#drop NMHC(GT) with 8443
df = df.drop(columns=['NMHC(GT)'], )

#Drop rows with NaN
df = df.dropna()
df.reset_index(drop=True, inplace=True)
#redution = 100-((7393*100)/9357) = 20% in data

#############
#1 - Escolha uma estratégia de tratamento de valores faltantes e 
#outliers e justifique sua escolha. 
#(de maior concentração de CO) ?
#############

#Escolha uma estratégia de tratamento de valores faltantes e outliers e justifique sua escolha.
#sentinel value = -200

# The difference between a good and an average machine learning model is 
# often its ability to clean data. One of the biggest challenges in data 
# cleaning is the identification and treatment of outliers. In simple terms,
#  outliers are observations that are significantly different from other 
#  data points. Even the best machine learning algorithms will underperform 
#  if outliers are not cleaned from the data because outliers can adversely 
#  affect the training process of a machine learning algorithm, resulting 
#  in a loss of accuracy.

# Outlier Identification
# There can be many reasons for the presence of outliers in the data. 
# Sometimes the outliers may be genuine, while in other cases, they 
# could exist because of data entry errors. It is important to understand 
# the reasons for the outliers before cleaning them.


# Identifying Outliers with Skewness
# Several machine learning algorithms make the assumption that 
# the data follow a normal (or Gaussian) distribution. This is 
# easy to check with the skewness value, which explains the extent 
# to which the data is normally distributed. Ideally, the skewness 
# value should be between -1 and +1, and any major deviation from 
# this range indicates the presence of extreme values.


#Understand Data to Make Wise Decisions
#Summary statistics
# def column2analysis(column_data):
#     print('Skew'      ,column_data,' = ',df[column_data].skew())
#     print('Mean'      ,column_data,' = ',df[column_data].mean())
#     print('Median'    ,column_data,' = ',df[column_data].median())
#     print('Mode'      ,column_data,' = ',df[column_data].mode())
#     print('Describle ',column_data)
#     print(df[column_data].describe())
#     plt.hist(df[column_data]) 
#     plt.show()
   
#Imputer Step
#If data is simetric, I use mean, if not, I use median to replace NaN

#PT08.S1 (CO) – Variável de predição
# column2analysis('PT08.S1(CO)')  
# df['PT08.S1(CO)'].fillna(value=df['PT08.S1(CO)'].mean(), inplace=True)
# df['PT08.S1(CO)'].hist()


# df['PT08.S1(CO)'].fillna(value=df['PT08.S1(CO)'].mean(), inplace=True)
# df['NMHC(GT)'].fillna(value=df['NMHC(GT)'].median(), inplace=True)
# df['C6H6(GT)'].fillna(value=df['C6H6(GT)'].mean(), inplace=True)
# df['PT08.S2(NMHC)'].fillna(value=df['PT08.S2(NMHC)'].mean(), inplace=True)
# df['NOx(GT)'].fillna(value=df['NOx(GT)'].median(), inplace=True)
# df['PT08.S3(NOx)'].fillna(value=df['PT08.S3(NOx)'].mean(), inplace=True)  
# df['NO2(GT)'].fillna(value=df['NO2(GT)'].mean(), inplace=True)
# df['PT08.S4(NO2)'].fillna(value=df['PT08.S4(NO2)'].mean(), inplace=True)
# df['PT08.S5(O3)'].fillna(value=df['PT08.S5(O3)'].mean(), inplace=True)
# df['T'].fillna(value=df['T'].mean(), inplace=True) 
# df['RH'].fillna(value=df['RH'].mean(), inplace=True)
# df['AH'].fillna(value=df['AH'].median(), inplace=True)

# Outlier Treatment
# In the previous sections, we learned about techniques for outlier 
# detection. However, this is only half of the task. Once we have 
# identified the outliers, we need to treat them. There are several 
# techniques for this, and we will discuss the most widely used ones below.

# Quantile-based Flooring and Capping
# In this technique, we will do the flooring (e.g., the 10th percentile) 
# for the lower values and capping (e.g., the 90th percentile) 
# for the higher values. The lines of code below print the 10th 
# and 90th percentiles of the variable 'Income', respectively. 
# These values will be used for quantile-based flooring and capping.

#Trimming
#IQR Score
#Replacing Outliers with Median Values


#Outliers Stage 
def column_outliers(column_data):
    print('Skew'      ,column_data,' = ',df[column_data].skew())
    # print('Describle ',column_data)
    # print(df[column_data].describe())
    plt.boxplot(df[column_data]) 
    plt.show()



#PT08.S1 (CO) – Variável de predição
#Replacing by column mean only outliers > 1550 to not delete data without authorization
#The Skew was reduced
column_outliers('PT08.S1(CO)')  
#median_column = df['PT08.S1(CO)'].median()
#df['PT08.S1(CO)'] = np.where(df['PT08.S1(CO)'] > 1550, median_column, df['PT08.S1(CO)'])
index = df[(df['PT08.S1(CO)'] >1650)].index
df.drop(index, inplace=True)
column_outliers('PT08.S1(CO)')  



#Benzene Concentration (mg/m^3)
#Replacing by column mean only outliers > 25 
#The Skew was reduced
column_outliers('C6H6(GT)') 
index = df[(df['C6H6(GT)'] >25)].index
df.drop(index, inplace=True)
column_outliers('C6H6(GT)')  

#PT08.S2 (NMHC)
#Replacing by column mean only outliers > 1650
#The Skew was reduced
column_outliers('PT08.S2(NMHC)') 


#NOx Concentration (ppb)
#Replacing by column mean only outliers >500
#The Skew was reduced
column_outliers('NOx(GT)')
index = df[(df['NOx(GT)'] > 500)].index
df.drop(index, inplace=True)
column_outliers('NOx(GT)')

#PT08.S3 (NOx)
#Replacing by column mean only outliers > 1350
#The Skew was reduced
column_outliers('PT08.S3(NOx)') 
index = df[(df['PT08.S3(NOx)'] > 1350)].index
df.drop(index, inplace=True)
column_outliers('PT08.S3(NOx)')

#NO2 Concentration (mg/m^3)
#Replacing by column mean only outliers > quantile(0.98) and < quantile(0.01)
column_outliers('NO2(GT)') 
index = df[(df['NO2(GT)'] > 200)].index
df.drop(index, inplace=True)
column_outliers('NO2(GT)')  

#PT08.S4 (NO2s)
column_outliers('PT08.S4(NO2)') 

# PT08.S5 (O3)
column_outliers('PT08.S5(O3)') 
index = df[(df['PT08.S5(O3)'] > 1800)].index
df.drop(index, inplace=True)
column_outliers('PT08.S5(O3)')  

# Temperature (C)
column_outliers('T') 
# Relative Humidity (%)
column_outliers('RH') 
# AH Absolute Humidity
column_outliers('AH') 

df.reset_index(drop=True, inplace=True)

# Outlier effect on the mean
# Outliers can significantly increase or decrease 
# the mean when they are included in the calculation. 
# Since all values are used to calculate the mean, it 
# can be affected by extreme outliers. An outlier is a 
# value that differs significantly from the others in a data set.

# When should you use the mean, median or mode?
# The 3 main measures of central tendency are best 
# used in combination with each other because they 
# have complementary strengths and limitations. But 
# sometimes only 1 or 2 of them are applicable to your 
#  set, depending on the level of measurement of the variable.

# The mode can be used for any level of measurement, 
# but it’s most meaningful for nominal and ordinal levels.
# The median can only be used on data that can be ordered 
# – that is, from ordinal, interval and ratio levels of measurement.
# The mean can only be used on interval and ratio 
# levels of measurement because it requires equal 
# spacing between adjacent values or scores in the scale.

# In skewed distributions, the median is the best measure 
# because it is unaffected by extreme outliers or 
# non-symmetric distributions of scores. The mean 
# and mode can vary in skewed distributions.

#############
#2 - Para as quartas-feiras, quais os horários de pico na cidade 
#(de maior concentração de CO) ?
#############
import datetime

#Function to change date format 
def format_date(dt): 
    #dt = '11/03/2004'
    day, month, year = (int(x) for x in dt.split('/'))    
    ans = datetime.date(year, month, day)
    week_day = ans.strftime("%A")
    return week_day
    #print (ans.strftime("%A"))

#create wednesday dataframe to analysis     
df_wednesday = pd.DataFrame([ ], columns = list(df.columns))
for i in range(len(df['Date'])):
   check_day = format_date(df.loc[i,'Date'])
   if check_day == 'Wednesday':
     a_row = df.loc[i,:]
     row_df = pd.DataFrame([a_row])
     df_wednesday = pd.concat([row_df, df_wednesday], ignore_index=True)

 

#create 2 lists to save hours of day and the CO mean to every our of wednesdays
hours = list(pd.date_range("00:00:00", "23:00:00", freq="60min").strftime('%H:%M:%S'))
mean_hours = [ ]
for i in range(len(hours)):
    acumulador = [ ]    
    for j in range(len(df_wednesday['Time'])):
        colon_hour =  df_wednesday.loc[j,"Time"].replace('.',":")
        if  colon_hour  == hours[i]:        
            acumulador.append(df_wednesday.loc[j,"PT08.S1(CO)"]) 
    mean_hours.append(mean(acumulador))      

#Hour and max concentration of CO
print('The hour of max concentration of CO is',mean_hours.index(max(mean_hours)),':00:00')
print('The concentration of CO at',8,':00 is',max(mean_hours))

#Showing the bar graph
y_pos = range(len(hours))
plt.bar(y_pos, mean_hours)
# Rotation of the bars names
plt.xticks(y_pos,hours, rotation=90)


# hours_day = pd.DataFrame([ ], columns = pd.date_range("00:00:00", "23:00:00", freq="60min").strftime('%H:%M:%S'))
# columns_hours_day = list(hours_day.columns)

#############
# 3- Quais as variáveis mais correlacionadas com a variável de predição?
#############
from sklearn import linear_model
from sklearn.metrics import r2_score
import seaborn as sns

#Correlation matrix to heat map
cormat = df.corr()
round(cormat,2)
sns.heatmap(cormat);

#The most most correleted data with CO concentration are C6H6(GT), PT08.S2(NMHC) e PT08.S5(O3)
# Correlation pearsonr()
# The Pearson product-moment correlation is one of the most commonly used 
# correlations in statistics. It’s a measure of the strength and the direction 
# of a linear relationship between two variables.

#             spearmanr()
# The nice thing about the Spearman correlation is that relies on nearly 
#  the same assumptions as the pearson correlation, but it doesn’t rely 
#  on normality, and your data can be ordinal as well. Thus, it’s a 
#  non-parametric test. 

#             kendalltau()
# The Kendall correlation is similar to the spearman correlation in that 
# it is non-parametric. It can be used with ordinal or continuous data. 
# It is a statistic of dependence between two variables.

# we can see pearson and spearman are roughly the same, but kendall is 
# very much different. That's because Kendall is a test of strength of 
# dependece (i.e. one could be written as a linear function of the other), 
# whereas Pearson and Spearman are nearly equivalent in the way they correlate 
# normally distributed data.
import scipy.stats

#'PT08.S1 (CO)' and 'C6H6(GT)'
X = np.array(df.iloc[:, 3])
y = np.array(df.iloc[: ,2])
print(scipy.stats.pearsonr(X, y)[0])
plt.scatter(X,y)
plt.show()

#'PT08.S2(NMHC)'
X = np.array(df.iloc[:, 4])
y = np.array(df.iloc[: ,2])
print(scipy.stats.pearsonr(X, y)[0])
plt.scatter(X,y)
plt.show()

#'PT08.S5(O3)'
X = np.array(df.iloc[:, 9])
y = np.array(df.iloc[: ,2])
print(scipy.stats.pearsonr(X, y)[0])
plt.scatter(X,y)
plt.show()

###########
# 4 - Crie um modelo de regressão de PT08.S1 a partir das
# demais variáveis. Avalie usando as métricas que julgar 
# pertinente para o problema.
########
from scipy import stats

# Execute a method that returns some important key values of 
# Linear Regression:
slope, intercept, r, p, std_err = stats.linregress(X, y)

# Create a function that uses the slope and intercept values
#  to return a new value. This new value represents where 
#  on the y-axis the corresponding x value will be placed
def myfunc(x):
  return slope * x + intercept

# Run each value of the x array through the function. 
# This will result in a new array with new values for the y-axis
mymodel = list(map(myfunc, X))

plt.scatter(X, y)
plt.plot(X, mymodel)
plt.show()


#Multiple Regression
# Multiple regression is like linear regression, but with
# more than one independent value, meaning that we try
# to predict a value based on two or more variables.

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X = df[['C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S5(O3)' ]]
y = df['PT08.S1(CO)']

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=4)

regr = linear_model.LinearRegression()
regr.fit(train_X,train_y)

#The R-squared score (Mean Square Error)
#r2=0 low correlation e r2=1 high correlation
#R2 for train dataset 
r2_train = round(r2_score(train_y,regr.predict(train_X)),ndigits=2)
print(r2_train,'is r2 for train dataset')
#R2 for  test dataset  
r2_test = round(r2_score(test_y,regr.predict(test_X)), ndigits=2)
print(r2_test,'is r2 for test dataset')
