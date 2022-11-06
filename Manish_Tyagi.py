#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:22:40 2022

@author: manish tyagi

"""

#import libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#set printing options
pd.options.display.max_columns = None
pd.options.display.max_rows = None

#load input file
loan_ip=pd.read_csv('/Users/priyankagharpure/Desktop/Manish/MS - ML:AI from IIIT&LJMU/Course 1 Statistics/Lending Case Studz/New Dataset/loan.csv')
loan_ip.shape


#################################INITIAL DATA ANALYSIS & CLEANING OF DATA

loan_ip.head()

#check null values % in each column

round(loan_ip.isnull().sum()/len(loan_ip.index), 2)*100

#drop null value columns
loan_ip=loan_ip.dropna(axis=1,how='all')


#check again null values % in each column

round(loan_ip.isnull().sum()/len(loan_ip.index), 2)*100


#Columns mths_since_last_record and next_pymnt_d have more than 90% null values. hence dropping the two columns from the dataframe

loan_ip = loan_ip.drop(['mths_since_last_record','next_pymnt_d'], axis=1)


#check again null values % in each column

round(loan_ip.isnull().sum()/len(loan_ip.index), 2)*100


#remove single valued columns 


unique_counts = loan_ip.nunique()

print(unique_counts)


#dropping all the columns that have just 1 distinct values

loan_ip = loan_ip.drop(['out_prncp','out_prncp_inv','pymnt_plan', 'initial_list_status','collections_12_mths_ex_med','policy_code','application_type','acc_now_delinq','chargeoff_within_12_mths','delinq_amnt','tax_liens'], axis=1)
loan_ip.shape   #44 columns left


#drop customer columns with no correlation to loan default

loan_ip=loan_ip.drop(['id', 'member_id','url','zip_code','addr_state'], axis=1)

#drop customer behaviour columns which are not required for analysis

loan_ip=loan_ip.drop(['delinq_2yrs', 'earliest_cr_line','inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','last_credit_pull_d'], axis=1)


loan_ip.shape   #21 columns left 


#analysing Loan_status 

sns.countplot(x = 'loan_status', data = loan_ip)

#as status 'current' is of no importance for our analysis, rows with this status should be filtered out 

loan_ip=loan_ip[loan_ip['loan_status']!="Current"]

loan_ip.shape   #38577,21

loan_ip.describe()

loan_ip.info()



#convert 'emp_length' to numeric 

loan_ip["emp_length"] = loan_ip["emp_length"].str.extract("(\d*\.?\d+)", expand=True)


#checking for missing values in columns 

round(loan_ip.isnull().sum()/len(loan_ip.index), 2)*100

loan_ip["emp_length"].mode()   #10 can be used to replace missing values in rows 


loan_ip['emp_length'] = loan_ip['emp_length'].fillna("10")


#cleaning int_rate


loan_ip["int_rate"].mode() 

loan_ip["int_rate"]=loan_ip["int_rate"].str.replace("%","")

loan_ip["int_rate"]= pd.to_numeric(loan_ip["int_rate"])


loan_ip["int_rate"].head()


loan_ip.shape


#convert 'loan status' to numeric 1 and 0

loan_ip['loan_status'].unique()

loan_ip['loan_status'] = loan_ip['loan_status'].replace(['Charged Off'],1)
loan_ip['loan_status'] = loan_ip['loan_status'].replace(['Fully Paid'],0)

loan_ip["loan_status"]= pd.to_numeric(loan_ip["loan_status"])


#outlier handling - annual_inc

sns.boxplot(loan_ip['annual_inc'])

quantile_info = loan_ip.annual_inc.quantile([0.5, 0.75,0.90, 0.95, 0.97,0.98, 0.99])
quantile_info

loan_ip = loan_ip[loan_ip["annual_inc"] < loan_ip["annual_inc"].quantile(0.95)]

loan_ip.shape

sns.boxplot(loan_ip['annual_inc'])


#DTI,loan_amnt and funded_amnt_inv  outlier correction not needed, data looks good 

sns.boxplot(loan_ip.dti)

sns.boxplot(loan_ip.loan_amnt)

loan_ip.loan_amnt.quantile([0.75,0.90,0.95,0.97,0.975, 0.98, 0.99, 1.0])


sns.boxplot(loan_ip.funded_amnt_inv)

loan_ip.funded_amnt_inv.quantile([0.5,0.75,0.90,0.95,0.97,0.975, 0.98,0.985, 0.99, 1.0])


#removing 'months'from 'term''

loan_ip['term'] = loan_ip['term'].str[0:3]

loan_ip["term"]= pd.to_numeric(loan_ip["term"])

loan_ip.shape



######################## UNIVARIATE ANALYSIS ##################################


#loan status 
sns.countplot(x = 'loan_status', data = loan_ip)


#ownership

#checking unique values for home_ownership
loan_ip['home_ownership'].unique()

#replacing 'NONE' with 'OTHERS'
loan_ip['home_ownership'].replace(to_replace = ['NONE'],value='OTHER',inplace = True)

#checking unique values for home_ownership again
loan_ip['home_ownership'].unique()

fig, ax = plt.subplots(figsize = (6,4))
ax.set(yscale = 'log')
sns.countplot(x='home_ownership', data=loan_ip[loan_ip['loan_status']==1])


# defining a function to plot loan_status across various categorical variables for analysis
def plot_status(var):
    sns.barplot(x=var, y='loan_status', data=loan_ip)
    plt.show()


#grade analysis

plot_status("grade")


#term analysis 

plot_status('term')


# sub-grade: not a deciding factor as all the values are quite close 
plt.figure(figsize=(10, 5))
plot_status('sub_grade')


# home ownership: all values close , not a great discriminator
plot_status('home_ownership')


# verification_status: surprisingly, verified loans default more than not verified
plot_status('verification_status')


# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(10, 4))
plot_status('purpose')


#fig, ax = plt.subplots(figsize = (12,8))
#ax.set(xscale = 'log')
#sns.countplot(y ='purpose', data=loan_ip[loan_ip.loan_status == 1])


# checking the distribution of loans across years
# convert the year column into datetime and then extract year and month from it
loan_ip['issue_d'].head()

from datetime import datetime
loan_ip['issue_d'] = loan_ip['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))

loan_ip['issue_d'].head()

# extracting month and year from issue_date
loan_ip['month'] = loan_ip['issue_d'].apply(lambda x: x.month)
loan_ip['year'] = loan_ip['issue_d'].apply(lambda x: x.year)

# check number of loans granted across years
loan_ip.groupby('year').year.count()

# number of loans across months
loan_ip.groupby('month').month.count()

# default rates across years
# the default rate was highest in 2007 and increased in 2011, inspite of reducing from 2008 till 2010
plot_status('year')


# default rates across months: not much variation across months
plt.figure(figsize=(16, 6))
plot_status('month')


########################## Binning of continous variables #########################

#binning funded_amnt_inv

loan_ip["funded_amnt_inv"].describe()

def funded_amnt_inv(n):
    if n < 5000:
        return 'low'
    elif n >=5000 and n < 15000:
        return 'medium'
    elif n >= 15000 and n < 25000:
        return 'high'
    else:
        return 'very high'


loan_ip['funded_amnt_inv'] = loan_ip['funded_amnt_inv'].apply(lambda x: funded_amnt_inv(x))


loan_ip['funded_amnt_inv'].value_counts()


loan_ip['funded_amnt_inv'].head()

loan_ip.shape

loan_ip.info()


# binning interest rate to low, medium, high  


loan_ip["int_rate"].describe()


def int_rate(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=15:
        return 'medium'
    else:
        return 'high'
    
    
loan_ip['int_rate'] = loan_ip['int_rate'].apply(lambda x: int_rate(x))

loan_ip["int_rate"].head()



# binning debt to income ratio


loan_ip["dti"].describe()

def dti(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=20:
        return 'medium'
    else:
        return 'high'
    

loan_ip['dti'] = loan_ip['dti'].apply(lambda x: dti(x))

loan_ip["dti"].head()


# binning annual income


loan_ip["annual_inc"].describe()

def annual_income(n):
    if n <= 50000:
        return 'low'
    elif n > 50000 and n <=100000:
        return 'medium'
    elif n > 100000 and n <=150000:
        return 'high'
    else:
        return 'very high'

loan_ip['annual_inc'] = loan_ip['annual_inc'].apply(lambda x: annual_income(x))

loan_ip["annual_inc"].head()





# binning employment length


loan_ip["emp_length"].describe()

loan_ip["emp_length"].unique()

#converting variable to numeric 

loan_ip["emp_length"]= pd.to_numeric(loan_ip["emp_length"])

#loan_ip_clean3 = loan_ip_clean3[~loan_ip_clean3['emp_length'].isnull()]

# binning the variable
def emp_length(n):
    if n <= 1:
        return 'fresher'
    elif n > 1 and n <=3:
        return 'junior'
    elif n > 3 and n <=7:
        return 'senior'
    else:
        return 'expert'

loan_ip['emp_length'] = loan_ip['emp_length'].apply(lambda x: emp_length(x))

loan_ip["emp_length"].unique()








######################## Analysis by binned variables ######################

#analysis by funded_amnt_inv
plot_status('funded_amnt_inv')


#analysis by int_rate     
plot_status('int_rate')


#analysis by dti    
plot_status('dti')


# annual income and default rate
# lower the annual income, higher the default rate
plot_status('annual_inc')


# emp_length and default rate
# not much of a predictor of default
plot_status('emp_length')


###################### Segmented Univariate Analysis #######################

#Annual income vs loan purpose

plt.figure(figsize=(10,10))
sns.barplot(data =loan_ip,x='annual_inc', y='purpose', hue ='loan_status',palette="deep")
plt.show()


loan_ip.info()
loan_ip['annual_inc'].unique()


# purpose vs term 

plt.figure(figsize=[10, 6])
sns.barplot(x='term', y="loan_status", hue='purpose', data=loan_ip)
plt.show()


#  a function which takes a categorical variable and plots the default rate
# segmented by purpose 

def plot_segmented(cat_var):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_var, y='loan_status', hue='purpose', data=loan_ip)
    plt.show()

    
plot_segmented('term')


# grade of loan
plot_segmented('grade')


# home ownership
plot_segmented('home_ownership')



# year
plot_segmented('year')



# emp_length
plot_segmented('emp_length')


# loan_amnt: same trend across loan purposes
plot_segmented('loan_amnt')


# interest rate
plot_segmented('int_rate')


# installment
#plot_segmented('installment') #takes lot of time 


# debt to income ratio
plot_segmented('dti')


# annual income
plot_segmented('annual_inc')


# variation of default rate across annual_inc
loan_ip.groupby('annual_inc').loan_status.mean().sort_values(ascending=False)


#################  BIVARIATE  ANALYSIS ################

#Heat Map
plt.figure(figsize=(8,8))
sns.heatmap(loan_ip.corr())
plt.show() 


# Checking correlation  and using heatmap to visualise it.
sns.set(rc={'figure.figsize':(8,8)})
sns.set_style('whitegrid')
# Heatmap
sns.heatmap(loan_ip.corr())
plt.show()


loan_ip.corr()