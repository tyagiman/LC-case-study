#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:12:31 2022

@author: manishtyagi
"""


#call lib
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


##################################################Analyse data############################################################### 

loan_ip.info()
print(loan_ip.shape)


loan_ip['annual_inc'].max()
loan_ip['annual_inc'].min()

print(loan_ip.head())

loan_ip.head()

loan_ip.columns





################################################DATA CLEANING & FORMATTING#####################################################
#filterout rows with loan_status="Current"

loan_ip=loan_ip[loan_ip['loan_status']!="Current"]
loan_ip.shape

loan_ip['annual_inc'].max()
loan_ip['annual_inc'].min()
loan_ip['annual_inc'].mean()
loan_ip['annual_inc'].median()



#check null values % in each column

round(loan_ip.isnull().sum()/len(loan_ip.index), 2)*100


#find columns that have null values 
print(loan_ip.isnull().sum())

#drop null value columns
loan_ip_nonna=loan_ip.dropna(axis=1,how='all')

#check null values % in each column

round(loan_ip_nonna.isnull().sum()/len(loan_ip_nonna.index), 2)*100



print(loan_ip_nonna.isnull().sum())

#check null values % in each column

round(loan_ip_nonna.isnull().sum()/len(loan_ip_nonna.index), 2)*100

print(loan_ip_nonna.shape)

#Columns mths_since_last_record have more than 90% null values. hence dropping the two columns from the dataframe

loan_ip_nonna = loan_ip_nonna.drop(['mths_since_last_record'], axis=1)

loan_ip_nonna.info()

round(loan_ip_nonna.isnull().sum()/len(loan_ip_nonna.index), 2)*100



#find columns with low distinct values 

unique_counts = loan_ip_nonna.nunique()

print(unique_counts)

#dropping all the columns that have just 1 distinct values

loan_ip_nonna = loan_ip_nonna.drop(['out_prncp','out_prncp_inv','pymnt_plan', 'initial_list_status','collections_12_mths_ex_med','policy_code','application_type','acc_now_delinq','chargeoff_within_12_mths','delinq_amnt','tax_liens'], axis=1)
loan_ip_nonna.info()


#drop columns with no correlation to loan default

loan_ip_clean=loan_ip_nonna.drop(['id', 'member_id','url','zip_code','addr_state'], axis=1)

print(loan_ip_clean.head())


#fixing missing value columns 

round(loan_ip_clean.isnull().sum()/len(loan_ip_clean.index), 2)*100


###########export csv


loan_ip_clean.to_csv(r'/Users/priyankagharpure/Desktop/Manish/MS - ML:AI from IIIT&LJMU/Course 1 Statistics/Lending Case Studz/New Dataset/loan_ip_clean.csv', index = False)




#convert 'term' to numeric

loan_ip_clean.term.dtype

loan_ip_clean['term'].dtypes

#print(loan_ip_clean['term'].str[0:3])


term=loan_ip_clean['term'].str[0:3]

loan_ip_clean1 = loan_ip_clean.drop(['term'], axis=1)

loan_ip_clean2=pd.concat([loan_ip_clean1,term],axis=1)

print(loan_ip_clean2['term'].head())



#replace '< 1 year' to '0.5'for column 'emp_length'

loan_ip_clean2['emp_length'] = loan_ip_clean2['emp_length'].replace(['< 1 year'], '0.5')


print(loan_ip_clean2['emp_length'].head())


#convert 'emp_length' to numeric 


loan_ip_clean2["emp_length"] = loan_ip_clean2["emp_length"].str.extract("(\d*\.?\d+)", expand=True)

print(loan_ip_clean2["emp_length"].head())


loan_ip_clean2.info()



#Fixing continuous value columns  'emp_length'


loan_ip_clean2.info()

loan_ip_clean2.shape

loan_ip_clean2["emp_length"].mode()    #mode is 10

#print(loan_ip_clean2.nunique())
#print(pd.pivot_table(data=loan_ip_clean2,index=['emp_length'],aggfunc={'emp_length':np.sum}))


loan_ip_clean2['emp_length'] = loan_ip_clean2['emp_length'].fillna(10)
loan_ip_clean2['emp_length'] = loan_ip_clean2['emp_length'].replace([10], '10')

loan_ip_clean2['emp_length'].unique()


#Fixing continuous value columns  'revol_util'

#loan_ip_clean2['revol_util'].unique()
#loan_ip_clean2['revol_util'].mode()


#convert 'loan status' to numeric

loan_ip_clean2['loan_status'].unique()

loan_ip_clean2['loan_status'] = loan_ip_clean2['loan_status'].replace(['Charged Off'],1)
loan_ip_clean2['loan_status'] = loan_ip_clean2['loan_status'].replace(['Fully Paid'],0)



#outlier Income handling  - removing outliers above 95 percentile 

print(loan_ip_clean2['annual_inc'].describe())

print(loan_ip_clean2['annual_inc'].mean())

#boxplot with outliers
sns.boxplot(loan_ip_clean2['annual_inc'])

quantile_info = loan_ip_clean2.annual_inc.quantile([0.5, 0.75,0.90, 0.95, 0.97,0.98, 0.99])
quantile_info

loan_ip_clean3 = loan_ip_clean2[loan_ip_clean["annual_inc"] < loan_ip_clean2["annual_inc"].quantile(0.95)]

loan_ip_clean3.info()
loan_ip_clean3.shape

#boxplot after removing outliers 
sns.boxplot(loan_ip_clean3.annual_inc)


#DTI,loan_amnt and funded_amnt_inv  outlier correction not needed, data looks good 

sns.boxplot(loan_ip_clean3.dti)

sns.boxplot(loan_ip_clean3.loan_amnt)

loan_ip_clean3.loan_amnt.quantile([0.75,0.90,0.95,0.97,0.975, 0.98, 0.99, 1.0])


sns.boxplot(loan_ip_clean3.funded_amnt_inv)

loan_ip_clean3.funded_amnt_inv.quantile([0.5,0.75,0.90,0.95,0.97,0.975, 0.98,0.985, 0.99, 1.0])

################################  UNIVARIATE ANALYSIS  ##########################

#loan status 
sns.countplot(x = 'loan_status', data = loan_ip_clean3)

loan_ip_clean3.sub_grade = pd.to_numeric(loan_ip_clean3.sub_grade.apply(lambda x : x[-1]))

loan_ip_clean3.sub_grade.head()


#ownership

#checking unique values for home_ownership
loan_ip_clean3['home_ownership'].unique()

#replacing 'NONE' with 'OTHERS'
loan_ip_clean3['home_ownership'].replace(to_replace = ['NONE'],value='OTHER',inplace = True)

#checking unique values for home_ownership again
loan_ip_clean3['home_ownership'].unique()

fig, ax = plt.subplots(figsize = (6,4))
ax.set(yscale = 'log')
sns.countplot(x='home_ownership', data=loan_ip_clean3[loan_ip_clean3['loan_status']==1])

# lets define a function to plot loan_status across categorical variables
def plot_cat(cat_var):
    sns.barplot(x=cat_var, y='loan_status', data=loan_ip_clean3)
    plt.show()

plot_cat('grade')


plot_cat('term')


# sub-grade: as expected - A1 is better than A2 better than A3 and so on 
plt.figure(figsize=(16, 6))
plot_cat('sub_grade')


# home ownership: not a great discriminator
plot_cat('home_ownership')


# verification_status: surprisingly, verified loans default more than not verifiedb
plot_cat('verification_status')



# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(16, 6))
plot_cat('purpose')


# let's also observe the distribution of loans across years
# first lets convert the year column into datetime and then extract year and month from it
loan_ip_clean3['issue_d'].head()

from datetime import datetime
loan_ip_clean3['issue_d'] = loan_ip_clean3['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))


# extracting month and year from issue_date
loan_ip_clean3['month'] = loan_ip_clean3['issue_d'].apply(lambda x: x.month)
loan_ip_clean3['year'] = loan_ip_clean3['issue_d'].apply(lambda x: x.year)

# let's first observe the number of loans granted across years
loan_ip_clean3.groupby('year').year.count()

# number of loans across months
loan_ip_clean3.groupby('month').month.count()


# lets compare the default rates across years
# the default rate had suddenly increased in 2011, inspite of reducing from 2008 till 2010
plot_cat('year')


# comparing default rates across months: not much variation across months
plt.figure(figsize=(16, 6))
plot_cat('month')


# loan amount: the median loan amount is around 10,000
sns.displot(loan_ip_clean3['loan_amnt'])
plt.show()


# binning loan amount
def loan_amount(n):
    if n < 5000:
        return 'low'
    elif n >=5000 and n < 15000:
        return 'medium'
    elif n >= 15000 and n < 25000:
        return 'high'
    else:
        return 'very high'
        
loan_ip_clean3['loan_amnt'] = loan_ip_clean3['loan_amnt'].apply(lambda x: loan_amount(x))

loan_ip_clean3['loan_amnt'].value_counts()

# let's compare the default rates across loan amount type
# higher the loan amount, higher the default rate
plot_cat('loan_amnt')

# let's also convert funded amount invested to bins
loan_ip_clean3['funded_amnt_inv'] = loan_ip_clean3['funded_amnt_inv'].apply(lambda x: loan_amount(x))

# funded amount invested
plot_cat('funded_amnt_inv')


# lets also convert interest rate to low, medium, high   ###ERROR
# binning loan amount
def int_rate(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=15:
        return 'medium'
    else:
        return 'high'
    
    
loan_ip_clean3['int_rate'] = loan_ip_clean3['int_rate'].apply(lambda x: int_rate(x))

# comparing default rates across rates of interest
# high interest rates default more, as expected
plot_cat('int_rate')


# debt to income ratio
def dti(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=20:
        return 'medium'
    else:
        return 'high'
    

loan_ip_clean3['dti'] = loan_ip_clean3['dti'].apply(lambda x: dti(x))

# comparing default rates across debt to income ratio
# high dti translates into higher default rates, as expected
plot_cat('dti')


# funded amount
def funded_amount(n):
    if n <= 5000:
        return 'low'
    elif n > 5000 and n <=15000:
        return 'medium'
    else:
        return 'high'
    
loan_ip_clean3['funded_amnt'] = loan_ip_clean3['funded_amnt'].apply(lambda x: funded_amount(x))

plot_cat('funded_amnt')


# installment
def installment(n):
    if n <= 200:
        return 'low'
    elif n > 200 and n <=400:
        return 'medium'
    elif n > 400 and n <=600:
        return 'high'
    else:
        return 'very high'
    
loan_ip_clean3['installment'] = loan_ip_clean3['installment'].apply(lambda x: installment(x))

# comparing default rates across installment
# the higher the installment amount, the higher the default rate
plot_cat('installment')



# annual income
def annual_income(n):
    if n <= 50000:
        return 'low'
    elif n > 50000 and n <=100000:
        return 'medium'
    elif n > 100000 and n <=150000:
        return 'high'
    else:
        return 'very high'

loan_ip_clean3['annual_inc'] = loan_ip_clean3['annual_inc'].apply(lambda x: annual_income(x))

# annual income and default rate
# lower the annual income, higher the default rate
plot_cat('annual_inc')


# employment length
# first, let's drop the missing value observations in emp length
loan_ip_clean3 = loan_ip_clean3[~loan_ip_clean3['emp_length'].isnull()]

# binning the variable     #####ERROR
def emp_length(n):
    if n <= 1:
        return 'fresher'
    elif n > 1 and n <=3:
        return 'junior'
    elif n > 3 and n <=7:
        return 'senior'
    else:
        return 'expert'

loan_ip_clean3['emp_length'] = loan_ip_clean3['emp_length'].apply(lambda x: emp_length(x))


# emp_length and default rate
# not much of a predictor of default
plot_cat('emp_length')


#Segmented Univariate Analysis


# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(16, 6))
plot_cat('purpose')

# lets first look at the number of loans for each type (purpose) of the loan
# most loans are debt consolidation (to repay otehr debts), then credit card, major purchase etc.
plt.figure(figsize=(16, 6))
sns.countplot(x='purpose', data=loan_ip_clean3)
plt.show()


# filtering the df for the 4 types of loans mentioned above    #### MAEKE it 4
main_purposes = ["credit_card","debt_consolidation","home_improvement","major_purchase"]
loan_ip_clean3 = loan_ip_clean3[loan_ip_clean3['purpose'].isin(main_purposes)]
loan_ip_clean3['purpose'].value_counts()


# plotting number of loans by purpose 
sns.countplot(x=loan_ip_clean3['purpose'])
plt.show()


# let's now compare the default rates across two types of categorical variables
# purpose of loan (constant) and another categorical variable (which changes)

plt.figure(figsize=[10, 6])
sns.barplot(x='term', y="loan_status", hue='purpose', data=loan_ip_clean3)
plt.show()


# lets write a function which takes a categorical variable and plots the default rate
# segmented by purpose 

def plot_segmented(cat_var):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_var, y='loan_status', hue='purpose', data=loan_ip_clean3)
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
plot_segmented('installment')


# debt to income ratio
plot_segmented('dti')


# annual income
plot_segmented('annual_inc')


# variation of default rate across annual_inc
loan_ip_clean3.groupby('annual_inc').loan_status.mean().sort_values(ascending=False)


# one can write a function which takes in a categorical variable and computed the average 
# default rate across the categories
# It can also compute the 'difference between the highest and the lowest default rate' across the 
# categories, which is a decent metric indicating the effect of the varaible on default rate

def diff_rate(cat_var):
    default_rates = loan_ip_clean3.groupby(cat_var).loan_status.mean().sort_values(ascending=False)
    return (round(default_rates, 2), round(default_rates[0] - default_rates[-1], 2))

default_rates, diff = diff_rate('annual_inc')
print(default_rates) 
print(diff)


#default rate 


round(np.mean(loan_ip_clean3['loan_status']), 2)


# plotting default rates across grade of the loan
sns.barplot(x='grade', y='loan_status', data=loan_ip_clean3)
plt.show()
 
# lets define a function to plot loan_status across categorical variables
def plot_cat(cat_var):
    sns.barplot(x=cat_var, y='loan_status', data=loan_ip_clean3)
    plt.show()


# filtering all the object type variables
df_categorical = loan_ip_clean3.loc[:, loan_ip_clean3.dtypes == object]
df_categorical['loan_status'] = loan_ip_clean3['loan_status']

# Now, for each variable, we can compute the incremental diff in default rates
print([i for i in loan_ip_clean3.columns])
# compare default rates across grade of loan
plot_cat('grade')


# storing the diff of default rates for each column in a dict
d = {key: diff_rate(key)[1]*100 for key in df_categorical.columns if key != 'loan_status'}
print(d)

