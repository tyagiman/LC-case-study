#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:12:31 2022

@author: manishtyagi
"""
#set printing options
pd.options.display.max_columns = None
pd.options.display.max_rows = 20

#call lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load input file
loan_ip=pd.read_csv('/Users/priyankagharpure/Desktop/Manish/MS - ML:AI from IIIT&LJMU/Course 1 Statistics/Lending Case Studz/New Dataset/loan.csv')
loan_ip.info()


#analyse data 

loan_ip.info()
print(loan_ip.shape)

print(loan_ip)

loan_ip.head()

loan_ip.columns

#find columns that have null values 
print(loan_ip.isnull().sum())

#drop null value columns
loan_ip_nonna=loan_ip.dropna(axis=1,how='all')


print(loan_ip_nonna.isnull().sum())

#check null values % in each column

round(loan_ip_nonna.isnull().sum()/len(loan_ip_nonna.index), 2)*100

print(loan_ip_nonna.shape)

#Columns next_pymnt_d & mths_since_last_record have more than 90% null values. hence dropping the two columns from the dataframe

loan_ip_nonna = loan_ip_nonna.drop(['next_pymnt_d', 'mths_since_last_record'], axis=1)

loan_ip_nonna.info()

#find columns with low distinct values 

unique_counts = loan_ip_nonna.nunique()

print(unique_counts)

#dropping all the columns that have just 1 distinct values

loan_ip_nonna = loan_ip_nonna.drop(['pymnt_plan', 'initial_list_status','collections_12_mths_ex_med','policy_code','application_type','acc_now_delinq','chargeoff_within_12_mths','delinq_amnt','tax_liens'], axis=1)
loan_ip_nonna.info()


#drop columns with no correlation to loan default

loan_ip_clean=loan_ip_nonna.drop(['id', 'member_id','url','zip_code','addr_state'], axis=1)

print(loan_ip_clean)