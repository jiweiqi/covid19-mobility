# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:23:13 2020

@author: hexx
"""


import pandas as pd
import numpy as np
import os
from myFunctions import def_add_datashift, createFolder
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import joblib


today = pd.to_datetime('today')
today =today.strftime("%Y-%m-%d")


# Model_Date = np.load("./Model_Parameter.npy",allow_pickle='TRUE').item()

PODA_Model = np.load("./PODA_Model_"+today+".npy",allow_pickle='TRUE').item()

start_Date = '2020-02-25'
end_Date= PODA_Model['ML_File_Date']
ML_File = PODA_Model['ML_File_Date']    #'5-2-2020'

# YYG_projection_Date='2020-05-13'
# ML_Model='2020-05-10'
# fuel_mobility_factor_file = '2020-04-24'
# model_mark =''

# yesterday =pd.to_datetime('today')-pd.DateOffset(days=1)
# yesterday= yesterday.strftime("%Y-%m-%d")
today = pd.to_datetime('today')
today =today.strftime("%Y-%m-%d")

Apple_File_Date=PODA_Model['Apple_File_Date']
# pd_all = pd.read_excel('./ML Files/State_Level_Data_forML_'+ML_File+'.xlsx', header=0)
# pd_all = PODA_Model['ML_Data'].reset_index()
# data_used = pd_all[['date', 'WeekDay', 'State Name', 'retail_and_recreation', 'grocery_and_pharmacy', 'workplaces', 'parks',
#                    'Apple US', 'Apple State']]

data_used = PODA_Model['Google_Apple_Mobility_Projection_mean']


data_used = data_used[(data_used['date']> (pd.to_datetime(start_Date)-pd.DateOffset(days=7))) & (data_used['date'] < pd.to_datetime(end_Date))]
# data_used = data_used[(data_used['date']> (pd.to_datetime(start_Date)-pd.DateOffset(days=7)))]
data_used = data_used.set_index('date')

NHTS_Category_Share = pd.read_excel('NHTS.xlsx', sheet_name='Category Share')
NHTS_State_Fuel_Share = pd.read_excel('NHTS.xlsx', sheet_name='State Fuel Share')
PODA_Model['NHTS Category Share'] = NHTS_Category_Share
PODA_Model['NHTS State Fuel Share'] = NHTS_State_Fuel_Share

df_StateName_Code = PODA_Model['StateName_StateCode']

   
cols = ['State Name']
data_used = data_used.join(df_StateName_Code.set_index(cols), on=cols, how='left')

data_used = data_used.join(NHTS_Category_Share.set_index('State Code'), on='State Code', how='left')

data_used = data_used.join(NHTS_State_Fuel_Share.set_index('State Name'), on='State Name', how='left')


fuel_Demand_EIA = PODA_Model['Fuel_Demand_EIA'].reset_index()
fuel_Demand_EIA = fuel_Demand_EIA[(fuel_Demand_EIA['Date'] > pd.to_datetime(start_Date)+pd.DateOffset(days=7))]
# fuel_Demand_EIA = fuel_Demand_EIA[(fuel_Demand_EIA['Date'] > pd.to_datetime(start_Date)+pd.DateOffset(days=7)) & (fuel_Demand_EIA['Date'] < pd.to_datetime(end_Date))]
fuel_Demand_EIA = fuel_Demand_EIA.set_index('Date')
fuel_Demand_EIA['apple'] =0

aa = data_used[['Apple State Mobility Predict', 'Percentage gasoline', 'WeekDay']]
aa['Apple State'] =aa['Apple State Mobility Predict'].to_numpy(dtype=np.float)

aa['Apple fuel factor'] = aa['Apple State']*aa['Percentage gasoline']
aa=aa.dropna()
day_Shift = int(2)
fuel_factor = aa.sum(level='date')
# fuel_factor = fuel_factor[['fuel factor','WeekDay']]
fuel_factor['WeekDay'] = fuel_factor['WeekDay']/50
fuel_factor['Shifted Date'] =fuel_factor.index+pd.DateOffset(days=day_Shift)
baseline =1            #average of EIA between Jan 03-Feb 07(thousand bpd)

for i, date_i in enumerate(fuel_Demand_EIA.index):
    print(i, date_i)
    apple_weekly = fuel_factor[(fuel_factor['Shifted Date']<=pd.to_datetime(date_i)) & (fuel_factor['Shifted Date']>(pd.to_datetime(date_i)-pd.DateOffset(days=7)))] 
    #
    # apple_weekly['fuel factor'].mean(afuel_factoris =0)
    fuel_Demand_EIA.loc[date_i, 'apple'] = apple_weekly['Apple fuel factor'].mean(axis =0)
                            
# fuel_Demand_EIA = fuel_Demand_EIA[fuel_Demand_EIA.index != pd.to_datetime('05-08-2020')]

fuel_Demand_EIA=fuel_Demand_EIA.dropna()
x = fuel_Demand_EIA['apple'].to_numpy()                                       
y = fuel_Demand_EIA['Gasoline'].to_numpy()


x_conv=x
y_conv = y
x_length = len(x)
x_conv = x_conv.reshape(x_length, 1)
y_conv = y_conv.reshape(x_length, 1)
regr = linear_model.LinearRegression()
regr.fit(x_conv, y_conv)

regr_coef = regr.coef_
print('reg_coeff: ', regr_coef)
regr_interp = regr.intercept_
print('reg_interp: ', regr_interp)
R2 = regr.score(x_conv, y_conv)
print('R2: ', R2)


fig1 = plt.figure(figsize=(6, 5))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(fuel_Demand_EIA.index, regr.predict(x_conv), '-o', label=['pred'])
ax1.plot(fuel_Demand_EIA.index, fuel_Demand_EIA['Gasoline'], '--s', label='EIA')
ax1.plot(fuel_Demand_EIA.index, x, '-o', label=['orig'])
ax1.set_xlabel('Date')
ax1.set_ylabel('Y')
ax1.set_title('fuel demand: shift:'+str(day_Shift)+' days'+ 'R2='+str(R2))
ax1.legend()
                 


PODA_Model['Apple_EIA_Regression'] = regr
np.save(("./PODA_Model_"+today+".npy"), PODA_Model)

createFolder('./Fuel Demand Projection')
filename = './Fuel Demand Projection/apple_fuel_model_'+Apple_File_Date+'.sav'
joblib.dump(regr, filename)
                        
