# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:03:20 2020

@author: hexx
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from myFunctions import def_add_datashift, createFolder

today = pd.to_datetime('today')
today =today.strftime("%Y-%m-%d")
# Model_Date = np.load("./Model_Parameter.npy",allow_pickle='TRUE').item()

PODA_Model = np.load("./PODA_Model_"+today+".npy",allow_pickle='TRUE').item()

google_Mobility_Day = PODA_Model['ML_File_Date']

start_Date = '03-01-2020'
YYG_projection_Date=PODA_Model['YYG_File_Date']
ML_Model=PODA_Model['ML_File_Date']
fuel_mobility_factor_file = ML_Model
apple_fuel_Factor_file = PODA_Model['ML_File_Date']
model_mark =''
isopen=''    #'_noreopen'


fuel_Demand_EIA = PODA_Model['Fuel_Demand_EIA'].reset_index()
fuel_Demand_EIA = fuel_Demand_EIA.set_index('Date')

caseID =['lower', 'mean', 'upper', 'MIT']   
for case in caseID:
    if case == 'mean':
        caseLabel = 'Reference'
    else:
        caseLabel = case
    
   
    data_used = PODA_Model['Google_Apple_Mobility_Projection_'+case].reset_index()
    
    data_used = data_used[(data_used['date']> pd.to_datetime(start_Date))]
    data_used = data_used.set_index('date')
    
   
    NHTS_Category_Share = PODA_Model['NHTS Category Share']
    NHTS_State_Fuel_Share = PODA_Model['NHTS State Fuel Share']
    df_StateName_Code = PODA_Model['StateName_StateCode']
    
    
    cols = ['State Name']
    data_used = data_used.join(df_StateName_Code.set_index(cols), on=cols, how='left')
    
    data_used = data_used.join(NHTS_Category_Share.set_index('State Code'), on='State Code', how='left')
    
    
    '''
    #Google mobility-fuel correlation model
    '''
    #load model correlation factors
    factor = PODA_Model['Google_Mobility_EIA_Factor']
    # data_used['work factor'] = 1 + data_used['Workplaces']/100*factor[0]
    # data_used['school factor'] = 1 + data_used['Workplaces']/100*factor[1]
    # data_used['medical factor'] = 1 + data_used['Grocery and Pharmacy']/100*factor[2]
    # data_used['shopping factor'] = 1 + data_used['Grocery and Pharmacy']/100*factor[3]
    # data_used['social factor'] = 1 + data_used['Retail and Recreation']/100*factor[4]
    # data_used['park factor'] = 1 + data_used['Parks']/100*factor[5]
    # data_used['transport someone factor'] = 1+ data_used['Retail and Recreation']/100*factor[7]
    # data_used['meals factor'] = 1 + data_used['Retail and Recreation']/100*factor[6]
    # data_used['else factor'] = 1+ data_used['Retail and Recreation']/100*factor[7]
    
    data_used['work factor'] = 1 + data_used['Workplaces']/100*factor[0]
    data_used['school factor'] = 1 + data_used['Workplaces']/100*factor[1]
    data_used['medical factor'] = 1 + data_used['Grocery and Pharmacy']/100*factor[2]
    data_used['shopping factor'] = 1 + data_used['Grocery and Pharmacy']/100*factor[3]
    data_used['social factor'] = 1 + data_used['Retail and Recreation']/100*factor[4]
    data_used['park factor'] = 1 + data_used['Parks']/100*factor[5]
    data_used['transport someone factor'] = 1+ data_used['Retail and Recreation']/100*factor[7]     #Workplaces
    data_used['meals factor'] = 1 + data_used['Retail and Recreation']/100*factor[6]
    data_used['else factor'] = 1+ data_used['Retail and Recreation']/100*factor[7]          #workplace
    
    data_used['Google State Mobility Predict'] = (data_used['Work']*data_used['work factor'] + \
        data_used['School/Daycare/Religious activity']*data_used['school factor'] + \
            data_used['Medical/Dental services']*data_used['medical factor'] + \
                data_used['Shopping/Errands']*data_used['shopping factor'] + \
                    data_used['Social/Recreational']*factor[8]*data_used['social factor'] + \
                        data_used['Social/Recreational']*(1-factor[8])*data_used['park factor'] + \
                            data_used['Meals']*data_used['meals factor'] +\
                                data_used['Transport someone']*data_used['transport someone factor'] + \
                                    data_used['Something else']*data_used['else factor'])/100 + factor[9]
        
    aa = data_used.join(NHTS_State_Fuel_Share.set_index('State Name'), on='State Name', how='left')
    
    aa['Google fuel factor'] = aa['Google State Mobility Predict']*aa['Percentage gasoline']
    
    aa['Apple fuel factor']=aa['Apple State Mobility Predict']*aa['Percentage gasoline']
    
    aa['Date'] = aa.index
    
    day_Shift = int(factor[10])
    x = aa.sum(level='date')
    x = x[['Google fuel factor', 'Apple fuel factor']]
    # x['Date'] =x.index+pd.DateOffset(days=day_Shift)
    
    '''
    apple mobility-fuel correlation 
    '''
    apple_x = x['Apple fuel factor'].to_numpy()
    apple_x_length = len(apple_x)
    apple_x=apple_x.reshape(apple_x_length, 1)
    regr = PODA_Model['Apple_EIA_Regression']
    # regr_coef = regr.coef_
    # print('reg_coeff: ', regr_coef)
    # regr_interp = regr.intercept_
    # print('reg_interp: ', regr_interp)
    Apple_fuel_Demand_Pred = regr.predict(apple_x)
    # aa['Apple Fuel Demand Predict'] = fuel_Demand_Apple_Pred
    baseline = 8722              #average of EIA between Jan 03-Feb 07(thousand bpd)
    PODA_Model['EIA_Baseline'] = baseline
    
    data_save = aa[['Date', 'State Name', 'State Code', 'Google State Mobility Predict', 'Apple State Mobility Predict']]
    data_save['Google State Mobility Predict'] = data_save['Google State Mobility Predict']*100
    data_save.to_excel('./Fuel Demand Projection/Mobility_State_'+YYG_projection_Date+case+isopen+'.xlsx')
    
    x['Google Fuel Demand Predict'] = x['Google fuel factor']*baseline
    x['Apple Fuel Demand Predict'] = Apple_fuel_Demand_Pred
    # x.to_excel('./Fuel Demand Projection/Mobility_US_'+YYG_projection_Date+case+isopen+'.xlsx')
    
    PODA_Model['Fuel_Demand_Projection_'+case]=x
    PODA_Model['Mobility_State_Level_Projection_'+case]=data_save
    
    fig1 = plt.figure(figsize=(6, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(x.index, x['Google Fuel Demand Predict'], '-', label='Google Mobility (Predicted')
    ax1.plot(x.index, x['Apple Fuel Demand Predict'], '--g', label='Apple Mobility (Predicted')
    ax1.plot(fuel_Demand_EIA.index - pd.DateOffset(days=day_Shift), fuel_Demand_EIA['Gasoline'], '--s', label='EIA Weekly Fuel Demand')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Motor Gasoline Demand (thousand BPD)')
    ax1.set_ylim(4000, 10000)
    ax1.set_title('Fuel Demand: '+model_mark+case+' YYG:'+YYG_projection_Date + ' MLmodel:' +ML_Model+isopen)
    ax1.legend()
        
       
    # if (case == 'mean') & (isopen ==''):
    #     data_save.to_excel('C:/Users/hexx/Box Sync/Energy-COVID-19/Data for Website/Mobility_State_'+YYG_projection_Date+case+'.xlsx')
        

np.save(("./PODA_Model_"+today+".npy"), PODA_Model)

createFolder('./PODA_Model')
copyfile('./PODA_Model_'+today+'.npy', './PODA_Model/PODA_Model_'+today+'.npy')







