# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:19:50 2020

@author: hexx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


Model_Date='2020-07-08'
PODA_Model = np.load("./PODA_Model/PODA_Model_"+Model_Date+".npy",allow_pickle='TRUE').item()
'''
Data preparation
'''
google_Mobility_Day = PODA_Model['ML_File_Date']
start_Date = '2-15-2020'
end_Date = PODA_Model['ML_File_Date'] 
ML_Model=PODA_Model['ML_File_Date']

fuel_Demand_EIA = PODA_Model['Fuel_Demand_EIA']

pd_all = PODA_Model['ML_Data'].reset_index()


data_used = pd_all[['date', 'WeekDay', 'State Name', 'retail_and_recreation', 'grocery_and_pharmacy', 'workplaces', 'parks',
                   'EmergDec', 'SchoolClose', 'NEBusinessClose', 
                   'RestaurantRestrict', 'StayAtHome']]

data_used = data_used[(data_used['date']> pd.to_datetime(start_Date)) & (data_used['date'] < pd.to_datetime(end_Date))]
data_used = data_used.set_index('date')
del pd_all

NHTS_Category_Share = PODA_Model['NHTS Category Share']
NHTS_State_Fuel_Share = PODA_Model['NHTS State Fuel Share']
df_StateName_Code = PODA_Model['StateName_StateCode']

cols = ['State Name']
data_used = data_used.join(df_StateName_Code.set_index(cols), on=cols, how='left')

data_used = data_used.join(NHTS_Category_Share.set_index('State Code'), on='State Code', how='left')


factor = PODA_Model['Google_Mobility_EIA_Factor']
print(factor)
data_used['work factor'] = 1 + data_used['workplaces']/100*factor[0]
data_used['school factor'] = 1 + data_used['workplaces']/100*factor[1]
data_used['medical factor'] = 1 + data_used['grocery_and_pharmacy']/100*factor[2]
data_used['shopping factor'] = 1 + data_used['grocery_and_pharmacy']/100*factor[3]
data_used['social factor'] = 1 + data_used['retail_and_recreation']/100*factor[4]
data_used['park factor'] = 1 + data_used['parks']/100*factor[5]
data_used['transport someone factor'] = 1+ data_used['retail_and_recreation']/100*factor[7]
data_used['meals factor'] = 1 + data_used['retail_and_recreation']/100*factor[6]
data_used['else factor'] = 1+ data_used['retail_and_recreation']/100*factor[7]

data_used['accumulated factor'] = (data_used['Work']*data_used['work factor'] + \
    data_used['School/Daycare/Religious activity']*data_used['school factor'] + \
        data_used['Medical/Dental services']*data_used['medical factor'] + \
            data_used['Shopping/Errands']*data_used['shopping factor'] + \
                data_used['Social/Recreational']*factor[8]*data_used['social factor'] + \
                    data_used['Social/Recreational']*(1-factor[8])*data_used['park factor'] + \
                        data_used['Meals']*data_used['meals factor'] +\
                            data_used['Transport someone']*data_used['transport someone factor'] + \
                                data_used['Something else']*data_used['else factor'])/100 + factor[9]
                            

aa = data_used.join(NHTS_State_Fuel_Share.set_index('State Name'), on='State Name', how='left')

aa['fuel factor'] = aa['accumulated factor']*aa['Percentage gasoline']

day_Shift = int(factor[10])
x = aa.sum(level='date')
x = x[['fuel factor','WeekDay']]
x['WeekDay'] = x['WeekDay']/50
x['Shifted Date'] =x.index+pd.DateOffset(days=day_Shift)

# demand_factor = 0.93840494
baseline = PODA_Model['EIA_Baseline']            #average of EIA between Jan 03-Feb 07(thousand bpd)

fuel_Demand_EIA = fuel_Demand_EIA.join(x.set_index('Shifted Date'), on='Date', how='left')
fuel_Demand_EIA['fuelpred'] = fuel_Demand_EIA['fuel factor']*baseline
fuel_Demand_EIA['least_square'] = ((fuel_Demand_EIA['Gasoline']-fuel_Demand_EIA['fuelpred'])/fuel_Demand_EIA['Gasoline'])**2

retu = fuel_Demand_EIA['least_square'].sum()

fig1 = plt.figure(figsize=(6, 5))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(x['Shifted Date'], x['fuel factor']*baseline, '-o', label=['pred'])
ax1.plot(fuel_Demand_EIA['Date'], fuel_Demand_EIA['Gasoline'], '--s', label='EIA')
ax1.set_xlabel('Date')
ax1.set_ylabel('Y')
ax1.set_title('fuel demand: shift:'+str(day_Shift)+' days')
ax1.legend()
    
    
    
