# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:19:50 2020

@author: hexx
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from myFunctions import createFolder
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

'''
Data preparation
'''
#weekly fuel demand

today = pd.to_datetime('today')
today =today.strftime("%Y-%m-%d")
today = '2020-09-12'

PODA_Model = np.load("./PODA_Model_"+today+".npy",allow_pickle='TRUE').item()


google_Mobility_Day = PODA_Model['ML_File_Date']
start_Date = '04-05-2020'
end_Date = PODA_Model['ML_File_Date']
# end_Date = today

fuel_Demand_EIA = pd.read_excel('https://www.eia.gov/dnav/pet/xls/PET_CONS_WPSUP_K_W.xls', 
                                sheet_name = 'Data 1', 
                                header=2)

fuel_Demand_EIA['Date'] = pd.to_datetime(fuel_Demand_EIA['Date'])

fuel_Demand_EIA.rename(columns={'Weekly U.S. Product Supplied of Finished Motor Gasoline  (Thousand Barrels per Day)':'Gasoline'}, 
                       inplace=True)

fuel_Demand_EIA = fuel_Demand_EIA.drop(columns=['Weekly U.S. Product Supplied of Petroleum Products  (Thousand Barrels per Day)', 
                             'Weekly U.S. Product Supplied of Kerosene-Type Jet Fuel  (Thousand Barrels per Day)', 
                             'Weekly U.S. Product Supplied of Distillate Fuel Oil  (Thousand Barrels per Day)', 
                             'Weekly U.S. Product Supplied of Residual Fuel Oil  (Thousand Barrels per Day)', 
                             'Weekly U.S. Product Supplied of Propane and Propylene  (Thousand Barrels per Day)',
                             'Weekly U.S. Product Supplied of Other Oils  (Thousand Barrels per Day)'])

fuel_Demand_EIA_save = fuel_Demand_EIA[(fuel_Demand_EIA['Date'] > 
                                        pd.to_datetime('01-01-2020'))]

PODA_Model['Fuel_Demand_EIA'] = fuel_Demand_EIA_save

fuel_Demand_EIA = fuel_Demand_EIA[(fuel_Demand_EIA['Date'] > pd.to_datetime(start_Date)) 
                                  & (fuel_Demand_EIA['Date'] <= pd.to_datetime(end_Date))]

fuel_Demand_EIA = fuel_Demand_EIA.set_index('Date')


case = 'mean'

data_used = PODA_Model['ML_Data']

data_used['date'] = data_used.index

data_used = data_used[(data_used['date'] > (pd.to_datetime(start_Date) - pd.DateOffset(days=7))) 
                      & (data_used['date'] < pd.to_datetime(end_Date))]

NHTS_Category_Share = pd.read_excel('./data/NHTS.xlsx', 
                                    sheet_name='Category Share')

NHTS_State_Fuel_Share = pd.read_excel('./data/NHTS.xlsx', 
                                      sheet_name='State Fuel Share')

PODA_Model['NHTS Category Share'] = NHTS_Category_Share
PODA_Model['NHTS State Fuel Share'] = NHTS_State_Fuel_Share

df_StateName_Code = pd.read_excel('./data/US_StateCode_List.xlsx', 
                                  sheet_name='Sheet1', 
                                  header=0)

cols = ['State Name']
data_used = data_used.join(df_StateName_Code.set_index(cols), 
                           on=cols, 
                           how='left')

data_used = data_used.join(NHTS_Category_Share.set_index('State Code'), 
                           on='State Code', 
                           how='left')

EIA_fuel = fuel_Demand_EIA[['Gasoline']]

def min_func(factor):
    global EIA_fuel
      
    data_used['work factor'] = 1 + data_used['workplaces']/100*factor[0]
    data_used['school factor'] = 1 + data_used['workplaces']/100*factor[1]
    data_used['medical factor'] = 1 + data_used['grocery_and_pharmacy']/100*factor[2]
    data_used['shopping factor'] = 1 + data_used['grocery_and_pharmacy']/100*factor[3]
    data_used['social factor'] = 1 + data_used['retail_and_recreation']/100*factor[4]
    data_used['park factor'] = 1 + data_used['parks']/100*factor[5]
    data_used['transport someone factor'] = 1 + data_used['retail_and_recreation']/100*factor[7]
    data_used['meals factor'] = 1 + data_used['retail_and_recreation']/100*factor[6]
    data_used['else factor'] = 1+ data_used['retail_and_recreation']/100*factor[7]
    
    data_used['accumulated factor'] = (
        data_used['Work'] * data_used['work factor'] +
        data_used['School/Daycare/Religious activity'] * data_used['school factor'] +
        data_used['Medical/Dental services']*data_used['medical factor'] + 
        data_used['Shopping/Errands']*data_used['shopping factor'] + 
        data_used['Social/Recreational']*factor[8]*data_used['social factor'] + 
        data_used['Social/Recreational']*(1-factor[8])*data_used['park factor'] + 
        data_used['Meals']*data_used['meals factor'] +
        data_used['Transport someone']*data_used['transport someone factor'] +
        data_used['Something else']*data_used['else factor'])/100 + factor[9]
                                                            
    DayShift = int(factor[10])
    aa = data_used.join(NHTS_State_Fuel_Share.set_index('State Name'), 
                        on='State Name', 
                        how='left')
    
    aa['fuel factor'] = aa['accumulated factor'] * aa['Percentage gasoline']

    x = aa.sum(level='date')
    x = x[['fuel factor','WeekDay']]
    x['WeekDay'] = x['WeekDay']/50

    baseline = 8722
    
    x['Shifted Date'] = x.index + pd.DateOffset(days=DayShift)
    
    for i, date_i in enumerate(fuel_Demand_EIA.index):
        
        Google_weekly = x[(x['Shifted Date']<=pd.to_datetime(date_i)) 
                          & (x['Shifted Date']>(pd.to_datetime(date_i)-pd.DateOffset(days=7)))] 
        
        EIA_fuel.loc[date_i, 'Google'] = Google_weekly['fuel factor'].mean(axis =0)
    
    EIA_fuel = EIA_fuel.dropna()

    EIA_fuel['fuelpred'] = EIA_fuel['Google']*baseline
    EIA_fuel['least_square'] = ((EIA_fuel['Gasoline']-EIA_fuel['fuelpred'])/EIA_fuel['Gasoline'])**2
    retu = EIA_fuel['least_square'].sum()
    return retu
    
#index            (0)  (1)  (2)  (3)   (4)  (5)  (6) (7)   (8)    (9)  (10)
x0 =            [  1,   1,   1,   1,   1,   1,   1,   1,    0.5,     0,   0]

bounds = Bounds([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,    0,     0,   0], 
                [1.2, 1.2, 1.2, 1.2, 1.2, 2, 1.2, 1.2,    1,  0.05,  10])

res = minimize(min_func, x0, method='SLSQP', bounds=bounds)

print('optim factor = ')
for index, val in np.ndenumerate(res.x):
    print('\t factor[{}] = {:.2e}'.format(index[0], val))
print('optim loss = {:.3e}'.format(res.fun))

a = res.x

createFolder('./Fuel Demand Projection')
np.savetxt('./Fuel Demand Projection/Fuel_mobility_factor' 
           + google_Mobility_Day +'.csv', a, delimiter = ",")

PODA_Model['Google_Mobility_EIA_Factor'] = a
np.save(("./PODA_Model_"+today+".npy"), PODA_Model)

r2 = r2_score(EIA_fuel['fuelpred'], EIA_fuel['Gasoline'])

print('r2 = {:.4e}'.format(r2))

fig1 = plt.figure(figsize=(6, 5))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(EIA_fuel.index, EIA_fuel['fuelpred'], '-', 
         label='pred')

ax1.plot(EIA_fuel.index, EIA_fuel['Gasoline'], '--o', 
         label='EIA')

ax1.set_xlabel('Date')
ax1.set_ylabel('Fuel Demand')
plt.xticks(rotation=45)
ax1.legend()


fig1 = plt.figure(figsize=(6, 5))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(EIA_fuel['Gasoline'], EIA_fuel['fuelpred'], 'o', 
         label='pred')

ax1.plot([EIA_fuel['Gasoline'].min(), EIA_fuel['Gasoline'].max()], 
         [EIA_fuel['Gasoline'].min(), EIA_fuel['Gasoline'].max()], 
         '--', 
         label='y = x')

ax1.set_xlabel('True')
ax1.set_ylabel('Pred')
ax1.legend()
