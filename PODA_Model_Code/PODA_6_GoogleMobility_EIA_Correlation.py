# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:19:50 2020

@author: hexx
"""

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize, Bounds
from myFunctions import def_add_datashift, createFolder, Google_factor

'''
Data preparation
'''
#weekly fuel demand

today = pd.to_datetime('today')
today =today.strftime("%Y-%m-%d")

PODA_Model = np.load("./PODA_Model_"+today+".npy",allow_pickle='TRUE').item()


# Model_Date = np.load("./Model_Parameter.npy",allow_pickle='TRUE').item()
google_Mobility_Day = PODA_Model['ML_File_Date']
start_Date = '03-25-2020'
end_Date = PODA_Model['ML_File_Date']   #'5-2-2020'
# google_Mobility_Day='2020-05-17'

fuel_Demand_EIA = pd.read_excel('https://www.eia.gov/dnav/pet/xls/PET_CONS_WPSUP_K_W.xls', sheet_name = 'Data 1', header=2)
fuel_Demand_EIA['Date'] = pd.to_datetime(fuel_Demand_EIA['Date'])
fuel_Demand_EIA.rename(columns={'Weekly U.S. Product Supplied of Finished Motor Gasoline  (Thousand Barrels per Day)':'Gasoline'}, inplace=True)
fuel_Demand_EIA = fuel_Demand_EIA.drop(columns=['Weekly U.S. Product Supplied of Petroleum Products  (Thousand Barrels per Day)', 
                             'Weekly U.S. Product Supplied of Kerosene-Type Jet Fuel  (Thousand Barrels per Day)', 
                             'Weekly U.S. Product Supplied of Distillate Fuel Oil  (Thousand Barrels per Day)', 
                             'Weekly U.S. Product Supplied of Residual Fuel Oil  (Thousand Barrels per Day)', 
                             'Weekly U.S. Product Supplied of Propane and Propylene  (Thousand Barrels per Day)',
                             'Weekly U.S. Product Supplied of Other Oils  (Thousand Barrels per Day)'])

fuel_Demand_EIA_save = fuel_Demand_EIA[(fuel_Demand_EIA['Date'] > pd.to_datetime('01-01-2020'))]
PODA_Model['Fuel_Demand_EIA'] = fuel_Demand_EIA_save
fuel_Demand_EIA = fuel_Demand_EIA[(fuel_Demand_EIA['Date'] > pd.to_datetime(start_Date)) & (fuel_Demand_EIA['Date'] <= pd.to_datetime(end_Date))]
fuel_Demand_EIA = fuel_Demand_EIA.set_index('Date')



case = 'mean'
cwd = os.getcwd()
# datafile = './ML Files/State_Level_Data_forML_'+google_Mobility_Day+'.xlsx'

# datafile =str(Path(cwd)) + datafile


# pd_all = pd.read_excel(datafile)
# pd_all = PODA_Model['ML_Data'].reset_index()
# projectionFile ='/MObility_Projection_'+google_Mobility_End_Day+'_'+case+'.xlsx'
# projectionFile = str(Path(cwd))+projectionFile
# mobility_Proj_Data = pd.read_excel(projectionFile)
# data_used = pd_all[['date', 'WeekDay', 'State Name', 'retail_and_recreation', 'grocery_and_pharmacy', 'workplaces', 'parks',
#                    'EmergDec', 'SchoolClose', 'NEBusinessClose', 
#                    'RestaurantRestrict', 'StayAtHome']]
# del pd_all

data_used = PODA_Model['Google_Apple_Mobility_Projection_mean']

data_used = data_used[(data_used['date']> (pd.to_datetime(start_Date)-pd.DateOffset(days=7))) & (data_used['date'] < pd.to_datetime(end_Date))]
data_used = data_used.set_index('date')

NHTS_Category_Share = pd.read_excel('NHTS.xlsx', sheet_name='Category Share')
NHTS_State_Fuel_Share = pd.read_excel('NHTS.xlsx', sheet_name='State Fuel Share')

PODA_Model['NHTS Category Share'] = NHTS_Category_Share
PODA_Model['NHTS State Fuel Share'] =NHTS_State_Fuel_Share

df_StateName_Code = pd.read_excel(cwd+'/US_StateCode_List.xlsx', sheet_name='Sheet1', header=0)

cols = ['State Name']
data_used = data_used.join(df_StateName_Code.set_index(cols), on=cols, how='left')

data_used = data_used.join(NHTS_Category_Share.set_index('State Code'), on='State Code', how='left')

EIA_fuel = fuel_Demand_EIA[['Gasoline']]

def min_func(factor):
    global EIA_fuel
      
    data_used['work factor'] = 1 + data_used['Workplaces']/100*factor[0]
    data_used['school factor'] = 1 + data_used['Workplaces']/100*factor[1]
    data_used['medical factor'] = 1 + data_used['Grocery and Pharmacy']/100*factor[2]
    data_used['shopping factor'] = 1 + data_used['Grocery and Pharmacy']/100*factor[3]
    data_used['social factor'] = 1 + data_used['Retail and Recreation']/100*factor[4]
    data_used['park factor'] = 1 + data_used['Parks']/100*factor[5]
    data_used['transport someone factor'] = 1+ data_used['Retail and Recreation']/100*factor[7]      #Workplaces
    data_used['meals factor'] = 1 + data_used['Retail and Recreation']/100*factor[6]
    data_used['else factor'] = 1+ data_used['Retail and Recreation']/100*factor[7]       #workplaces
    
    data_used['accumulated factor'] = (data_used['Work']*data_used['work factor'] + \
        data_used['School/Daycare/Religious activity']*data_used['school factor'] + \
            data_used['Medical/Dental services']*data_used['medical factor'] + \
                data_used['Shopping/Errands']*data_used['shopping factor'] + \
                    data_used['Social/Recreational']*factor[8]*data_used['social factor'] + \
                        data_used['Social/Recreational']*(1-factor[8])*data_used['park factor'] + \
                            data_used['Meals']*data_used['meals factor'] +\
                                data_used['Transport someone']*data_used['transport someone factor'] + \
                                    data_used['Something else']*data_used['else factor'])/100 + factor[9]
    
                                                         
    DayShift = int(factor[10])
    aa = data_used.join(NHTS_State_Fuel_Share.set_index('State Name'), on='State Name', how='left')
    
    
    aa['fuel factor'] = aa['accumulated factor']*aa['Percentage gasoline']

    x = aa.sum(level='date')
    x = x[['fuel factor','WeekDay']]
    x['WeekDay'] = x['WeekDay']/50

    # demand_factor = 0.93840494
    baseline = 8722              #average of EIA between Jan 03-Feb 07(thousand bpd)
    
    x['Shifted Date'] = x.index+pd.DateOffset(days=DayShift)
    
    for i, date_i in enumerate(fuel_Demand_EIA.index):
        # print(i, date_i)
        Google_weekly = x[(x['Shifted Date']<=pd.to_datetime(date_i)) & (x['Shifted Date']>(pd.to_datetime(date_i)-pd.DateOffset(days=7)))] 
        #
        # apple_weekly['fuel factor'].mean(afuel_factoris =0)
        EIA_fuel.loc[date_i, 'Google'] = Google_weekly['fuel factor'].mean(axis =0)
    
    # EIA_Fuel = fuel_Demand_EIA.join(x.set_index('Date'), on='Date', how='left')
    EIA_fuel = EIA_fuel.dropna()

    EIA_fuel['fuelpred'] = EIA_fuel['Google']*baseline
    # EIA_fuel = EIA_fuel[EIA_fuel.index !=  pd.to_datetime('05-08-2020')]
    EIA_fuel['least_square'] = ((EIA_fuel['Gasoline']-EIA_fuel['fuelpred'])/EIA_fuel['Gasoline'])**2
    retu = EIA_fuel['least_square'].sum()
    return retu
    

x0=[1, 1, 1, 1, 1, 1, 1, 1 , 0.5, 0, 3]
bounds = Bounds([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0], [1.2, 1.2, 1.2, 1.2, 1.2, 1, 1.3, 1.3, 1, 0.05, 10])

# res = minimize(min_func, x0, method='trust-constr', options={'verbose': 1, 'disp': True}, bounds=bounds)
res = minimize(min_func, x0, method='SLSQP', bounds=bounds)
print(res.x)
print(res.fun)

a = res.x

createFolder('./Fuel Demand Projection')
np.savetxt('./Fuel Demand Projection/Fuel_mobility_factor' + google_Mobility_Day +'.csv', a, delimiter=",")

PODA_Model['Google_Mobility_EIA_Factor']=a
np.save(("./PODA_Model_"+today+".npy"), PODA_Model)

# fig1 = plt.figure(figsize=(6, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# ax1.plot(x.index, x['fuel factor']*baseline/100, '-o', label=['pred'])
# ax1.plot(fuel_Demand_EIA.index-pd.DateOffset(days=day_Shift), fuel_Demand_EIA['Gasoline']*demand_factor, '--s', label='EIA')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Y')
# ax1.set_title('fuel demand: shift:'+str(day_Shift)+' days')
# ax1.legend()
    
    
    
