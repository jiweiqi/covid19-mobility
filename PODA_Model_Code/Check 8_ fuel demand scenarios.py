# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:03:20 2020

@author: hexx
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import joblib
from myFunctions import def_add_datashift, createFolder
import matplotlib as mpl
import seaborn as sns

# mpl.style.use('classic')
'''needs input'''
start_Date = '3-1-2020'

sns.set_style("ticks")
Line_Style =['.-m', '-.r', '-b', '--g']

model_mark =''

# today = pd.to_datetime('today')
# today =today.strftime("%Y-%m-%d")
PODA_Model_Date='2020-07-08'

moving_avg =7


PODA_Model = np.load("./PODA_Model_"+PODA_Model_Date+".npy",allow_pickle='TRUE').item()
fuel_mobility_factor_file = PODA_Model['ML_File_Date']
apple_fuel_Factor_file = PODA_Model['Apple_File_Date']
ML_Model = PODA_Model['ML_File_Date']
YYG_projection_Date=PODA_Model['YYG_File_Date']

# fuel_Demand_EIA = pd.read_excel('https://www.eia.gov/dnav/pet/xls/PET_CONS_WPSUP_K_W.xls', sheet_name = 'Data 1', header=2)
fuel_Demand_EIA = PODA_Model['Fuel_Demand_EIA'].reset_index()
fuel_Demand_EIA = fuel_Demand_EIA.set_index('Date')

data_plot = pd.DataFrame()   #define an empty data frame
caseList =['mean', 'lower', 'upper', 'MIT']

fuel_Demand_Mean = PODA_Model['Fuel_Demand_Projection_mean'][['Google Fuel Demand Predict']]
fuel_Demand_upper = PODA_Model['Fuel_Demand_Projection_upper'][['Google Fuel Demand Predict']]
fuel_Demand_lower = PODA_Model['Fuel_Demand_Projection_lower'][['Google Fuel Demand Predict']]
fuel_Demand_MIT = PODA_Model['Fuel_Demand_Projection_MIT'][['Google Fuel Demand Predict']]
fuel_Demand_MIT['Fuel_Demand_Projection_MIT_7daymoving']= fuel_Demand_MIT['Google Fuel Demand Predict'].rolling(window=moving_avg).mean()
# for case in caseList:
        


fuel_Demand_Normal = pd.read_excel('./No pandemic fuel demand in 2020.xlsx', sheet_name='Sheet1', header=0)


fig1 = plt.figure(figsize=(6, 5))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(fuel_Demand_Normal['Date'], fuel_Demand_Normal['Average value (2016-2019)']/1000, '.-k', label='Non-pandemic')
ax1.plot(fuel_Demand_MIT.index, fuel_Demand_MIT['Google Fuel Demand Predict'].rolling(window=moving_avg).mean()/1000, '.-m', label='MIT')
ax1.plot(fuel_Demand_lower.index, fuel_Demand_lower['Google Fuel Demand Predict'].rolling(window=moving_avg).mean()/1000, '-.r', label='YYG Optimistic')
ax1.plot(fuel_Demand_Mean.index, fuel_Demand_Mean['Google Fuel Demand Predict'].rolling(window=moving_avg).mean()/1000, '-', label='YYG Reference')
ax1.plot(fuel_Demand_upper.index, fuel_Demand_upper['Google Fuel Demand Predict'].rolling(window=moving_avg).mean()/1000, '--g', label='YYG Pessimistic')

data_Save = fuel_Demand_Mean
data_Save['Fuel_YYG_Pessimistic_7daymovingavg'] = fuel_Demand_upper['Google Fuel Demand Predict'].rolling(window=moving_avg).mean()
data_Save['Fuel_YYG_Reference_7daymovingavg'] = fuel_Demand_Mean['Google Fuel Demand Predict'].rolling(window=moving_avg).mean()
data_Save['Fuel_YYG_optismistic_7daymovingavg'] = fuel_Demand_lower['Google Fuel Demand Predict'].rolling(window=moving_avg).mean()
data_Save['Fuel_YYG_Pessimistic'] = fuel_Demand_upper['Google Fuel Demand Predict']
data_Save['Fuel_YYG_Reference'] = fuel_Demand_Mean['Google Fuel Demand Predict']
data_Save['Fuel_YYG_optismistic'] = fuel_Demand_lower['Google Fuel Demand Predict']


ax1.fill_between(fuel_Demand_upper.index, fuel_Demand_upper['Google Fuel Demand Predict'].rolling(window=moving_avg).mean()/1000, fuel_Demand_lower['Google Fuel Demand Predict'].rolling(window=moving_avg).mean()/1000, 
                 # where = (data_plot['Google Fuel Demand upper'] > data_plot['Google Fuel Demand lower']),
                 alpha=0.2, 
                 color='C1')
ax1.plot(fuel_Demand_EIA.index-pd.DateOffset(days=0), fuel_Demand_EIA['Gasoline']/1000, '--d', label='EIA Weekly')
ax1.set_xlabel('Date')
ax1.set_xlim(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-11-01'))
fig1.autofmt_xdate(rotation=45)
ax1.set_ylim([4, 10])
ax1.set_ylabel('Motor Gasoline Demand (Million BPD)')
ax1.set_title(' Gasoline Demand Projection ('+str(moving_avg)+'-day Moving Average)', FontSize=14)
ax1.legend()


# fig2 = plt.figure(figsize=(6, 5))
# ax1 = fig2.add_subplot(1, 1, 1)
# ax1.plot(data_plot['Date mean'], data_plot['Google Fuel Demand lower'], '--g', label='Optimistic')
# ax1.plot(data_plot['Date mean'], data_plot['Google Fuel Demand mean'], '-', label='Reference')
# ax1.plot(data_plot['Date mean'], data_plot['Google Fuel Demand upper'], '-.r', label='Pessimistic')
# ax1.plot(fuel_Demand_EIA.index-pd.DateOffset(days=5), fuel_Demand_EIA['Gasoline'], '--d', label='EIA Weekly Fuel Demand')
# ax1.set_xlabel('Date')
# ax1.set_xlim(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-09-01'))
# ax1.set_ylim([4000, 11000])
# ax1.set_ylabel('Motor Gasoline Demand (x1000 BPD)')
# ax1.set_title('Gasoline Demand Projection', FontSize=14)
# ax1.legend()


plt.savefig('./Figures_for_Paper/Fig4_FuelDemand_3_Scenarios_'+str(moving_avg)+'daymoving.png', dpi=300, format="png", bbox_inches='tight')
# plt.savefig('./Figures_for_Paper/Fig4_FuelDemand_3_Scenarios.png', dpi=300, format="pdf", bbox_inches='tight')



data_plot.to_excel('C:/Users/hexx/Box Sync/Energy-COVID-19/Data for Website/Fuel_Demand_3_Scenarios_'+ML_Model+'.xlsx')




# fuel_Demand_MIT.to_excel('Figure4_MIT_Projection.xlsx')

# data_Save.to_excel('Figure4_YYG_Google_Projection.xlsx')



















    