# -*- coding: utf-8 -*-
"""
Created on Tue May  12 17:07:00 2020

@author: hexx
This code do the following: 
    (1)saves policy, COVID, and Projection data downloaded online to local folder
    (2)process and saved data to be usded to project mobility

"""
'''
Need to download the MIT projection data and save it in the model folder
Update the file name in line 25
'''


import pandas as pd
import numpy as np
import os
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import copy
from myFunctions import def_add_datashift, createFolder

today_x = pd.to_datetime('today')
today =today_x.strftime("%Y-%m-%d")

'''
Please download the file from https://www.covidanalytics.io/projections
'''
MIT_file_name = 'MIT_covid_analytics_projections_'+today+'.csv'

createFolder('./Mobility projection')
        
scenario_cases = ['mean']      #'upper', 'lower',
shiftDay = 16   #YYG model shift between new infected and confirmed

today_x = pd.to_datetime('today')
today =today_x.strftime("%Y-%m-%d")

PODA_Model = np.load(("./PODA_Model_"+today+".npy"),allow_pickle='TRUE').item()


MIT_Data = pd.read_csv(MIT_file_name, header=0)

MIT_Data = MIT_Data[(MIT_Data['Country'] == 'US') & (MIT_Data['Province']!='District of Columbia')]

MIT_Data = MIT_Data[['Country', 'Province', 'Day', 'Total Detected', 'Total Detected Deaths']]
MIT_Data.rename(columns={'Province':'State Name', 'Day':'date', 'Total Detected': 'State Total Confirmed', 
                         'Total Detected Deaths':'State Total Deaths'}, inplace=True)

MIT_Data['date'] = pd.to_datetime(MIT_Data['date'])
MIT_Data['Date'] = MIT_Data['date']
MIT_Data.set_index('date', inplace=True)
PODA_Model['MIT_Data'] = MIT_Data
PODA_Model['MIT_File_Date'] = today
PODA_Model['MIT_Projection'] = MIT_Data



YYG_Date = PODA_Model['YYG_File_Date']


#create folder to save YYG Projection 
createFolder('./YYG Data/'+YYG_Date)

# createFolder('./COVID/'+today)


df_StateName_Code = PODA_Model['StateName_StateCode']
ML_Data = PODA_Model['ML_Data']

# load Policy Data
df_Policy = pd.read_csv('https://raw.githubusercontent.com/COVID19StatePolicy/SocialDistancing/master/data/USstatesCov19distancingpolicy.csv', encoding= 'unicode_escape')
createFolder('./Policy File')
df_Policy.to_excel('./Policy File/Policy'+today+'.xlsx')     # save policy data

# Read Population Data
df_Population = PODA_Model['State Population']

#Read County Area
df_Area = PODA_Model['State Area']

#Employment 
df_Employee = PODA_Model['State Employment']


'''
Calculate the new infected to confirmed correlation
'''
# df_US_Projection = pd.read_csv('https://raw.githubusercontent.com/youyanggu/covid19_projections/master/projections/'+YYG_Date+'/US.csv')
# df_US_Projection.to_csv('./YYG Data/'+YYG_Date+'/US.csv')     # save US Projection data
# df_US_Projection['date'] = pd.to_datetime(df_US_Projection['date'])
# df_US_Projection.set_index('date', inplace=True)

MIT_State_Projection = PODA_Model['MIT_Projection'][['State Total Confirmed', 'State Total Deaths', 'State Name']]

MIT_US_Projection = MIT_State_Projection[MIT_State_Projection['State Name'] == 'None']
MIT_US_Projection.rename(columns={'State Total Confirmed': 'US Total Confirmed', 
                         'State Total Deaths': 'US Total Death'}, inplace=True)


'''
compare MIT and YYG data
'''
#load YYG Data
YYG_Pred_mean = PODA_Model['Data_for_Mobility_Projection_mean'][['US Daily Confirmed', 'US Daily Death']].drop_duplicates()
YYG_Pred_lower = PODA_Model['Data_for_Mobility_Projection_lower'][['US Daily Confirmed', 'US Daily Death']].drop_duplicates()
YYG_Pred_upper = PODA_Model['Data_for_Mobility_Projection_upper'][['US Daily Confirmed', 'US Daily Death']].drop_duplicates()


merged_select = MIT_US_Projection
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(1, 1, 1)
# normalized scale
ax.plot(merged_select.index, merged_select['US Total Confirmed'].diff(), '-.o', label='MIT Predicted Confirmed')
# ax.plot(merged_select.index, merged_select['predicted_total_infected_mean'], 'o', label='YYG Predicted')

ax.plot(YYG_Pred_mean.index, YYG_Pred_mean['US Daily Confirmed'], '--b', label='YYG Predicted Confirmed mean')
ax.plot(YYG_Pred_lower.index, YYG_Pred_lower['US Daily Confirmed'], '-r', label='YYG Predicted Confirmed lower')
ax.plot(YYG_Pred_upper.index, YYG_Pred_upper['US Daily Confirmed'], '--k', label='YYG Predicted Confirmed upper')


ax.set_xlabel('Label')
ax.set_ylabel('Prediction')
ax.set_xlim(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-10-01'))
ax.legend()
ax.set_title('US Daily Confirmed')


fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(1, 1, 1)
# normalized scale
ax.plot(merged_select.index, merged_select['US Total Death'].diff(), 'o', label='MIT Predicted Death')
# ax.plot(merged_select.index, merged_select['predicted_total_infected_mean'], 'o', label='YYG Predicted')

ax.plot(YYG_Pred_mean.index, YYG_Pred_mean['US Daily Death'], '--b', label='YYG Predicted Death mean')
ax.plot(YYG_Pred_lower.index, YYG_Pred_lower['US Daily Death'], '-r', label='YYG Predicted Death lower')
ax.plot(YYG_Pred_upper.index, YYG_Pred_upper['US Daily Death'], '--k', label='YYG Predicted Death upper')

ax.set_xlabel('Label')
ax.set_ylabel('Prediction')
ax.set_xlim(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-10-01'))
ax.legend()
ax.set_title('US Daily Death')

'''
'''


'''
Organize data
'''
df_US_Projection = copy.deepcopy(MIT_US_Projection)
all_Data=pd.DataFrame()

#YYG State level projection

df_US_Projection['State Name']='US'
df_US_Projection['country_region_code'] = 'US'
df_US_Projection['country_region'] = 'United States'
df_US_Projection['retail_and_recreation'] =1
df_US_Projection['grocery_and_pharmacy'] =1
df_US_Projection['parks'] = 1
df_US_Projection['transit_stations'] = 1
df_US_Projection['workplaces'] = 1
df_US_Projection['residential'] = 1


# df_US_Projection['US Total Confirmed'] = df_US_Projection['predicted_total_infected_'+scenario].shift(new_row['shiftDay'])*new_row['regr_coef'] + new_row['regr_interp']
df_US_Projection['US Daily Confirmed'] = df_US_Projection['US Total Confirmed'].diff()
# df_US_Projection['US Daily Confirmed'] = (df_US_Projection['predicted_new_infected_'+scenario].shift(shiftDay))/infected_Confirmed_Ratio
df_US_Projection['US Daily Confirmed Dfdt'] = df_US_Projection['US Daily Confirmed'].diff() 
# df_US_Projection = def_add_datashift (df_US_Projection, 'US Total Confirmed', [1, 3, 7, 10])
df_US_Projection = def_add_datashift (df_US_Projection, 'US Daily Confirmed', [1, 3, 7, 10])
df_US_Projection = def_add_datashift (df_US_Projection, 'US Daily Confirmed Dfdt', [1, 3, 7, 10])
    


# df_US_Projection['US Total Death'] = df_US_Projection['predicted_total_deaths_'+scenario].fillna(0) + df_US_Projection['total_deaths'].fillna(0)
# df_US_Projection['US Daily Death'] = df_US_Projection['predicted_deaths_'+scenario].fillna(0) + df_US_Projection['actual_deaths'].fillna(0)
df_US_Projection['US Daily Death'] = df_US_Projection['US Total Death'].diff()
# a = df_US_Projection['US Daily Death'].diff()

df_US_Projection['US Daily Death Dfdt'] = df_US_Projection['US Daily Death'].diff()

# df_US_Projection = def_add_datashift (df_US_Projection, 'US Total Death', [1, 3, 7, 10])
df_US_Projection = def_add_datashift (df_US_Projection, 'US Daily Death', [1, 3, 7, 10])
df_US_Projection = def_add_datashift (df_US_Projection, 'US Daily Death Dfdt', [1, 3, 7, 10])

# df_US_Projection = df_US_Projection.iloc[:, 18:100]

stateNameList = df_StateName_Code['State Name'].drop_duplicates().dropna().tolist()

ML_Data_StateDailyDeath=pd.DataFrame()

for stateName in stateNameList:
        
    if stateName == 'District of Columbia':
        continue
    
    state_Code = df_StateName_Code.loc[df_StateName_Code['State Name'] == stateName, 'State Code'].iloc[0]

    print (stateName)

    
    df_State_Data =  MIT_State_Projection[MIT_State_Projection['State Name'] == stateName]
    df_State_Data['State Daily Confirmed'] = df_State_Data['State Total Confirmed'].diff()
    df_State_Data['State Daily Death']=df_State_Data['State Total Deaths'].diff()
    df_State_Data['State Daily Death Dfdt'] = df_State_Data['State Daily Death'].diff()
    # df_State_Data = def_add_datashift (df_State_Data, 'State Total Death', [1, 3, 7, 10])
    df_State_Data = def_add_datashift (df_State_Data, 'State Daily Death', [1, 3, 7, 10])
    df_State_Data = def_add_datashift (df_State_Data, 'State Daily Death Dfdt', [1, 3, 7, 10])
    
    merged1 = pd.merge_asof(df_US_Projection, df_State_Data, left_index=True,right_index=True, direction='backward')
    # merged1 = pd.merge_asof(merged1, df_State_Projection, left_index=True,right_index=True, direction='backward')
    
    # df_Population = df_Population['STNAME', 'CTYNAME', 'POPESTIMATE2019']
    state_Population = df_Population.loc[df_Population['County Name'] == stateName, 'Population']
    
    #___________________________________
    #read county size

    state_Area = df_Area.loc[df_Area['County Name'] == stateName, 'Area']
    #___________________________________
    #read county size
    state_Unemployment_Rate = df_Employee.loc[df_Employee['County Name'] == stateName, 'Unemployment_rate_2018']
    state_Household_Income = df_Employee.loc[df_Employee['County Name'] == stateName, 'Median_Household_Income_2018']
    
    merged1['State Population'] = state_Population.iloc[0]
    merged1['State_D_Confirmed_Per1000'] = merged1['State Daily Confirmed']/merged1['State Population']*1000
    merged1['State_D_Death_Per1000'] = merged1['State Daily Death']/merged1['State Population']*1000
    merged1['State_Area'] = state_Area.iloc[0]
    merged1['State_Population_Density'] = merged1['State Population']/merged1['State_Area']
    merged1['State_Daily_Death_perArea'] = merged1['State Daily Death']/merged1['State_Area']
    merged1['State_Unemployment_Rate'] = state_Unemployment_Rate.iloc[0] if not(state_Unemployment_Rate.empty) else np.nan
    merged1['State_Household_Income'] = state_Household_Income.iloc[0] if not(state_Household_Income.empty) else np.nan
    
    
    #read policy data
    df_Policy['DateEnacted'] = pd.to_datetime(df_Policy['DateEnacted'], format='%Y%m%d')
    df_Policy['DateEnded'] = pd.to_datetime(df_Policy['DateEnded'], format='%Y%m%d')
    df_Policy['DateEased'] = pd.to_datetime(df_Policy['DateEased'], format='%Y%m%d')
    df_Policy['DateExpiry'] = pd.to_datetime(df_Policy['DateExpiry'], format='%Y%m%d')
    PolicyList = ['EmergDec', 'SchoolClose', 'NEBusinessClose', 'RestaurantRestrict', 'StayAtHome']
    for policy_Name in PolicyList:
        policy_Data = df_Policy[(df_Policy['StateName'] == stateName) & (df_Policy['StatePolicy'] == policy_Name)]
        policy_Start = policy_Data['DateEnacted'].min()
        policy_Eased = policy_Data['DateEased'].max()
        policy_Expiry = policy_Data['DateExpiry'].max()
        policy_End = policy_Data['DateEnded'].max()
        merged1[policy_Name] =0
        merged1.loc[(merged1.index >= policy_Start) & (pd.notnull(policy_Start)), policy_Name] = 1 
        merged1.loc[(merged1.index >= policy_Eased) & (pd.notnull(policy_Eased)), policy_Name] = 0.5 
        if pd.notnull(policy_Expiry):
            policy_End = max(policy_Expiry, policy_End)
        merged1.loc[(merged1.index > policy_End) & (pd.notnull(policy_End)), policy_Name] = 0 
    
    merged1['WeekDay'] = merged1.index.weekday+1 
    merged1['State Name'] = stateName
    merged1['County Name'] = ''
    merged1['Apple US'] = 1
    merged1['Apple State'] =1
    
    
    all_Data = all_Data.append(merged1)
    
  
all_Data['statecode'] = pd.factorize(all_Data['State Name'].to_numpy())[0]
all_Data = all_Data.dropna()

PODA_Model['Data_for_Mobility_Projection_MIT']=all_Data
# all_Data['statecode'] = pd.factorize(all_Data.iloc[:, 2])[0]
save_File = "./Mobility projection/MIT_State_Level_Data_for_Projection_"+today+".xlsx"
all_Data.to_excel(save_File)

    
np.save(("./PODA_Model_"+today+".npy"), PODA_Model)
  
