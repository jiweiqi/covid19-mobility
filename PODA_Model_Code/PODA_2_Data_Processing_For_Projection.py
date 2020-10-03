# -*- coding: utf-8 -*-
"""
Created on Tue May  12 17:07:00 2020

@author: hexx
This code do the following: 
    (1)saves policy, COVID, and Projection data downloaded online to local folder
    (2)process and saved data to be usded to project mobility

"""


import pandas as pd
import numpy as np
import os
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


from myFunctions import def_add_datashift, createFolder

import warnings
warnings.filterwarnings("ignore")

createFolder('./Mobility projection')
        
scenario_cases = ['lower', 'mean', 'upper']      #'upper', 'lower',

startDate = '2020-02-24'

today_x = pd.to_datetime('today')
today =today_x.strftime("%Y-%m-%d")
# today ='2020-07-08'
PODA_Model = np.load("./PODA_Model_"+today+".npy",allow_pickle='TRUE').item()
YYG_Date = PODA_Model['YYG_File_Date']

moving_avg = PODA_Model['Moving_Average']
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
confirmed = ML_Data[ML_Data['State Name']=='Michigan']
confirmed = confirmed[['US Total Confirmed', 'US Daily Confirmed', 'US Daily Death']]

confirmed = confirmed.rename(columns={"US Total Confirmed":"ML US Total Confirmed", "US Daily Confirmed":"ML US Daily Confirmed", 
                                      "US Daily Death":"ML US Daily Death"})

infected_to_Confirmed = pd.DataFrame(columns = ['Country Name', 'scenario', 'shiftDay', 'regr_coef', 'regr_interp'])

infected_to_Confirmed_State = pd.DataFrame(columns = ['State Name', 'scenario', 'shiftDay', 'regr_coef', 'regr_interp'])

for zz, scenario in enumerate(scenario_cases):

    '''
    Calculate the new infected to confirmed correlation
    '''
    df_US_Projection = pd.read_csv('https://raw.githubusercontent.com/youyanggu/covid19_projections/master/projections/'+YYG_Date+'/US.csv')
    df_US_Projection.to_csv('./YYG Data/'+YYG_Date+'/US.csv')     # save US Projection data
    df_US_Projection['date'] = pd.to_datetime(df_US_Projection['date'])
    df_US_Projection.set_index('date', inplace=True)
    
    YYG_Daily_Infected = df_US_Projection[['predicted_new_infected_'+scenario]]
    
    YYG_Daily_Infected = YYG_Daily_Infected[(YYG_Daily_Infected.index < today_x) & (YYG_Daily_Infected.index > pd.to_datetime('2020-05-01'))]


    
    R2_old=0
    for j in range(0, 20):
        YYG_Data_shifted = YYG_Daily_Infected['predicted_new_infected_'+scenario].shift(j).to_frame()
        YYG_Data_shifted['date']=YYG_Data_shifted.index
        YYG_Data_shifted=YYG_Data_shifted.set_index('date')
        # merged = pd.merge_asof(YYG_Data_shifted, confirmed, left_index=True, right_index=True).dropna()
        merged = confirmed.join(YYG_Data_shifted).dropna()
        
        
        x_conv=merged['predicted_new_infected_'+scenario].to_numpy()
        y_conv = merged['ML US Daily Confirmed'].to_numpy()
        x_length = len(x_conv)
        x_conv = x_conv.reshape(x_length, 1)
        y_conv = y_conv.reshape(x_length, 1)
        regr = linear_model.LinearRegression(fit_intercept = False)
        regr.fit(x_conv, y_conv)
        
        R2_new = regr.score(x_conv, y_conv)
        
        if R2_new > R2_old:
            new_row = {'Country Name': 'US', 'scenario': scenario, 'shiftDay': j, 
                       'regr_coef': regr.coef_[0][0], 'regr_interp':regr.intercept_, 'R2': R2_new}
            merged_select = merged
            regr_selected = regr
            R2_old = R2_new
            
    infected_to_Confirmed=infected_to_Confirmed.append(new_row, ignore_index =True)    
    
    
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    # normalized scale
    ax.plot(merged_select.index, merged_select['predicted_new_infected_'+scenario]*new_row['regr_coef'] + new_row['regr_interp'], 'o', label='YYG Predicted')
    # ax.plot(merged_select.index, merged_select['predicted_total_infected_mean'], 'o', label='YYG Predicted')

    ax.plot(merged_select.index, merged_select['ML US Daily Confirmed'], label='confirmed')
    ax.set_xlabel('Label')
    ax.set_ylabel('Prediction')
    ax.set_xlim(pd.to_datetime('2020-05-01'), pd.to_datetime('today'))
    fig.autofmt_xdate(rotation=45)
    ax.legend()
    ax.set_title('US'+scenario)
    
    '''
    '''
    
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

    
    df_US_Projection['US Daily Confirmed'] = df_US_Projection['predicted_new_infected_'+scenario].shift(new_row['shiftDay'])*new_row['regr_coef'] + new_row['regr_interp']
    df_US_Projection['US Daily Confirmed'] = df_US_Projection['US Daily Confirmed'].rolling(window=moving_avg).mean()

    # df_US_Projection['US Daily Confirmed'] = df_US_Projection['US Total Confirmed'].diff().rolling(window=moving_avg).mean()
    
    for i, da in enumerate(confirmed.index):
        df_US_Projection.loc[da,'US Total Confirmed']= confirmed.loc[da, 'ML US Total Confirmed']
        df_US_Projection.loc[da,'US Daily Confirmed']= confirmed.loc[da, 'ML US Daily Confirmed']
        
    # df_US_Projection['US Daily Confirmed'] = (df_US_Projection['predicted_new_infected_'+scenario].shift(shiftDay))/infected_Confirmed_Ratio
    df_US_Projection['US Daily Confirmed Dfdt'] = df_US_Projection['US Daily Confirmed'].diff() 
    # df_US_Projection = def_add_datashift (df_US_Projection, 'US Total Confirmed', [1, 3, 7, 10])
    df_US_Projection = def_add_datashift (df_US_Projection, 'US Daily Confirmed', [1, 3, 7, 10])
    df_US_Projection = def_add_datashift (df_US_Projection, 'US Daily Confirmed Dfdt', [1, 3, 7, 10])
        

    
    df_US_Projection['US Total Death'] = df_US_Projection['predicted_total_deaths_'+scenario].fillna(0) + df_US_Projection['total_deaths'].fillna(0)
    df_US_Projection['US Daily Death'] = (df_US_Projection['predicted_deaths_'+scenario].fillna(0) + df_US_Projection['actual_deaths'].fillna(0)).rolling(window=moving_avg).mean()
    
    for i, da in enumerate(confirmed.index):
        df_US_Projection.loc[da,'US Daily Death']= confirmed.loc[da, 'ML US Daily Death']
        
    
    df_US_Projection['US Daily Death Dfdt'] = df_US_Projection['US Daily Death'].diff()
    
    # df_US_Projection = def_add_datashift (df_US_Projection, 'US Total Death', [1, 3, 7, 10])
    df_US_Projection = def_add_datashift (df_US_Projection, 'US Daily Death', [1, 3, 7, 10])
    df_US_Projection = def_add_datashift (df_US_Projection, 'US Daily Death Dfdt', [1, 3, 7, 10])
    
    df_US_Projection = df_US_Projection.iloc[:, 18:100]
    df_US_Projection = df_US_Projection[df_US_Projection.index > pd.to_datetime(startDate)]
    
    stateNameList = df_StateName_Code['State Name'].drop_duplicates().dropna().tolist()
    
    ML_Data_StateDailyDeath=pd.DataFrame()

    for stateName in stateNameList:
        

        
        if stateName == 'District of Columbia':
            continue
        
        state_Code = df_StateName_Code.loc[df_StateName_Code['State Name'] == stateName, 'State Code'].iloc[0]

        print (scenario +': '+ stateName)
        
        
        YYG_State_Proj_Location ='https://raw.githubusercontent.com/youyanggu/covid19_projections/master/projections/'+ YYG_Date +'/US_'+ state_Code+'.csv'
        
        df_State_Projection = pd.read_csv(YYG_State_Proj_Location, header=0)
        
        # save YYG State Projection data
        if zz==0:
            df_State_Projection.to_csv('./YYG Data/'+YYG_Date+'/US_'+state_Code+'.csv')     
            
        df_State_Projection['date'] = pd.to_datetime(df_State_Projection['date'])
        df_State_Projection.set_index('date', inplace=True)
        
        ML_Data_State = ML_Data[ML_Data['State Name'] == stateName]
        ML_Data_StateDailyDeath = ML_Data_State[['State Daily Death']]
        ML_Data_StateDailyDeath.rename(columns={'State Daily Death': 'ML State Daily Death'},inplace=True)
        ML_Data_StateDailyDeath = ML_Data_StateDailyDeath[ML_Data_StateDailyDeath.index > df_State_Projection.index[0]]
        
          
            
        '''
        Calculate the new infected to confirmed correlation
        '''
        # df_State_Projection = pd.read_csv('https://raw.githubusercontent.com/youyanggu/covid19_projections/master/projections/'+YYG_Date+'/US.csv')
        # df_US_Projection.to_csv('./YYG Data/'+YYG_Date+'/US.csv')     # save US Projection data
        # df_US_Projection['date'] = pd.to_datetime(df_US_Projection['date'])
        # df_US_Projection.set_index('date', inplace=True)
        
        YYG_Total_Infected = df_State_Projection[['predicted_total_infected_'+scenario]]
        
        YYG_Total_Infected = YYG_Total_Infected[(YYG_Total_Infected.index < today_x) & (YYG_Total_Infected.index > pd.to_datetime('2020-05-01'))]
    
        confirmed_State = ML_Data_State[['State Total Confirmed', 'State Daily Confirmed']]

        confirmed_State = confirmed_State.rename(columns={"State Total Confirmed":"ML State Total Confirmed", "State Daily Confirmed":"ML State Daily Confirmed"})
        
        R2_old=0
        for j in range(0, 20):
            YYG_Data_shifted = YYG_Total_Infected['predicted_total_infected_'+scenario].shift(j)
            
            
            YYG_Data_shifted = YYG_Total_Infected['predicted_total_infected_'+scenario].shift(j).to_frame()
            YYG_Data_shifted['date']=YYG_Data_shifted.index
            YYG_Data_shifted=YYG_Data_shifted.set_index('date')
            merged = confirmed_State.join(YYG_Data_shifted).dropna()
               
            
            # merged = pd.merge_asof(YYG_Data_shifted, confirmed_State, left_index=True, right_index=True).dropna()
            
            x_conv=merged['predicted_total_infected_'+scenario].to_numpy()
            y_conv = merged['ML State Total Confirmed'].to_numpy()
            x_length = len(x_conv)
            x_conv = x_conv.reshape(x_length, 1)
            y_conv = y_conv.reshape(x_length, 1)
            regr = linear_model.LinearRegression(fit_intercept = True)
            regr.fit(x_conv, y_conv)
            
            R2_new = regr.score(x_conv, y_conv)
            
            if R2_new > R2_old:
                new_row_State = {'State Name': stateName, 'scenario': scenario, 'shiftDay': j, 
                           'regr_coef': regr.coef_[0][0], 'regr_interp':regr.intercept_, 'R2': R2_new}
                merged_select = merged
                regr_selected = regr
                R2_old = R2_new
                
        infected_to_Confirmed_State=infected_to_Confirmed_State.append(new_row_State, ignore_index =True)    
        
        
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        # normalized scale
        ax.plot(merged_select.index, merged_select['predicted_total_infected_'+scenario]*new_row_State['regr_coef'] + new_row_State['regr_interp'], 'o', label='YYG Predicted')
        # ax.plot(merged_select.index, merged_select['predicted_total_infected_mean'], 'o', label='YYG Predicted')
    
        ax.plot(merged_select.index, merged_select['ML State Total Confirmed'], label='confirmed')
        ax.set_xlabel('Label')
        ax.set_ylabel('Prediction')
        ax.set_xlim(pd.to_datetime('2020-05-01'), pd.to_datetime('today'))
        fig.autofmt_xdate(rotation=45)
        ax.legend()
        ax.set_title(stateName+scenario)
        
        '''
        '''
        
        df_State_Projection['State Total Confirmed'] = df_State_Projection['predicted_total_infected_'+scenario].shift(new_row_State['shiftDay'])*new_row_State['regr_coef'] + new_row_State['regr_interp']
        
        df_State_Projection['State Daily Confirmed'] = df_State_Projection['State Total Confirmed'].diff().rolling(window=moving_avg).mean()
        
        for i, da in enumerate(confirmed_State.index):
            df_State_Projection.loc[da,'State Total Confirmed']= confirmed_State.loc[da, 'ML State Total Confirmed']
            df_State_Projection.loc[da,'State Daily Confirmed']= confirmed_State.loc[da, 'ML State Daily Confirmed']
            
        df_State_Projection=df_State_Projection[df_State_Projection.index >= pd.to_datetime('2020-03-01')].sort_index()
        # df_US_Projection['US Daily Confirmed'] = (df_US_Projection['predicted_new_infected_'+scenario].shift(shiftDay))/infected_Confirmed_Ratio
        df_State_Projection['State Daily Confirmed Dfdt'] = df_State_Projection['State Daily Confirmed'].diff() 
        # df_US_Projection = def_add_datashift (df_US_Projection, 'US Total Confirmed', [1, 3, 7, 10])
        df_State_Projection = def_add_datashift (df_State_Projection, 'State Daily Confirmed', [1, 3, 7, 10])
        df_State_Projection = def_add_datashift (df_State_Projection, 'State Daily Confirmed Dfdt', [1, 3, 7, 10]) 
        
        # df_State_Projection = df_State_Projection[df_State_Projection.index >= pd.to_datetime('2020-03-01')].sort_index()
        df_State_Data = (df_State_Projection['predicted_total_deaths_'+scenario].fillna(0) + df_State_Projection['total_deaths'].fillna(0)).rename('State Total Death').to_frame()
        # df_State_Data = df_State_Data[df_State_Data.index >= pd.to_datetime('2020-03-01')].sort_index()
        df_State_Data = pd.merge_asof(df_State_Data, ML_Data_StateDailyDeath, left_index=True,right_index=True, direction='forward')
        
        # df_State_Data['State Daily Death'] = df_State_Projection['predicted_deaths_'+scenario].fillna(0) + df_State_Data['ML State Daily Death'].fillna(0)
        df_State_Data['State Daily Death'] = (df_State_Projection['predicted_deaths_'+scenario].fillna(0) + df_State_Projection['actual_deaths'].fillna(0)).rolling(window=moving_avg).mean()

        '''
        replace the YYG historical daily death data by the ML historical data
        '''
        for i, da in enumerate(ML_Data_StateDailyDeath.index):
            df_State_Data.loc[da,'State Daily Death']= ML_Data_StateDailyDeath.loc[da, 'ML State Daily Death']
        
        df_State_Data = df_State_Data[df_State_Data.index >= pd.to_datetime('2020-03-01')].sort_index()
        df_State_Data['State Daily Death Dfdt'] = df_State_Data['State Daily Death'].diff()
        # df_State_Data = def_add_datashift (df_State_Data, 'State Total Death', [1, 3, 7, 10])
        df_State_Data = def_add_datashift (df_State_Data, 'State Daily Death', [1, 3, 7, 10])
        df_State_Data = def_add_datashift (df_State_Data, 'State Daily Death Dfdt', [1, 3, 7, 10])
        
        
        merged1 = pd.merge_asof(df_US_Projection, df_State_Data, left_index=True,right_index=True, direction='backward')
        merged1 = pd.merge_asof(merged1, df_State_Projection, left_index=True,right_index=True, direction='backward')
        
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
        merged1['State_D_Confirmed_Per1000'] = ''
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
            
            
            
        Policy2=['PublicMask']
        merged1['PublicMask']= 0
        policy_Data_na = df_Policy[(df_Policy['StateName'] == stateName) & (df_Policy['StatePolicy'] == 'PublicMask') &(pd.isnull(df_Policy['PublicMaskLevel']))]
        policy_Data_1 = df_Policy[(df_Policy['StateName'] == stateName) & (df_Policy['StatePolicy'] == 'PublicMask') &(df_Policy['PublicMaskLevel'] == 'Mandate1')]
        policy_Data_2 = df_Policy[(df_Policy['StateName'] == stateName) & (df_Policy['StatePolicy'] == 'PublicMask') &(df_Policy['PublicMaskLevel'] == 'Mandate2')]
        policy_Data_3 = df_Policy[(df_Policy['StateName'] == stateName) & (df_Policy['StatePolicy'] == 'PublicMask') &(df_Policy['PublicMaskLevel'] == 'Mandate3')]
        
        if not(policy_Data_na.empty):
            merged1.loc[(merged1.index >= policy_Data_na['DateEnacted'].min()), 'PublicMask'] = 0.5 
        if not(policy_Data_1.empty):
            merged1.loc[(merged1.index >= policy_Data_1['DateEnacted'].min()), 'PublicMask'] = 1
        if not(policy_Data_2.empty):
            merged1.loc[(merged1.index >= policy_Data_2['DateEnacted'].min()), 'PublicMask'] = 2
        if not(policy_Data_3.empty):
            merged1.loc[(merged1.index >= policy_Data_3['DateEnacted'].min()), 'PublicMask'] = 3
        
        merged1['WeekDay'] = merged1.index.weekday+1 
        merged1['State Name'] = stateName
        merged1['County Name'] = ''
        merged1['Apple US'] = 1
        merged1['Apple State'] =1
        
        # print(merged1.index[0])
        
        
        all_Data = all_Data.append(merged1)
        
      
    all_Data['statecode'] = pd.factorize(all_Data['State Name'].to_numpy())[0]
    PODA_Model['Data_for_Mobility_Projection_'+scenario]=all_Data
    # all_Data['statecode'] = pd.factorize(all_Data.iloc[:, 2])[0]
    save_File = "./Mobility projection/State_Level_Data_for_Projection_"+YYG_Date+'_'+scenario+".xlsx"
    all_Data.to_excel(save_File)
    
    
    
#plot results

Line_Style =['.-m', '-.r', '-b', '--g']
fig2 = plt.figure(figsize=(6, 5))
fig3 = plt.figure(figsize=(6, 5))

ax2 = fig2.add_subplot(1, 1, 1)
ax3 = fig3.add_subplot(1, 1, 1)

caseID =['lower', 'mean', 'upper']   
for case_i, case in enumerate(caseID):

    
    COVID = PODA_Model['Data_for_Mobility_Projection_'+case]
    COVID = COVID[COVID['State Name']== 'Michigan']
    
    ax2.plot(COVID.index, COVID['US Daily Confirmed'], Line_Style[case_i], label=case)
    
    ax3.plot(COVID.index, COVID['US Daily Death'],  Line_Style[case_i], label=case)
    # if (case == 'mean') & (isopen ==''):
    #     data_save.to_excel('C:/Users/hexx/Box Sync/Energy-COVID-19/Data for Website/Mobility_State_'+YYG_projection_Date+case+'.xlsx')
        
ax2.legend()
ax3.legend()

PODA_Model['infected_to_Confirmed']=infected_to_Confirmed
    
np.save(("./PODA_Model_"+today+".npy"), PODA_Model)
  
warnings.filterwarnings("default")
