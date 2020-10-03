# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:58:47 2020

@author: hexx
THis code compile COVID pandemic, mobility data, demograhpic, policy data 
for Machine Learning (ML)
The file will be saved on local folder, and saved as PODA_Model_2020_MM_YY.npy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from myFunctions import def_add_datashift, createFolder
'''
Need to check YYG website to adjust the date difference between today and 
the latest YYG projection
https://github.com/youyanggu/covid19_projections/tree/master/projections

Check Apple mobility website: https://www.apple.com/covid19/mobility
'''
YYG_date_adjust = 2
Apple_Date_adjust = 2

Weekly_average = False
moving_avg = 7

# today = pd.to_datetime('2020-08-07')
today = pd.to_datetime('today')

YYG_Date = today - pd.DateOffset(days=YYG_date_adjust)
YYG_Date = YYG_Date.strftime("%Y-%m-%d")

Apple_File_Date = today - pd.DateOffset(days=Apple_Date_adjust)
Apple_File_Date = Apple_File_Date.strftime("%Y-%m-%d")

PODA_Model = {'Date': today.strftime("%Y-%m-%d"),
              'ML_File_Date': today.strftime("%Y-%m-%d"),
              'Apple_File_Date': Apple_File_Date,
              'YYG_File_Date': YYG_Date,
              'Moving_Average': moving_avg}

df_StateName_Code = pd.read_excel('./data/US_StateCode_List.xlsx', 
                                  sheet_name='Sheet1', 
                                  header=0)

PODA_Model['StateName_StateCode'] = df_StateName_Code

'''
Get Apple and Google Mobility Data
'''
# Get Apple mobility data
print('Read Apple Data')
'''
Need to check Apple Website to find the right url path
'''
df_Apple_Mobility = pd.read_csv("https://covid19-static.cdn-apple.com/covid19-mobility-data/2016HotfixDev16/v3/en-us/applemobilitytrends-"
                                +Apple_File_Date+".csv")

createFolder('./Mobility Google-Apple')
df_Apple_Mobility.to_csv('./Mobility Google-Apple/applemobilitytrends-'+Apple_File_Date+'.csv')  

apple_US = df_Apple_Mobility[(df_Apple_Mobility['region'] == 'United States') & 
                             (df_Apple_Mobility['transportation_type'] == 'driving')].transpose()
apple_US = apple_US.iloc[6:,]
apple_US.rename(columns = {apple_US.columns[0]: "Apple US" }, inplace = True)
apple_US.index = pd.to_datetime(apple_US.index)

############################################################################################################################
# get Google Mobility data
print('Read Google Mobility Data')
df_Google = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=911a386b6c9c230f', 
                        header=0)
df_Google.to_csv('./Mobility Google-Apple/Google_download_'+today.strftime("%Y-%m-%d")+'.csv')
googleMobilityData = df_Google[df_Google['country_region_code'] == 'US']   #
original_List = ['sub_region_1', 'sub_region_2', 
                  'retail_and_recreation_percent_change_from_baseline',	
                  'grocery_and_pharmacy_percent_change_from_baseline',
                  'parks_percent_change_from_baseline',
                  'transit_stations_percent_change_from_baseline',
                  'workplaces_percent_change_from_baseline',
                  'residential_percent_change_from_baseline']
rename_List =['State Name', 'County Name', 'retail_and_recreation',	
              'grocery_and_pharmacy',	'parks',
              'transit_stations', 'workplaces',	'residential']
j=0
for i in original_List:
    googleMobilityData.rename(columns={i: rename_List[j]}, inplace=True)
    j=j+1
    
googleMobilityStateLevel = googleMobilityData[(googleMobilityData['County Name'].isna()) 
                                              & (googleMobilityData['State Name'].notna())]

googleMobilityUSLevel = googleMobilityData[(googleMobilityData['County Name'].isna())]


'''
Get COVID-19 Pandemic Data
'''
############################################################################################################################
# Get COVID confirmed case Data
df_Confirmed_Ini = pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv')
createFolder('./COVID')
df_Confirmed_Ini.to_csv('./COVID/Confirmed_'+today.strftime("%Y-%m-%d")+'.csv')     # save US Confirmed case data
df_Confirmed_Ini = df_Confirmed_Ini.iloc[:, :-1]
df_Confirmed = df_Confirmed_Ini[df_Confirmed_Ini['countyFIPS'] != 1]
df_Confirmed = df_Confirmed[df_Confirmed['countyFIPS'] != 0]
df_Confirmed.rename(columns={"State": "State Code"}, inplace=True)   #rename the column name
df_Confirmed= pd.melt(df_Confirmed, 
            id_vars=['State Code', 'County Name'],
            value_vars=list(df_Confirmed.columns[5:]), # list of days of the week
            var_name='Date', 
            value_name='County Total Confirmed')
df_Confirmed = df_Confirmed.dropna()
df_Confirmed['Date'] = pd.to_datetime(df_Confirmed['Date'])
cols = ['State Code']
df_Confirmed = df_Confirmed.join(df_StateName_Code.set_index(cols), on=cols, how='left')

df_Confirmed.sort_values(by=['State Name', 'County Name',  'Date'], inplace=True)
#remove DC data
df_Confirmed = df_Confirmed[df_Confirmed['State Name'] != "District of Columbia"]


'''
Read COVID Death Data
'''
df_Death_Ini = pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv')
df_Death_Ini.to_csv('./COVID/Death_'+today.strftime("%Y-%m-%d")+'.csv')     # save US Confirmed case data

df_Death = df_Death_Ini[df_Death_Ini['countyFIPS'] != 1]
df_Death = df_Death[df_Death['countyFIPS'] != 0]
df_Death.rename(columns={"State": "State Code"}, inplace=True)   #rename the column name
df_Death= pd.melt(df_Death, 
            id_vars=['State Code', 'County Name'],
            value_vars=list(df_Death.columns[5:]), # list of days of the week
            var_name='Date', 
            value_name='County Total Death')
df_Death=df_Death.dropna()

df_Death['Date'] = pd.to_datetime(df_Death['Date'])
cols = ['State Code']
df_Death = df_Death.join(df_StateName_Code.set_index(cols), on=cols, how='left')

df_Death.sort_values(by=['State Name', 'County Name',  'Date'], inplace=True)
#remove DC data
df_Death = df_Death[df_Death['State Name'] != "District of Columbia"]

df_Confirmed_Ini.rename(columns={"State": "State Code"}, inplace=True)   #rename the column name
df_Confirmed_Ini= df_Confirmed_Ini.join(df_StateName_Code.set_index(cols), on=cols)

df_Death_Ini.rename(columns={"State": "State Code"}, inplace=True)   #rename the column name
df_Death_Ini = df_Death_Ini.join(df_StateName_Code.set_index(cols), on=cols)


'''
read Policy Data
'''
df_Policy = pd.read_csv('https://raw.githubusercontent.com/COVID19StatePolicy/SocialDistancing/master/data/USstatesCov19distancingpolicy.csv', 
                        encoding= 'unicode_escape')

# Read Population Data
population_File = './data/Population_County_Level.xlsx'
df_Population = pd.read_excel(population_File, sheet_name ='Population_County_Level', header=0)
PODA_Model['State Population'] = df_Population
#Read County Area
area_File = './data/County_Area.xls'
df_Area = pd.read_excel(area_File, sheet_name ='Sheet1', header=0)
PODA_Model['State Area'] = df_Area
#Employment 
#https://www.census.gov/library/publications/2011/compendia/usa-counties-2011.html
employee_File = './data/Unemployment.xlsx'
df_Employee = pd.read_excel(employee_File, sheet_name ='Sheet1', header=0)
PODA_Model['State Employment'] = df_Employee


'''
Data processing
'''
#######################################################################################
#Calculate National level confirmed cases
US_Confirmed = df_Confirmed_Ini.sum().drop(labels=['State Name', 'stateFIPS', 
                                                   'countyFIPS', 'County Name', 
                                                   'State Code']).rename("US Total Confirmed")

US_Confirmed_Daily = US_Confirmed.diff().rename('US Daily Confirmed')
if Weekly_average:
    US_Confirmed_Daily = US_Confirmed_Daily.rolling(window=moving_avg).mean()
    
US_Confirmed_Daily_Dfdt = US_Confirmed_Daily.diff().rename('US Daily Confirmed Dfdt')
US_Confirmed.to_frame()
US_Confirmed.index = pd.to_datetime(US_Confirmed.index)
US_Confirmed_Daily.to_frame()
US_Confirmed_Daily.index = pd.to_datetime(US_Confirmed_Daily.index)
US_Confirmed_Daily_Dfdt.to_frame()
US_Confirmed_Daily_Dfdt.index = pd.to_datetime(US_Confirmed_Daily_Dfdt.index)
US_Confirmed = pd.merge_asof(US_Confirmed, US_Confirmed_Daily, left_index=True, right_index=True)
US_Confirmed = pd.merge_asof(US_Confirmed, US_Confirmed_Daily_Dfdt, left_index=True, right_index=True)
# US_Confirmed = def_add_datashift (US_Confirmed, 'US Total Confirmed', [1, 3, 7, 10])
US_Confirmed = def_add_datashift(US_Confirmed, 'US Daily Confirmed', [1, 3, 7, 10])
US_Confirmed = def_add_datashift(US_Confirmed, 'US Daily Confirmed Dfdt', [1, 3, 7, 10])


#Calculate National level death cases
US_Death = df_Death_Ini.sum().drop(labels=['State Name', 'stateFIPS', 
                                           'countyFIPS', 'County Name', 
                                           'State Code'
                                           ]).rename("US Total Death")

US_Death_Daily = US_Death.diff().rename("US Daily Death")
US_Death_Daily['6/27/20'] = 447           # data correction due to NY state adjustment

if Weekly_average:
    US_Death_Daily = US_Death_Daily.rolling(window=moving_avg).mean()
    
US_Death_Daily_Dfdt = US_Death_Daily.diff().rename("US Daily Death Dfdt")
US_Death.to_frame()
US_Death.index = pd.to_datetime(US_Death.index)
US_Death_Daily.to_frame()
US_Death_Daily.index = pd.to_datetime(US_Death_Daily.index)
US_Death_Daily_Dfdt.to_frame()
US_Death_Daily_Dfdt.index = pd.to_datetime(US_Death_Daily_Dfdt.index)
US_Death = pd.merge_asof(US_Death, US_Death_Daily, left_index=True, right_index=True)
US_Death = pd.merge_asof(US_Death, US_Death_Daily_Dfdt, left_index=True, right_index=True)
# US_Death = def_add_datashift (US_Death, 'US Total Death', [1, 3, 7, 10])
US_Death = def_add_datashift (US_Death, 'US Daily Death', [1, 3, 7, 10])
US_Death = def_add_datashift (US_Death, 'US Daily Death Dfdt', [1, 3, 7, 10])

'''
#Loop to merge data sets
'''
all_Data = pd.DataFrame()
stateNameList = df_Confirmed['State Name'].drop_duplicates().dropna().tolist()

for stateName in stateNameList:
    print (stateName)
    state_Code = df_StateName_Code.loc[df_StateName_Code['State Name'] == stateName, 'State Code'].iloc[0]
            
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #State Level Data
    #Calculate State level confirmed cases

    state_Confirmed = df_Confirmed_Ini[df_Confirmed_Ini['State Name'] == stateName]
    state_Confirmed = state_Confirmed.sum().drop(labels = ['State Name', 'stateFIPS', 'countyFIPS', 'County Name', 'State Code'
                                                           ]).rename("State Total Confirmed")
    state_Confirmed_Daily = state_Confirmed.diff().rename("State Daily Confirmed")
    if Weekly_average:
        state_Confirmed_Daily = state_Confirmed_Daily.rolling(window=moving_avg).mean()
    
    state_Confirmed_Daily_Dfdt = state_Confirmed_Daily.diff().rename("State Daily Confirmed Dfdt")
    state_Confirmed.to_frame()
    state_Confirmed.index = pd.to_datetime(state_Confirmed.index)
    state_Confirmed_Daily.to_frame()
    state_Confirmed_Daily.index = pd.to_datetime(state_Confirmed_Daily.index)
    state_Confirmed_Daily_Dfdt.to_frame()
    state_Confirmed_Daily_Dfdt.index = pd.to_datetime(state_Confirmed_Daily_Dfdt.index)
    state_Confirmed = pd.merge_asof(state_Confirmed, state_Confirmed_Daily, left_index=True, right_index=True)
    state_Confirmed = pd.merge_asof(state_Confirmed, state_Confirmed_Daily_Dfdt, left_index=True, right_index=True)
    
    state_Confirmed = def_add_datashift (state_Confirmed, 'State Daily Confirmed', [1, 3, 7, 10])
    state_Confirmed = def_add_datashift (state_Confirmed, 'State Daily Confirmed Dfdt', [1, 3, 7, 10])
      
    #Calculate State level death cases

    state_Death = df_Death_Ini[df_Death_Ini['State Name'] == stateName]
    state_Death = state_Death.sum().drop(labels=['State Name', 'stateFIPS', 'countyFIPS', 'County Name', 'State Code'
                                                 ]).rename("State Total Death")
    state_Death_Daily = state_Death.diff().rename("State Daily Death")
    if Weekly_average:
        state_Death_Daily = state_Death_Daily.rolling(window=moving_avg).mean()
        
    state_Death_Daily_Dfdt = state_Death_Daily.diff().rename("State Daily Death Dfdt")
    state_Death.to_frame()
    state_Death.index = pd.to_datetime(state_Death.index)
    state_Death_Daily.to_frame()
    state_Death_Daily.index = pd.to_datetime(state_Death_Daily.index)
    state_Death_Daily_Dfdt.to_frame()
    state_Death_Daily_Dfdt.index = pd.to_datetime(state_Death_Daily_Dfdt.index)
    state_Death = pd.merge_asof(state_Death, state_Death_Daily, left_index=True, right_index=True)
    state_Death = pd.merge_asof(state_Death, state_Death_Daily_Dfdt, left_index=True, right_index=True)
    # state_Death = def_add_datashift (state_Death, 'State Total Death', [1, 3, 7, 10])
    state_Death = def_add_datashift (state_Death, 'State Daily Death', [1, 3, 7, 10])
    state_Death = def_add_datashift (state_Death, 'State Daily Death Dfdt', [1, 3, 7, 10])
    
    
    #YYG State level projection
    #YYG data not used for ML model development, but can be used to predcit confirmed and deadth. 
    YYG_Proj_File = 'https://raw.githubusercontent.com/youyanggu/covid19_projections/master/projections/' + YYG_Date +'/US_'+ state_Code+'.csv'
    df_Projection = pd.read_csv(YYG_Proj_File, header=0)
    df_Projection['date'] = pd.to_datetime(df_Projection['date'])
    df_Projection = df_Projection[df_Projection['date'] <= (today+pd.DateOffset(days=7))]
    df_Projection.set_index('date', inplace=True)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #Google Mobility data
    google_Mobility = googleMobilityStateLevel[googleMobilityStateLevel['State Name'] == stateName]
    google_Mobility.set_index ('date', inplace=True)
    google_Mobility.index = pd.to_datetime(google_Mobility.index)
    
    '''
    merge data 
    '''
    merged_Confirmed = pd.merge_asof(US_Confirmed, state_Confirmed, left_index=True, right_index=True) 
    merged_Death = pd.merge_asof(US_Death, state_Death, left_index=True, right_index=True) 
    merged_Case = pd.merge_asof(merged_Confirmed, merged_Death, left_index=True, right_index=True)  
    merged1 = pd.merge_asof(google_Mobility, merged_Case, left_index=True,right_index=True, direction='forward')
    merged1 = pd.merge_asof(merged1, df_Projection, left_index=True,right_index=True)

    '''
    add demographic data
    '''
    state_Population = df_Population.loc[df_Population['County Name'] == stateName, 'Population']
    state_Area = df_Area.loc[df_Area['County Name'] == stateName, 'Area']
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
    
    '''
    add policy data
    '''
    df_Policy['DateEnacted'] = pd.to_datetime(df_Policy['DateEnacted'], format='%Y%m%d')
    df_Policy['DateEnded'] = pd.to_datetime(df_Policy['DateEnded'], format='%Y%m%d')
    df_Policy['DateEased'] = pd.to_datetime(df_Policy['DateEased'], format='%Y%m%d')
    df_Policy['DateExpiry'] = pd.to_datetime(df_Policy['DateExpiry'], format='%Y%m%d')
    PolicyList = ['EmergDec', 'SchoolClose', 'NEBusinessClose', 'RestaurantRestrict', 'StayAtHome']
    for policy_Name in PolicyList:
        policy_Data = df_Policy[(df_Policy['StateName'] == stateName) 
                                & (df_Policy['StatePolicy'] == policy_Name)]
        policy_Start = policy_Data['DateEnacted'].min()
        policy_Eased = policy_Data['DateEased'].max()
        policy_Expiry = policy_Data['DateExpiry'].max()
        policy_End = policy_Data['DateEnded'].max()
        merged1[policy_Name] =0
        merged1.loc[(merged1.index >= policy_Start) 
                    & (pd.notnull(policy_Start)), policy_Name] = 1 
        merged1.loc[(merged1.index >= policy_Eased) 
                    & (pd.notnull(policy_Eased)), policy_Name] = 0.5 
        if pd.notnull(policy_Expiry):
            policy_End = max(policy_Expiry, policy_End)
        merged1.loc[(merged1.index > policy_End) 
                    & (pd.notnull(policy_End)), policy_Name] = 0 
    merged1['WeekDay'] = merged1.index.weekday+1 
    
    Policy2=['PublicMask']
    merged1['PublicMask']= 0
    policy_Data_na = df_Policy[(df_Policy['StateName'] == stateName) 
                               & (df_Policy['StatePolicy'] == 'PublicMask') 
                               & (pd.isnull(df_Policy['PublicMaskLevel']))]
    policy_Data_1 = df_Policy[(df_Policy['StateName'] == stateName) 
                              & (df_Policy['StatePolicy'] == 'PublicMask') 
                              & (df_Policy['PublicMaskLevel'] == 'Mandate1')]
    policy_Data_2 = df_Policy[(df_Policy['StateName'] == stateName) 
                              & (df_Policy['StatePolicy'] == 'PublicMask') 
                              & (df_Policy['PublicMaskLevel'] == 'Mandate2')]
    policy_Data_3 = df_Policy[(df_Policy['StateName'] == stateName) 
                              & (df_Policy['StatePolicy'] == 'PublicMask') 
                              & (df_Policy['PublicMaskLevel'] == 'Mandate3')]
    
    if not(policy_Data_na.empty):
        merged1.loc[(merged1.index >= policy_Data_na['DateEnacted'].min()), 'PublicMask'] = 0.5 
    if not(policy_Data_1.empty):
        merged1.loc[(merged1.index >= policy_Data_1['DateEnacted'].min()), 'PublicMask'] = 1
    if not(policy_Data_2.empty):
        merged1.loc[(merged1.index >= policy_Data_2['DateEnacted'].min()), 'PublicMask'] = 2
    if not(policy_Data_3.empty):
        merged1.loc[(merged1.index >= policy_Data_3['DateEnacted'].min()), 'PublicMask'] = 3
        
    #read apple state mobility data
    apple_State = df_Apple_Mobility[(df_Apple_Mobility['region'] == stateName) & 
                             (df_Apple_Mobility['transportation_type'] == 'driving')].transpose()
    
    apple_State = apple_State.iloc[6:,]
    apple_State.rename(columns = {apple_State.columns[0]: "Apple State" }, 
                       inplace = True)
    apple_State.index = pd.to_datetime(apple_State.index)
    apple_Mobility = pd.merge_asof(apple_US, apple_State, left_index=True, 
                                   right_index=True)
    merged1 = pd.merge_asof(merged1, apple_Mobility, 
                            left_index=True, right_index=True)
    
    all_Data = all_Data.append(merged1)
    
    
all_Data['statecode'] = pd.factorize(all_Data['State Name'].to_numpy())[0]

#remove some outlier days 
# all_Data = all_Data[all_Data.index != pd.to_datetime('4-12-2020')] #Easter Day

createFolder('./ML Files')
all_Data.to_excel("./ML Files/State_Level_Data_forML_"
                  +today.strftime("%Y-%m-%d")+".xlsx")

'''
save to PODA_Model
'''
PODA_Model['ML_Data'] = all_Data
np.save(("./PODA_Model_"+today.strftime("%Y-%m-%d")+".npy"), PODA_Model)


fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(US_Confirmed.index, US_Confirmed['US Daily Confirmed'], '--b', 
        label='US Daily Confirmed')
ax.set_ylabel('Daily Conformed')
ax.legend()
ax2 = ax.twinx()
ax2.plot(US_Death.index, US_Death['US Daily Death'], '-.r', 
         label='US Daily Death')
ax2.set_ylabel('Daily Death')
ax2.legend()
