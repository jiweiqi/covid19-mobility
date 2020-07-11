#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:07:49 2020

@author: weiqi
modified: Xin He
    load model parameter
    process multiple cases
    save to file

Example for predict Albama State data

"""

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from torch import nn

from myFunctions import def_add_datashift, createFolder
import shap

class Net(torch.nn.Module):
    def __init__(self, n_feature, layers, nodes, n_output):
        super(Net, self).__init__()

        self.seq = nn.Sequential()

        self.seq.add_module('fc_1', nn.Linear(n_feature, nodes))
        self.seq.add_module('relu_1', nn.ReLU())

        for i in range(layers):
            self.seq.add_module('fc_' + str(i + 2), nn.Linear(nodes, nodes))
            self.seq.add_module('relu_' + str(i + 2), nn.ReLU())

        self.seq.add_module('fc_last', nn.Linear(nodes, n_output))

    def forward(self, input):

        return self.seq(input)
sns.set(style="darkgrid")


today = pd.to_datetime('today')
today =today.strftime("%Y-%m-%d")
today = '2020-07-08'

PODA_Model = np.load(("./PODA_Model_"+today+".npy"),allow_pickle='TRUE').item()
google_Mobility_Day = PODA_Model['ML_File_Date']
ML_Model_Para = PODA_Model['ML_Model_Para_layer_2node_25ReLU']

# Model_Date = np.load("./Model_Parameter.npy",allow_pickle='TRUE').item()
# google_Mobility_Day = Model_Date['ML_File_Date']

ML_Model=PODA_Model['ML_File_Date']
case_all = ['mean']  #three cases 'mean', 'upper', 'lower'

model_mark =''
# isopen='_noreopen'
isopen=''
YYG_projection_Date = PODA_Model['YYG_File_Date']
'''
load model parameter
'''

# model_Para = np.load("./ML Files/model_Para_"+model_mark+ML_Model+".npy",allow_pickle='TRUE').item()


X_norm_mean = ML_Model_Para['xNormMean']
X_norm_std = ML_Model_Para['xNormStd']
Y_norm_mean = ML_Model_Para['yNormMean']
Y_norm_std = ML_Model_Para['yNormStd']
col_X_Name = ML_Model_Para['col_X_Name']
col_Y_Name = ML_Model_Para['col_Y_Name']
checkfile = ML_Model_Para['model_File']
layers_number=ML_Model_Para['Layers_num']
nodes_number=ML_Model_Para['nodes_num']
print(col_X_Name)

for case in case_all: 
####################################################################################################
#get projection file
    print(case)
    # datafile = './Mobility projection/State_Level_Data_for_Projection_'+YYG_projection_Date+'_'+case+isopen+'.xlsx'
    # pd_all = pd.read_excel(datafile)
    
    
    pd_all = PODA_Model['Data_for_Mobility_Projection_'+case]
    pd_all = pd_all[pd_all.index<pd.to_datetime('2020-07-01')]
    # pd_all['date']=pd_all.index
    pd_all = pd_all.reset_index()
    x_Data=pd_all[col_X_Name]
    y_Data=pd_all[col_Y_Name]
    pd_used = pd.concat([x_Data, y_Data], axis=1)

    # Drop rows with Nan value
    X_all_indexing = pd_used.dropna().iloc[:, :len(col_X_Name)]

    plot_index = X_all_indexing.index
    
    
    X_all = pd_used.dropna().iloc[:, :len(col_X_Name)].to_numpy()
    Y_all = pd_used.dropna().iloc[:, len(col_X_Name):].to_numpy()
    X_all_norm = (X_all - X_norm_mean) / X_norm_std
    Y_all_norm = (Y_all - Y_norm_mean) / Y_norm_std
    # convert numpy to tensor
    X_all_tensor = torch.Tensor(X_all_norm)
    Y_all_tensor = torch.Tensor(Y_all_norm)
    
        
    # this is one way to define a network
    # define the network
    net = Net(n_feature=X_all.shape[1],
              layers=layers_number,
              nodes=nodes_number,
              n_output=len(col_Y_Name))
    
    # load pre-trained model
    is_restart = True
    if is_restart is True:
        checkpoint = torch.load('./ML Files/'+checkfile + '.tar')
        net.load_state_dict(checkpoint['model_state_dict'])
    
    net.eval()
    
    """
    
    Make prediction
    
    """
    
    inputs = X_all_tensor
    label = Y_all_tensor
    
    prediction = net(inputs)
    
    
    
        
    """
    Shap for feature selections
    Install: conda install -c conda-forge shap
    # """
    
    shap_col_name = ['US Daily Confirmed', 'US Daily Confirmed Dfdt', 'US Daily Confirmed_shifted_1', 
                     'US Daily Confirmed_shifted_3', 'US Daily Confirmed_shifted_7', 'US Daily Confirmed_shifted_10', 
                     'US Daily Confirmed Dfdt_shifted_1', 'US Daily Confirmed Dfdt_shifted_3', 'US Daily Confirmed Dfdt_shifted_7', 
                     'US Daily Confirmed Dfdt_shifted_10', 'US Daily Death', 'US Daily Death Dfdt', 'US Daily Death_shifted_1', 
                     'US Daily Death_shifted_3', 'US Daily Death_shifted_7', 'US Daily Death_shifted_10', 
                     'US Daily Death Dfdt_shifted_1', 'US Daily Death Dfdt_shifted_3', 'US Daily Death Dfdt_shifted_7', 
                     'US Daily Death Dfdt_shifted_10', 'State Daily Death', 'State Daily Death Dfdt', 'State Daily Death_shifted_1', 
                     'State Daily Death_shifted_3', 'State Daily Death_shifted_7', 'State Daily Death_shifted_10', 
                     'State Daily Death Dfdt_shifted_1', 'State Daily Death Dfdt_shifted_3', 'State Daily Death Dfdt_shifted_7', 
                     'State Daily Death Dfdt_shifted_10', 'State Population', 'State_D_Death_Per1000', 'State_Daily_Death_perArea',
                     'State_Area', 'State_Population_Density', 'State_Unemployment_Rate', 'State_Household_Income', 'EmergDec', 'SchoolClose', 'NEBusinessClose', 
                     'RestaurantRestrict', 'StayAtHome', 'WeekDay', 'statecode'] 
    
    labels = ['retail_and_recreation', 'grocery_and_pharmacy','parks', 'workplaces', 'Apple State']
    # k_explainer = shap.DeepExplainer(net, inputs)
    # k_shap_values = k_explainer.shap_values(inputs)
    
    shap_n_sample =
    p = 100
    k_explainer = shap.DeepExplainer(net, inputs[:shap_n_sample, :])
    k_shap_values = k_explainer.shap_values(inputs[:shap_n_sample, :])
    
    # average over datasets
    # summarize the effects of all the features
    shap.summary_plot(k_shap_values, 
                      inputs, 
                      feature_names=shap_col_name,
                      class_names = labels,
                      max_display = 20,
                      show=False)
    plt.tight_layout()
    plt.savefig('./Supplementary Figure 11.png', dpi=300, format="png", bbox_inches='tight')
    plt.savefig('./Supplementary Figure 11.pdf', format="pdf", bbox_inches='tight')
    
    plt.show()


    

