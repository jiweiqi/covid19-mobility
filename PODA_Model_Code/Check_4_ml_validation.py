#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:07:49 2020

@author: weiqi

Example for predict Albama State data

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import max_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.dates as mdates
sns.set(style="darkgrid")



Model_Date='2020-07-02'

stateNamePlot='Texas'
PODA_Model = np.load("./PODA_Model_"+Model_Date+".npy",allow_pickle='TRUE').item()
ML_Model_Para = PODA_Model['ML_Model_Para_layer_2node_25ReLU']

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


# load raw data
# In the excel, I added one column: the weekday of the date.


pd_all = PODA_Model['ML_Data'].reset_index()


model_Para = PODA_Model['ML_Model_Para_layer_2node_25ReLU']


X_norm_mean = model_Para['xNormMean']
X_norm_std = model_Para['xNormStd']
Y_norm_mean = model_Para['yNormMean']
Y_norm_std = model_Para['yNormStd']
col_X_Name = model_Para['col_X_Name']
col_Y_Name = model_Para['col_Y_Name']
checkfile = model_Para['model_File']
layers_number=model_Para['Layers_num']
nodes_number=model_Para['nodes_num']



# Drop rows with Nan value
pd_all_State = pd_all[pd_all['State Name'] == stateNamePlot]
x_Data=pd_all_State[col_X_Name]
y_Data=pd_all_State[col_Y_Name]
pd_used = pd.concat([x_Data, y_Data], axis=1)

date_indexing = pd_all[pd_all['State Name']==stateNamePlot]

X_all_indexing = pd_used.dropna().iloc[:, :len(col_X_Name)]

plot_index = X_all_indexing.index

X_all = pd_used.dropna().iloc[:, :len(col_X_Name)].to_numpy(dtype=np.float)
Y_all = pd_used.dropna().iloc[:, len(col_X_Name):].to_numpy(dtype=np.float)
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
# For example, we are going to predict the first 72 rows of the data for Alabama

# regression plot
inputs = X_all_tensor
label = Y_all_tensor

prediction = net(inputs)


"""
Shap for feature selections
Install: conda install -c conda-forge shap
"""

# import shap

# k_explainer = shap.DeepExplainer(net, inputs)
# k_shap_values = k_explainer.shap_values(inputs)

# # average over datasets
# # summarize the effects of all the features
# shap.summary_plot(k_shap_values, 
#                   inputs, 
#                   feature_names=pd_used.columns[:-1],
#                   show=False)
# plt.tight_layout()
# plt.savefig('shap.png', dpi=300)
# plt.show()

"""
Visulize prediction
"""



fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(1, 2, 1)
# normalized scale
ax.plot(label.numpy(), prediction.detach().numpy(), 'o')
ax.set_xlabel('Label')
ax.set_ylabel('Prediction')
ax.set_title(stateNamePlot)

# original scale
ax = fig.add_subplot(1, 2, 2)

label_origin = label.numpy() * Y_norm_std + Y_norm_mean

prediction_origin = prediction.detach().numpy() * Y_norm_std + \
    Y_norm_mean
r2_score(label_origin[:, 4], prediction_origin[:,4])


ax.plot(label_origin[:, 0], prediction_origin[:, 0], 'o')

ax.set_xlabel('Label')
ax.set_ylabel('Prediction')
ax.set_title(stateNamePlot)

fig.tight_layout()

plt.show()
plt.close()




fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(1, 2, 1)
# normalized scale
ax.plot(label.numpy(), prediction.detach().numpy(), 'o')
ax.set_xlabel('Label')
ax.set_ylabel('Prediction')
ax.set_title(stateNamePlot)

# original scale
ax = fig.add_subplot(1, 2, 2)

label_origin = label.numpy() * Y_norm_std + Y_norm_mean

prediction_origin = prediction.detach().numpy() * Y_norm_std + \
    Y_norm_mean

ax.plot(pd_all.iloc[plot_index, 0], label_origin[:, 0], '-o', label='Historical')
ax.plot(pd_all.iloc[plot_index, 0], prediction_origin[:, 0], '--s', label='Prediction')

ax.set_xlabel('Date')
ax.set_ylabel('Y')
ax.set_title(stateNamePlot +", retail_and_recreation")
ax.legend()

fig.tight_layout()


fig2 = plt.figure(figsize=(6, 5))
ax1 = fig2.add_subplot(1, 1, 1)
ax1.plot(pd_all.iloc[plot_index, 0], label_origin[:, 0], '-o', label='Historical')
ax1.plot(pd_all.iloc[plot_index, 0], prediction_origin[:, 0], '--s', label='Prediction')
ax1.set_xlabel('Date')
ax1.set_ylabel('Google Retail and Recreation')
ax1.set_title(stateNamePlot +", Google Retail and Recreation", fontsize =14)
ax1.set_xlim(pd.Timestamp('2020-02-15'), pd.Timestamp('2020-06-01'))
ax1.legend()
fig2.autofmt_xdate()

fig3 = plt.figure(figsize=(6, 5))
ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(pd_all.iloc[plot_index, 0], label_origin[:, 1], '-o', label='Historical')
ax3.plot(pd_all.iloc[plot_index, 0], prediction_origin[:, 1], '--s', label='Prediction')
ax3.set_xlabel('Date')
ax3.set_ylabel('Google Grocery and Pharmacy')
ax3.set_xlim(pd.Timestamp('2020-02-15'), pd.Timestamp('2020-06-01'))
# ax3.set_xlim(pd.Timestamp('02/15/2020'), pd.Timestamp('05/01/2020'))
ax3.set_title(stateNamePlot +", Google Grocery and Pharmacy", fontsize =14)
ax3.legend()
fig3.autofmt_xdate()

fig4 = plt.figure(figsize=(6, 5))
ax4 = fig4.add_subplot(1, 1, 1)
ax4.plot(pd_all.iloc[plot_index, 0], label_origin[:, 2], '-o', label='Historical')
ax4.plot(pd_all.iloc[plot_index, 0], prediction_origin[:, 2], '--s', label='Prediction')
ax4.set_xlabel('Date')
ax4.set_ylabel('Google Parks')
ax4.set_title(stateNamePlot +", Google Parks", fontsize =14)
ax4.set_xlim(pd.Timestamp('2020-02-15'), pd.Timestamp('2020-06-01'))
ax4.legend()
fig4.autofmt_xdate()

fig5 = plt.figure(figsize=(6, 5))
ax4 = fig5.add_subplot(1, 1, 1)
ax4.plot(pd_all.iloc[plot_index, 0], label_origin[:, 3], '-o', label='Historical')
ax4.plot(pd_all.iloc[plot_index, 0], prediction_origin[:, 3], '--s', label='Prediction')
ax4.set_xlabel('Date')
ax4.set_ylabel('Google Workplaces')
ax4.set_title(stateNamePlot +", Google Workplaces", fontsize =14)
ax4.set_xlim(pd.Timestamp('2020-02-15'), pd.Timestamp('2020-06-01'))
ax4.legend()
fig5.autofmt_xdate()

fig6 = plt.figure(figsize=(6, 5))
ax4 = fig6.add_subplot(1, 1, 1)
ax4.plot(pd_all.iloc[plot_index, 0], label_origin[:, 4], '-o', label='Historical')
ax4.plot(pd_all.iloc[plot_index, 0], prediction_origin[:, 4], '--s', label='Prediction')
ax4.set_xlabel('Date')
ax4.set_ylabel('Apple Mobility')
ax4.set_title(stateNamePlot +", Apple Mobility", fontsize =14)
ax4.set_xlim(pd.Timestamp('2020-02-15'), pd.Timestamp('2020-06-01'))
ax4.format_xdata = mdates.DateFormatter('%m-%d')
ax4.legend()
fig6.autofmt_xdate()

plt.show()
plt.close()
