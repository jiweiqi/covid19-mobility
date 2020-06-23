#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:07:49 2020

@author: weiqi
modified: Xin He 5/9/2020
Enable saved model parameters
 
It taks about 10 mintues to train the model (5000 steps), but you can change "trainSteps" to do more training. 

#TODO:
    + Identify key predictor via Sharp value: https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30
    + Multiple output prediction: currently only predict the transit station mobility, other values such as 
      stay home and office can also be predicted using same neural network
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import max_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from myFunctions import def_add_datashift, createFolder

from os import path

'''
You can increase this number to do more trainings, but usually 10,000 is enough
'''
trainSteps =5000
matplotlib.rcParams.update({'font.size': 22})
sns.set(style="ticks")
today = pd.to_datetime('today')
today =today.strftime("%Y-%m-%d")

PODA_Model = np.load(("./PODA_Model_"+today+".npy"),allow_pickle='TRUE').item()
google_Mobility_Day = PODA_Model['ML_File_Date']

class MyDataSet(Dataset):
    def __init__(self, data, label):

        self.data = data
        self.label = label

    def __getitem__(self, item):

        return self.data[item], self.label[item]

    def __len__(self):

        return self.data.shape[0]


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

createFolder('./ML Files')
#nn.ReLU() -> nn.Tanh()


# load raw data
pd_all = PODA_Model['ML_Data']


# layers_number=4
# nodes_number = 20



# Assign the ML inputs
col_X_Name = ['US Daily Confirmed', 'US Daily Confirmed Dfdt', 'US Daily Confirmed_shifted_1', 
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
             'RestaurantRestrict', 'StayAtHome', 'WeekDay', 'statecode']  # 

#Assign the ML outputs
col_Y_Name = ['retail_and_recreation', 'grocery_and_pharmacy','parks', 
              'workplaces', 'Apple State']

x_Data=pd_all[col_X_Name]
y_Data=pd_all[col_Y_Name]
pd_used = pd.concat([x_Data, y_Data], axis=1)

# pd_used = pd_all.iloc[:, col_used]
print(list(pd_used.columns))

# Drop rows with Nan value
X_all = np.array(pd_used.dropna().iloc[:, :len(col_X_Name)].to_numpy(), dtype=np.float)
Y_all = np.array(pd_used.dropna().iloc[:, len(col_X_Name):].to_numpy(), dtype=np.float)


# Normalize the input and output datasets, benefit training
X_all_norm = (X_all - X_all.mean(axis=0))/X_all.std(axis=0)
Y_all_norm = (Y_all - Y_all.mean(axis=0))/Y_all.std(axis=0)

#Calculate normalization parameters and save it
xNormMean = X_all.mean(axis=0)
xNormStd = X_all.std(axis=0)
yNormMean = Y_all.mean(axis=0)
yNormStd = Y_all.std(axis=0)

R2_save = pd.DataFrame(columns = ['Layer', 'Node', 'Apple training', 'Apple test', 'google workplace training',
                                  'Google workplaces test', 'Google retail training', 'Google retail test',
                                  'Google park training', 'Google park test', 'Google grocery training', 
                                  'Google grocery test'])

layer_List =[2]   #[1, 2, 3, 4]
nodes_List =[25]  #[15, 20, 25, 30]
for i, layers_number in enumerate(layer_List):
    for j, nodes_number in enumerate(nodes_List):
        
        checkfile = 'checkpoint_MLmodel_'+google_Mobility_Day+'layer_'+str(layers_number)+'node_'+str(nodes_number)+'ReLU'
        
        if path.exists("./ML Files/"+checkfile+'.tar'):
            is_restart = True
        else:
            is_restart= False
            
        ML_Model_Para ={'trainSteps': trainSteps,
                        'col_X_Name': col_X_Name,
                        'col_Y_Name':col_Y_Name,
                        'xNormMean': xNormMean,
                        'xNormStd': xNormStd,
                        'yNormMean': yNormMean,
                        'yNormStd': yNormStd,
                        'model_File': checkfile,
                        'Layers_num':layers_number,
                        'nodes_num':nodes_number,
                        'func':'ReLU',
                        'google_Mobility_Day': google_Mobility_Day}
        np.save(("./ML Files/model_Para_"+google_Mobility_Day+'Layer_'+str(layers_number)+'node_'+str(nodes_number)+'ReLU'+".npy"), ML_Model_Para)
        
        PODA_Model['ML_Model_Para_'+'layer_'+str(layers_number)+'node_'+str(nodes_number)+'ReLU'] = ML_Model_Para
        #save ML Model parameter to file
        np.save(("./PODA_Model_"+today+".npy"), PODA_Model)
        
        '''
        Training the ML Model
        '''
        # train/test dataset spilt for cross validation
        X_train, X_test, Y_train, Y_test = train_test_split(X_all_norm, Y_all_norm,
                                                            test_size=0.33,
                                                            random_state=32)
        
        # convert to tensor
        X_train = torch.Tensor(X_train)
        Y_train = torch.Tensor(Y_train)
        X_test = torch.Tensor(X_test)
        Y_test = torch.Tensor(Y_test)
        
        batchSize = 64
        train_data = MyDataSet(data=X_train, label=Y_train)
        train_loader = DataLoader(train_data, batch_size=batchSize,
                                  shuffle=True, drop_last=True, pin_memory=False)
        
        test_data = MyDataSet(data=X_test, label=Y_test)
        test_loader = DataLoader(test_data, batch_size=batchSize,
                                 shuffle=True, drop_last=True, pin_memory=False)
        
        
        # build pytorch model
        torch.manual_seed(1)  # reproducible
        
        loss_list = {'epoch': [], 'train': [], 'test': []}
        
        # this is one way to define a network
        # define the network
        net = Net(n_feature=X_train.shape[1],
                  layers=layers_number,
                  nodes=nodes_number,
                  n_output=len(col_Y_Name))
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-2)
        
        if is_restart is True:
            checkpoint = torch.load('./ML Files/'+checkfile + '.tar')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_old = checkpoint['epoch']
            loss_list = checkpoint['loss_list']
        
        # train the network
        for epoch in tqdm(range(trainSteps)):
            if is_restart:
                if epoch < epoch_old:
                    continue
        
            loss_list['epoch'].append(epoch)
        
            loss_train = 0
            loss_test = 0
            for i_sample, (inputs, outputs) in enumerate(train_loader):
                prediction = net(inputs)
                loss = criterion(prediction, outputs)
                loss_train += loss.item()
        
                optimizer.zero_grad()
        
                loss.backward()
        
                optimizer.step()
        
            loss_list['train'].append(loss_train/(i_sample+1))
        
            with torch.no_grad():
                for i_sample, (inputs, outputs) in enumerate(test_loader):
                    prediction = net(inputs)
                    loss = criterion(prediction, outputs)
                    loss_test += loss.item()
        
            loss_list['test'].append(loss_test/(i_sample+1))
        
            if epoch % 1000 == 0:
        
                print("@epoch {:8d} oss_train {:.2e} loss_test {:.2e}".format(
                    epoch, loss_list['train'][-1], loss_list['test'][-1]))
        
                torch.save({'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss_list': loss_list,
                            }, './ML Files/'+checkfile+'.tar')
        
                # regression plot
                prediction_train = net(X_train)
                prediction_test = net(X_test)
        
                fig = plt.figure(figsize=(12, 12))
        
                ax = fig.add_subplot(2, 2, 1)
                
                ax.plot(Y_train[:300, 4].detach().numpy(),
                        prediction_train[:300, 4].detach().numpy(), 'o', label='Train')
                ax.plot(Y_test[:300, 4].detach().numpy(),
                        prediction_test[:300, 4].detach().numpy(), 's', label='Test')
                ax.text(-1.8, 1.5, 'Apple Mobility', fontsize=22)
                ax.set_xlabel('Label', fontsize=22)
                ax.set_ylabel('Prediction', fontsize=22)
                ax.legend(loc='lower right', fontsize=20)
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.tick_params(labelsize=18)
                ax.set_title(" R2_Train={:.2f}, R2_Test={:.2f}".format(r2_score(Y_train[:, 4].detach().numpy(), prediction_train[:, 4].detach().numpy()),
                                                                      r2_score(Y_test[:, 4].detach().numpy(), prediction_test[:, 4].detach().numpy())), fontsize=22)
        
                ax = fig.add_subplot(2, 2, 2)
                # ax.text(x, y, s, fontsize=12)
                ax.plot(Y_train[:300, 0].detach().numpy(),
                        prediction_train[:300, 0].detach().numpy(), 'o', label='Train')
                ax.plot(Y_test[:300, 0].detach().numpy(),
                        prediction_test[:300, 0].detach().numpy(), 's', label='Test')
                ax.text(-1.8, 1.5, 'Retail and Recreation', fontsize=22)
                ax.set_xlabel('Label', fontsize=22)
                ax.set_ylabel('Prediction', fontsize=22)
                ax.legend(loc='lower right', fontsize = 20)
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.tick_params(labelsize=18)
                ax.set_title(" R2_Train={:.2f}, R2_Test={:.2f}".format(r2_score(Y_train[:, 0].detach().numpy(), prediction_train[:, 0].detach().numpy()),
                                                                      r2_score(Y_test[:, 0].detach().numpy(), prediction_test[:, 0].detach().numpy())), fontsize=22)
        
                ax = fig.add_subplot(2, 2, 3)
                ax.plot(Y_train[:300, 1].detach().numpy(),
                        prediction_train[:300, 1].detach().numpy(), 'o', label='Train')
                ax.plot(Y_test[:300, 1].detach().numpy(),
                        prediction_test[:300, 1].detach().numpy(), 's', label='Test')
                ax.text(-1.8, 1.5, 'Grocery and Pharmacy', fontsize=22)
                ax.set_xlabel('Label', fontsize=22)
                ax.set_ylabel('Prediction', fontsize=22)
                ax.legend(loc='lower right', fontsize = 20)
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.tick_params(labelsize=18)
                ax.set_title(" R2_Train={:.2f}, R2_Test={:.2f}".format(r2_score(Y_train[:, 1].detach().numpy(), prediction_train[:, 1].detach().numpy()),
                                                                      r2_score(Y_test[:, 1].detach().numpy(), prediction_test[:, 1].detach().numpy())), fontsize=22)
                ax = fig.add_subplot(2, 2, 4)
                ax.plot(Y_train[:300, 3].detach().numpy(),
                        prediction_train[:300, 3].detach().numpy(), 'o', label='Train')
                ax.plot(Y_test[:300, 3].detach().numpy(),
                        prediction_test[:300, 3].detach().numpy(), 's', label='Test')
                ax.text(-1.8, 1.5, 'Workplaces', fontsize=22)
                ax.set_xlabel('Label', fontsize=22)
                ax.set_ylabel('Prediction', fontsize=22)
                ax.legend(loc='lower right', fontsize = 20)
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.tick_params(labelsize=18)
                ax.set_title(" R2_Train={:.2f}, R2_Test={:.2f}".format(r2_score(Y_train[:, 3].detach().numpy(), prediction_train[:, 3].detach().numpy()),
                                                                      r2_score(Y_test[:, 3].detach().numpy(), prediction_test[:, 3].detach().numpy())), fontsize=22)
                
                
                new_row = {'Layer':layers_number, 'Node':nodes_number, 
                           'Apple training': r2_score(Y_train[:, 4].detach().numpy(), prediction_train[:, 4].detach().numpy()), 
                           'Apple test': r2_score(Y_test[:, 4].detach().numpy(), prediction_test[:, 4].detach().numpy()), 
                           'google workplace training': r2_score(Y_train[:, 3].detach().numpy(), prediction_train[:, 3].detach().numpy()),
                           'Google workplaces test': r2_score(Y_test[:, 3].detach().numpy(), prediction_test[:, 3].detach().numpy()), 
                           'Google retail training': r2_score(Y_train[:, 0].detach().numpy(), prediction_train[:, 0].detach().numpy()), 
                           'Google retail test': r2_score(Y_test[:, 0].detach().numpy(), prediction_test[:, 0].detach().numpy()),
                           'Google park training': r2_score(Y_train[:, 2].detach().numpy(), prediction_train[:, 2].detach().numpy()), 
                           'Google park test': r2_score(Y_test[:, 2].detach().numpy(), prediction_test[:, 2].detach().numpy()), 
                           'Google grocery training': r2_score(Y_train[:, 1].detach().numpy(), prediction_train[:, 1].detach().numpy()), 
                           'Google grocery test': r2_score(Y_test[:, 1].detach().numpy(), prediction_test[:, 1].detach().numpy())
                           }
                
                
                R2_save=R2_save.append(new_row, ignore_index =True)  
                
                
                
                fig.tight_layout()
        
                fig.savefig('./ML Files/'+checkfile, 
                            dpi=600, facecolor="w", edgecolor="w", 
                            orientation="portrait", bbox_inches='tight')
                ax.cla()
                fig.clf()
                plt.close()
                
                
PODA_Model['ML_R2_ReLU'] = R2_save
np.save(("./PODA_Model_"+today+".npy"), PODA_Model)

R2_save.to_csv('./R2_ReLU.csv')
