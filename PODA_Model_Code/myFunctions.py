# -*- coding: utf-8 -*-
"""
Created on Tue May 12 00:52:39 2020

@author: hexx
"""
import os

def def_add_datashift (data_frame, column_name, x):
    for i in x:
        shift_i = column_name + '_shifted_'+str(i)
        data_frame[shift_i] = data_frame[column_name]
        data_frame[shift_i] = data_frame[shift_i].shift(0-i)
    return(data_frame)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def Google_factor (data_used, factor):
    data_used['work factor'] = 1 + data_used['workplaces']/100*factor[0]
    data_used['school factor'] = 1 + data_used['workplaces']/100*factor[1]
    data_used['medical factor'] = 1 + data_used['grocery_and_pharmacy']/100*factor[2]
    data_used['shopping factor'] = 1 + data_used['grocery_and_pharmacy']/100*factor[3]
    data_used['social factor'] = 1 + data_used['workplaces']/100*factor[4]
    data_used['park factor'] = 1 + data_used['parks']/100*factor[5]
    data_used['transport someone factor'] = 1+ data_used['workplaces']/100*factor[7]
    data_used['meals factor'] = 1 + data_used['workplaces']/100*factor[6]
    data_used['else factor'] = 1+ data_used['workplaces']/100*factor[7]
    
    data_used['accumulated factor'] = (data_used['Work']*data_used['work factor'] + \
        data_used['School/Daycare/Religious activity']*data_used['school factor'] + \
            data_used['Medical/Dental services']*data_used['medical factor'] + \
                data_used['Shopping/Errands']*data_used['shopping factor'] + \
                    data_used['Social/Recreational']*factor[8]*data_used['social factor'] + \
                        data_used['Social/Recreational']*(1-factor[8])*data_used['park factor'] + \
                            data_used['Meals']*data_used['meals factor'] +\
                                data_used['Transport someone']*data_used['transport someone factor'] + \
                                    data_used['Something else']*data_used['else factor'])/100 + factor[9]
    return data_used