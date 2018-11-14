# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 03:26:59 2018

@author: 선승엽
"""

import utilfunc
import numpy as np

class Preprocessor:
    def __init__(self):
        self.alg_list =  ['categorical', 'standard_norm', 'min-max']
    def preprocess(self, alg_idx, tr_data, ts_data):
        _index=utilfunc.data_type_indicator(tr_data, return_list = True)
        pre_tr_data=np.array(tr_data)
        pre_ts_data=np.array(ts_data)
        if alg_idx == 0:
            for i in range(tr_data.shape[1]):
                if _index[i]==2 or _index[1]==0:
                    pre_tr_data.T[i]=self.label_encoder(tr_data.T[i])
                    pre_ts_data.T[i]=self.label_encoder(ts_data.T[i])
                else:
                    pre_tr_data.T[i]=self.categorical(tr_data.T[i])
                    pre_ts_data.T[i]=self.categorical(ts_data.T[i])
            return pre_tr_data, pre_ts_data
        elif alg_idx == 1:
            for i in range(tr_data.shape[1]):
                if _index[i]==2 or _index[1]==0:
                    pre_tr_data.T[i]=self.label_encoder(tr_data.T[i])
                    pre_ts_data.T[i]=self.label_encoder(ts_data.T[i])
                else:
                    pre_tr_data.T[i]=self.standard_normalization(tr_data.T[i])
                    pre_ts_data.T[i]=self.standard_normalization(ts_data.T[i])
            return pre_tr_data, pre_ts_data
        elif alg_idx ==2:
            _min = 0
            _max = 1

            for i in range(tr_data.shape[1]):
                if _index[i]==2 or _index[1]==0:
                    pre_tr_data.T[i]=self.label_encoder(tr_data.T[i])
                    pre_ts_data.T[i]=self.label_encoder(ts_data.T[i])
                else:
                    pre_tr_data.T[i]=self.min_max(tr_data.T[i], _min, _max)
                    pre_ts_data.T[i]=self.min_max(ts_data.T[i], _min, _max)
        
            return pre_tr_data, pre_ts_data
  
        
    def label_encoder(self, data):
        temp_list = []
        for i in range(data.shape[0]):
            string = str(data[i])
            if string in temp_list:
                data[i] = int(temp_list.index(string))
            else:
                temp_list.append(string)
                data[i] = int(temp_list.index(string))
        return np.array(data)

    def standard_normalization(self, data):
        return (data-np.mean(data))/np.std(data)

    def min_max(self, data, min_val, max_val):
        data = ((max_val-min_val)*((data-data.min())/(data.max()-data.min()))) + min_val
        
        return data
    def categorical(self, data):
        z= (data - data.mean())/data.std()
        for i in range(z.shape[0]):
            if z[i] <= -1.28:
                z[i] = 0
            elif z[i] <= -0.84:
                z[i] = 1
            elif z[i] <= -0.54:
                z[i] = 2
            elif z[i] <= -0.25:
                z[i] = 3
            elif z[i] <= 0:
                z[i] = 4
            elif z[i] <= 0.25:
                z[i] = 5
            elif z[i] <= 0.54:
                z[i] = 6
            elif z[i] <= 0.84:
                z[i] = 7
            elif z[i] <= 1.28:
                z[i] = 8
            else:
                z[i] = 9
        return z
  

    def get_alg_list(self):
        return self.alg_list



