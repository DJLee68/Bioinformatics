# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 03:26:59 2018

@author: 선승엽
"""
from sklearn import preprocessing
from tkinter import simpledialog
import utilfunc
import numpy as np

class Preprocessor:
    def __init__(self):
        self.alg_list =  ['None','standard_norm', 'min-max']
    def preprocessing(self, alg_idx, tr_data, ts_data):
        _index=utilfunc.data_type_indicator(tr_data, return_list = True)
        pre_tr_data=np.array(tr_data)
        pre_ts_data=np.array(ts_data)
        
        if alg_idx == 0:
            for i in range(tr_data.shape[1]):
                if _index[i]==2 or _index[1]==1:
                    pre_tr_data.T[i]=self.label_encoder(tr_data.T[i])
                    pre_ts_data.T[i]=self.label_encoder(ts_data.T[i])
            print(pre_tr_data.shape==tr_data.shape)
            return pre_tr_data, pre_ts_data
        elif alg_idx == 1:
            for i in range(tr_data.shape[1]):
                if _index[i]==2 or _index[1]==1:
                    pre_tr_data.T[i]=self.label_encoder(tr_data.T[i])
                    pre_ts_data.T[i]=self.label_encoder(ts_data.T[i])
                else:
                    pre_tr_data.T[i]=self.standard_normalization(tr_data.T[i])
                    pre_ts_data.T[i]=self.standard_normalization(ts_data.T[i])
            print(pre_tr_data.shape==tr_data.shape)
            return pre_tr_data, pre_ts_data
        elif alg_idx ==2:
            _min = 0
            _max = 1
            """
            _min=simpledialog.askinteger("Input", "최소값을 입력하시오")
            _max=simpledialog.askinteger("Input", "최대값을 입력하시오")
            """
            for i in range(tr_data.shape[1]):
                if _index[i]==2 or _index[1]==1:
                    pre_tr_data.T[i]=self.label_encoder(tr_data.T[i])
                    pre_ts_data.T[i]=self.label_encoder(ts_data.T[i])
                else:
                    pre_tr_data.T[i]=self.min_max(tr_data.T[i], _min, _max)
                    pre_ts_data.T[i]=self.min_max(ts_data.T[i], _min, _max)
            print(pre_tr_data.shape==tr_data.shape)
            return pre_tr_data, pre_ts_data
        
        
    def label_encoder(self, data):
        temp_list = []
        for i in range(data.shape[0]):
            string = str(data[i])
            string.replace(" ","")
            string.replace("'","")
            string.replace("`","")
            if string not in temp_list:
                temp_list.append(string)
            data[i] = int(temp_list.index(data[i]))
        print(data)
        return data

    def standard_normalization(self, data):
        return (data-np.mean(data))/np.std(data)

    def min_max(self, data, min_val, max_val):
        data = ((max_val-min_val)*((data-data.min())/(data.max()-data.min()))) + min_val
        
        return data


    def get_alg_list(self):
        return self.alg_list



