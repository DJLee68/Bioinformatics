# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 21:10:55 2018

@author: 선승엽
"""

import numpy as np
import pandas as pd

#def data_type_indicator(data):  # 데이터 타입을 배열로 반환
def data_type_indicator(data, return_list=False):    
    """
    if n!=0, return list of each features type
    """
    temp = list()

    for i in range(data.shape[1]):
        # string data
        if np.any(data.T[i] == data.T[i].astype(str)):
            temp.append(2)

        # categorical data
        elif np.array(pd.get_dummies(pd.DataFrame(data))).shape[1] < 7:
            temp.append(0)

        # continuous data
        elif np.any(data.T[i].astype(int) != data.T):
            temp.append(1)

    result = np.array(temp)
    
    
    if len(result) == 1 and return_list==False:
        return result[0]
    elif return_list==False:
        return 3
    
    else:
        return result