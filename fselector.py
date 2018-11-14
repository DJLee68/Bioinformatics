import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import wrapper as wp
from tkinter import simpledialog
import pandas as pd
class FSelector:
    alg_list = None
    

    def __init__(self):
        self.alg_list = ['PCC', 'PCA', 'filter', 'Wrapper']
        self.fs_size = 0
    def start_fs(self, alg_idx, tr_data, tr_ans, ts_data, ts_ans, calg_idx):
        if alg_idx == 0:

            return self.pcc(tr_data, tr_ans, ts_data)
        elif alg_idx == 1:
            return self.pca(tr_data, ts_data)
        elif alg_idx == 2:
            return self._filter(tr_data, ts_data)
        elif alg_idx == 3:
            return wp.wrapper(tr_data, tr_ans, ts_data, ts_ans)

    def pcc(self, tr_data, tr_ans, ts_data):
        corr_array = []
        
        for i in range(tr_data.shape[1]):
            corr_array.append(np.corrcoef(tr_data.T[i].astype(float), tr_ans.astype(float))[0, 1])
        corr_array = np.square(corr_array)
        pcc_feature_idx = np.flip(np.argsort(corr_array), 0)
        fs_tr_data = tr_data[:, pcc_feature_idx[0:self.fs_size]]
        fs_ts_data = ts_data[:, pcc_feature_idx[0:self.fs_size]]

        return fs_tr_data, fs_ts_data

    def pca(self, tr_data, ts_data):
        pca = PCA(n_components=self.fs_size)
        pca.fit(tr_data)

        fs_tr_data = pca.transform(tr_data)
        fs_ts_data = pca.transform(ts_data)

        return fs_tr_data, fs_ts_data

    def _filter(self, tr_data, ts_data):
        fs_tr_data=self.filter_func(tr_data)
        fs_ts_data=self.filter_func(ts_data)
        
        return fs_tr_data, fs_ts_data
    
    def filter_func(self, data):
        index=[]
        for i in range (data.shape[1]):
            for j in range (i):
                if mutual_info_score(data.T[i], data.T[j])==1:
                    index.append(j)
        index=np.array(index)
        return np.delete(data, np.unique(index).astype(int), axis=1)
     
    def wrapper(self, tr_data, tr_ans, ts_data, ts_ans):
        return wp.wrapper(tr_data, tr_ans, ts_data, ts_ans)

    
        
        
    def set_fs_size(self, fs_size):
        self.fs_size = fs_size

    def get_alg_list(self):
        return self.alg_list






