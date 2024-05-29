# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:21:09 2024

@author: umroot
"""
from copy import deepcopy as dc
from torch.utils.data import Dataset


def prepare_dataframe_for_lstm(df, n_steps,xaxis,yaxis):
    df = dc(df)
    df.set_index(xaxis, inplace=True)#time
    
    for i in range(1, n_steps+1):
        df[f'load(t-{i})'] = df[yaxis].shift(i)#load
        
    df.dropna(inplace=True)
    
    return df




class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]