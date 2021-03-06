# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------+
#
# @author: Doctorini
# This module creates a custom vector data loader using Dataset module form
# torch
#
# VectorsDataloader class using a panda Dataframe to build a lazy dictionary
# splited into features and labels keys
#
# VectorsDataloaderSplited is same but the example is divided into two variables
#
#------------------------------------------------------------------------------+

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class VectorsDataloader(Dataset):

    def __init__(self,data_root,padding_idx):
        train = torch.load(data_root)
        self.padding_idx=padding_idx
        self.data = pd.DataFrame(train, columns=['features', 'labels'])
        self.max_len_sentence=pd.Series(self.data['features']).map(len).max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx,0]
        sentence = sentence+[self.padding_idx for _ in range(abs(len(sentence)-self.max_len_sentence))]
        return {'features':np.asarray(sentence), 'labels': self.data.iloc[idx,1]}

    def max_len_sentence(self):
        return self.max_len_sentence

    def set_max_len_sentence(self,max_len_new):
        self.max_len_sentence = max_len_new


class VectorsDataloaderSplited(Dataset):

    def __init__(self,data_root,padding_idx):
        train = torch.load(data_root)
        self.padding_idx=padding_idx
        self.data = pd.DataFrame(train, columns=['features', 'labels'])
        self.X = self.data['features']
        self.Y = self.data['labels']
        self.max_len_sentence=pd.Series(self.data['features']).map(len).max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.X.iloc[idx]
        sentence = sentence+[self.padding_idx for _ in range(abs(len(sentence)-self.max_len_sentence))]
        return np.asarray(sentence, dtype=np.longlong), self.Y.iloc[idx]

    def max_len_sentence(self):
        return self.max_len_sentence

    def set_max_len_sentence(self,max_len_new):
        self.max_len_sentence = max_len_new


class CustomDataLoader(Dataset):

    def __init__(self, data_root="",explicit_data=None):
        data = torch.load(data_root) if explicit_data is None else explicit_data
        self.X = data['features']
        self.Y = data['labels']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sentence = self.X[idx]
        return {'features': np.asarray(sentence), 'labels': self.Y[idx]}

    def instances_count(self):
        return len(self.X)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.X = data['source_ids']
        self.y = data['target_ids']
        self.labels=data['labels']
        self.attention = data['attention_mask']

    def __getitem__(self, idx):
        return {'source_ids': torch.tensor(self.X[idx]), 'target_ids': torch.tensor(self.y[idx]), 'attention_mask': torch.tensor(self.attention[idx]), 'labels': self.labels[idx]}

    def __len__(self):
        return len(self.X)


class CustomDatasetRestMexTwoTask(Dataset):
    def __init__(self, data):
        self.X = data['source_ids']
        self.y = data['target_ids']
        self.y_attraction= data['target_ids_attraction']
        self.labels=data['labels']
        self.labels_attraction=data['labels_attraction']
        self.attention = data['attention_mask']

    def __getitem__(self, idx):
        return {'source_ids': torch.tensor(self.X[idx]), 'target_ids': torch.tensor(self.y[idx]), 'attention_mask': torch.tensor(self.attention[idx]),
                'labels': self.labels[idx], 'target_ids_attraction': torch.tensor(self.y_attraction[idx]), 'labels_attraction': self.labels_attraction[idx]}

    def __len__(self):
        return len(self.X)


class CustomDatasetRestMexTestTwoTask(Dataset):
    def __init__(self, data):
        self.X = data['source_ids']
        self.attention = data['attention_mask']
        self.id = data['id']

    def __getitem__(self, idx):
        return {'source_ids': torch.tensor(self.X[idx]), 'attention_mask': torch.tensor(self.attention[idx]), 'id': self.id[idx]}

    def __len__(self):
        return len(self.X)