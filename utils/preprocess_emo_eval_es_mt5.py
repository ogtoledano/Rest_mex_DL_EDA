# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------+
#
# @author: Doctorini
# This module preprocces a text file dataset: 20newsgroups removing first
# stop_words and discretize each word form dictionary
#------------------------------------------------------------------------------+

import sys
import os
import torch
import nltk
import sys



sys.path.append('..\\..\\Text_Cat_Based_EDA')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\utils')
from utils.embedding_builder import build_word_embedding,build_tensor
from nltk.corpus import stopwords
from utils.custom_dataloader import VectorsDataloader
from torch.utils.data import DataLoader
from utils.logging_custom import make_logger
from sklearn.datasets import fetch_20newsgroups,fetch_20newsgroups_vectorized
import pandas as pd
from sklearn.model_selection import train_test_split
from random import sample

MAX_LEN_SENTENCE=0

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd


def removing_stop_words(texts):
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        sentence = [word for word in tokens if word not in stop_words]
        texts[i] = ' '.join(sentence)


def build_dataset_and_dict():
    os.chdir('../')
    wdir = os.getcwd()
    path_dataset = wdir + "/datasets/emo_eval_es"
    X = []
    y = []
    labels=['anger','disgust','fear','joy','others', 'sadness','surprise']

    for i, label in enumerate(labels):
        file = path_dataset + "/entrenamiento/" + label+".txt"
        lines = open(file, "r", encoding="utf8").readlines()[1:]
        for text in lines:
            text = text.split('\t')[1]
            X.append(text)
            y.append(i)

    x_dev = []
    y_dev = []

    for i, label in enumerate(labels):
        file = path_dataset + "/dev/" + label+".txt"
        lines = open(file, "r", encoding="utf8").readlines()[1:]
        for text in lines:
            text = text.split('\t')[1]
            x_dev.append(text)
            y_dev.append(i)

    x_test=[]
    y_test=[]

    for i, label in enumerate(labels):
        file = path_dataset + "/test/" + label+".txt"
        lines = open(file, "r", encoding="utf8").readlines()[1:]
        for text in lines:
            text = text.split('\t')[1]
            x_test.append(text)
            y_test.append(i)

    emo_eval_train={'input_text':X,'target_text':y}
    emo_eval_test = {'input_text': x_test, 'target_text': y_test}
    emo_eval_dev = {'input_text': x_dev, 'target_text': y_dev}

    train_df= pd.DataFrame.from_dict(emo_eval_train)
    train_df['input_text']=train_df['input_text'].replace("\n"," ")
    train_df['prefix'] = "multilabel classification"

    test_df = pd.DataFrame.from_dict(emo_eval_test)
    test_df['input_text'] = train_df['input_text'].replace("\n", " ")
    test_df['prefix'] = "multilabel classification"

    dev_df = pd.DataFrame.from_dict(emo_eval_dev)
    dev_df['input_text'] = train_df['input_text'].replace("\n", " ")
    test_df['prefix'] = "multilabel classification"

    return train_df,test_df,dev_df


if __name__ == "__main__":
    build_dataset_and_dict()
