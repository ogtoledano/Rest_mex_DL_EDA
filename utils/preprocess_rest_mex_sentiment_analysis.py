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
import pandas as pd



sys.path.append('..\\..\\Topic_Rec_Based_EDA')
sys.path.append('..\\..\\Topic_Rec_Based_EDA\\utils')
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


def removing_stop_words(texts):
    # Removing stop words
    stop_words = set(stopwords.words('spanish'))
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        sentence = [word for word in tokens if word not in stop_words]
        texts[i] = ' '.join(sentence)


def build_dataset_and_dict():
    os.chdir('../')
    wdir=os.getcwd()
    file_train=wdir+"/datasets/rest-mex_2021_Sentiment_data_train.csv"
    df=pd.read_csv(file_train)
    print(df.iloc[350,1])
    print(df.iloc[350,2])
    print(df.shape)


    y=[]
    X=[]

    for i in range(df.shape[0]):
        whole_text=str(df.iloc[i,1])+". "+str(df.iloc[i,2])
        X.append(whole_text)
        y.append(int(df.iloc[i,8])-1)

    X_train, X_test, y_train,y_test=train_test_split(X, y, test_size=0.30, random_state=142)
    bbc_train={'data':X_train,'target':y_train}
    bbc_test = {'data': X_test, 'target': y_test}
    texts = bbc_train['data']
    labels_target = bbc_train['target']

    log_exp_run = make_logger()

    log_exp_run.experiments("Number of instances for training: ")
    log_exp_run.experiments(len(bbc_train['data']))
    log_exp_run.experiments("Number of instances for testing: ")
    log_exp_run.experiments(len(bbc_test['data']))

    removing_stop_words(texts)
    dataset_train = {'features': [], 'labels': []}
    max_sequence_length = 1000
    max_nb_words = 2000
    tokenizer=Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(texts)
    sequences_train=tokenizer.texts_to_sequences(texts)

    if not os.path.exists(wdir + '/datasets/dataset_train_rest_mex_sentiment_nosw'):
        dataset_train['features']=pad_sequences(sequences_train, maxlen=max_sequence_length)#[0:5]
        dataset_train['labels']=labels_target#[0:5]
        torch.save(dataset_train, wdir + "/datasets/dataset_train_rest_mex_sentiment_nosw")

    dataset_test = {'features': [], 'labels': []}
    texts = bbc_test['data']
    labels_target = bbc_test['target']
    removing_stop_words(texts)

    sequences_test = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    log_exp_run.experiments("Found unique tokens: " + str(len(word_index)))

    if not os.path.exists(wdir + '/datasets/dataset_test_rest_mex_sentiment_nosw'):
        dataset_test['features']=pad_sequences(sequences_test, maxlen=max_sequence_length)#[0:5]
        dataset_test['labels']=labels_target#[0:5]
        torch.save(dataset_test, wdir + "/datasets/dataset_test_rest_mex_sentiment_nosw")

    if not os.path.exists(wdir + '/datasets/dictionary_rest_mex_sentiment_nosw'):
        torch.save(word_index, wdir + "/datasets/dictionary_rest_mex_sentiment_nosw")


if __name__ == "__main__":
    build_dataset_and_dict()
