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

    emo_eval_train={'data':X,'target':y}
    emo_eval_test = {'data': x_test, 'target': y_test}
    emo_eval_dev = {'data': x_dev, 'target': y_dev}

    texts = emo_eval_train['data']
    labels_target = emo_eval_train['target']

    print(texts[0])

    log_exp_run = make_logger()
    log_exp_run.experiments("Categories-labels: ")
    log_exp_run.experiments(labels)
    log_exp_run.experiments("Number of instances for training: ")
    log_exp_run.experiments(len(emo_eval_train['data']))
    log_exp_run.experiments("Number of instances for testing: ")
    log_exp_run.experiments(len(emo_eval_test['data']))

    removing_stop_words(texts)
    dataset_train = {'features': [], 'labels': []}
    max_sequence_length = 1000
    max_nb_words = 2000
    tokenizer=Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(texts)
    sequences_train=tokenizer.texts_to_sequences(texts)

    wdir = os.getcwd()
    if not os.path.exists(wdir + '/datasets/dataset_train_emo_eval_nosw'):
        dataset_train['features']=pad_sequences(sequences_train, maxlen=max_sequence_length)
        dataset_train['labels']=labels_target
        torch.save(dataset_train, wdir + "/datasets/dataset_train_emo_eval_nosw")

    dataset_dev = {'features': [], 'labels': []}
    texts = emo_eval_dev['data']
    labels_target = emo_eval_dev['target']
    removing_stop_words(texts)

    sequences_test = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    log_exp_run.experiments("Found unique tokens: " + str(len(word_index)))

    if not os.path.exists(wdir + '/datasets/dataset_dev_emo_eval_nosw'):
        dataset_dev['features'] = pad_sequences(sequences_test, maxlen=max_sequence_length)
        dataset_dev['labels'] = labels_target
        torch.save(dataset_dev, wdir + "/datasets/dataset_dev_emo_eval_nosw")

    dataset_test = {'features': [], 'labels': []}
    texts = emo_eval_test['data']
    labels_target = emo_eval_test['target']
    removing_stop_words(texts)

    sequences_test = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    log_exp_run.experiments("Found unique tokens: " + str(len(word_index)))

    if not os.path.exists(wdir + '/datasets/dataset_test_emo_eval_nosw'):
        dataset_test['features']=pad_sequences(sequences_test, maxlen=max_sequence_length)
        dataset_test['labels']=labels_target
        torch.save(dataset_test, wdir + "/datasets/dataset_test_emo_eval_nosw")

    if not os.path.exists(wdir + '/datasets/dictionary_emo_eval_nosw'):
        torch.save(word_index, wdir + "/datasets/dictionary_emo_eval_nosw")


if __name__ == "__main__":
    build_dataset_and_dict()
