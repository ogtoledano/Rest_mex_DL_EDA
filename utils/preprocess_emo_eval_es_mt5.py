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

from nltk.corpus import stopwords
from transformers import T5Tokenizer,AutoTokenizer


def removing_stop_words(texts):
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        sentence = [word for word in tokens if word not in stop_words]
        texts[i] = ' '.join(sentence)


def prepare_input(texts):
    for i, text in enumerate(texts):
        texts[i] = "multilabel classification: {} </s>".format(text)

def tokenize(data):
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

    source_encoding = tokenizer(
        data['input_text'],
        max_length=200,
        padding='max_length',
        pad_to_max_length=True,
        truncation=True,
    )

    target_encoding = tokenizer(
        data['target_text'],
        max_length=1,
        pad_to_max_length=False,
        truncation=False,
    )

    encodings = {
        'source_ids': source_encoding['input_ids'],
        'target_ids': target_encoding['input_ids'],
        'attention_mask': source_encoding['attention_mask'],
    }

    return encodings

def build_dataset_and_dict():
    os.chdir('../')
    wdir = os.getcwd() + "/" if not os.path.exists("/content/Rest_mex_DL_EDA/") else "/content/Rest_mex_DL_EDA/"  # only colab
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
            y.append(str(i))

    x_dev = []
    y_dev = []

    for i, label in enumerate(labels):
        file = path_dataset + "/dev/" + label+".txt"
        lines = open(file, "r", encoding="utf8").readlines()[1:]
        for text in lines:
            text = text.split('\t')[1]
            x_dev.append(text)
            y_dev.append(str(i))

    x_test=[]
    y_test=[]

    for i, label in enumerate(labels):
        file = path_dataset + "/test/" + label+".txt"
        lines = open(file, "r", encoding="utf8").readlines()[1:]
        for text in lines:
            text = text.split('\t')[1]
            x_test.append(text)
            y_test.append(str(i))

    emo_eval_train={'input_text':X,'target_text':y}
    emo_eval_test = {'input_text': x_test, 'target_text': y_test}
    emo_eval_dev = {'input_text': x_dev, 'target_text': y_dev}

    removing_stop_words(emo_eval_train['input_text'])
    removing_stop_words(emo_eval_test['input_text'])
    removing_stop_words(emo_eval_dev['input_text'])

    prepare_input(emo_eval_train['input_text'])
    prepare_input(emo_eval_test['input_text'])
    prepare_input(emo_eval_dev['input_text'])

    emo_eval_train=tokenize(emo_eval_train)
    emo_eval_test = tokenize(emo_eval_test)
    emo_eval_dev = tokenize(emo_eval_dev)

    torch.save(emo_eval_train, wdir + "datasets/dataset_train_emo_eval_mt5")

    torch.save(emo_eval_test, wdir + "datasets/dataset_test_emo_eval_mt5")

    torch.save(emo_eval_dev, wdir + "datasets/dataset_dev_emo_eval_mt5")


if __name__ == "__main__":
    build_dataset_and_dict()
