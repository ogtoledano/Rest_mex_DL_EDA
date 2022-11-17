import pandas as pd
import os
import torch
import nltk

import pandas as pd
from nltk.corpus import stopwords

from utils.logging_custom import make_logger


from sklearn.model_selection import train_test_split


from transformers import T5Tokenizer,AutoTokenizer
MAX_LEN_SENTENCE = 0


def extract_text_label(text):
    text_post = text[:len(text) - 3]
    label = text[len(text)-2:len(text)-1]
    text_array = text_post.split(",")[3:]
    text_post = ','.join(text_array)
    return text_post, int(label)


def removing_stop_words(texts):
    # Removing stop words
    stop_words = set(stopwords.words('spanish'))
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

    targets_polarity = [value[0] for value in data['target_text']]

    target_encoding_polarity = tokenizer(
        targets_polarity,
        max_length=2,
        pad_to_max_length=False,
        truncation=False,
    )

    targets_polarity = [int(value) for value in targets_polarity]

    encodings = {
        'source_ids': source_encoding['input_ids'],
        'target_ids': target_encoding_polarity['input_ids'],
        'attention_mask': source_encoding['attention_mask'],
        'labels': torch.tensor(targets_polarity, dtype=torch.long)
    }

    return encodings


def build_dataset_and_dict():
    os.chdir("../")
    wdir = os.getcwd()
    path_dataset = "C:\\Users\\Laptop\\Desktop\\youtube"

    file_train = path_dataset + "/Youtube01-Psy.csv"

    f = open(file_train, "r", encoding="utf8")
    lines = f.readlines()
    lines = lines[1:]
    y = []
    X = []

    for line in lines:
        text, label = extract_text_label(line)
        X.append(text)
        y.append(str(label))

    file_train = path_dataset + "/Youtube02-KatyPerry.csv"

    f = open(file_train, "r", encoding="utf8")
    lines = f.readlines()
    lines = lines[1:]

    for line in lines:
        text, label = extract_text_label(line)
        X.append(text)
        y.append(str(label))

    file_train = path_dataset + "/Youtube03-LMFAO.csv"

    f = open(file_train, "r", encoding="utf8")
    lines = f.readlines()
    lines = lines[1:]

    for line in lines:
        text, label = extract_text_label(line)
        X.append(text)
        y.append(str(label))

    file_train = path_dataset + "/Youtube04-Eminem.csv"

    f = open(file_train, "r", encoding="utf8")
    lines = f.readlines()
    lines = lines[1:]

    for line in lines:
        text, label = extract_text_label(line)
        X.append(text)
        y.append(str(label))

    file_train = path_dataset + "/Youtube05-Shakira.csv"

    f = open(file_train, "r", encoding="utf8")
    lines = f.readlines()
    lines = lines[1:]

    for line in lines:
        text, label = extract_text_label(line)
        X.append(text)
        y.append(str(label))

    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=12345)
    stt_train = {'input_text': x_train, 'target_text': y_train}
    stt_dev = {'input_text': x_test, 'target_text': y_test}

    log_exp_run = make_logger()

    log_exp_run.experiments("Number of instances for training: ")
    log_exp_run.experiments(len(stt_train['input_text']))
    log_exp_run.experiments("Number of instances for dev set: ")
    log_exp_run.experiments(len(stt_dev['input_text']))

    removing_stop_words(stt_train['input_text'])
    removing_stop_words(stt_dev['input_text'])

    prepare_input(stt_train['input_text'])
    prepare_input(stt_dev['input_text'])

    stt_train = tokenize(stt_train)
    stt_dev = tokenize(stt_dev)

    torch.save(stt_train, wdir + "/datasets/dataset_train_ysc_mt5")

    torch.save(stt_dev, wdir + "/datasets/dataset_test_ysc_mt5")


if __name__ == "__main__":
    build_dataset_and_dict()
