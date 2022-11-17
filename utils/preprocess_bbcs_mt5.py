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
    path_dataset = "C:\\Users\\Laptop\\Desktop\\bbcsport"
    X = []
    y = []
    labels = ['athletics', 'cricket', 'football', 'rugby', 'tennis']

    for i, label in enumerate(labels):
        folder = path_dataset + "/" + label
        for file in os.listdir(folder):
            f = open(folder + "/" + file, "r")
            text = f.read()
            X.append(text)
            y.append(str(i))

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

    torch.save(stt_train, wdir + "/datasets/dataset_train_bbcs_mt5")

    torch.save(stt_dev, wdir + "/datasets/dataset_test_bbcs_mt5")


if __name__ == "__main__":
    build_dataset_and_dict()
