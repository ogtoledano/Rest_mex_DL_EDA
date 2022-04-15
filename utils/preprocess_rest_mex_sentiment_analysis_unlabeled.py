import pandas as pd
import os
import torch
import nltk

import pandas as pd
from nltk.corpus import stopwords

from utils.logging_custom import make_logger


from sklearn.model_selection import train_test_split

MAX_LEN_SENTENCE=0
from transformers import T5Tokenizer,AutoTokenizer

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

    ids = [int(value) for value in data['id']]

    source_encoding = tokenizer(
        data['input_text'],
        max_length=200,
        padding='max_length',
        pad_to_max_length=True,
        truncation=True,
    )

    encodings = {
        'source_ids': source_encoding['input_ids'],
        'attention_mask': source_encoding['attention_mask'],
        'id': torch.tensor(ids, dtype=torch.long)
    }

    return encodings


def build_dataset_and_dict():
    os.chdir('../')
    wdir=os.getcwd()
    file_train=wdir+"/datasets/Rest_Mex_2022_Sentiment_Analysis_Track_Test.xlsx"
    df=pd.read_excel(r''+file_train, engine='openpyxl')
    print(df.iloc[0,1])
    print(df.iloc[350,2])
    print(df.shape)

    X = []
    id =[]
    for i in range(df.shape[0]):
        whole_text = str(df.iloc[i, 1]) + ": " + str(df.iloc[i, 2])
        X.append(whole_text)
        id.append(str(df.iloc[i, 0]))

    stt_test = {'input_text': X, 'id': id}

    log_exp_run = make_logger()

    log_exp_run.experiments("Number of instances for training: ")
    log_exp_run.experiments(len(stt_test['input_text']))

    removing_stop_words(stt_test['input_text'])

    prepare_input(stt_test['input_text'])

    stt_test = tokenize(stt_test)

    torch.save(stt_test, wdir + "/datasets/dataset_test_stt_mt5")


if __name__ == "__main__":
    build_dataset_and_dict()
