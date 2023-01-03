import pandas as pd
import os
import torch
import nltk

import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split

MAX_LEN_SENTENCE=0
from transformers import T5Tokenizer,AutoTokenizer

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.es.examples import sentences
import es_core_news_md
from nltk import SnowballStemmer
from transformers import MarianMTModel, MarianTokenizer
import nltk
from nltk.corpus import wordnet as wn


class CustomDataset(Dataset):
    def __init__(self, data):
        self.X = data['input_ids']
        self.attention = data['attention_mask']

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.X[idx]), 'attention_mask': torch.tensor(self.attention[idx])}

    def __len__(self):
        return len(self.X)


def removing_stop_words(texts):
    # Removing stop words
    stop_words = set(stopwords.words('spanish'))
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        sentence = [word for word in tokens if word not in stop_words]
        texts[i] = ' '.join(sentence)


def lemmatization(texts):
    nlp = es_core_news_md.load()
    for i, text in enumerate(texts):
        doc = nlp(text)
        sentence = [word.lemma_.lower() for word in doc]
        texts[i] = ' '.join(sentence)


def normalize(texts):
    nlp = es_core_news_md.load()
    for i, text in enumerate(texts):
        doc = nlp(text)
        words = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
        lexical_tokens = [t.lower() for t in words if len(t) > 3 and t.isalpha()]
        texts[i] = ' '.join(lexical_tokens)


def get_mt_model(translation, device):
    model_name = 'Helsinki-NLP/opus-mt-' + translation
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, ' '.join(nltk.word_tokenize(text[:500]))) for text in
                     batch_texts]
    return formated_bach


def perform_translation(batch_texts, device, model, tokenizer, language="es"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    model.to(device)
    source_encoding = tokenizer(formated_batch_texts, return_tensors="pt", padding=True)

    encodings = {
        'input_ids': source_encoding['input_ids'],
        'attention_mask': source_encoding['attention_mask']
    }

    dataset = None

    if language == 'en':
        torch.save(encodings, "chs_mt5_translate_es.pt")
        dataset = CustomDataset(torch.load("chs_mt5_translate_es.pt"))
    else:
        torch.save(encodings, "chs_mt5_translate_en.pt")
        dataset = CustomDataset(torch.load("chs_mt5_translate_en.pt"))

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Generate translation using model
    translated = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        translated += out

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_texts


def combine_texts(original_texts, back_translated_batch):
    return set(original_texts + back_translated_batch)


def perform_back_translation_with_augmentation(batch_texts, device,first_model_tkn, first_model, second_model_tkn, second_model, original_language="es", temporary_language="en"):

    # Translate from Original to Temporary Language
    tmp_translated_batch = perform_translation(batch_texts, device, first_model, first_model_tkn, temporary_language)

    # Translate Back to English
    back_translated_batch = perform_translation(tmp_translated_batch, device, second_model, second_model_tkn,
                                                original_language)

    # Return The Final Result
    # return combine_texts(original_texts, back_translated_batch)
    return back_translated_batch


def prepare_input(texts):
    for i, text in enumerate(texts):
        texts[i] = "multilabel classification: {} </s>".format(text)


def tokenize(data,tokenizer):

    source_encoding = tokenizer(
        data['input_text'],
        max_length=512,
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
    os.chdir('../')
    wdir = os.getcwd()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')
    file_train = wdir + "/datasets/cuba_hotels_sentiment.xlsx"
    df = pd.read_excel(r'' + file_train, engine='openpyxl')
    print(df.iloc[0, 1])
    print(df.iloc[350, 2])
    print(df.shape)
    y = []
    X = []

    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    dictionary_data = {'X': [], 'y': []}

    for i in range(df.shape[0]):
        whole_text = str(df.iloc[i, 0]) + ": " + str(df.iloc[i, 1])
        dictionary_data['X'].append(whole_text)
        dictionary_data['y'].append(str(df.iloc[i, 2] - 1))

    X_train, X_dev, y_train, y_dev = train_test_split(dictionary_data['X'], dictionary_data['y'], test_size=0.30,
                                                      random_state=142)

    # Perform data augmentation
    dictionary_data['X'] = X_train
    dictionary_data['y'] = y_train

    df = pd.DataFrame(dictionary_data)
    rslt_df_4 = df.loc[df['y'] == '4']
    count_major_class = rslt_df_4.shape[0]

    augmentation = {'0': 1206,
                    '1': 1103,
                    '2': 1200,
                    '3': 1200}

    first_model_tkn, first_model = get_mt_model('es-en', device)
    second_model_tkn, second_model = get_mt_model('en-es', device)

    for i in ['0', '1', '2', '3']:
        rslt_df = df.loc[df['y'] == i]
        rslt_df_i = rslt_df.sample(n=augmentation[i])
        bt = perform_back_translation_with_augmentation(rslt_df_i['X'].tolist(), device,first_model_tkn, first_model, second_model_tkn, second_model)
        len_before = len(rslt_df['X'].tolist())
        augmented_list = list(set(rslt_df['X'].tolist() + bt))
        X += augmented_list
        len_increased = len(augmented_list)
        y += [i for _ in range(len_increased - len_before)]
        print("done in {} label, and {} examples increased".format(i, (len_increased - len_before)))

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.0, random_state=142)
    stt_train = {'input_text': X_train, 'target_text': y_train}
    stt_dev = {'input_text': X_dev, 'target_text': y_dev}

    print("Number of instances for training: ")
    print(len(stt_train['input_text']))
    print("Number of instances for dev set: ")
    print(len(stt_dev['input_text']))

    print(stt_train['input_text'][0])
    normalize(stt_train['input_text'])
    print(stt_train['input_text'][0])
    normalize(stt_dev['input_text'])

    lemmatization(stt_train['input_text'])
    lemmatization(stt_dev['input_text'])

    prepare_input(stt_train['input_text'])
    prepare_input(stt_dev['input_text'])

    stt_train = tokenize(stt_train,tokenizer)
    stt_dev = tokenize(stt_dev,tokenizer)

    torch.save(stt_train, wdir + "/datasets/dataset_train_chs_mt5_bt_lemm")

    torch.save(stt_dev, wdir + "/datasets/dataset_test_chs_mt5_bt_lemm")


if __name__ == "__main__":
    build_dataset_and_dict()

