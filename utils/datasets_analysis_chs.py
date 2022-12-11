from sklearn.model_selection import train_test_split
import os
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def removing_stop_words_1(texts):
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        sentence = [word for word in tokens if word not in stop_words]
        texts[i] = ' '.join(sentence)


# Removing and counting stop-words, and word frequency
def removing_stop_words(texts):
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    count = 0
    whole_text = ''
    whole_text_sw = ''
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(text)
        sentence = []
        whole_text_sw += ' '+text
        for word in tokens:
            if word not in stop_words:
                sentence.append(word)
            else:
                count += 1

        texts[i] = ' '.join(sentence)
        whole_text += ' '+texts[i]

    fdist1 = nltk.FreqDist(nltk.word_tokenize(whole_text))

    return count, fdist1, whole_text, whole_text_sw


def autolabel(rects, ax):
    # Attach a text label above each bar in *rects*, displaying its height.
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_hist(x_train):
    print("min: {}, max: {}".format(min(x_train), max(x_train)))
    fig, ax = plt.subplots(figsize=(8, 4))

    # plot the cumulative histogram
    ax.plot(sorted(x_train,reverse=True))
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_xlabel("Palabras en el vocabulario")
    ax.set_ylabel('Procentage de ocurrencia')
    plt.title("Cuba Hotel Sentiment")
    plt.savefig('analysechs.eps', format='eps')
    plt.show()


def make_analysis(label_index,y_counter):
    fig, ax = plt.subplots()
    rect = ax.bar(label_index, y_counter.values(), width=0.5)
    ax.set_xlabel('Categorías')
    ax.set_ylabel('Ejemplos')
    autolabel(rect, ax)
    ax.set_xticks(label_index)
    ax.set_xticklabels(labels)
    plt.title("Cuba Hotel Sentiment")
    plt.savefig('analysebbcs1.eps', format='eps')
    plt.show()

    # Split dataset into train and test sets, 70% and 30% partitions
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12345)

    train = {'data': x_train, 'target': y_train}
    test = {'data': x_test, 'target': y_test}

    texts = train['data']
    removing_stop_words_1(texts)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index

    labels_target = test['target']

    print("Number of instances for training: {}".format(len(texts)))

    print("Number of instances for testing: {}".format(len(test['data'])))

    count, wf_1, whole_text_1, whole_text_sw_1 = removing_stop_words(texts)
    count2, wf_2, whole_text_2, whole_text_sw_2 = removing_stop_words(test['data'])

    wf_texts_train = nltk.FreqDist(nltk.word_tokenize(whole_text_sw_1))
    # wf_texts_test = nltk.FreqDist(nltk.word_tokenize(whole_text_sw_1))
    plot_hist(tokenizer.word_counts.values())

    print("Amount of English stop-words for train set: {}".format(count))
    print("Amount of English stop-words for test set: {}".format(count2))
    print("Amount of English stop-words for test set: {}".format(count2 + count))

    print("Word frequency for training set:\n{}".format(wf_1))

    print("Cloud for trainig set")
    word_cloud = WordCloud(collocations=False, background_color='white').generate(whole_text_1)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    print("Word frequency for test set:\n{}".format(wf_2))

    print("Cloud for test set")
    word_cloud = WordCloud(collocations=False, background_color='white').generate(whole_text_2)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    os.chdir('../')
    wdir = os.getcwd()
    file_train = wdir + "/datasets/cuba_hotels_sentiment.xlsx"
    df = pd.read_excel(r'' + file_train, engine='openpyxl')
    print(df.iloc[0, 1])
    print(df.iloc[350, 2])
    print(df.shape)

    y = []
    X = []
    labels = ['1', '2', '3', '4', '5']
    label_index = [x for x in range(len(labels))]
    for i in range(df.shape[0]):
        whole_text = str(df.iloc[i, 0]) + ": " + str(df.iloc[i, 1])
        X.append(whole_text)
        y.append(str(df.iloc[i, 2] - 1))

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.30, random_state=142)
    stt_train = {'input_text': X_train, 'target_text': y_train}
    stt_dev = {'input_text': X_dev, 'target_text': y_dev}

    y_counter={}
    for value in y:
        if value in y_counter.keys():
            y_counter[value]=y_counter[value]+1
        else:
            y_counter[value]=1

    make_analysis(label_index,y_counter)