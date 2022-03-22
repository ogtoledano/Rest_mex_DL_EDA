<p align="center">
  <img src="https://github.com/ogtoledano/Transformer_Based_EDA/blob/main/logo.PNG" />
</p>

# A hybrid EDA-based algorithm for fine-tuning transformers in sentiment analysis

This project allows text categorization/sentiment analysis for Spanish tweets. For this purpose, a new optimization method based on Estimation of Distributions Algorithms (EDA) is introduced for the fine-tuning of Transformers. A MT5-base model is used

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pEFVMd4IWjLZ_Xg9Kc2deuC-FsgbYeZW?usp=sharing)[![visitors](https://visitor-badge.vercel.app/p/Rest_mex_DL_EDA?color=brightgreen)](https://github.com/ogtoledano/Rest_mex_DL_EDA)

*********** update at March 21, 2022 *************

## Requirements

In order to run this project you will need a working installation of:

+ deap
+ gensim
+ scikit-learn
+ pandas
+ datasets
+ transformers==4.0.1
+ numpy
+ nltk
+ matplotlib
+ seaborn 

For loading pre-trained models, Dataloaders, hyper-parameters tuning, Transformers fine-tuning and testing, you will need:
+ torch == 1.7.1
+ skorch == 0.11.0

## Pre-trained models

We use two pre-trained models taken off [Hugging Face ðŸ¤—](https://huggingface.co/):

+ MT5-base: `google/mt5-base`. Pre-trained model is available at [here](https://huggingface.co/google/mt5-base)
+ MT5-base-small: `google/mt5-small`. Pre-trained model is available at [here](https://huggingface.co/google/mt5-small)

You can specify the model as follow:
```shell
    # Defining tokenizer and model from pre-trained
    tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained('google/mt5-base')
```

## Datasets
Raw text of all dataset are available at `datasets` folder
### Spanish
+ Rest_Mex_2022_Sentiment_Analysis_Track. Classification task where the participating system can predict the polarity of an opinion issued by a tourist who traveled to the most representative places, restaurants and hotels in Mexico.  [here](https://sites.google.com/cicese.edu.mx/rest-mex-2022/tracks/sentiment-analysis-task)
