import time

# ------ Scikit-learn ----------------------------------------------------------+
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import Checkpoint, LoadInitState,EarlyStopping

# ------ Tranformesrs ----------------------------------------------------------+
from transformers import T5Tokenizer
from transformers.models.mt5 import MT5ForConditionalGeneration
import torch
from torch.optim import AdamW

from algorithms_models.model_mt5_emoeval_builder import CustomMT5Model
from algorithms_models.trainer_mt5 import Trainer
from algorithms_models.trainer_mt5_custom import TrainerMT5Custom
from algorithms_models import model_mt5_emoeval_builder

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def confusion_matrix_chart(test_accs,train_accs, confusion_mtxes, labels, epoches, url_img, figsize=(20, 8)):
    cm = confusion_mtxes[np.argmax(test_accs)]
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    fig = plt.figure(figsize=figsize)
    x_axis= np.asarray([x for x in range(epoches)])
    plt.subplot(1, 2, 1)
    plt.plot(x_axis,test_accs, 'g')
    plt.xlabel("Epoches")
    plt.plot(x_axis,train_accs,'r')
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.savefig(url_img+'figure.png')
    plt.show()


def train_model_t5_aqg(dic_param, log_exp_run, wdir, device, train_data, test_data, gscv_best_model):
    # Defining a param distribution for hyperparameter-tuning for model and fit params
    param_grid = {
        'lr': dic_param['alpha_distribution'],
        'mode': ["train"]
    }

    fit_param = {
        'patientia': dic_param['sgd_early_stopping_patientia'],
        'min_diference': dic_param['sgd_min_difference'],
        'checkpoint_path': wdir + "checkpoints/", 'test_data': test_data, 'is_unbalanced': False, 'task': 'main'
    }

    checkpoint = Checkpoint(dirname=fit_param['checkpoint_path'], f_params=dic_param['f_params_name'],
                            f_optimizer=dic_param['f_optimizer_name'], f_history=dic_param['f_history_name'],
                            f_criterion=dic_param['f_criterion_name'],
                            monitor=None)

    load_state = LoadInitState(checkpoint)

    # Defining skorch-based neural network
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

    trainer = Trainer(
        module=model,
        max_epochs=dic_param['epochs'],
        tokenizer=tokenizer,
        iterator_train__shuffle=True,
        train_split=None,
        batch_size=dic_param['sgd_batch_size'],
        device=device,
        callbacks=[checkpoint],
        criterion=torch.nn.CrossEntropyLoss,
        optimizer= AdamW,
        lr=5e-5,
        mode="train"
    )

    trainer.fit(train_data, fit_param=fit_param)
    trainer.score_unbalance(test_data)
    trainer.score_unbalance(train_data, is_unbalanced=True)
    confusion_matrix_chart(trainer.test_accs, trainer.train_accs, trainer.confusion_mtxes,
                           range(dic_param['labels']), dic_param['epochs'], wdir + "experiments/")

    return trainer


def train_model_t5_custom(dic_param, log_exp_run, wdir, device, train_data, test_data, gscv_best_model):
    # Defining a param distribution for hyperparameter-tuning for model and fit params
    param_grid = {
        'lr': dic_param['alpha_distribution'],
        'mode': ["train"]
    }

    fit_param = {
        'patientia': dic_param['sgd_early_stopping_patientia'],
        'min_diference': dic_param['sgd_min_difference'],
        'checkpoint_path': wdir + "checkpoints/", 'test_data': test_data, 'is_unbalanced': False, 'task': 'main'
    }

    checkpoint = Checkpoint(dirname=fit_param['checkpoint_path'], f_params=dic_param['f_params_name'],
                            f_optimizer=dic_param['f_optimizer_name'], f_history=dic_param['f_history_name'],
                            f_criterion=dic_param['f_criterion_name'],
                            monitor=None)

    load_state = LoadInitState(checkpoint)

    # Defining skorch-based neural network
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

    trainer = TrainerMT5Custom(
        module=CustomMT5Model,
        module__labels=dic_param['labels'],
        max_epochs=dic_param['epochs'],
        tokenizer=tokenizer,
        iterator_train__shuffle=True,
        train_split=None,
        batch_size=dic_param['sgd_batch_size'],
        device=device,
        callbacks=[checkpoint],
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=AdamW,
        lr=6e-5,
        optimizer__weight_decay=dic_param['weight_decay'],
        mode="train"
    )

    trainer.fit(train_data, fit_param=fit_param)
    trainer.score_unbalance(test_data)
    trainer.score_unbalance(train_data, is_unbalanced=False)
    confusion_matrix_chart(trainer.test_accs, trainer.train_accs, trainer.confusion_mtxes,
                           range(dic_param['labels']), dic_param['epochs'], wdir + "experiments/")

    return trainer




