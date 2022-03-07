import sys

#from docutils.nodes import label

from algorithms_models.model_cnn_builder import ModelCNN

import torch
from algorithms_models.trainer import Trainer
import time

# ------ Scikit-learn ----------------------------------------------------------+
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import Checkpoint, LoadInitState,EarlyStopping

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.standard_output import make_txt_file_out


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


def train_model_sgd(dic_param, log_exp_run, wdir, device, tensor_embedding, train_data, test_data, gscv_best_model):
    # Defining a param distribution for hyperparameter-tuning for model and fit params
    param_grid = {
        'lr': dic_param['alpha_distribution'],
        'optimizer__momentum': dic_param['momentum_distribution'],
        'mode': ["SGD"]  # Modes: Adam,SGD
    }

    fit_param = {
        'patientia': dic_param['sgd_early_stopping_patientia'],
        'min_diference': dic_param['sgd_min_difference'],
        'checkpoint_path': wdir + "checkpoints/"
    }

    checkpoint = Checkpoint(dirname=fit_param['checkpoint_path'], f_params=dic_param['f_params_name'],
                            f_optimizer=dic_param['f_optimizer_name'], f_history=dic_param['f_history_name'],
                            f_criterion=dic_param['f_criterion_name'],
                            monitor=None)

    load_state = LoadInitState(checkpoint)

    # Defining skorch-based neural network
    model = Trainer(
        module=ModelCNN,
        module__word_embedding_size=dic_param['word_embedding_size'],
        module__labels=dic_param['labels'],
        module__weights_tensor=tensor_embedding,
        module__batch_size=dic_param['sgd_batch_size'],
        max_epochs=dic_param['epochs_gs_cv'],
        iterator_train__shuffle=True,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=None,
        device=device,
        callbacks=[checkpoint],
        optimizer=torch.optim.SGD,
        mode="SGD"
        # optimizer__weight_decay=dic_param['l2_reg'] #L2 regularization
    )

    # Defining GridSearch using k-fold cross validation
    log_exp_run.experiments("GridSearch using k-fold cross validation with for SGD")
    start_time = time.time()
    gs = GridSearchCV(model, param_grid, cv=dic_param['grid_search_cross_val_cv'], verbose=2)

    if gscv_best_model is None:
        gs.fit(train_data, fit_param=fit_param)
        log_exp_run.experiments(
            "Time elapsed for GridSearch using k-fold cross validation with k=5 for SGD: " + str(
                time.time() - start_time))

        log_exp_run.experiments("Best param estimated for SGD: ")
        log_exp_run.experiments(gs.best_params_)
        log_exp_run.experiments("Best score for SGD: ")
        log_exp_run.experiments(gs.best_score_)
        log_exp_run.experiments("GridSearch scores")
        log_exp_run.experiments(gs.cv_results_)
        gscv_best_model = gs.best_estimator_

    best_model = gscv_best_model

    best_model.set_params(max_epochs=dic_param['epochs'])
    start_time = time.time()
    best_model.fit(train_data, fit_param=fit_param)
    log_exp_run.experiments("Time elapsed for SGD : " + str(time.time() - start_time))
    best_model.score(test_data)
    best_model.score(train_data)
    log_exp_run.experiments("SGD as optimizer: Process ends successfully!")
    log_exp_run.experiments("--------------------------\n\n\n")
    return gscv_best_model


def train_model_adam(dic_param, log_exp_run, wdir, device, tensor_embedding, train_data, test_data, dev_data, gscv_best_model):
    # Defining a param distribution for hyperparameter-tuning for model and fit params
    param_grid = {
        'lr': dic_param['alpha_distribution'],
        'mode': ["Adam"]  # Modes: Adam,SGD
    }

    fit_param = {
        'patientia': dic_param['sgd_early_stopping_patientia'],
        'min_diference': dic_param['sgd_min_difference'],
        'checkpoint_path': wdir + "checkpoints/", 'test_data': test_data, 'is_unbalanced': False
    }

    checkpoint = Checkpoint(dirname=fit_param['checkpoint_path'], f_params=dic_param['f_params_name'],
                            f_optimizer=dic_param['f_optimizer_name'], f_history=dic_param['f_history_name'],
                            f_criterion=dic_param['f_criterion_name'],
                            monitor=None)

    load_state = LoadInitState(checkpoint)

    # Defining skorch-based neural network
    model = Trainer(
        module=ModelCNN,
        module__word_embedding_size=dic_param['word_embedding_size'],
        module__labels=dic_param['labels'],
        module__weights_tensor=tensor_embedding,
        module__batch_size=dic_param['sgd_batch_size'],
        max_epochs=dic_param['epochs_gs_cv'],
        iterator_train__shuffle=True,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=None,
        device=device,
        callbacks=[checkpoint],
        optimizer=torch.optim.Adam,
        mode="Adam"
        # optimizer__weight_decay=dic_param['l2_reg'] #L2 regularization
    )

    # model.initialize()
    # print(summary(model.module_,torch.zeros((1,1000),dtype=torch.long), show_input=True))

    # Defining GridSearch using k-fold cross validation
    log_exp_run.experiments("GridSearch using k-fold cross validation with for Adam")
    start_time = time.time()
    gs = GridSearchCV(model, param_grid, cv=dic_param['grid_search_cross_val_cv'], verbose=2)

    if gscv_best_model is None:
        gs.fit(train_data, fit_param=fit_param)

        log_exp_run.experiments(
            "Time elapsed for GridSearch using k-fold cross validation with k=5 for Adam: " + str(
                time.time() - start_time))

        log_exp_run.experiments("Best param estimated for Adam: ")
        log_exp_run.experiments(gs.best_params_)
        log_exp_run.experiments("Best score for Adam: ")
        log_exp_run.experiments(gs.best_score_)
        log_exp_run.experiments("GridSearch scores")
        log_exp_run.experiments(gs.cv_results_)
        gscv_best_model = gs.best_estimator_

    best_model = gscv_best_model

    best_model.set_params(max_epochs=dic_param['epochs'])
    start_time = time.time()
    fit_param['is_unbalanced'] = True
    best_model.fit(train_data, fit_param=fit_param)
    log_exp_run.experiments("Time elapsed for Adam : " + str(time.time() - start_time))
    best_model.score(test_data)
    best_model.score(dev_data)
    best_model.score_unbalanced(train_data, print_logs=True)
    confusion_matrix_chart(best_model.test_accs, best_model.train_accs, best_model.confusion_mtxes, range(dic_param['labels']), dic_param['epochs'], wdir + "experiments/")
    make_txt_file_out("sentiment",test_data,best_model.get_module(),device,wdir + "experiments/")
    log_exp_run.experiments("Adam as optimizer: Process ends successfully!")
    log_exp_run.experiments("--------------------------\n\n\n")
    return gscv_best_model

