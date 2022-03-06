import sys
sys.path.append('..\\..\\Text_Cat_Based_EDA')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\utils')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\script')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\pretrained_models')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\evolutionary_algorithms')

from algorithms_models.model_cnn_builder import ModelCNN
from utils.embedding_builder import build_glove_from_pretrained, build_spanish_glove_from_pretrained

import torch
from utils.custom_dataloader import CustomDataLoader
from utils.logging_custom import make_logger
from algorithms_models.evolutionary_optimizer import EDA_Optimizer
from utils.file_arguments_reader import load_param_from_file
import os
import time
from script.main_gradient_based import train_model_adam, train_model_sgd

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.standard_output import make_txt_file_out


def confusion_matrix_chart(test_accs,train_accs, confusion_mtxes, labels, generations, url_img, figsize=(20, 8)):
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
    x_axis= np.asarray([x for x in range(generations)])
    plt.subplot(1, 2, 1)
    plt.plot(x_axis,test_accs, 'g')
    plt.xlabel("Generations")
    plt.plot(x_axis,train_accs,'r')
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.savefig(url_img+'figure.png')
    plt.show()


if __name__ == "__main__":
    # Load train arguments from file
    os.chdir("../")
    wdir = os.getcwd() + "/" if not os.path.exists("/content/Rest_mex_DL_EDA/") else "/content/Rest_mex_DL_EDA/"  # only colab
    dic_param = load_param_from_file(wdir + "script/arguments.txt")
    log_exp_run = make_logger(name="" + dic_param['name_log_experiments_result'])
    device = "cuda:" + str(dic_param['cuda_device_id']) if torch.cuda.is_available() else "cpu"

    # Load pre-trained word embedding model with specific language: Spanish or English
    tensor_embedding = build_spanish_glove_from_pretrained(wdir + 'utils/pretrained_models',
                                                            wdir + 'datasets/' + dic_param['dataset_dictionary']) if \
                                                            dic_param['word_embedding_pretrained_glove_language'] == 'Spanish' \
                                                            else build_glove_from_pretrained(wdir + 'utils/pretrained_models',
                                                            wdir + 'datasets/' + dic_param['dataset_dictionary'])

    train_data = CustomDataLoader(wdir + 'datasets/' + dic_param['dataset_train'])
    test_data = CustomDataLoader(wdir + 'datasets/' + dic_param['dataset_test'])

    gscv_best_model = None

    for i in range(dic_param['num_executions']): # Total measurement
        # Do STAGE 1: Learning convolutional layers by gradient-based method
        print("Training and test results of {}: ".format(dic_param['cnn_optimizer']))
        start_time = time.time()
        if dic_param['cnn_optimizer'] == 'Adam':
            gscv_best_model = train_model_adam(dic_param, log_exp_run, wdir, device, tensor_embedding, train_data, test_data,gscv_best_model)
        if dic_param['cnn_optimizer'] == 'SGD':
            gscv_best_model = train_model_sgd(dic_param, log_exp_run, wdir, device, tensor_embedding, train_data, test_data,gscv_best_model)

        # Defining skorch-based neural network
        model = EDA_Optimizer(
            module=ModelCNN,
            module__word_embedding_size=dic_param['word_embedding_size'],
            module__labels=dic_param['labels'],
            module__weights_tensor=tensor_embedding,
            module__batch_size=dic_param['sgd_batch_size'],
            train_split=None,
            criterion=torch.nn.CrossEntropyLoss,
            device=device,
            sigma=dic_param['sigma'],
            centroid=dic_param['centroid'],
            mode="EDA_CMA_ES"
        )

        param_distribution = {'sigma': dic_param['sigma_distribution'], 'mode': ["EDA_CMA_ES"]}

        param_model = {"generations": dic_param['generations'], 'mode': "EDA_CMA_ES",
                       "population_size": dic_param['individuals'],
                       "checkpoint": wdir + "checkpoints/" + dic_param['cnn_checkpoint'],'test_data': test_data}

        # Do STAGE 2: Learning full-connected layer by EDA optimization
        model.fit(train_data, fit_param=param_model)
        log_exp_run.experiments("Time elapsed: " + str(time.time() - start_time))
        model.score(test_data)
        model.score_unbalanced(train_data)
        print("Training and test results of {} and {}: ".format(dic_param['cnn_optimizer'],param_distribution['mode']))
        confusion_matrix_chart(model.test_accs, model.train_accs, model.confusion_mtxes,
                               range(dic_param['labels']), dic_param['generations'], wdir + "experiments/")
        make_txt_file_out("sentiment", test_data, model.get_module(), device, wdir + "experiments/")
        log_exp_run.experiments(dic_param['cnn_optimizer'] + " + EDA_CMA_ES: Process ends successfully!")
        log_exp_run.experiments("--------------------------\n\n\n")
