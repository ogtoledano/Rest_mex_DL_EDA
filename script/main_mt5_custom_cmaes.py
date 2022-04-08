import os
from utils.file_arguments_reader import load_param_from_file

from script.main_gradient_based_mt5 import train_model_t5_custom
from utils.logging_custom import make_logger
import torch
from utils.custom_dataloader import CustomDataset,CustomDatasetRestMexTwoTask
import numpy as np
from transformers import T5Tokenizer
from algorithms_models.model_mt5_encoder_builder import CustomMT5Model
from algorithms_models.evolutionary_optimizer_mt5_custom import EDA_Optimizer

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def confusion_matrix_chart_eda(test_accs, confusion_mtxes, labels, url_img, figsize=(20, 8)):
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
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.savefig(url_img+'figure.png')
    plt.show()


if __name__ == "__main__":
    # For deterministic results
    random_seed = 64
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load train arguments from file
    os.chdir("../")
    wdir = os.getcwd() + "/" if not os.path.exists("/content/Rest_mex_DL_EDA/") else "/content/Rest_mex_DL_EDA/"  # only colab
    dic_param = load_param_from_file(wdir + "script/arguments.txt")
    log_exp_run = make_logger(name="" + dic_param['name_log_experiments_result'])
    device = "cuda:" + str(dic_param['cuda_device_id']) if torch.cuda.is_available() else "cpu"

    train_dataset = CustomDatasetRestMexTwoTask(torch.load(wdir + "/datasets/" + dic_param['dataset_train']))
    val_dataset = CustomDatasetRestMexTwoTask(torch.load(wdir + "/datasets/" + dic_param['dataset_dev']))

    gscv_best_model = None

    # Defining skorch-based neural network
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

    for i in range(dic_param['num_executions']):  # Total measurement
        # Do STAGE 1: Learning convolutional layers by gradient-based method
        #gscv_best_model = train_model_t5_custom(dic_param, log_exp_run, wdir, device, train_dataset, val_dataset, gscv_best_model)

        # Defining skorch-based neural network
        trainer = EDA_Optimizer(
            module=CustomMT5Model,
            module__labels=dic_param['labels'],
            batch_size=dic_param['sgd_batch_size'],
            train_split=None,
            criterion=torch.nn.CrossEntropyLoss,
            tokenizer=tokenizer,
            device=device,
            sigma=dic_param['sigma'],
            centroid=dic_param['centroid'],
            mode="EDA_CMA_ES"
        )

        param_distribution = {'sigma': dic_param['sigma_distribution'], 'mode': ["EDA_CMA_ES"]}

        param_model = {"generations": dic_param['generations'], 'mode': "EDA_CMA_ES",
                       "population_size": dic_param['individuals'],
                       "checkpoint": wdir + "checkpoints/" + dic_param['cnn_checkpoint'], 'test_data': val_dataset, 'is_unbalanced': False, 'task': 'main'}

        # Do STAGE 2: Learning full-connected layer by EDA optimization
        trainer.fit(train_dataset, fit_param=param_model)
        trainer.score_unbalance(val_dataset)
        trainer.score_unbalance(train_dataset, is_unbalanced=False)
        confusion_matrix_chart_eda(trainer.test_accs, trainer.confusion_mtxes, range(dic_param['labels']), wdir + "experiments/")
        log_exp_run.experiments(dic_param['cnn_optimizer'] + " + EDA_CMA_ES: Process ends successfully!")
        log_exp_run.experiments("--------------------------\n\n\n")