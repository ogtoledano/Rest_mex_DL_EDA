import os
from utils.file_arguments_reader import load_param_from_file

from script.main_gradient_based_mt5 import train_model_t5_custom
from utils.logging_custom import make_logger
import torch
from utils.standard_output import make_txt_file_out_two_task
from utils.custom_dataloader import CustomDataset,CustomDatasetRestMexTwoTask, CustomDatasetRestMexTestTwoTask
import numpy as np
from transformers import T5Tokenizer
from algorithms_models.model_mt5_encoder_builder import CustomMT5Model
from algorithms_models.evolutionary_optimizer_mt5_custom import EDA_Optimizer

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Print numpy array without truncation
import sys
np.set_printoptions(threshold=sys.maxsize)

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

    test_dataset = CustomDatasetRestMexTestTwoTask(torch.load(wdir + "/datasets/dataset_test_stt_mt5"))
    val_dataset = CustomDatasetRestMexTwoTask(torch.load(wdir + "/datasets/dataset_dev_stt_mt5"))

    # Defining skorch-based neural network
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

    model_cma_es_polarity = EDA_Optimizer(
        module=CustomMT5Model,
        module__labels=5,
        batch_size=dic_param['sgd_batch_size'],
        train_split=None,
        criterion=torch.nn.CrossEntropyLoss,
        tokenizer=tokenizer,
        device=device,
        sigma=dic_param['sigma'],
        centroid=dic_param['centroid'],
        mode="EDA_CMA_ES"
    )

    model_cma_es_polarity.initialize()
    model_cma_es_polarity.load_state(path=wdir + '/checkpoints/params_Adam_stt_1_EDA_CMA_ES.pt', trainable=False)

    acc_polarity, conf_matrix_polarity = model_cma_es_polarity.score_unbalance(val_dataset)
    print('Acc polarity: ',acc_polarity)

    model_cma_es_attraction = EDA_Optimizer(
        module=CustomMT5Model,
        module__labels=3,
        batch_size=dic_param['sgd_batch_size'],
        train_split=None,
        criterion=torch.nn.CrossEntropyLoss,
        tokenizer=tokenizer,
        device=device,
        sigma=dic_param['sigma'],
        centroid=dic_param['centroid'],
        mode="EDA_CMA_ES"
    )

    model_cma_es_attraction.initialize()
    model_cma_es_attraction.load_state(path=wdir + '/checkpoints/params_Adam_stt_2_EDA_CMA_ES.pt', trainable=False)

    acc_attraction, conf_matrix_attraction = model_cma_es_attraction.score_unbalance(val_dataset, task="attraction")
    print('Acc polarity: ', acc_attraction)

    make_txt_file_out_two_task(test_dataset, model_cma_es_polarity.get_module(), model_cma_es_attraction.get_module(),
                               device, wdir + "experiments/")
