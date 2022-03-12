import os
from utils.file_arguments_reader import load_param_from_file

from script.main_gradient_based_mt5 import train_model_t5_aqg
from utils.logging_custom import make_logger
import torch
from utils.custom_dataloader import CustomDataset

if __name__ == "__main__":
    # Load train arguments from file
    os.chdir("../")
    wdir = os.getcwd() + "/" if not os.path.exists("/content/Rest_mex_DL_EDA/") else "/content/Rest_mex_DL_EDA/"  # only colab
    dic_param = load_param_from_file(wdir + "script/arguments.txt")
    log_exp_run = make_logger(name="" + dic_param['name_log_experiments_result'])
    device = "cuda:" + str(dic_param['cuda_device_id']) if torch.cuda.is_available() else "cpu"

    train_dataset = CustomDataset(torch.load(wdir + "/datasets/" + dic_param['dataset_train']))
    val_dataset = CustomDataset(torch.load(wdir + "/datasets/" + dic_param['dataset_test']))

    gscv_best_model = None
    gscv_best_model = train_model_t5_aqg(dic_param, log_exp_run, wdir, device, train_dataset, val_dataset,
                                                   gscv_best_model)

