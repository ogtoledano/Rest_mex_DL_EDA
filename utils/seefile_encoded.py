import os
from utils.file_arguments_reader import load_param_from_file
from utils.logging_custom import make_logger
import torch
from utils.custom_dataloader import CustomDataset
from torch.utils.data import DataLoader
from utils.imbalanced_dataset_sampling_mt5 import ImbalancedDatasetSamplerMT5

if __name__ == "__main__":
    os.chdir("../")
    wdir = os.getcwd() + "/" if not os.path.exists(
        "/content/Rest_mex_DL_EDA/") else "/content/Rest_mex_DL_EDA/"  # only colab
    dic_param = load_param_from_file(wdir + "script/arguments.txt")
    log_exp_run = make_logger(name="" + dic_param['name_log_experiments_result'])
    device = "cuda:" + str(dic_param['cuda_device_id']) if torch.cuda.is_available() else "cpu"

    train_dataset = CustomDataset(torch.load(wdir + "/datasets/" + dic_param['dataset_train']))
    val_dataset = CustomDataset(torch.load(wdir + "/datasets/" + dic_param['dataset_test']))

    iter_data = DataLoader(train_dataset, batch_size=3, sampler=ImbalancedDatasetSamplerMT5(train_dataset))

    with torch.no_grad():
        for batch in iter_data:
            input_ids = batch['source_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target_ids'].to(device)
            print(input_ids)

