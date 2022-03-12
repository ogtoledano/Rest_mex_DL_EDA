import time

# ------ Scikit-learn ----------------------------------------------------------+
from sklearn.model_selection import GridSearchCV
from skorch.callbacks import Checkpoint, LoadInitState,EarlyStopping

# ------ Tranformesrs ----------------------------------------------------------+
from transformers import T5Tokenizer, MT5Model
import torch
from transformers import AdamW
from algorithms_models.trainer_mt5 import Trainer


def train_model_t5_aqg(dic_param, log_exp_run, wdir, device, train_data, test_data, gscv_best_model):
    # Defining a param distribution for hyperparameter-tuning for model and fit params
    param_grid = {
        'lr': dic_param['alpha_distribution'],
        'mode': ["train"]
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
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    model = MT5Model.from_pretrained("google/mt5-small")

    trainer = Trainer(
        module=model,
        max_epochs=dic_param['epochs'],
        tokenizer=tokenizer,
        iterator_train__shuffle=True,
        train_split=None,
        batch_size=dic_param['batch_size'],
        device=device,
        callbacks=[checkpoint],
        criterion=torch.nn.CrossEntropyLoss,
        optimizer= AdamW,
        lr=5e-5,
        mode="train"
    )

    trainer.fit(train_data, fit_param=fit_param)

    return gscv_best_model




