import sys
sys.path.append('..\\..\\Text_Cat_Based_EDA')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\utils')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\pretrained_models')
sys.path.append('..\\..\\Text_Cat_Based_EDA\\evolutionary_algorithms')

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.logging_custom import make_logger

# Scikit-learn ----------------------------------------------------------+
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score, accuracy_score
from skorch import NeuralNet
from transformers import AdamW
from torch.utils.data import DataLoader
from utils.imbalanced_dataset_sampling_mt5 import ImbalancedDatasetSamplerMT5
import sklearn.metrics as sm


class TrainerMT5Custom(NeuralNet):

    def __init__(self,*args,mode="AdamW",tokenizer,batch_size,**kargs):
        super().__init__(*args, **kargs)
        self.mode=mode
        self.tokenizer=tokenizer
        self.batch_size=batch_size
        log_exp_run = make_logger(name="experiment_" + self.mode)
        log_exp_run.experiments("Running on device: "+str(self.device))
        log_exp_run.experiments("Training model by Back-propagation with optimizer: "+mode)

    def initialize_criterion(self,*args,**kargs):
        super().initialize_criterion(*args,**kargs)
        return self

    def initialize_module(self,*args,**kargs):
        super().initialize_module(*args, **kargs)
        param_length = sum([p.numel() for p in self.module_.parameters() if p.requires_grad])
        log_exp_run = make_logger(name="experiment_" + self.mode)
        log_exp_run.experiments("Amount of parameters: " + str(param_length))
        """
        for name,p in self.module_.named_parameters():
            log_exp_run.experiments("Params per layer {}, {}".format(name,p.numel()))
        """
        return self

    # Sci-kit methods
    def predict(self, X):
        input_ids = X['source_ids'].to(self.device)
        attention_mask = X['attention_mask'].to(self.device)
        labels = X['target_ids'].to(self.device)
        labels_ids = X['labels'].to(self.device)
        labels[labels == -100] = self.module_.config.pad_token_id
        self.module_.to(self.device)
        output = self.module_(input_ids=input_ids, labels=labels, attention_mask=attention_mask, labels_ids=labels_ids)
        logits = output.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions

    # Skorch methods: Compute the criterion metrics
    def score(self, X, y=None):
        train_loss = 0
        iter_data = DataLoader(X, batch_size=self.batch_size, shuffle=True)
        log_exp_run = make_logger(name="experiment_" + self.mode)

        self.module_.to(self.device)
        self.module_.eval()

        with torch.no_grad():
            for batch in iter_data:
                input_ids = batch['source_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['target_ids'].to(self.device)
                labels_ids = batch['labels'].to(self.device)
                labels[labels == -100] = self.module_.config.pad_token_id
                output = self.module_(input_ids=input_ids, labels=labels, attention_mask=attention_mask, labels_ids=labels_ids)
                loss = output.loss
                train_loss += loss.item()

        log_exp_run.experiments("Cross-entropy loss for each fold: {}".format(train_loss))
        return train_loss

    def compute_metrics(self, labels, preds):

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def score_unbalance(self, X, y=None, is_unbalanced=False):
        train_loss = 0
        iter_data = DataLoader(X, batch_size=self.batch_size, sampler=ImbalancedDatasetSamplerMT5(X)) if is_unbalanced else DataLoader(X, batch_size=self.batch_size, shuffle=True)
        log_exp_run = make_logger(name="experiment_" + self.mode)

        self.module_.to(self.device)
        self.module_.eval()

        predictions = []
        labels_ref = []

        with torch.no_grad():
            for batch in iter_data:
                input_ids = batch['source_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['target_ids'].to(self.device)
                labels_ids = batch['labels'].to(self.device)
                labels[labels == -100] = self.module_.config.pad_token_id
                output = self.module_(input_ids=input_ids, labels=labels, attention_mask=attention_mask, labels_ids=labels_ids)
                loss = output.loss
                train_loss += loss.item()

                logits = output.logits
                preds_batch = torch.argmax(logits, dim=-1)
                predictions.extend(preds_batch.cpu().numpy())
                labels_ref.extend(labels_ids.cpu().numpy())

        accuracy = accuracy_score(labels_ref, predictions)
        # mae = mean_absolute_error(labels_ref, predictions)
        macro_f1 = f1_score(labels_ref, predictions, average='macro')

        log_exp_run.experiments("Cross-entropy loss for each fold: {}".format(train_loss))
        log_exp_run.experiments("Accuracy for each fold: " + str(accuracy))
        log_exp_run.experiments("\n" + classification_report(labels_ref, predictions))
        # log_exp_run.experiments("\nMean Absolute Error (MAE): " + str(mae))
        log_exp_run.experiments("\nMacro F1: " + str(macro_f1))
        confusion_mtx = sm.confusion_matrix(labels_ref, predictions)
        metrics = self.compute_metrics(labels_ref,predictions)
        log_exp_run.experiments("All metrics (weighted) \nF1= {}, precision= {}, recall= {}".format(metrics['f1'], metrics['precision'], metrics['recall']))
        return accuracy, confusion_mtx

    # Skorch methods: this method fits the estimator by back-propagation and an optimizer
    def fit(self, X, y=None, **fit_params):
        log_exp_run = make_logger(name="experiment_" + self.mode)

        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.X_ = X

        self.train_loss_acc = []

        self.test_accs = []
        self.train_accs = []
        self.confusion_mtxes = []

        self.module_.to(self.device)
        optimizer = self.optimizer_
        criterion = self.criterion_

        is_unbalanced = fit_params["is_unbalanced"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["is_unbalanced"]
        task = fit_params["task"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["task"]

        iter_data = DataLoader(X, batch_size=self.batch_size, sampler=ImbalancedDatasetSamplerMT5(X)) if is_unbalanced else DataLoader(X, batch_size=self.batch_size, shuffle=True)

        self.notify('on_train_begin', X=X, y=y)

        isinstance(optimizer,AdamW)
        log_exp_run.experiments("Run using {} as optimizer".format("AdamW" if isinstance(optimizer,AdamW) else "SGD"))

        log_exp_run.experiments("lr: {}".format(self.lr))

        on_epoch_kwargs = {
            'dataset_train': X,
            'dataset_valid': None,
        }

        for epoch in range(self.max_epochs):
            train_loss = 0
            predictions = []
            labels_ref = []
            self.notify('on_epoch_begin', **on_epoch_kwargs)
            for batch in iter_data:
                optimizer.zero_grad()
                input_ids = batch['source_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['target_ids'].to(self.device) if task == 'main' else batch['target_ids_attraction'].to(self.device)
                labels_ids = batch['labels'].to(self.device) if task == 'main' else batch['labels_attraction'].to(self.device)

                labels[labels == -100] = self.module_.config.pad_token_id
                self.notify("on_batch_begin", X=input_ids, y=labels, training=True)
                outputs = self.module_(input_ids=input_ids, labels=labels, attention_mask=attention_mask, labels_ids=labels_ids)

                loss = outputs.loss

                train_loss += loss.item()

                logits = outputs.logits
                preds_batch = torch.argmax(logits, dim=-1)
                #labels_batch = [int(self.tokenizer.decode(ids, skip_special_tokens=True)) for ids in labels]
                predictions.extend(preds_batch.cpu().numpy())
                labels_ref.extend(labels_ids.cpu().numpy())
                loss.backward()
                optimizer.step()
                self.notify("on_batch_end", X=input_ids, y=labels, training=True)

            self.notify('on_epoch_end', **on_epoch_kwargs)
            log_exp_run.experiments("Epoch ran: " + str(epoch) + " loss: " + str(train_loss))
            self.train_loss_acc.append(train_loss)

            # Test acc and confusion matrix  charts
            test_acc, confusion_mtx = self.score_unbalance(X=fit_params["test_data"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["test_data"], is_unbalanced=False)
            self.test_accs.append(test_acc)
            self.confusion_mtxes.append(confusion_mtx)
            self.train_accs.append(accuracy_score(labels_ref, predictions))

        log_exp_run.experiments("Train loss series:")
        log_exp_run.experiments(self.train_loss_acc)
        self.notify('on_train_end', X=X, y=y)
        return self
