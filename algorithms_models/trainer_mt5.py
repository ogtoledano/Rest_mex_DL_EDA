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
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, f1_score
from skorch import NeuralNet
from transformers import AdamW
from torch.utils.data import DataLoader


class Trainer(NeuralNet):

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
        labels[labels == -100] = self.module_.config.pad_token_id
        self.module_.to(self.device)
        output = self.module_(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        return output[1]

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
                labels[labels == -100] = self.module_.config.pad_token_id
                output = self.module_(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = output[0]
                train_loss += loss.item()

        log_exp_run.experiments("Cross-entropy loss for each fold: {}".format(train_loss))
        return train_loss

    def score_unbalance(self, X, y=None):
        train_loss = 0
        iter_data = DataLoader(X, batch_size=self.batch_size, shuffle=True)
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
                labels[labels == -100] = self.module_.config.pad_token_id
                output = self.module_(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = output[0]
                train_loss += loss.item()

                outs = self.module_.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=2,
                    num_beams=4,
                )

                preds_batch = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
                labels_batch = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
                predictions.extend(preds_batch)
                labels_ref.extend(labels_batch)

        log_exp_run.experiments("Predictions \n{}".format(predictions))
        log_exp_run.experiments("Labels \n{}".format(labels_ref))

        accuracy = accuracy_score(labels_ref, predictions)
        mae = mean_absolute_error(labels_ref, predictions)
        macro_f1 = f1_score(labels_ref, predictions, average='macro')

        log_exp_run.experiments("Cross-entropy loss for each fold: {}".format(train_loss))
        log_exp_run.experiments("Accuracy for each fold: " + str(accuracy))
        log_exp_run.experiments("\n" + classification_report(labels_ref, predictions))
        log_exp_run.experiments("\nMean Absolute Error (MAE): " + str(mae))
        log_exp_run.experiments("\nMacro F1: " + str(macro_f1))
        return train_loss

    # Skorch methods: this method fits the estimator by back-propagation and an optimizer
    def fit(self, X, y=None, **fit_params):
        log_exp_run = make_logger(name="experiment_" + self.mode)

        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.X_ = X

        train_loss_acc=[]
        self.module_.to(self.device)
        optimizer = self.optimizer_
        criterion = self.criterion_
        iter_data = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        patientia = fit_params["patientia"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["patientia"]
        cont_early_stoping = fit_params["patientia"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["patientia"]
        min_diference = fit_params["min_diference"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["min_diference"]
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
            self.notify('on_epoch_begin', **on_epoch_kwargs)
            for batch in iter_data:
                optimizer.zero_grad()
                input_ids = batch['source_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['target_ids'].to(self.device)
                """
                if self.device == 'cuda:0':
                    print(torch.cuda.memory_summary(device=None, abbreviated=False))
                """
                labels[labels == -100] = self.module_.config.pad_token_id
                self.notify("on_batch_begin", X=input_ids, y=labels, training=True)
                outputs = self.module_(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                """
                lprobs = torch.nn.functional.log_softmax(outputs[1], dim=-1)
                loss, nll_loss = label_smoothed_nll_loss(
                    lprobs, labels, 1, ignore_index=model.config.pad_token_id
                )
                """
                loss = outputs[0]

                train_loss += loss.item()
                #print("Bach loss: {}".format(loss.item()))
                loss.backward()
                optimizer.step()
                self.notify("on_batch_end", X=input_ids, y=labels, training=True)

            self.notify('on_epoch_end', **on_epoch_kwargs)
            log_exp_run.experiments("Epoch ran: " + str(epoch) + " loss: " + str(train_loss))
            train_loss_acc.append(train_loss)

        log_exp_run.experiments("Train loss series:")
        log_exp_run.experiments(train_loss_acc)
        self.notify('on_train_end', X=X, y=y)
        return self

    def tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True, max_length=512):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )

        return inputs

    def prepare_input(self, inputs):
        code = "generate question: {} <h1> {} <h1> </s>".format(inputs['code'], inputs['answer'])
        answer = inputs['answer']
        qg_examples = {"answer": answer, "source_text": code}
        return [qg_examples['source_text']]

    def generate_questions(self, inputs):
        inputs = self.tokenize(self.prepare_input(inputs), padding=True, truncation=True)

        self.module_.to(self.device)
        self.module_.eval()

        outs = self.module_.generate(
            input_ids=inputs.data['input_ids'].to(self.device),
            attention_mask=inputs.data['attention_mask'].to(self.device),
            max_length=32,
            num_beams=4,
        )

        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions
