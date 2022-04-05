#------------------------------------------------------------------------------+
# This impementation is based on library gensim for word embeddings
#
# @author: Doctorini
# Implementation of Evolutionary Optimizer
# Using EMNA (Estimation of Multivariate Normal Algorithm), CMA-ES, cUMDA
# implementation as EDA (Estimation of distribution algortihms)
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES for EDA and Torch modules----------------------+

import sys
import torch
import re

from utils.logging_custom import make_logger
from algorithms_models.eda.EDA import EMNA
from algorithms_models.eda.CUMDA import CUMDA
from deap import base
from deap import creator
from deap import tools
from deap import cma
import time
import numpy as np
from numpy import random

import torch.nn as nn
from torch.utils.data import DataLoader

# Scikit-learn ----------------------------------------------------------+
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score, accuracy_score, mean_absolute_error
from skorch import NeuralNet
import sklearn.metrics as sm
from utils.imbalanced_dataset_sampling_mt5 import ImbalancedDatasetSamplerMT5
import os
#--- CONSTANTS ----------------------------------------------------------------+


# Defining the optimization problem as argmin(lossFuntion) and the individual
# with continuous codification as update rules for weight based on evolutionary algorithms

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


class EDA_Optimizer(NeuralNet):

    def __init__(self,*args,mode="EDA_EMNA",centroid=0.5,sigma=0.8,individuals=10,generations=5,param_length=0,batch_size=28,tokenizer,**kargs):
        super().__init__(*args, **kargs)
        #self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        log_exp_run = make_logger(name="experiment_"+self.mode)
        log_exp_run.experiments("Running on device: " + str(self.device))
        self.centroid = centroid
        self.sigma = sigma
        self.generations=generations
        self.param_length=param_length
        self.individuals=individuals
        self.batch_size=batch_size
        self.tokenizer = tokenizer

    # Train mode is: "SGD","SGD_MINI_BATCH","EDA_EMNA","EDA_CMA_ES"
    def set_train_mode(self, mode):
        self.mode = mode

    def initialize_criterion(self,*args,**kargs):
        super().initialize_criterion(*args,**kargs)
        return self

    def initialize_module(self,*args,**kargs):
        super().initialize_module(*args, **kargs)
        #self.param_length = sum([p.numel() for p in self.module_.parameters() if p.requires_grad])
        fc_pathern = re.compile("fc\w*") # matching only with full final_layer_norm layers (fc)
        self.param_length = sum([p.numel() for name, p in self.module_.named_parameters() if p.requires_grad and fc_pathern.match(name)])
        log_exp_run = make_logger(name="experiment_" + self.mode)
        log_exp_run.experiments("Amount of parameters: " + str(self.param_length))
        return self

    def get_module(self):
        return self.module_

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

    # Sckorch methods: this method fits the estimator using a determinate way defined by
    # mode attr. The main modes for training are: EDA_EMNA, EDA_CMA_ES, EDA_CUMDA for all
    # representation model torch ann
    def fit(self, X, y=None, **fit_params):

        if not self.warm_start or not self.initialized_:
            self.initialize()
        self.X_ = X

        log_exp_run = make_logger(name="experiment_" + self.mode)

        self.load_params(f_params=fit_params["fit_param"]["checkpoint"]) # Continue fitting by metaheuristic from previous model
        log_exp_run.experiments("Loaded fit params from previous model")

        self.mode = fit_params["fit_param"]["mode"]
        path = fit_params["fit_param"]["checkpoint"].split('.')[0] +"_"+self.mode+".pt"
        self.individuals=fit_params["fit_param"]["population_size"]

        is_unbalanced = fit_params["is_unbalanced"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["is_unbalanced"]
        task = fit_params["task"] if fit_params.get('fit_param') is None else fit_params["fit_param"]["task"]

        if self.mode == "EDA_EMNA":
            log_exp_run.experiments("Training with EDA_EMNA...")
            start_time = time.time()
            self.train_eda_enma_early_stopping(self.sigma, self.centroid, fit_params["fit_param"]["generations"], X)
            log_exp_run.experiments("Time elapsed for EDA_EMNA: " + str(time.time() - start_time))

        if self.mode == "EDA_CMA_ES":
            log_exp_run.experiments("Training with EDA_CMA_ES...")
            start_time = time.time()
            self.train_eda_cma_es_early_stopping(self.sigma, self.centroid, fit_params["fit_param"]["generations"], X, is_unbalanced, task)
            log_exp_run.experiments("Time elapsed for EDA_CMA_ES: " + str(time.time() - start_time))

        if self.mode == "EDA_CUMDA":
            log_exp_run.experiments("Training with EDA_CUMDA...")
            start_time = time.time()
            self.train_eda_cumda_early_stopping(self.sigma, fit_params["fit_param"]["generations"], X)
            log_exp_run.experiments("Time elapsed for EDA_CUMDA: " + str(time.time() - start_time))

        #self.save_state(path)

        return self

    def save_state(self, path):
        log_exp_run = make_logger(name="experiment_" + self.mode)

        torch.save({
            'generation': self.generations,
            'model_state_dict': self.module_.state_dict(),
            'mode': self.mode
        }, path)

        log_exp_run.experiments("Checkpoint saved")

    def load_state(self, path, trainable=True):
        if not self.warm_start or not self.initialized_:
            self.initialize()

        path = path+"params_eda.pt"
        if os.path.exists(path):
            checkpoint = torch.load(path)
            log_exp_run = make_logger(name="experiment_" + self.mode)
            if checkpoint is not None:
                self.module_.load_state_dict(checkpoint["model_state_dict"])
                self.mode=checkpoint["mode"]

                if trainable:
                    self.module_.train()
                else:
                    self.module_.eval()

                log_exp_run.experiments("Loaded state from check point with mode: "+self.mode)

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
                output = self.module_(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                      labels_ids=labels_ids)
                loss = output.loss
                train_loss += loss.item()

        log_exp_run.experiments("Cross-entropy loss for each fold: {}".format(train_loss))
        return train_loss

    # Training of tensor model using EMNA as EDA algorithms, with early stopping
    def train_eda_enma_early_stopping(self, sigma, centroid, generations, data):
        log_exp_run = make_logger(name="experiment_" + self.mode)
        iter_data = DataLoader(data, batch_size=self.module__batch_size, shuffle=True)

        # LAMBDA is the size of the population
        # N is the size of individual, the number of parameters on ANN
        N, LAMBDA = self.param_length, self.individuals

        # MU intermediate set of LAMBDA
        MU = int(LAMBDA / 4)

        # Creating an instance of EMNA
        strategy = EMNA(centroid=[centroid] * N, sigma=sigma, mu=MU, lambda_=LAMBDA)

        toolbox = base.Toolbox()
        toolbox.register("evaluate", loss_function, model=self.module_, training_data=iter_data, device=self.device)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)
        random.seed(int(time.time()))

        hof = tools.HallOfFame(1, similar=np.array_equal)

        # Define statisticals metrics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Param for early stoping
        t = 0
        STAGNATION_ITER = 100  # int(np.ceil(0.2 * t + 120 + 30. * N / LAMBDA))
        min_std = 1e-3
        conditions = {"MaxIter": False, "Stagnation": False}

        for gen in range(generations):
            # Generate a new population
            population = toolbox.generate()
            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            if hof is not None:
                hof.update(population)

            # Update the strategy with the evaluated individuals
            toolbox.update(population)

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)

            t += 1

            if t >= generations:
                # The maximum number of iteration
                conditions["MaxIter"] = True

            means_values = logbook.select("avg")
            if len(means_values) > STAGNATION_ITER and np.median(means_values[-20:]) >= np.median(
                    means_values[-STAGNATION_ITER:-STAGNATION_ITER + 20]) or record["std"] <= min_std:
                # The stagnation condition
                conditions["Stagnation"] = True

            if any(conditions.values()):
                break

        stop_causes = [k for k, v in conditions.items() if v]
        log_exp_run.experiments("Stopped because of condition")
        for cause in stop_causes:
            log_exp_run.experiments(cause)

        best_solution = hof[0]
        series_fitness = [i.fitness.values[0] for i in population]
        fix_individual_to_fln_layers(best_solution, self.module_, self.device)

        log_exp_run.experiments("Results for EDA_EMNA\r\n")
        log_exp_run.experiments("Parameters: \r\n")
        log_exp_run.experiments("Sigma: " + str(sigma) + " centroid: " + str(centroid) + "\r\n")
        log_exp_run.experiments("\r\n" + str(logbook) + "\r\n")
        log_exp_run.experiments(list(series_fitness))

    # Training tensor model using CUMDA as EDA algorithms, with early stopping
    def train_eda_cumda_early_stopping(self, sigma, generations, data):
        log_exp_run = make_logger(name="experiment_" + self.mode)
        iter_data = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        # LAMBDA is the size of the population
        # N is the size of individual, the number of parameters on ANN

        N, LAMBDA = self.param_length, self.individuals

        MU = int(LAMBDA / 4)

        toolbox = base.Toolbox()

        toolbox.register("evaluate", loss_function, model=self.module_, training_data=iter_data, device=self.device)
        random.seed(int(time.time()))

        # creating an instance of CUMDA
        strategy = CUMDA(N, sigma=sigma, mu=MU, lambda_=LAMBDA)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)
        hof = tools.HallOfFame(1)

        # Define statisticals metrics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Param for early stopping
        t = 0
        STAGNATION_ITER = 100  # int(np.ceil(0.2 * t + 120 + 30. * N / LAMBDA))
        min_std = 1e-4
        conditions = {"MaxIter": False, "Stagnation": False}

        # population_result=[]
        for gen in range(generations):
            # Generate a new population
            population = toolbox.generate()
            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            if hof is not None:
                hof.update(population)

            # Update the strategy with the evaluated individuals
            toolbox.update(population)

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)

            # print(self.logbook.stream)
            t += 1

            if t >= generations:
                # The maximum number of iteration
                conditions["MaxIter"] = True

            means_values = logbook.select("avg")
            if len(means_values) > STAGNATION_ITER and np.median(means_values[-20:]) >= np.median(
                    means_values[-STAGNATION_ITER:-STAGNATION_ITER + 20]) or record["std"] <= min_std:
                # The stagnation condition
                conditions["Stagnation"] = True

            if any(conditions.values()):
                break

        stop_causes = [k for k, v in conditions.items() if v]
        log_exp_run.experiments("Stopped because of condition")
        for cause in stop_causes:
            log_exp_run.experiments(cause)

        best_solution = hof[0]
        series_fitness = [i.fitness.values[0] for i in population]
        fix_individual_to_fln_layers(best_solution, self.module_, self.device)

        log_exp_run.experiments("Results for EDA_CUMDA\r\n")
        log_exp_run.experiments("Parameters: \r\n")
        log_exp_run.experiments("Sigma: " + str(sigma) + "\r\n")
        log_exp_run.experiments("\r\n" + str(logbook) + "\r\n")
        log_exp_run.experiments(list(series_fitness))

    # Training tensor model using CMA-ES as EDA algorithms, with early stopping
    def train_eda_cma_es_early_stopping(self, sigma, centroid, generations, data, is_unbalanced=False, task="main"):
        log_exp_run = make_logger(name="experiment_" + self.mode)
        iter_data = DataLoader(data, batch_size=self.batch_size, sampler=ImbalancedDatasetSamplerMT5(data)) if is_unbalanced else DataLoader(data, batch_size=self.batch_size, shuffle=True)
        # LAMBDA is the size of the population
        # N is the size of individual, the number of parameters on ANN
        N, LAMBDA = self.param_length, self.individuals

        toolbox = base.Toolbox()
        toolbox.register("evaluate", loss_function, model=self.module_, training_data=iter_data, device=self.device, task=task)
        np.random.seed(int(time.time()))

        # creating an instance of CMA
        strategy = cma.Strategy(centroid=[centroid] * N, sigma=sigma, lambda_=LAMBDA)#cma.Strategy
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)
        hof = tools.HallOfFame(1)

        # Define statisticals metrics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Param for early stopping
        t = 0
        STAGNATION_ITER = 100  # int(np.ceil(0.2 * t + 120 + 30. * N / LAMBDA))
        min_std = 1e-4
        conditions = {"MaxIter": False, "Stagnation": False}

        for gen in range(generations):
            # Generate a new population
            population = toolbox.generate()
            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            if hof is not None:
                hof.update(population)

            # Update the strategy with the evaluated individuals
            toolbox.update(population)

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)

            # print(self.logbook.stream)
            t += 1

            if t >= generations:
                # The maximum number of iteration
                conditions["MaxIter"] = True

            means_values = logbook.select("avg")
            if len(means_values) > STAGNATION_ITER and np.median(means_values[-20:]) >= np.median(
                    means_values[-STAGNATION_ITER:-STAGNATION_ITER + 20]) or record["std"] <= min_std:
                # The stagnation condition
                conditions["Stagnation"] = True

            if any(conditions.values()):
                break

        stop_causes = [k for k, v in conditions.items() if v]
        log_exp_run.experiments("Stopped because of condition")
        for cause in stop_causes:
            log_exp_run.experiments(cause)

        best_solution = hof[0]
        series_fitness = [i.fitness.values[0] for i in population]
        fix_individual_to_fln_layers(best_solution, self.module_, self.device)

        log_exp_run.experiments("Results for EDA_CMA-ES\r\n")
        log_exp_run.experiments("Parameters: \r\n")
        log_exp_run.experiments("Sigma: " + str(sigma) + " centroid: " + str(centroid) + "\r\n")
        log_exp_run.experiments("\r\n" + str(logbook) + "\r\n")
        log_exp_run.experiments(list(series_fitness))

    def compute_metrics(self, labels, preds):

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def score_unbalance(self, X, y=None, is_unbalanced=False, task='main'):
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
                labels = batch['target_ids'].to(self.device) if task == 'main' else batch['target_ids_attraction'].to(self.device)
                labels_ids = batch['labels'].to(self.device) if task == 'main' else batch['labels_attraction'].to(self.device)
                labels[labels == -100] = self.module_.config.pad_token_id
                output = self.module_(input_ids=input_ids, labels=labels, attention_mask=attention_mask, labels_ids=labels_ids)
                loss = output.loss
                train_loss += loss.item()

                logits = output.logits
                preds_batch = torch.argmax(logits, dim=-1)
                predictions.extend(preds_batch.cpu().numpy())
                labels_ref.extend(labels_ids.cpu().numpy())

        accuracy = accuracy_score(labels_ref, predictions)
        mae = mean_absolute_error(labels_ref, predictions)
        macro_f1 = f1_score(labels_ref, predictions, average='macro')

        log_exp_run.experiments("Cross-entropy loss for each fold: {}".format(train_loss))
        log_exp_run.experiments("Accuracy for each fold: " + str(accuracy))
        log_exp_run.experiments("\n" + classification_report(labels_ref, predictions))
        log_exp_run.experiments("\nMean Absolute Error (MAE): " + str(mae))
        log_exp_run.experiments("\nMacro F1: " + str(macro_f1))
        confusion_mtx = sm.confusion_matrix(labels_ref, predictions)
        metrics = self.compute_metrics(labels_ref,predictions)
        log_exp_run.experiments("All metrics (weighted) \nF1= {}, precision= {}, recall= {}".format(metrics['f1'], metrics['precision'], metrics['recall']))
        return accuracy, confusion_mtx


#  Compute loss with examples giving a single individual and Tensor model
def loss_function(individual, model, training_data, device, task):
    fix_individual_to_fln_layers(individual, model, device)
    train_loss = 0
    model.to(device)

    with torch.no_grad():
        for batch in training_data:
            input_ids = batch['source_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target_ids'].to(device) if task == 'main' else batch['target_ids_attraction'].to(device)
            labels_ids = batch['labels'].to(device) if task == 'main' else batch['labels_attraction'].to(device)
            labels[labels == -100] = model.config.pad_token_id
            output = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, labels_ids=labels_ids)
            loss = output.loss
            train_loss += loss.item()

    return train_loss,


#  Fix individual to full-connected layer in params model
def fix_individual_to_fln_layers(individual, model, device):
    index = 0
    individual_tensor = torch.tensor(individual)
    individual_tensor.to(device)
    model.to(device)
    fc_pathern = re.compile("fc\w*") # matching only with full connected layers (fc)
    for name, p in model.named_parameters():
        if p.requires_grad and fc_pathern.match(name):
            len_p = p.numel()
            aux_tensor = individual_tensor[index:index + len_p]
            p.data = aux_tensor.reshape(p.shape).data.type(torch.FloatTensor)
            index += len_p