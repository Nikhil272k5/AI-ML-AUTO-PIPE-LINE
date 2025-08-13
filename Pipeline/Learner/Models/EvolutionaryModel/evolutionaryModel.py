import warnings
from random import randrange
from pandas import DataFrame
import time
from sklearn.model_selection import train_test_split
import numpy as np

from ..abstractModel import AbstractModel
from ....Exceptions import EvolutionaryModelException
from ..modelTypes import EVOLUTIONARY_MODEL
from .population import Population
from ..constants import AVAILABLE_TASKS
from ..Callbacks import EvolutionaryFeedback, ModelTriedCallback, EvolutionaryNewBestFeedback, ModelTrainingCallback


class EvolutionaryModel(AbstractModel):
    """
        The evolutionary model is responsible, like any other AbstractModel, to learn from data and eventually predict,
    but it also searches for the best performing model using evolutionary algorithms.

        The API is similar to the AbstractModel's API
    """

    def __init__(self, in_size: int, out_size: int, task: str = "", config: dict = None, predicted_name: list = None):
        """
                 Initializes a evolutionary model
        :param in_size: the size of the input data
        :param out_size: the size that needs to be predicted
        :param config: the configuration dictionary (expected to receive the EVOLUTIONARY_MODEL_CONFIG configuration)
        :param task: the task to be done (REGRESSION / CLASSIFICATION)
        :param predicted_name: the name of the attribute to be predicted to be predicted
        """
        AbstractModel.__init__(self)

        if config is None:
            config = {}

        # model parameters
        self._predicted_name = predicted_name
        self._task = task
        self._config = config
        self._input_size = in_size
        self._output_size = out_size

        # evolutionary attributes
        if task in AVAILABLE_TASKS:
            self._population = self._create_population(in_size, out_size, task,
                                                       config)  # the population for the evolutionary algorithm
        else:
            self._population = None  # to be created when analysing the data

        self._model = None  # the final model, after the evolutionary phase
        self._model_score = None

        # learning statistics
        self._models_tried = []
        self._epoch_bests = []
        self._best_model = None

    @staticmethod
    def _create_population(in_size: int, out_size: int, task: str, config: dict = None) -> Population:
        """
            Creates a population as configured in the config file
        :param in_size: the size of the input data
        :param out_size: the size that needs to be predicted
        :param task: the task of the model (CLASSIFICATION / REGRESSION)
        :param config: the configuration dictionary
        :return: a population of models
        """
        if config is None:
            config = {}

        population_size = config.get("POPULATION_SIZE", 10)
        population = Population(in_size, out_size, task, population_size, config)
        return population

    def _model_train(self, X: DataFrame, Y: DataFrame, train_time: int = 600, callbacks: list = None,
                     validation_split: float = 0.2, verbose: bool = True) -> 'AbstractModel':
        """
                Trains the model with the data provided.
            :param validation_split: percentage of the data to be used in validation; None if validation should not be used
            :param callbacks: a list of predefined callbacks that get called at every epoch
            :param train_time: time of the training session in seconds: default 10 minutes
            :param X: the independent variables in form of Pandas DataFrame
            :param Y: the dependents(predicted) values in form of Pandas DataFrame
            :param verbose: decides whether or not the model prints intermediary outputs
            :return: the model
        """
        if callbacks is None:
            callbacks = []

        # valid callbacks
        valid_callbacks = []
        model_tried_callback = None
        epoch_best_callback = None
        for callback in callbacks:
            if type(callback) is ModelTrainingCallback:
                valid_callbacks.append(callback)

            if type(callback) is EvolutionaryFeedback:
                valid_callbacks.append(callback)

            if type(callback) is ModelTriedCallback:
                model_tried_callback = callback

            if type(callback) is EvolutionaryNewBestFeedback:
                epoch_best_callback = callback


        # define the task
        if self._task not in AVAILABLE_TASKS or self._population is None:
            # if the task is not defined, neither is the population
            self._task = self._determine_task_type(Y)
            self._population = self._create_population(self._input_size, self._output_size, self._task, self._config)

        # define the predicted names
        if self._predicted_name is None:
            self._predicted_name = list(Y.columns)

        # get the training time parameters
        search_time = self._config.get("SEARCHING_TIME_SHARE", 0.5) * train_time  # search for models

        # set up the train test split logic: train the models using a dataset and evaluate them using a validation one
        if validation_split is None:
            x_train = X
            x_val = X
            y_train = Y
            y_val = Y
        else:
            if type(validation_split) != float:
                validation_split = 0.2
            validation_split = max(validation_split, 0.1)
            validation_split = min(validation_split, 0.9)

            x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=validation_split,
                                                              random_state=randrange(2048))

        # set up time tracking
        start_time = time.time()
        search_final = start_time + search_time
        epochs = 0
        seconds_count = 0
        keep_searching = True

        # initial evaluation - needed in order to evaluate the fitness of all the chromosomes in the population
        # as a general rule, we aim to generate a number of p*10 models, where p is the desired population
        # thus, the total search time will be split accordingly so each model has an even time of evaluation
        # this means that 9*p epochs will be used, since one model is created per epoch
        model_eval_time = search_time / (self._config.get("POPULATION_SIZE", 10) * 10)

        # call any necessary callbacks
        for callback in valid_callbacks:
            if type(callback) is EvolutionaryFeedback:
                callback({
                    "epoch": epochs,
                    "message": "Evaluating population...",
                })

        evaluation_feedback = None
        if model_tried_callback is not None:
            def evaluation_feedback(chromosome):
                data = {
                    "MODEL_SUMMARY": chromosome.get_model().summary(),
                    "DESCRIPTION": str(chromosome.get_model()),
                    "SCORE": chromosome.get_fitness()
                }
                if len(data["MODEL_SUMMARY"].get("TRAIN_DATA", {}).get("EPOCH_LOSS_TRAIN", [])) > 2 and \
                        not (np.nan in data["MODEL_SUMMARY"].get("TRAIN_DATA", {}).get("EPOCH_LOSS_TRAIN", [])):
                    model_tried_callback(data)

        print("Evaluating population...") if verbose else None
        self._population.eval(x_train, y_train, self._task, self._config.get("GENERAL_CRITERION"), model_eval_time,
                              validation_split=None, per_chromosome_call=evaluation_feedback)

        # add population models to statistics
        self._models_tried = []
        for chromosome in self._population.get_chromosomes():
            data = {
                "MODEL_SUMMARY": chromosome.get_model().summary(),
                "DESCRIPTION": str(chromosome.get_model()),
                "SCORE": chromosome.get_fitness()
            }
            self._models_tried.append(data)

            # if model_tried_callback is not None:
            #     if len(data["MODEL_SUMMARY"].get("TRAIN_DATA", {}).get("EPOCH_LOSS_TRAIN", [])) > 2 and \
            #             not (np.nan in data["MODEL_SUMMARY"].get("TRAIN_DATA", {}).get("EPOCH_LOSS_TRAIN", [])):
            #         model_tried_callback(data)

        # searches for the best model
        print("Searching for the best model...") if verbose else None

        # call any necessary callbacks
        for callback in valid_callbacks:
            if type(callback) is EvolutionaryFeedback:
                callback({
                    "epoch": epochs,
                    "message": "Searching for the best model...",
                })
                break

        while keep_searching:
            keep_searching = False
            epoch_start = time.time()

            # gather two chromosomes
            mother = self._population.selection()  # get the
            father = self._population.selection()  # parents

            # combine them
            offspring = self._population.XO(mother, father)  # combine them
            # mutate the result
            offspring_m = self._population.mutation(offspring)  # perform a mutation

            # evaluate the result
            offspring_m.eval(x_train, y_train, self._task, self._config.get("GENERAL_CRITERION"), model_eval_time,
                             validation_split=None)

            # add it in the population
            self._population.replace(offspring_m)

            # update the best model
            population_best = self._population.get_best()
            if self._best_model is None or self._model_score > population_best.get_fitness():
                self._best_model = population_best.get_model()
                self._model_score = population_best.get_fitness()
                data = {
                    "MODEL_SUMMARY": self._best_model.summary(),
                    "DESCRIPTION": str(self._best_model),
                    "SCORE": self._model_score
                }
                self._epoch_bests.append(data)

            data = {
                "MODEL_SUMMARY": self._best_model.summary(),
                "DESCRIPTION": str(self._best_model),
                "SCORE": self._model_score
            }

            if epoch_best_callback is not None and data["SCORE"] is not np.nan:
                epoch_best_callback(data)

            # gather statistics
            data = {
                "MODEL_SUMMARY": offspring_m.get_model().summary(),
                "DESCRIPTION": str(offspring_m.get_model()),
                "SCORE": offspring_m.get_fitness()
            }
            self._models_tried.append(data)
            if model_tried_callback is not None:
                if len(data["MODEL_SUMMARY"].get("TRAIN_DATA", {}).get("EPOCH_LOSS_TRAIN", [])) > 2 and \
                        not (np.nan in data["MODEL_SUMMARY"].get("TRAIN_DATA", {}).get("EPOCH_LOSS_TRAIN", [])):
                    model_tried_callback(data)

            # epoch end: gather time data
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            seconds_count += epoch_duration
            epochs += 1

            if search_final - epoch_end > epoch_duration * .5:  # the remaining time is more than half of the last epoch
                keep_searching = True  # train one more epoch

            # output epoch details
            validation_data = ""
            if validation_split is not None:
                validation_data = " Validation Score: {:.5f} |".format(
                    self._best_model.eval(x_val, y_val, self._task, self._config.get("GENERAL_CRITERION"))
                )

            print_string = "Epoch {:3d} -  Best Score: {:.5f} |{} Search time: {:.2f} seconds".format(
                epochs, population_best.get_fitness(), validation_data, epoch_duration)

            print(print_string) if verbose else None

            # call any necessary callbacks
            for callback in valid_callbacks:
                if type(callback) is EvolutionaryFeedback:
                    callback({
                        "epoch": epochs,
                        "message": print_string,
                    })
                    break

        # training the best model
        print("Training the best model...") if verbose else None
        for callback in valid_callbacks:
            if type(callback) is EvolutionaryFeedback:
                callback({
                    "epoch": epochs,
                    "message": "Training the best model...",
                })
                break

        self._model = self._population.get_best().get_model()
        best_model_time = int(train_time - (time.time() - start_time))  # the amount of seconds remaining

        _callbacks = []
        for callback in valid_callbacks:
            if type(callback) is ModelTrainingCallback:
                _callbacks.append(callback)

        self._model.train(X, Y, train_time=best_model_time, verbose=verbose, validation_split=validation_split,
                          callbacks=_callbacks)

        # return the trained best model
        return self._model

    def _model_predict(self, X: DataFrame, raw_output: bool = False) -> DataFrame:
        """
            Predicts the output of X based on previous learning
        :param X: DataFrame; the X values to be predicted into some Y Value
        :param raw_output: returns the exact output of the model, without rebasing into the initial classes
        :return: DataFrame with the predicted data
        """
        if self._model is None:
            raise EvolutionaryModelException("Train the model before performing a prediction.")

        return self._model.predict(X, raw_output=raw_output)

    def to_dict(self) -> dict:
        """
            Returns a dictionary representation of the model for further file saving.
        :return: dictionary with model encoding
        """
        if self._model is None:
            raise EvolutionaryModelException("Cannot convert EvolutionaryModel to dict unless a train() is performed.")

        return self._model.to_dict()

    def model_type(self) -> str:
        return EVOLUTIONARY_MODEL

    def _description_string(self) -> str:
        if self._model is None:
            return "Evolutionary Model - Not configured"
        else:
            TOP = 10
            best_models = self._population.get_best_n_description(TOP)
            best_models_string = '\n'.join(best_models)
            return "Evolutionary Model - \n  - Best model: \n{best} \n  - Top models: \n{top}".format(
                best=str(self._best_model),
                top=best_models_string
            )

    def get_config(self) -> dict:
        return self._config

    def summary(self) -> dict:
        """
            Returns a dictionary with statistics about training.
            "EPOCHS_BESTS" and "MODELS_TRIED" contain list with dictionaries of form:
                    {
                        "MODEL_SUMMARY": model summary, computed with the summary() method,
                        "DESCRIPTION": model description,
                        "SCORE": model score
                    }
        :return: dictionary with statistics
        """
        metadata = {}

        train_data = {
            "EPOCH_BESTS": self._epoch_bests,
            "BEST_MODEL": self._best_model.summary(),
            "MODELS_TRIED": self._models_tried
        }

        return {
            "MODEL_TYPE": self.model_type(),
            "METADATA": metadata,
            "TRAIN_DATA": train_data
        }

    def get_labels(self) -> list:
        warnings.warn("Method get_labels is not available for EvolutionaryModel")
        return []
