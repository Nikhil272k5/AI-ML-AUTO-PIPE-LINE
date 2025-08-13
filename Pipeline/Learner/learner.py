from ..Mapper import Mapper
from pandas import DataFrame
from .Models import AbstractModel
from .Models import ModelFactory
from ..Exceptions.learnerException import LearnerException


class Learner:
    """
        The class that handles the learning inside the pipeline.
        It's main task is to learn from a dataset and return a model.
        Based on a configuration file given as constructor parameter it is able to do a series of tasks:
            - fit the data on a dataset with a default predefined model (defined in config)
            - fit the data using a series of models and evolutionary algorithms for finding the best one #TO BE DONE

        Methods:
            - learn: creates a model and learns it based on a given dataset
            - get_model: returns the last trained model
            - get_mapper: gets the mapper with attributes
    """

    def __init__(self, config: dict = None, model: 'AbstractModel' = None):
        """
            Creates a learner instance based on the configuration file.
            :param config: dictionary with the configurations for the learning module
                        - expected to get the TRAINING_CONFIG section of the config file
        """

        if config is None:
            config = {}

        self._config = config
        self._mapper = Mapper('Learner')
        self._model_factory = ModelFactory(self._config)
        self._model = model

    def learn(self, X: DataFrame, Y: DataFrame, verbose: bool = True, callbacks: list = None, time: int = None) \
            -> AbstractModel:
        """
            Learns based on the configuration provided.
        :param callbacks: callbacks to be executed by the model when training it
        :param X: the data to learn from
        :param Y: the predictions to compare to

        :param verbose: decides whether the learn() method should produce any output
        :param time: time in seconds for the training session; if not specified the time in config is used
        :return: the trained model
        """
        if callbacks is None:
            callbacks = []

        # input and output size
        input_size = X.shape[1]
        output_size = Y.shape[1]

        self._mapper.set("input_size", input_size)
        self._mapper.set("output_size", output_size)

        # creates a model
        model = self._model
        if model is None:
            model = self._model_factory.create_model(in_size=input_size, out_size=output_size)

        # trains the model
        if time is None:
            train_time = self._convert_train_time(self._config.get("TIME", "10m"))
        else:
            train_time = time

        # here's where the magic happens
        model.train(X, Y, train_time, verbose=verbose, callbacks=callbacks)

        # returns it
        self._model = model
        return model

    @staticmethod
    def _convert_train_time(time: str) -> int:
        """
            Converts the time from "xd yh zm ts" into seconds
        :param time: string containing the time in textual format -number of days , hours, minutes and seconds
        :return: the time in seconds
        """
        mapping = {}
        crt_count = 0

        for c in time:  # for each character
            if c.isnumeric():
                crt_count = crt_count * 10 + int(c)
            elif c in "dhms":  # days hours minutes seconds
                mapping[c] = mapping.get(c, 0) + crt_count
                crt_count = 0
            else:
                crt_count = 0

        seconds = mapping.get("s", 0) + mapping.get("m", 0) * 60 + \
                  mapping.get("h", 0) * (60 * 60) + mapping.get("d", 0) * 24 * 60 * 60

        return seconds

    def get_model(self) -> AbstractModel:
        """
            Returns the model after training.
        :return: the model that has been trained with the learn method
        :exception LearnerException if this method is called before learn is called
        """
        if self._model is None:
            raise LearnerException("Could not retrieve model before 'learn' is called.")

        return self._model

    def get_mapper(self) -> 'Mapper':
        """
            Returns the mapper that contains data about training
        :return: the mapper
        """
        model_map = None
        if not (self._model is None):
            model_map = self._model.to_dict()

        self._mapper.set("MODEL", model_map)
        return self._mapper
