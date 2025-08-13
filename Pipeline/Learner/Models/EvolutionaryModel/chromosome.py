from pandas import DataFrame

from ..abstractModel import AbstractModel


class Chromosome:
    """
        The individual unit from a population.
        It's main task is to hold a model and evaluate it.

        Methods:
            - eval(): evaluates the model's performance on a dataset
            - get_fitness(): returns the fitness/score of the model
            - get_model(): returns the model within the chromosome
    """

    def __init__(self, model: AbstractModel):
        """
            Initializes a chromosome with a model
        :param model: the model that the chromosome operates on
        """
        self._genotype = model
        self._phenotype = None

    def eval(self, X: DataFrame, Y: DataFrame, task: str, criterion: str, time: int, validation_split: float) -> float:
        """
            Evaluates the model and returns a score (the fitness of the chromosome).
        By default, the model is trained with verbose = None, so outputs do not combine with evolutionary model output.
        :param validation_split: percentage of the data to be used in validation; None if validation should not be used
        :param time: time of the training session in seconds: default 10 minutes
        :param criterion: the criterion from the configuration file
        :param task: the task (CLASSIFICATION/REGRESSION)
        :param X: the data to predict an output from
        :param Y: the data to compare the output to
        :return: chromosome's fitness
        """
        model = self.get_model()
        model.train(X, Y, train_time=time, validation_split=validation_split, verbose=False)
        score = self._genotype.eval(X, Y, task, criterion, include_train_stats=True)
        self._phenotype = score
        return score

    def get_fitness(self) -> float:
        """
            Returns the model evaluation score
        :return: the phenotype
        """
        return self._phenotype

    def get_model(self) -> AbstractModel:
        """
            Returns the model
        :return: the model within the chromosome
        """
        return self._genotype

    def __repr__(self):
        score = "not evaluated"
        if self._phenotype:
            score = "{:.5f}".format(self._phenotype)
        return str("Score: {} Model: {}".format(score, str(self._genotype)))

