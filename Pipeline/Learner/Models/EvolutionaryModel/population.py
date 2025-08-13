from pandas import DataFrame
from random import choices

from ..abstractModel import AbstractModel
from .chromosome import Chromosome
from .model_creation import create_random_model
from ..modelTypes import *
from .model_crossover import *
from .model_mutation import *


class Population:
    """
        The main scope of the class is to aggregate chromosomes(models) into one place,
    to train them and eventually find the best one.

        Methods:
            - eval(): evaluates the whole population based on a X,Y dataset
            - get_best(): finds the best chromosome and returns it
            - replace(): replaces the worst performing model(chromosome) with a new chromosome
            - selection(): returns a chromosome from the population (the better it's model performance,
                            the higher the chances of returning that chromosome)
            - XO(): crossover between two chromosomes: return another one
            - mutation(): performs a random mutation on a chromosome
    """

    def __init__(self, input_size: int, output_size: int, task: str, population_size: int = 10, config: dict = None):
        """
            Creates a population of chromosomes (models)
        :param input_size: the size of the input data
        :param output_size: the desired size of the prediction
        :param task: the task of the population's models (CLASSIFICATION / REGRESSION)
        :param population_size: the population size generated
        :param config: the configuration for evolutionary choices and ranges
            (expected to get the EVOLUTIONARY_MODEL_CONFIG part of the config file)
        """
        if config is None:
            config = {}

        self._config = config
        self._input_size = input_size
        self._output_size = output_size
        self._task = task
        self._criterion = config.get("GENERAL_CRITERION")

        self._population_size = population_size
        self._population = self._create_population(input_size, output_size, task, population_size, config)
        self._best_model = None
        self._fitness = None

        self._best_chromosome = None

    def eval(self, X: DataFrame, Y: DataFrame, task: str, criterion: str, time: int, validation_split: float,
             per_chromosome_call=None) -> Chromosome:
        """
            Evaluates the population
        :param per_chromosome_call: lambda function; if not None, call it with every evaluated chromosome
        :param validation_split: percentage of the data to be used in validation; None if validation should not be used
        :param time: time of the training session for each model in seconds
        :param criterion: the criterion from the configuration file
        :param task: the task (CLASSIFICATION/REGRESSION)
        :param X: the data to predict an output from
        :param Y: the data to compare the output to
        :return: the best model in the population
        """
        best = None
        best_fitness = None

        for chromosome in self._population:
            chromosome_fitness = chromosome.eval(X, Y, task, criterion, time, validation_split)

            if best_fitness is None or self._is_fitter(chromosome_fitness, best_fitness):
                best = chromosome
                best_fitness = chromosome_fitness

            if per_chromosome_call is not None:
                per_chromosome_call(chromosome)

        self._best_chromosome = best  # cache the best for further usage
        return best

    def get_best(self) -> Chromosome:
        """
            Returns the best model in the population.
            The method assumes that at every moment the population has done eval on every chromosome,
        thus they all have a phenotype.
        :return:
        """
        if not (self._best_chromosome is None):  # if the best chromosome is unchanged since the last calculation
            return self._best_chromosome

        best = None
        best_fitness = None

        for chromosome in self._population:
            chromosome_fitness = chromosome.get_fitness()

            if best_fitness is None or self._is_fitter(chromosome_fitness, best_fitness):
                best = chromosome
                best_fitness = chromosome_fitness

        return best

    def replace(self, chromosome: Chromosome) -> list:
        """
            Adds the model into the population by replacing the worst model with the new one
        :param chromosome: the new model to be added
        :return: the new population ( the same list as before, but changed)
        """
        # compare to the current best
        if not (self._best_chromosome is None) and \
                self._is_fitter(chromosome.get_fitness(), self._best_chromosome.get_fitness()):
            self._best_chromosome = chromosome

        # determine the position of the worst
        model = chromosome.get_model()

        worst_fitness = None
        worst_position = None

        for i in range(len(self._population)):
            chromosome_crt = self._population[i]
            chromosome_fitness = chromosome_crt.get_fitness()

            if worst_fitness is None or self._is_fitter(worst_fitness, chromosome_fitness):
                worst_fitness = chromosome_fitness
                worst_position = i

        # delete the worst
        to_delete = self._population[worst_position]
        self._population[worst_position] = None
        del to_delete

        # replace the worst
        if self._population[worst_position] == self._best_chromosome:  # we remove the current best
            self._best_chromosome = None

        self._population[worst_position] = chromosome

        return self._population

    def selection(self) -> Chromosome:
        """
            Selects a model from the population
        :return: the selected model
        """
        # each chromosome has a fitness, and the lower the fitness, the higher the probability of election
        choices_list = list(range(len(self._population)))
        weights = [1 / chromosome.get_fitness() for chromosome in self._population]

        index = choices(choices_list, weights=weights)[0]

        return self._population[index]

    def XO(self, chromosome1: Chromosome, chromosome2: Chromosome) -> Chromosome:
        """
            Performs cross over between two population members (chromosomes).
        :param chromosome1: the first chromosome
        :param chromosome2: the second chromosome
        :return: the offspring
        """
        model_type1 = chromosome1.get_model().model_type()
        model_type2 = chromosome2.get_model().model_type()

        if model_type1 == DEEP_LEARNING_MODEL:
            if model_type2 == DEEP_LEARNING_MODEL:
                return Chromosome(
                    deep_learning_XO_deep_learning(chromosome1.get_model(), chromosome2.get_model(), self._input_size,
                                                   self._output_size, self._task)
                )

        # TODO more to be added

    def mutation(self, chromosome: Chromosome) -> Chromosome:
        """
            Performs an in-place mutation on the model, returning it.
        :param chromosome: the model to be mutated
        :return: the same model, mutated
        """
        model_type = chromosome.get_model().model_type()

        if model_type == DEEP_LEARNING_MODEL:
            return Chromosome(
                deep_learning_mutation(chromosome.get_model(), self._input_size, self._output_size, self._task,
                                       self._config.get("NEURAL_NETWORK_EVOL_CONFIG", {}))
            )

        return chromosome

    def _create_population(self, input_size: int, output_size: int, task: str, population_size: int,
                           config: dict) -> list:
        """
            Creates a random population of the given size and configuration.
        :param input_size: the input size for the data to be trained
        :param output_size: the output size of the data to be predicted
        :param task: the task that needs to be performed (CLASSIFICATION / REGRESSION)
        :param population_size: the number of chromosomes in the population
        :param config: the configuration that states the rules for population creation.
        :return: list of chromosomes
        """
        population = []
        for i in range(population_size):
            model = self._random_model(input_size, output_size, task, config)
            chromosome = Chromosome(model)
            population.append(chromosome)
        return population

    def _random_model(self, input_size, output_size, task, config: dict) -> AbstractModel:
        """
            Generates a random chromosome
        :param input_size: the input size for the data to be trained
        :param output_size: the output size of the data to be predicted
        :param task: the task that needs to be performed (CLASSIFICATION / REGRESSION)
        :param config: the configuration for the evolutionary models
                (expected EVOLUTIONARY_MODEL_CONFIG part of the config file)
        :return: the created model (untrained)
        """
        return create_random_model(input_size, output_size, config, task)

    def get_chromosomes(self) -> list:
        """
            Returns all the models in the population
        :return: list of Chromosomes
        """
        return [chromosome for chromosome in self._population]

    @staticmethod
    def _is_fitter(actual_fitness: float, best_fitness: float) -> bool:
        """
            Compare the two metrics and decides whether actual_fitness is better(fitter) than best_fitness
        :param actual_fitness: float, score of the actual model
        :param best_fitness: float, the best score so far
        :return: bool
        """
        return actual_fitness < best_fitness  # important: the general problem is considered
        # to be a minimization problem, thus a lower fitness is better

    def get_best_n_description(self, n: int) -> list:
        """
            Returns the best n chromosomes' descriptions
        :param n: the number of chromosomes to be returned
        :return: list of strings
        """
        self._population.sort(key=lambda chromosome: chromosome.get_fitness())
        descriptions = []

        number = min(n, len(self._population))
        for i in range(number):
            descriptions.append(str(self._population[i]))

        return descriptions
