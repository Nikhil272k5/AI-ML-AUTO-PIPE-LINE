from .abstractModel import AbstractModel
from .SpecializedModels import DeepLearningModel, RandomForestModel, SvmModel
from .EvolutionaryModel import EvolutionaryModel
from ...Exceptions.learnerException import ModelSelectionException


class ModelFactory:

    def __init__(self, config: dict = None):
        """
            Inits a model factory, responsible of returning untrained models as specified in the config.
            Based on the configuration provided it will create a new class derived from AbstractModel which will implement
        the model asked for.
            The class is responsible for the aggregation of different deep learning/ machine learning libraries,
        since it only has to return an AbstractModel class instance, regardless of what framework is used behind
        the train and predict methods.

        :param config: the configuration dictionary, expected to get the TRAINING_CONFIG part of the config file
        """
        if config is None:
            config = {}

        self._config = config
        self._task = config.get("TASK", "")

    def create_model(self, in_size: int, out_size: int) -> AbstractModel:
        """
            Creates a model as specified in the configuration.
        :return: the model created
        :param in_size: the input size of the model
        :param out_size: the predicted size of the model
        :raise: ModelSelectionException
        """

        # decide which kind of model has to be created and call the right method

        requested_type = self._config.get("TYPE", "default")

        if requested_type == "evolutionary":
            return self._create_evolutionary_model(in_size=in_size, out_size=out_size)

        elif requested_type == "default":  # choose the default model
            return self._create_default_model(in_size=in_size, out_size=out_size)

        else:
            raise ModelSelectionException("could not create model of type {}".format(requested_type))

    def _create_default_model(self, in_size, out_size):
        """
            Creates a default model which is expected to receive in_size variables and predict out_size variables
        :param in_size: the size of the input
        :param out_size: the size of the predicted output
        :return: the initial model (untrained)
        """
        model = None
        default_model_type = self._config.get("DEFAULT_MODEL", "neural_network")

        if default_model_type == "neural_network":
            model = DeepLearningModel(in_size, out_size, task=self._task,
                                      config=self._config.get("NEURAL_NETWORK_CONFIG", {}))

        elif default_model_type == "random_forest":
            model = RandomForestModel(task=self._task, config=self._config.get("RANDOM_FOREST_CONFIG", {}))

        elif default_model_type == "svm":
            model = SvmModel(task=self._task, config=self._config.get("SVM_CONFIG", {}))

        else:  # TODO add other methods as they are added in the SpecializedModels package
            pass

        return model

    def _create_evolutionary_model(self, in_size, out_size):
        """
        Creates an evolutionary model which is expected to find the best possible model which receives in_size
        variables and predicts out_size variables.

        :param in_size: the size of the input
        :param out_size: the size of the desired output
        :return: the initial model (untrained)
        """

        return EvolutionaryModel(in_size, out_size, task=self._task,
                                 config=self._config.get("EVOLUTIONARY_MODEL_CONFIG", {}))
