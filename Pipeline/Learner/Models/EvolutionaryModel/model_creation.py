"""
    This file is responsible for gathering all the possible model creation functions
since the population needs random models at the beginning.

    The main method is create_random_model() which calls the specialized methods.
    Each specialized method is responsible for converting the ranges and possible values from the
"EVAL" configuration, into a configuration able to be read by a model, like the DEFAULT configurations
in the config file.
"""

from random import choice, uniform, randint, choices, random

from ..SpecializedModels import DeepLearningModel
from ..abstractModel import AbstractModel
from ..constants import CLASSIFICATION


def create_random_model(in_size: int, out_size: int, config: dict, task: str) -> AbstractModel:
    """
        Creates a random model as specified in the config file
    :param in_size: the input size of the network
    :param out_size: the output size of the network
    :param config: the configuration dictionary with the possible models and their possible configurations
            (expected to receive the EVOLUTIONARY_MODEL_CONFIG configuration dictionary)
    :param task: the task that needs to be done (REGRESSION/CLASSIFICATION)
    :return: the initialized model
    """
    model_options = config.get("MODELS", ["neural_network"])
    model_type = choice(model_options)

    criterion = config.get("GENERAL_CRITERION", "MSE")

    if model_type == "neural_network":
        return create_deep_learning_model(in_size, out_size, config.get("NEURAL_NETWORK_EVOL_CONFIG", {}), task,
                                          criterion)

    elif model_type == "some_future_type":
        # TODO add types as they are added in config
        pass


def create_deep_learning_model(in_size: int, out_size: int, config: dict, task: str,
                               criterion: str) -> DeepLearningModel:
    """
        Creates a random neural Network model
    :param criterion: the criterion used in evaluation (ex: "MSE")
    :param in_size: the input size of the network
    :param out_size: the output size of the network
    :param config: the configuration dictionary with the possible ranges and values
        (expected to receive the NEURAL_NETWORK_EVOL_CONFIG part of the config file)
    :param task: the task that needs to be done (REGRESSION/CLASSIFICATION)
    :return: the initialized model
    """

    # create a configuration dictionary for the model
    optimizer = choice(config.get("OPTIMIZER_CHOICE", ["Adam", "SGD"]))
    learning_rate = uniform(*config.get("LEARNING_RATE_RANGE", [0.000001, 1]))
    momentum = uniform(*config.get("MOMENTUM_RANGE", [0, 1]))
    regularization = uniform(*config.get("REGULARIZATION_RANGE", [0, 0.01]))
    # in the layer choices, the random choice is more biased towards creating a custom weight range.
    # since using smooth all over the place could be too mainstream
    layer_choice = choices(config.get("HIDDEN_LAYERS_CHOICES", ["smooth", [10, 128, 6]]), weights=[0.1, 0.9])[0]
    if type(layer_choice) is list:
        layer_count = randint(1, max(layer_choice[2], 1))
        layers = [randint(layer_choice[0], layer_choice[1]) for _ in range(layer_count)]
    else:
        layers = "smooth"

    # the same as with layers, we put more bias on a list of random activations rather than a smooth activation choice
    activation_choice = choices(["uniform", "list"], weights=[0.3, 0.7])[0]
    if activation_choice == "uniform" or layers == "smooth":
        activation = choice(config.get("ACTIVATION_CHOICES", ["sigmoid", "relu", "linear"]))
    else:
        activation = [choice(config.get("ACTIVATION_CHOICES", ["sigmoid", "relu", "linear"]))
                      for _ in range(len(layers) + 1)]

    if task == CLASSIFICATION:  # for classification "sigmoid" is used in the last layer by default
        if type(activation) is str:
            activation = "sigmoid"
        else:
            activation[-1] = "sigmoid"

    # dropout
    if random() < 0.5:
        dropout = uniform(*config.get("DROPOUT_RANGE", [0, 0.6]))
    else:
        desired_len = 6
        if type(layers) is list:
            desired_len = len(layers)

        dropout = [uniform(*config.get("DROPOUT_RANGE", [0, 0.6])) for _ in range(desired_len)]

    # batch size
    batch_size = randint(*config.get("BATCH_SIZE_RANGE", [1, 128]))

    model_config = {
        "CRITERION": criterion,
        "OPTIMIZER": optimizer,
        "LEARNING_RATE": learning_rate,
        "MOMENTUM": momentum,
        "REGULARIZATION": regularization,
        "HIDDEN_LAYERS": layers,
        "ACTIVATIONS": activation,
        "DROPOUT": dropout,
        "BATCH_SIZE": batch_size
    }

    # create the model with the previously created dictionary
    model = DeepLearningModel(in_size=in_size, out_size=out_size, task=task, config=model_config)
    return model
