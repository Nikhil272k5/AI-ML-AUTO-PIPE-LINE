"""
    This file contains all the XO (crossover) methods for chromosomes (models)
"""
from random import choice, random, randint

from ..SpecializedModels import *


def deep_learning_XO_deep_learning(model1: DeepLearningModel, model2: DeepLearningModel,
                                   in_size: int, out_size: int, task:str):
    """
        Performs crossover between 2 deep learning models.
        For each possible parameter, it performs a combination of the parameters of the parents with a
    probability of 60%. Otherwise, it takes the parameter of one random parent.
    :param task: the task carried out by the model ("REGRESSION" / "CLASSIFICATION" )
    :param out_size: the output of the model
    :param in_size: the input of the model
    :param model1: deep learning model
    :param model2: deep learning model
    :return: deep learning model
    """
    config1 = model1.get_config()
    config2 = model2.get_config()
    XO_PROBAB = 0.6  # the probability that the parents are combined rather than choosing attributes from only one

    # optimizer
    optimizer = choice([config1.get("OPTIMIZER"), config2.get("OPTIMIZER")])

    # learning rate
    if random() <= XO_PROBAB:
        learning_rate = .5 * (config1.get("LEARNING_RATE") + config2.get("LEARNING_RATE"))
    else:
        learning_rate = choice([config1.get("LEARNING_RATE"), config2.get("LEARNING_RATE")])

    # momentum
    if random() <= XO_PROBAB:
        momentum = .5 * (config1.get("MOMENTUM") + config2.get("MOMENTUM"))
    else:
        momentum = choice([config1.get("MOMENTUM"), config2.get("MOMENTUM")])

    # regularization
    if random() <= XO_PROBAB:
        regularization = .5 * (config1.get("REGULARIZATION") + config2.get("REGULARIZATION"))
    else:
        regularization = choice([config1.get("REGULARIZATION"), config2.get("REGULARIZATION")])

    # hidden_layers
    if type(config1.get("HIDDEN_LAYERS")) != type(config2.get("HIDDEN_LAYERS")):
        layers = choice([config1.get("HIDDEN_LAYERS"), config2.get("HIDDEN_LAYERS")])
    else:
        if type(config1.get("HIDDEN_LAYERS")) is list:
            min_len = min(len(config1.get("HIDDEN_LAYERS")), len(config2.get("HIDDEN_LAYERS")))
            desired_len = int(.5 * (len(config1.get("HIDDEN_LAYERS")) + len(config2.get("HIDDEN_LAYERS"))))

            layers = [0]*desired_len

            for i in range(desired_len):
                if i < min_len:
                    layers[i] = choice([config1.get("HIDDEN_LAYERS")[i], config2.get("HIDDEN_LAYERS")[i]])
                else:
                    if len(config1.get("HIDDEN_LAYERS")) == min_len:    # take from the longest one
                        layers[i] = config2.get("HIDDEN_LAYERS")[i]
                    else:
                        layers[i] = config1.get("HIDDEN_LAYERS")[i]

        else:
            layers = choice([config1.get("HIDDEN_LAYERS"), config2.get("HIDDEN_LAYERS")])

    # activations
    if random() > XO_PROBAB or type(config1.get("ACTIVATIONS")) != type(config2.get("ACTIVATIONS")):
        activation = choice([config1.get("ACTIVATIONS"), config2.get("ACTIVATIONS")])
    else:
        if type(config1.get("ACTIVATIONS")) is list:
            # doing exactly like in the case of layers
            min_len = min(len(config1.get("ACTIVATIONS")), len(config2.get("ACTIVATIONS")))
            desired_len = int(.5 * (len(config1.get("ACTIVATIONS")) + len(config2.get("ACTIVATIONS"))))

            activation = []

            for i in range(desired_len):
                if i < min_len:
                    activation.append(choice([config1.get("ACTIVATIONS")[i], config2.get("ACTIVATIONS")[i]]))
                else:
                    if len(config1.get("ACTIVATIONS")) == min_len:  # take from the longest one
                        activation.append(config2.get("ACTIVATIONS")[i])
                    else:
                        activation.append(config1.get("ACTIVATIONS")[i])

            if config1.get("ACTIVATIONS")[-1] == "sigmoid" and config2.get("ACTIVATIONS")[-1] == "sigmoid":
                activation[-1] = "sigmoid"
        else:
            activation = choice([config1.get("ACTIVATIONS"), config2.get("ACTIVATIONS")])

    # dropout
    if random() <= XO_PROBAB and \
            type(config1.get("DROPOUT")) in [int, float] and \
            type(config2.get("DROPOUT")) in [int, float]:
        dropout = .5 * (config1.get("DROPOUT") + config2.get("DROPOUT"))
    else:
        dropout = choice([config1.get("DROPOUT"), config2.get("DROPOUT")])

    # batch_size
    if random() <= XO_PROBAB:
        batch_size = int(.5 * (config1.get("BATCH_SIZE") + config2.get("BATCH_SIZE")))
    else:
        batch_size = choice([config1.get("BATCH_SIZE"), config2.get("BATCH_SIZE")])

    offspring_config = {
        "CRITERION": config1.get("CRITERION", "undefined"),
        "OPTIMIZER": optimizer,
        "LEARNING_RATE": learning_rate,
        "MOMENTUM": momentum,
        "REGULARIZATION": regularization,
        "HIDDEN_LAYERS": layers,
        "ACTIVATIONS": activation,
        "DROPOUT": dropout,
        "BATCH_SIZE": batch_size
    }

    return DeepLearningModel(in_size, out_size, task, offspring_config)
