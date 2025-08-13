import pickle

from .modelTypes import *
from .SpecializedModels import DeepLearningModel, RandomForestModel, SvmModel
from .EvolutionaryModel import EvolutionaryModel
from ...Exceptions.learnerException import ModelLoaderException


def load_model(source):
    """
        Loads a generic model from file. After successful loading the model is ready to be used
    for training or prediction.
        Contains all the logic for loading all model types. Requires files that have 2 keys:
            - MODEL_TYPE
            - MODEL_DATA
            ;which must be returned in each instance of to_dict of any AbstractModel implementation

    :param source: str(file with saved dictionary) or dictionary(with model)
    :return: model instance
    """
    if type(source) is str:
        with open(source, 'rb') as f:
            dictionary = pickle.load(f)

    elif type(source) is dict:
        dictionary = source

    else:
        raise ModelLoaderException(
            "Could not load model from data type {}".format(type(source)))

    model_type = dictionary.get("MODEL_TYPE", "undefined")

    if model_type == DEEP_LEARNING_MODEL:
        model = DeepLearningModel(0, 0, dictionary=dictionary)
    elif model_type == RANDOM_FOREST_MODEL:
        model = RandomForestModel(dictionary=dictionary)
    elif model_type == SVM_MODEL:
        model = SvmModel(dictionary=dictionary)
    elif model_type == EVOLUTIONARY_MODEL:
        model = EvolutionaryModel(0, 0, dictionary=model_type)
    # TODO add models as they are added in the SpecializedModels module
    else:
        raise ModelLoaderException(
            "Could not load model of type {}".format(model_type))

    return model
