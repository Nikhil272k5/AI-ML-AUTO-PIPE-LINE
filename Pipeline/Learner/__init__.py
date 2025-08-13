"""Learner

    The package that handles learning within the pipeline.

    Attributes :
        - Learner: class that handles the learning phases
        - LearnerException: possible exception raised by Learner
        - modelFactory: retrieved specialized models given a configuration
        - model_loader: utility script for loading models from file
        - AbstractModel: the generic model behaviour
"""

# Package level imports

from .learner import Learner
from .Models import ModelFactory, load_model, AbstractModel
