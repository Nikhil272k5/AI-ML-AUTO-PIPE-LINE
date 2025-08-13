"""Models

    This package handles the model logic within the application and
encapsulates the logic of different frameworks like PyTorch and Sklearn
into a unified interface(AbstractModel)
"""

# Package level imports
from .model_loader import load_model
from .abstractModel import AbstractModel
from .modelFactory import ModelFactory
from .modelTypes import *
