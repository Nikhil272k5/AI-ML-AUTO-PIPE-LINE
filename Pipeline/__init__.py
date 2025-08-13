"""Pipeline

    The main object from the package is the Pipeline class.

    Pipeline flows:
        FLOW1: The fully automation goal --- done - still open to improvements
            Raw data -> Data processing -> Model learning -> Trained model

        FLOW2: The feature engineering flow --- done
            Raw data -> Data processing -> Processed data

        FLOW3: The default learning flow --- done
            Raw data -> Data processing -> Default learning -> Trained model

        FLOW4: The evolutionary learning flow --- done
            Raw data -> Data processing -> Evolutionary learning -> Trained model

        FLOW5: The data conversion flow --- done
            Raw data --Saved pipeline--> Data converting -> Converted data

        FLOW6: The raw data prediction flow --- done
            Raw data --Saved pipeline--> Data converting -> Prediction -> Result

        FLOW7: The converted data prediction flow --- done
            Raw data --Saved pipeline/model--> Prediction -> Result

        Each flow has an example in  ./examples
"""
from pandas import read_csv                 # for data input

from .pipeline import Pipeline              # main class
from .Learner import load_model             # method for model loading from file
from .pipeline import load_pipeline         # method for loading a pipeline from file
from .DataProcessor import Splitter
from .Learner.Models.Callbacks import *
