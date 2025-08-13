import json
import os
import time

from pandas import DataFrame, concat

from .Mapper import Mapper
from .DataProcessor.processor import Processor
from .Exceptions.pipelineException import PipelineException
from .Learner.Models.abstractModel import AbstractModel
from .Learner.learner import Learner
from .DataProcessor.DataSplitting.splitter import Splitter
from .Learner.Models.model_loader import load_model
from .configuration_manager import complete_configuration
from .Learner.Models.Callbacks import PipelineFeedback, DataConversionCallback


def load_pipeline(file: str) -> 'Pipeline':
    """
        Loads the pipeline from a file where it was previously saved
    :param file: path to the file where the pipeline was previously saved
    :return: the pipeline
    """
    mapper = Mapper("Pipeline", file)
    return Pipeline(mapper=mapper)


class Pipeline:
    """
        Represents the core of the program.
        Aims to convert raw data to trained model.
        Pipeline steps:
            1. Data cleaning & feature engineering module.
            2. Model training


        Methods:
            - process: processes a dataset according to the specifications in the config file
            - convert: converts data according to the rules learnt from a previous process call
            - learn: given a dataset and a configuration fits a model to the data
            - predict: provided that the pipeline has previously learnt a model, it predicts the output of data
            - fit: does all the steps activated in the configuration file
            - save: saves the pipeline (including the model) to a file
            - get_model: returns the model (none if it has not learnt a model previously)
            - load_pipeline(defined outside the class): reads a saved pipeline from a file and returns it
    """

    # define possible states - the state in which the pipeline is at a given moment
    DYNAMIC_CALL = False  # calling the object behaves differently depending on the state of the pipeline if true
    # if set to false, the flow will respect the configuration
    STATE_MACRO = "PIPELINE_STATUS"
    RAW_STATE = "RAW_STATE"  # the pipeline is new and no operation has been done to it
    PROCESSED_STATE = "PROCESSED_STATE"  # the pipeline has been used to process data
    LEARNT_STATE = "LEARNT_STATE"
    CONVERTED_STATE = "CONVERTED_STATE"  # the pipeline has been last used to convert data
    PREDICTED_STATE = "PREDICTED_STATE"  # the pipeline has last been used to predict data

    def __init__(self, config: dict = None, mapper_file: str = None, mapper: 'Mapper' = None,
                 default_config_path: str = None, **kwargs):
        """
            Inits the pipeline
        :param config: configuration dictionary
        :param mapper_file: the file where the mapper is saved, if existing
        :param mapper:the dictionary (in Mapper format) containing the data previously saved by the Pipeline instance
        :param default_config_path: if the pipeline is used with a configuration file located elsewhere than
                    the default location; if provided, this path will be used when creating the configuration
        :param kwargs
                - include "dynamic_call=True" in the argument list to enable dynamic pipeline call
        Usage:
            if provided any data, the Pipeline will init itself from that dictionary
            otherwise, if provided a config it will use that, if not it will try to read the config from file
                       if a mapper file is provided the processor will be initialized with that
        """
        if kwargs.get("dynamic_call", False):
            self.DYNAMIC_CALL = True

        # data processing attributes
        if mapper is None:  # initialized by the user
            self._processor = None
            self._mapper_file = None

            if config is None:
                self._config = Pipeline._read_config_file(default_config_path)
            else:
                self._config = complete_configuration(config, self._read_config_file())
            if self._config.get("DATA_PROCESSING", False):
                self._processor = Processor(self._config.get("DATA_PROCESSING_CONFIG"), file=mapper_file)

            self._mapper = Mapper("Pipeline")

            # learner attributes
            self._learner = Learner(self._config.get("TRAINING_CONFIG", {}))
            self._model = None

            self._mapper.set("CONVERSION_DONE", False)

        else:  # initialized by the load_pipeline method
            self._mapper = mapper
            self._config = mapper.get("CONFIG", default={})
            self._processor = Processor(self._config, data=mapper.get_mapper("PROCESSOR_DATA", {}))

            model_map = mapper.get("MODEL", default=None)
            if model_map is None:
                self._model = None
            else:
                self._model = load_model(model_map)

            self._learner = Learner(self._config.get("TRAINING_CONFIG", {}), model=self._model)

        # set status of pipeline
        current_status = self._mapper.get(self.STATE_MACRO, False)
        if current_status is False:
            self._mapper.set(self.STATE_MACRO, self.RAW_STATE)

    def _record_data_information(self, data: DataFrame, source: str, *args, **kwargs) -> None:
        """
            Maps metadata about the data fed into the pipeline.
        :param data: DataFrame containing the dataset
        :param source: string containing the source of the data
                        - data is recorded on process() and on learn(), to be used later in convert() and predict()
                        - source should be either "process" or "learn"
        :return: None
        """
        info = self._mapper.get("DATA_METADATA", default={})

        if source == "process":
            new_info = {
                "shape": data.shape,
                "columns": data.columns.to_list()
            }
        elif source == "learn":
            new_info = {
                "shape": data.shape,
                "y_column": kwargs.get("y_column", "undefined")
            }
        else:
            new_info = {}

        info[source] = new_info
        self._mapper.set("DATA_METADATA", info)

    def process(self, data: DataFrame, verbose: bool = True, callbacks: list = None) -> DataFrame:
        """
            Processes the data according to the configuration in the config file
        :param callbacks: list of AbstractCallback objects that might get called at some points in the app
        :param verbose: decides if the process() method will produce any output
        :param data: DataFrame containing the raw data that has to be transformed
        :return: DataFrame with the modified data
        :raises: Pipeline exception
        """
        if callbacks is None:
            callbacks = []

        start = time.time()
        self._record_data_information(data, "process")
        result = data

        valid_callbacks = []
        for callback in callbacks:
            if type(callback) in [PipelineFeedback, ]:
                valid_callbacks.append(callback)

        # 1. Data processing
        if self._config.get("DATA_PROCESSING", False):
            try:
                result = self._processor.process(result, verbose=verbose, callbacks=valid_callbacks)
            except Exception as err:
                # TODO: add logs
                raise PipelineException("Data processing error: {}.".format(err))

        self._mapper.set("COLUMNS_PROCESS", list(data.columns))
        self._mapper.set("CONVERSION_DONE", True)
        end = time.time()
        print("Processed in {0:.4f} seconds.".format(end - start)) if verbose else None
        for callback in valid_callbacks:
            callback({
                "message": "Processed in {0:.4f} seconds.".format(end - start)
            })
        self._mapper.set(self.STATE_MACRO, self.PROCESSED_STATE)
        return result

    def convert(self, data: DataFrame, verbose: bool = True) -> DataFrame:
        """
            Converts the data to the representation previously learned by the DataProcessor
        :param verbose: decides if the convert() method will produce any output
        :param data: DataFrame containing data similar to what the
        :return: DataFrame containing the converted data
        :exception: PipelineException
        """
        start = time.time()

        if self._processor is None:
            if self._mapper_file is None:
                raise PipelineException(
                    "Mapper file not set. In order to convert data, provide a mapper file to the constructor.")
            self._processor = Processor(self._config.get("DATA_PROCESSING_CONFIG"), self._mapper_file)

        try:
            result = self._processor.convert(data, verbose=verbose)
        except Exception as err:
            # TODO: add logs
            raise PipelineException("Conversion error: {}.".format(err))

        end = time.time()
        print("Converted in {0:.4f} seconds.".format(end - start)) if verbose else None
        self._mapper.set(self.STATE_MACRO, self.CONVERTED_STATE)
        return result

    def learn(self, data: DataFrame, y_column: str = None, verbose: bool = True,
              callbacks: list = None,train_time:int =None) -> AbstractModel:
        """
            Learns a model from the data.

        :param verbose: decides if the the learn() method should produce any output
        :param y_column: the name of the predicted column
        :param data: DataFrame containing the dataset to learn
        :param callbacks: callbacks to be executed by the model when training it
        :param train_time: time in seconds for the training session; if not specified the time in config is used
        :return: trained model or None if trained is not set to true in config
        :raises: Pipeline exception
        """
        if callbacks is None:
            callbacks = []

        start = time.time()
        self._record_data_information(data, "learn")

        if y_column is None:
            y_column = self._config.get("TRAINING_CONFIG", {}).get("PREDICTED_COLUMN_NAME", "undefined")

        result = None
        # 2. Model learning
        if self._config.get("TRAINING", False):
            x, y = Splitter.XYsplit(data, y_column)
            self._mapper.set("X_COLUMNS_TRAIN", list(x.columns))

            try:
                result = self._learner.learn(X=x, Y=y, verbose=verbose, callbacks=callbacks, time=train_time)
            except Exception as err:
                # TODO: add logs
                raise PipelineException("Learn error: {}.".format(err))

        end = time.time()
        print("Learnt in {0:.4f} seconds.".format(end - start)) if verbose else None
        self._model = result
        self._mapper.set(self.STATE_MACRO, self.LEARNT_STATE)
        return result

    def _copy_columns(self, X: DataFrame, columns: list = None) -> DataFrame:
        """
            Makes a copy of the columns marked in columns and returns it

        :param X: the data to be passed to the model
        :param columns: list of columns to be discarded
        :return: X without the marked columns
        """
        if columns is None:
            return X

        to_discard_valid = []
        for col in columns:
            if col in X.columns:
                to_discard_valid.append(col)

        if len(to_discard_valid) == 0:
            return None

        return X.loc[:, to_discard_valid]

    def _append_discarded_columns(self, X: DataFrame, columns: DataFrame) -> DataFrame:
        """
            Append the previously cached columns to the dataset X
        :param X: data frame to have data appended to
        :return: the merged data frame
        """
        merged_data = concat([columns, X], axis=1)
        return merged_data

    def predict(self, data: DataFrame, verbose: bool = False, discard_columns: list = None) -> DataFrame:
        """
            Predicts the output of the data using a previously learnt model.
        :param discard_columns: list with columns names that will be copied from the data frame and
                                    appended to the prediction
        :param verbose: decide is the predict method should output information to the console
        :param data: DataFrame with the x values to be predicted
        :return: DataFrame with the predicted values
        :raises PipelineException: when no model has been previously learnt
        """
        if self._model is None:
            raise PipelineException("Could not predict unless a training has been previously done.")

        if discard_columns is None:
            discard_columns = []

        y_column = self._config.get("DATA_PROCESSING_CONFIG", {}).get("PREDICTED_COLUMN_NAME", "")
        if y_column in data.columns:
            data = data.drop(y_column, axis=1)

        discarded_data = self._copy_columns(data, discard_columns)

        columns = list(data.columns)
        columns.sort()

        learnt_columns = self._mapper.get("X_COLUMNS_TRAIN", [])
        learnt_columns.sort()

        # in the unlikely case that the data contains the predicted column, remove it

        if columns == learnt_columns:  # the columns to predicts are the learnt columns
            self._mapper.set(self.STATE_MACRO, self.PREDICTED_STATE)
            prediction = self._model.predict(data)
            return self._append_discarded_columns(prediction, discarded_data)

        elif self._mapper.get("CONVERSION_DONE", False):  # the columns differ (maybe conversion has to be done)
            processed_cols = self._mapper.get("COLUMNS_PROCESS", [])
            processed_cols.sort()

            converted = self.convert(data)
            self._mapper.set(self.STATE_MACRO, self.PREDICTED_STATE)
            prediction = self._model.predict(converted)
            return self._append_discarded_columns(prediction, discarded_data)

        else:
            raise PipelineException("Expected model with columns {}; received {}"
                                    .format(self._mapper.get("X_COLUMNS_TRAIN", []), list(data.columns)))

    def fit(self, data: DataFrame, verbose: bool = True, training_callbacks: list = None):
        """
            Completes the pipeline as specified in the configuration file.

        :param verbose: decides if the method fit() and all the methods called in it should produce any output
        :param data: DataFrame with raw data
        :param training_callbacks: callbacks to be executed by the model when training it
        :return: data/ cleaned data/ processed data/ trained model ( based on the choices in the config file)
        """
        if training_callbacks is None:
            training_callbacks = []


        # Iterating over the pipeline steps
        # 1. Data processing
        result = self.process(data, verbose=verbose, callbacks=training_callbacks)

        # If data processing is activated and if the method has the dataConversionCallback passed as an argument,
        # return the intermediary data preview
        if self._config.get("DATA_PROCESSING", False):
            for callback in training_callbacks:
                if type(callback) is DataConversionCallback:
                    callback({
                        "data": result.head(10).to_dict()
                    })
                    break

        # 2. Learning
        result = self.learn(result, verbose=verbose, callbacks=training_callbacks)

        return result

    def __call__(self, data: DataFrame, verbose: bool = True):
        """
            Calls the fit method by calling the pipeline.
        :param verbose: decides if the call on a pipeliene should produce any output
        :param data: DataFrame with raw data
        :return: data/ cleaned data/ processed data/ trained model ( based on the choices in the config file)
        """
        if self.DYNAMIC_CALL is False:
            return self.fit(data, verbose=verbose)

        else:  # decide what to do depending on the state
            metadata = self._mapper.get("DATA_METADATA", {})
            state = self._mapper.get(self.STATE_MACRO, self.RAW_STATE)

            if state == self.RAW_STATE:  # follow the configuration
                print("Pipeline Dynamic Call: fit()") if verbose else None
                return self.fit(data, verbose=verbose)

            if data.shape == metadata.get("process", {}).get("shape", ()):  # probably a conversion is wanted
                if state == self.LEARNT_STATE:
                    print("Pipeline Dynamic Call: learn()") if verbose else None
                    return self.learn(data, metadata.get("learn", {}).get("y_column", "undefined"))

                else:
                    print("Pipeline Dynamic Call: convert()") if verbose else None
                    return self.convert(data, verbose=verbose)

            elif data.shape[1] == metadata.get("process", {}).get("shape", (-1, -1))[1] - 1:
                # if a model is present -> prediction ; else -> conversion
                if self._model is None:
                    print("Pipeline Dynamic Call: convert()") if verbose else None
                    return self.convert(data, verbose=verbose)

                else:
                    print("Pipeline Dynamic Call: fit()") if verbose else None
                    return self.predict(data, verbose=verbose)

            return self.fit(data, verbose=verbose)

    def save(self, file: str, include_model: bool = True) -> 'Pipeline':
        """
            Saves the pipeline logic to the specified file for further re-usage.
        :param file: the path(or name) of the save file
        :param include_model: default true | decides whether or not the model is included in the save file with the pipeline
        :return: None
        """
        # save the initial configuration for further operations on the pipeline
        self._mapper.set("CONFIG", self._config)

        # save the processor mapper for further data processing or conversion
        self._mapper.set_mapper(self._processor.get_mapper(), "PROCESSOR_DATA")

        # save the model to file
        model_map = None
        if include_model and (not (self._model is None)):
            model_map = self._model.to_dict()

        self._mapper.set("MODEL", model_map)

        # save the mapper to file
        self._mapper.save_to_file(file)

        return self

    def get_save_map(self, include_model: bool = True) -> dict:
        """
            Returns the dictionary to be save to a binary file with pickle.dump()/.dumps()
        :return: dictionary with data
        """
        self._mapper.set("CONFIG", self._config)

        # save the processor mapper for further data processing or conversion
        self._mapper.set_mapper(self._processor.get_mapper(), "PROCESSOR_DATA")

        # save the model to file
        model_map = None
        if include_model and (not (self._model is None)):
            model_map = self._model.to_dict()

        self._mapper.set("MODEL", model_map)

        return self._mapper.get_map()

    def get_model(self) -> AbstractModel:
        """
            Returns the trained model or None is no training has been done
        :return:
        """
        return self._model

    @staticmethod
    def _read_config_file(path: str = None) -> dict:
        """
            Reads the default configuration file
        :param path: the explicit path for the configuration file
        :return: dictionary with the encodings
        """
        if path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(dir_path, 'config.json')

        if not os.path.exists(path):
            raise PipelineException("Configuration Json file could not be parsed from source {}.".format(path))

        # print(path)
        with open(path) as json_file:
            data = json.load(json_file)
        return data
