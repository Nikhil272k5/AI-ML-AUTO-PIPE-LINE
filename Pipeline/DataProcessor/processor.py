import pandas as pd
from pandas import DataFrame, get_dummies
from pandas import concat
import numpy as np

from .DataCleaning import Cleaner
from .DataSplitting import Splitter
from .FeatureEngineering import Engineer
from ..Mapper import Mapper
from ..Exceptions.dataProcessorException import DataProcessorException


class Processor:
    """
        Data processing module; the first component of the pipeline.
        Converts data from raw format to a representation that can understood by further learning algorithms.
        Unless a configuration is passed as an argument the default one is used.


        Methods:
            - process: processes a raw dataset according to the configurations
            - convert: based on a previous conversion mapping it converts a new dataset with the same logic
            - get_mapper: returns a mapper with all the relevant data for file saving
            - save_processor: save the processor data to file
            - load_processor: static method for creating a Processor instance from a previous save file
    """

    def __init__(self, config: dict = None, file: str = None, data: 'Mapper' = None):
        """
            Inits the data processor with the configuration parsed from the json file
            Usage: pass a mapper dictionary and the processor will init itself from that
                   otherwise, pass the configuration dictionary and, optionally, a file with the saved mapper
        :param config: configuration dictionary that contains the logic of processing data
        :param file: the file where the processor mapper has been previously saved
        :param data: the mapper data already parsed into a dictionary
        """
        if data is None:
            if file is None:
                self._mapper = Mapper("Processor")  # maps the changes in the raw data, for future prediction tasks
            else:
                self._mapper = Mapper("Processor", file=file)

            if config is None:
                config = self._mapper.get("PROCESSOR_CONFIG", {})
            self._config = config

        else:
            if config is None:
                config = {}
            self._mapper = Mapper("Processor", dictionary=data.get_map())
            self._config = self._mapper.get("PROCESSOR_CONFIG", config)

    def get_mapper(self) -> 'Mapper':
        """
            Returns the mapper for saving purposes.
        :return: Mapper
        """
        return self._mapper

    def process(self, data: DataFrame, verbose: bool = True, callbacks: list = None) -> DataFrame:
        """
            Completes the whole cycle of automated feature engineering.
            Received raw data and returns data ready to be fed into the next step of the pipeline or
        into a learning algorithm.
        :param callbacks: list of AbstractCallback instances that might get called later
        :param verbose: decides if the process() method will produce any output
        :param data: Raw data input, in form of DataFrame
        :return: cleaned and processed data, in form of DataFrame
        :exception: DataProcessorException
        """

        if self._config.get("NO_PROCESSING", True):  # no processing configured in the configuration file
            self._mapper.set("NO_PROCESSING", True)
            return data

        # go over all the steps in the data processing pipeline

        # 1. Data cleaning
        if self._config.get("DATA_CLEANING", False):  # data cleaning set to be done
            self._mapper.set("DATA_CLEANING", True)
            cleaner = Cleaner(self._config.get("DATA_CLEANING_CONFIG", {}))
            y_column = self._config.get('PREDICTED_COLUMN_NAME', None)
            data = cleaner.clean(data, self._mapper, y_column, verbose=verbose)

        # 2. Data splitting
        y_column = self._config.get('PREDICTED_COLUMN_NAME', None)
        self._mapper.set("Y_COLUMN_NAME", y_column)
        result = Splitter.XYsplit(data, y_column)
        if result is None:
            raise DataProcessorException("Expected (X,Y) tuple of DataFrames from XYsplit but got None instead")

        X, Y = result  # init the X and Y variables

        # 3. Feature engineering
        if self._config.get("FEATURE_ENGINEERING", False):  # feature engineering set to be done
            self._mapper.set("FEATURE_ENGINEERING", True)

            engineer = Engineer(self._config.get("FEATURE_ENGINEERING_CONFIG", {}))
            X = engineer.process(X, self._mapper, {}, verbose=verbose, callbacks=callbacks)

        # 4. Retrieve mappings
        # mappings are already in the mapper field, which would be saved to file as soon as the save_processor is called

        # 5. Process Y column: if it is categorical and it is set to be processed, one_hot_encode it
        Y = self._process_Y_column(Y, self._config.get('PREDICTED_COLUMN_NAME', None))

        # 6. Create the output
        data = concat([X, Y], axis=1)

        return data

    def _process_Y_column(self, data: DataFrame, column_name: str) -> DataFrame:
        """
            Processes Y column only if it is categorical and if it set to do so
        :param data: DataFrame containing the predicted column
        :return:
        """
        self._mapper.set("PROCESSED_Y", False)
        if self._config.get("FEATURE_ENGINEERING_CONFIG", {}).get("PROCESS_CATEGORICAL_PREDICTED_COLUMN", False):
            categorical_threshold = self._config.get("FEATURE_ENGINEERING_CONFIG", {}).get("CATEGORICAL_THRESHOLD", 0)
            ratio = 1. * data[column_name].nunique() / data[column_name].count()

            distribution = None
            if ratio < categorical_threshold:
                distribution = 'discrete'
            else:
                distribution = 'continuous'

            if distribution == 'discrete':
                self._mapper.set("PROCESSED_Y", True)
                data = get_dummies(data, prefix=column_name, columns=[column_name])
                self._mapper.set('Y_NEW_NAMES', data.columns.to_list())

        return data

    def convert(self, data: DataFrame, verbose: bool = True) -> DataFrame:
        """
            Converts data to a format previously determined by the process method.
            Used after data processing for further predictions.
        :param verbose: decides whether the convert() method will produce any output
        :param data: Raw data input for prediction purpose
        :return: data transformed in a format previously determined by the logic within process method
        :exception:
        """
        if self._mapper.get("NO_PROCESSING", False):
            return data

        # go over all the steps in the data processing pipeline
        # 1. Data cleaning
        if self._mapper.get("DATA_CLEANING", False):  # data cleaning set to be done
            data = Cleaner.convert(data, self._mapper)

        # 2. if the y column is present, split after it
        split = False
        y_column = self._mapper.get("Y_COLUMN_NAME", "")
        if y_column in data.columns:
            split = True

            result = Splitter.XYsplit(data, y_column)
            if result is None:
                raise DataProcessorException("Expected (X,Y) tuple of DataFrames from XYsplit but got None instead")

            data, Y = result  # init the X and Y variables

        # 3. Feature engineering
        if self._mapper.get("FEATURE_ENGINEERING", False):  # feature engineering set to be done
            data = Engineer.convert(data, self._mapper, verbose=verbose)

        # 4. Process y column
        if split:
            Y = self._convert_Y_column(Y, y_column)
            data = concat([data, Y], axis=1)

        return data

    def _convert_Y_column(self, data: DataFrame, y_column: str) -> DataFrame:
        """
            Converts the y column as specified in the mapper
        :param data: DataFrame containing the y column
        :param y_column: string with the column name
        :return: DataFrame with the converted data
        """
        if self._mapper.get("PROCESSED_Y", False):
            new_data = pd.DataFrame(0, index=np.arange(data.shape[0]), columns=self._mapper.get("Y_NEW_NAMES"))
            for i, r in data[[y_column]].iterrows():
                if str(y_column + "_" + str(r[0])) in new_data.columns:
                    new_data.iloc[i][str(y_column + "_" + str(r[0]))] = 1

            data = new_data

        return data

    def save_processor(self, file: str) -> None:
        """
            Saves the processor logic to disc.
        :param file: text file for saving the data
        :return: None on error | processor on success for chaining reasons
        :exception: DataProcessorException
        """
        try:
            self._mapper.set("PROCESSOR_CONFIG", self._config)
            self._mapper.save_to_file(file)
        except Exception as e:
            raise DataProcessorException("Error while saving processor to file {}.Base error: {}.".format(file, e))

    @staticmethod
    def load_processor(file: str) -> 'Processor':
        """
        :param file: the file where a processor has been previously saved with the save_processor method
        :return: the instance of a processor class with the logic within the file
        :exception:
        """
        # the file contains the mapper, and withing the mapper it already exists a configuration
        return Processor(file=file)
