import pickle
import warnings
from abc import ABC, abstractmethod
from pandas import DataFrame, concat
from sklearn.metrics import log_loss, \
    mean_absolute_error, mean_squared_error, mean_squared_log_error

from .constants import CLASSIFICATION, REGRESSION
from ...Exceptions import AbstractModelException


class AbstractModel(ABC):
    """
        The end result of the pipeline.
        It's main task is to predict after a training session has been done.

        Methods:
            - train: trains the actual model based on a dataset
            - predict: predicts the output of a dataset
            - to_dict: returns a serializable dictionary
            - save: saves the model to file
            - model_type: returns the model type, as defined in "SpecializedModel/modelTypes.py"

        Behaviour:
            - calling an object ( model_instance(data) ), will return the prediction
    """
    ACCEPTED_CLASSIFICATION_METRICS = ["BCE", "CrossEntropy", "LogLikelihood"]
    ACCEPTED_REGRESSION_METRICS = ["mean_absolute_error", "MSE", "mean_squared_log_error"]

    METRICS_TO_FUNCTION_MAP = {
        "BCE": log_loss,
        "CrossEntropy": log_loss,
        "LogLikelihood": log_loss,
        "PoissonLogLikelihood": log_loss,

        "mean_absolute_error": mean_absolute_error,
        "MSE": mean_squared_error,
        "mean_squared_log_error": mean_squared_log_error
    }

    def __init__(self):
        """
            Initializes an abstract model
        """
        self._discarded_column_names = []
        self._discarded_data = None


    def _discard_columns(self, X: DataFrame, columns: list = None, caching: bool = False) -> DataFrame:
        """
            Removes the columns marked explicitly to be removed in columns
            If caching is enabled, the removed columns are saved to the class so they can be appended
        later; useful in case of prediction
        :param X: the data to be passed to the model
        :param columns: list of columns to be removed
        :param caching: decides if the removed columns are saved to be later used
        :return: X without the marked columns
        """
        if columns is None:
            return X

        to_discard_valid = []
        for col in columns:
            if col in X.columns:
                to_discard_valid.append(col)

        if caching:
            self._discarded_data = X.loc[:, to_discard_valid]

        return X.drop(to_discard_valid, axis=1)

    def _append_discarded_columns(self, X: DataFrame) -> DataFrame:
        """
            Append the previously cached columns to the dataset X
        :param X: data frame to have data appended to
        :return: the merged data frame
        """
        if self._discarded_data is None:
            return X

        merged_data = concat([self._discarded_data, X], axis=1)
        self._discarded_data = None  # so memory is freed
        return merged_data

    @abstractmethod
    def get_labels(self) -> list:
        """
            Returns the classification labels it learnt from.
            In case the task is not classification, it returns an empty list.
        :return: list with text labels
        """

    def train(self, X: DataFrame, Y: DataFrame, train_time: int = 600, validation_split: float = 0.2,
              callbacks: list = None, verbose: bool = True):
        """
            Trains the model with the data provided.

        :param validation_split: percentage of the data to be used in validation; None if validation should not be used
        :param callbacks: a list of predefined callbacks that get called at every epoch
        :param train_time: time of the training session in seconds: default 10 minutes
        :param X: the independent variables in form of Pandas DataFrame
        :param Y: the dependents(predicted) values in form of Pandas DataFrame
        :param verbose: decides whether or not the model prints intermediary outputs
        :raises AbstractModelException: on any actual model training error
                                        on constant value for Y
        :return: the model
        """
        # always sort the columns in alphabetical order, as a rule for both train and predict
        columns = list(X.columns)
        columns.sort()
        X = X[columns]

        # check if there is something to learn; if Y has just one value, throw exception
        if len(Y.drop_duplicates()) == 1:   # the same value for each row
            raise AbstractModelException("Could not train model with constant value for Y.")

        # train the actual model
        try:
            return self._model_train(X, Y, train_time, validation_split=validation_split, callbacks=callbacks,
                                     verbose=verbose)
        except Exception as err:
            raise AbstractModelException(err)

    @abstractmethod
    def _model_train(self, X: DataFrame, Y: DataFrame, train_time: int = 600, validation_split: float = 0.2,
                     callbacks: list = None, verbose: bool = True) -> 'AbstractModel':
        """
            Trains the specific model with the data given. To be implemented in each child class.
        :param validation_split: percentage of the data to be used in validation; None if validation should not be used
        :param callbacks: a list of predefined callbacks that get called at every epoch
        :param train_time: time of the training session in seconds: default 10 minutes
        :param X: the independent variables in form of Pandas DataFrame
        :param Y: the dependents(predicted) values in form of Pandas DataFrame
        :param verbose: decides whether or not the model prints intermediary outputs
        :return: the model
        """

    def predict(self, X: DataFrame, discard_columns: list = None, raw_output: bool = False) -> DataFrame:
        """
            Predicts the output of X based on previous learning

        :param X: DataFrame; the X values to be predicted into some Y Value
        :param discard_columns: the list of column names to discard from the actual prediction
                            if not provided, the discarded columns used in training are used now
        :param raw_output: returns the exact output of the model, without rebasing into the initial classes
        :raises AbstractModelException: on any actual model prediction error
        :return: DataFrame with the predicted data
        """
        # remove columns if necessary
        X = self._discard_columns(X, discard_columns, caching=True)

        # get the prediction
        columns = list(X.columns)
        columns.sort()
        X = X[columns]
        try:
            prediction = self._model_predict(X)
        except Exception as err:
            raise AbstractModelException(err)

        if raw_output:
            return prediction

        # append the removed columns
        prediction = self._append_discarded_columns(prediction)

        # return the result
        return prediction

    @abstractmethod
    def _model_predict(self, X: DataFrame, raw_output: bool = False) -> DataFrame:
        """
            Predicts the output of X based on previous learning. To be implemented in child classes
        :param X: DataFrame; the X values to be predicted into some Y Value
        :param raw_output: returns the exact output of the model, without rebasing into the initial classes
        :return: DataFrame with the predicted data
        """

    def __call__(self, X: DataFrame) -> DataFrame:
        """
            Calls the predict method.
        :param X: data to be predicted
        :return: predicted data
        """
        return self.predict(X)

    def eval(self, X: DataFrame, Y: DataFrame, task: str, metric: str, include_train_stats: bool = False):
        """
            Evaluates the model's performance and returns a score
        :param include_train_stats: decides whether to include the last training's stats or not
        :param task: the task of the model (REGRESSION / CLASSIFICATION)
        :param X: the input dataset
        :param Y: the dataset to compare the prediction to
        :param metric: the metric used
        :raises AbstractModelException: on any actual model prediction error
        :return: the score
        """
        if task not in [CLASSIFICATION, REGRESSION]:
            raise AbstractModelException("Task type {} not understood.".format(task))

        if metric not in self.ACCEPTED_CLASSIFICATION_METRICS + self.ACCEPTED_REGRESSION_METRICS:
            # go with a default metric type
            old_metric = metric
            if task == CLASSIFICATION:
                metric = "BCE"
            else:
                metric = "MSE"
            # TODO write to log file

        if task is REGRESSION and metric not in self.ACCEPTED_REGRESSION_METRICS:
            metric = "MSE"

        if task is CLASSIFICATION and metric not in self.ACCEPTED_CLASSIFICATION_METRICS:
            metric = "BCE"

        scorer = self.METRICS_TO_FUNCTION_MAP[metric]

        pred = self._model_predict(X, raw_output=True)
        pred = pred.reindex(sorted(pred.columns), axis=1)

        # y_true = Y.to_numpy()         # FIXME it seems like the scorer works with DataFrames
        # y_pred = pred.to_numpy()              # change if not working
        try:
            if task is CLASSIFICATION:
                labels = self.get_labels()
                if len(labels) <= 1:
                    labels = None
                score = scorer(Y, pred, labels=labels)
            else:
                score = scorer(Y, pred)
        except Exception as err:
            raise AbstractModelException("Could not score the results.")
        return score

    @abstractmethod
    def to_dict(self) -> dict:
        """
            Returns a dictionary representation that encapsulates all the details of the model
        :return: dictionary with 2 mandatory keys : MODEL_TYPE, MODEL_DATA
        """

    def save(self, file: str):
        """
            Saves the model to file
        :param file: the name of the file or the absolute path to it
        :raises AbstractModelException: on any file saving error
        :return: self for chaining purposes
        """
        try:
            with open(file, 'wb') as f:
                data = self.to_dict()
                pickle.dump(data, f)
            return self
        except Exception as err:
            raise AbstractModelException(err)

    @abstractmethod
    def model_type(self) -> str:
        """
            Returns the model type from available model types in file "model_types.py"
        :return: string with the model type
        """

    @staticmethod
    def _determine_task_type(Y: DataFrame) -> str:
        """
            Determines heuristically the task type given the output variable.
        :return: string from constants.py/AVAILABLE_TASKS with the specific task
        """
        string_dtypes = ["object", "string"]
        data_types = Y.dtypes
        for column in Y.columns:
            dtype = data_types[column]
            if str(dtype) in string_dtypes:
                return CLASSIFICATION

        total_number = len(Y)
        unique_number = len(Y.drop_duplicates(ignore_index=True))

        if unique_number / total_number > 0.08:  # there have to be at least 8% unique values from the total number
            return REGRESSION  # of values in order to be considered regression
        else:
            return CLASSIFICATION

    @staticmethod
    def _categorical_mapping(data: DataFrame) -> dict:
        """
            Checks all the unique columns, creates categorical features and returns mapping
            Return type dict {
                new_col_name : {
                    column1: value1,
                    column2: value2
                    ...
                }
                ...
            }
            The returned type represents how an entry should be in order to be part of one column
        :param data: the output variable to be encoded
        :return: the mapping dictionary
        """
        class_mappings = {}

        uniques = data.drop_duplicates(ignore_index=True)  # get the unique rows

        for row in uniques.iterrows():
            row = row[1]
            values = {}
            for col in data.columns.to_list():  # for each unique row get the values that determine it
                values[col] = row[col]

            new_class_name = '&'.join([key + "_" + str(values[key]) for key in values.keys()])  # get a new class
            # name reuniting all
            class_mappings[new_class_name] = values  # set the values to the new created class name

        return class_mappings

    @staticmethod
    def _to_categorical(data: DataFrame, mapping: dict) -> DataFrame:
        """
            According to the dictionary previously created, returns a converted dataset
        :param mapping: the mapping dictionary created with method _categorical_mapping on similar dataset
        :param data: dataset to be converted
        :return: converted dataset
        """
        new_columns = list(mapping.keys())
        new_values = []

        for row in data.iterrows():  # for each entry in the dataset

            final_column = None
            for possible_col in sorted(new_columns):  # check for every possible column
                ok = True
                for column in list(mapping[possible_col].keys()):  # for evey column, check if it matches the condition
                    if mapping[possible_col][column] != row[1][column]:
                        ok = False
                        break

                if ok:  # if it matches, set the column
                    final_column = possible_col
                    break

            d = {col: 0 for col in new_columns}  # set the row
            if not (final_column is None):
                d[final_column] = 1
            new_values.append(d)

        return DataFrame(new_values)

    @staticmethod
    def _from_categorical(data: DataFrame, mapping: dict) -> DataFrame:
        """
            Based on the mapping computed with _categorial_mapping function on a similar dataset,
        converts the encoded data into initial data
        :param data: dataset to be converted back to the inital form
        :param mapping: the mapping computed with _categorical mapping function
        :return: reverted dataset
        """
        categories = data.idxmax(axis=1)  # get the categories
        return DataFrame([mapping[c] for c in categories])  # easily construct the dataframe from list of
        # dictionaries

    @abstractmethod
    def _description_string(self) -> str:
        """
            Returns the description string for printing on user output.
        :return: string
        """

    def __repr__(self):
        return self._description_string()

    @abstractmethod
    def get_config(self) -> dict:
        """
            Returns the configuration that was used to build the model
        :return: dictionary with the configuration
        """

    @abstractmethod
    def summary(self) -> dict:
        """
            Returns the summary of a model.
            Includes details about the structure of the model (METADATA)
            And also details about the last training session (if available)
            Must contain 3 keys: MODEL_TYPE, METADATA and TRAIN_DATA
        """
