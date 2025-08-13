from pandas import DataFrame, get_dummies
import pandas as pd
import numpy as np

from ...Mapper import Mapper
from sklearn import preprocessing as pp
from ...Exceptions import DataEngineeringException

import spacy
from nltk import word_tokenize


class Engineer:
    """
        Class responsible with feature engineering.
        Handles continuous and categorical features, both numeric and textual

        Methods:
            - process: processes a dataframe with the rules form the config file
            - convert: converts a dataset with the rules used before for process
    """

    def __init__(self, config: dict = None):
        """
            Inits a Engineer object.
        :param config: dictionary with the configuration for the Engineer class
                        expected to receive the FEATURE_ENGINEERING_CONFIG part of the config file
        """
        if config is None:
            config = {}
        self._config = config
        self._mapper = Mapper("Engineer")
        self._numeric_dtypes = ["float64", "int64", "bool", "float32", "int32", "int8", "float8", 'uint8', 'uint32']
        self._textual_dtypes = ["object", "string"]

        # loading libraries for natural language processing
        self._nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def process(self, data: DataFrame, mapper: 'Mapper', column_type: dict = None, verbose: bool = True,
                callbacks: list = None) -> DataFrame:
        """
            Processes the dataset in a way that a learning algorithms can benefit more from it.
            Does outlier detection, feature engineering, data normalization/standardization, missing value filling, polynomial features and more.
        :param callbacks: list of AbstractCallback instances that might be called later
        :param verbose: defines if the process() method will print output to stdout
        :param data: DataFrame consisting of the data
        :param mapper: parent mapper that keeps track of changes
        :param column_type: describes whether features are continuous or discrete in form of a dictionary
                            (if not provided, the algorithm will try to figure out by itself - may reduce overall performance)
        :return: processed data in form of DataFrame
        """

        if column_type is None:
            column_type = {}

        if callbacks is None:
            callbacks = []

        # iterate through each column and process it according to it's type
        modified_data = DataFrame()
        data_types = data.dtypes
        not_processed = []

        for column in data.columns:

            dtype = data_types[column]

            if column in self._config.get("DO_NOT_PROCESS", []):
                interm_data = data[[column]]
                not_processed.append(column)
            else:
                print("Engineer: process column {}.".format(column)) if verbose else None
                for callback in callbacks:
                    callback({
                        "message": "Engineer: process column {}.".format(column)
                    })

                if str(dtype) in self._numeric_dtypes:
                    col_type = column_type.get(column, 'undefined')
                    interm_data = self._process_numeric(data.loc[:, [column]], column, col_type)

                elif str(dtype) in self._textual_dtypes:
                    col_type = column_type.get(column, 'undefined')
                    interm_data = self._process_text(data.loc[:, [column]], column, col_type)

                else:
                    raise DataEngineeringException("Unknown column type {}".format(str(dtype)))

            if not (interm_data is None):
                modified_data = pd.concat([modified_data, interm_data],
                                          axis=1)  # add the modified data to the new dataframe

        self._mapper.set("NOT_PROCESSED", not_processed)
        mapper.set_mapper(self._mapper)
        return modified_data

    """""""""""""""""""""""""""""""""""""""""""""   Numeric processing    """""""""""""""""""""""""""""""""""""""""""""

    def _process_numeric(self, data: DataFrame, column_name: str, column_type: str) -> DataFrame:
        """
            Processes a numeric column; Does nan replacing and feature engineering as specified in the config file.
        :param data: DataFrame containing the column to process
        :param column_name: string containing the column name
        :param column_type: 'discrete' or 'continuous' or 'undefined'; if 'undefined' the method will try to determine which
                                of the first 2 better suits the column
        :return: DataFrame containing the processed data
        """
        result = data

        if column_type == 'undefined':  # determine data distribution type
            column_type = self._determine_cont_discrete(data, column_name)

        if column_type == 'continuous':  # process continuous data
            result = self._process_numeric_continuous(data, column_name)
        else:  # process discrete(categorical) data
            result = self._process_numeric_discrete(data, column_name)

        return result

    def _process_numeric_continuous(self, data: DataFrame, column_name: str) -> DataFrame:
        """
            Processes a numeric column which is known to have a continuous values distribution.
            1. Capping outliers which are more/less than x/-x standard deviations from the mean (x from config file)
            2. Get default value
            3. Fill nans with default
            4. Polynomial features
            5. Normalization, standardization

        :param data: DataFrame with the column to be modified
        :param column_name: string representing the column name
        :return:
        """
        column_meta_data = {"distribution": "continuous", "data_type": "numeric", "name": column_name}

        if self._config.get("CONTINUOUS_DATA_CONFIG", {}).get("NUMERIC", {}).get("NOT_PROCESS",
                                                                                 False):  # explicitly marked to not be processed
            column_meta_data["not_process"] = True
            self._mapper.set(column_name, column_meta_data)
            return data

        # 1. Capping outliers
        stdev_from_mean = self._config.get("CONTINUOUS_DATA_CONFIG", {}).get("NUMERIC", {}).get(
            "OUTLIER_STDEV_FROM_MEAN", 3)
        lower_limit = data[column_name].mean() - stdev_from_mean * data[column_name].std()
        upper_limit = data[column_name].mean() + stdev_from_mean * data[column_name].std()

        data[column_name][data[column_name] < lower_limit] = lower_limit
        data[column_name][data[column_name] > upper_limit] = upper_limit

        column_meta_data["lower_limit"] = lower_limit
        column_meta_data["upper_limit"] = upper_limit

        # 2. Get default value
        default_value = data[column_name].median()
        column_meta_data["default_value"] = default_value

        # 3. Fill nans with default
        data = data[[column_name]].fillna(default_value)

        # 4. Polynomial features
        polynomial_features = self._config.get("CONTINUOUS_DATA_CONFIG", {}).get("NUMERIC", {}).get(
            "POLYNOMIAL_FEATURES", 3)
        column_meta_data["polynomial_features"] = polynomial_features
        poly = pp.PolynomialFeatures(polynomial_features, include_bias=False)
        poly_data = poly.fit_transform(data)
        poly_feature_names = poly.get_feature_names([column_name])

        data = DataFrame(poly_data, columns=poly_feature_names)

        # 5. Normalization, standardization
        method = self._config.get("CONTINUOUS_DATA_CONFIG", {}).get("NUMERIC", {}).get("NORMALIZATION_METHOD", None)
        if method == "min_max":
            max_values = data.max()
            min_values = data.min()

            column_meta_data['normalization_method'] = method
            column_meta_data['normalization_max'] = max_values
            column_meta_data['normalization_min'] = min_values

            data = (data - min_values) / (max_values - min_values)

        elif method == "z_score":
            mean = data.mean().tolist()
            std = data.std().to_list()

            column_meta_data['normalization_method'] = method
            column_meta_data['normalization_mean'] = mean
            column_meta_data['normalization_std'] = std

            any_std_0 = False
            for s in std:
                if s == 0:
                    any_std_0 = True
                    break

            if any_std_0 is False:
                data = (data - mean) / std
        else:
            column_meta_data['normalization_method'] = 'none'

        self._mapper.set(column_name, column_meta_data)
        return data

    def _process_numeric_discrete(self, data: DataFrame, column_name: str) -> DataFrame:
        """
            Processes a numeric column which is known to have a discrete values distribution.
            Applies one hot encoding and returns the result.
        :param data: column to analyse; assumes that no value is nan
        :param column_name: the name of the column to process
        :return: DataFrame containing the processed data
        """
        column_meta_data = {"distribution": "discrete", "data_type": "numeric", "name": column_name}

        if self._config.get("CATEGORICAL_DATA_CONFIG", {}).get("NUMERIC", {}).get("NOT_PROCESS",
                                                                                  False):  # explicitly marked to not be processed
            column_meta_data["not_process"] = True
            self._mapper.set(column_name, column_meta_data)
            return data
        if self._config.get("CATEGORICAL_DATA_CONFIG", {}).get("NUMERIC", {}).get("METHOD",
                                                                                  "one_hot_encode") == 'one_hot_encode':
            column_meta_data["method"] = "one_hot_encode"
            data = get_dummies(data, prefix=column_name, columns=[column_name])
            column_meta_data['onehotencoded_names'] = data.columns.to_list()

        self._mapper.set(column_name, column_meta_data)
        return data

    """""""""""""""""""""""""""""""""""""""""""""   Text processing    """""""""""""""""""""""""""""""""""""""""""""

    def _process_text(self, data: DataFrame, column_name: str, column_type: str) -> DataFrame:
        """
            Processes a text column; Does nan replacing and feature engineering as specified in the config file.
        :param data: DataFrame containing the column to process
        :param column_name: string with the column name
        :param column_type: 'discrete' or 'continuous' or 'undefined'; if 'undefined' the method will try to determine which
                                of the first 2 better suits the column
        :return: DataFrame containing the processed data
        """
        result = data

        if column_type == 'undefined':  # determine data distribution type
            column_type = self._determine_cont_discrete(data, column_name)

        if column_type == 'continuous':  # process continuous data
            result = self._process_text_continuous(data, column_name)
        else:  # process discrete(categorical) data
            result = self._process_text_discrete(data, column_name)

        return result

    def _process_text_discrete(self, data: DataFrame, column_name: str) -> DataFrame:
        """
            Processes a text column which is known to have a discrete values distribution.
            Applies one hot encoding and returns the result.
        :param data: column to analyse; assumes that no value is nan
        :param column_name: the name of the column to process
        :return: DataFrame containing the processed data
        """
        column_meta_data = {"distribution": "discrete", "data_type": "text", "name": column_name}

        if self._config.get("CATEGORICAL_DATA_CONFIG", {}).get("TEXTUAL", {}).get("NOT_PROCESS",
                                                                                  False):  # explicitly marked to not be processed
            column_meta_data["not_process"] = True
            self._mapper.set(column_name, column_meta_data)
            return data

        if self._config.get("CATEGORICAL_DATA_CONFIG", {}).get("TEXTUAL", {}).get("METHOD",
                                                                                  "one_hot_encode") == 'one_hot_encode':
            column_meta_data["method"] = "one_hot_encode"
            data = get_dummies(data, prefix=column_name, columns=[column_name])
            column_meta_data['onehotencoded_names'] = data.columns.to_list()

        self._mapper.set(column_name, column_meta_data)
        return data

    def _process_text_continuous(self, data: DataFrame, column_name: str) -> DataFrame:
        """
            Processes a text column the is known to have a continuous distribution, thus
        no categorical values.
            Does feature stemming and lemmatization, and then marks occurences of specific
        tokens in the entries in the data.

        :param data: DataFrame containing the column to be parsed
        :param column_name: string: the name of the column to be parsed
        :return: DataFrame with the embedding
        """
        column_meta_data = {"distribution": "continuous", "data_type": "text", "name": column_name}

        if self._config.get("CONTINUOUS_DATA_CONFIG", {}).get("TEXTUAL", {}).get("NOT_PROCESS",
                                                                                 False):  # explicitly marked to not be processed
            column_meta_data["not_process"] = True
            self._mapper.set(column_name, column_meta_data)
            return data

        # fill nan's
        data = data[[column_name]].fillna("")
        column_meta_data['default_value'] = ""

        # determine the set of values across al the columns and their frequencies
        word_frequency = {}  # the frequency of every token
        word_occurence = {}  # at which lines is each token located

        for row in range(data.shape[0]):
            text = data.at[row, column_name]
            tokens = self._get_world_cloud(text)
            for token in tokens:
                word_frequency[token] = 1 + word_frequency.get(token, 0)
                word_occurence[token] = [row] + word_occurence.get(token, [])

        # sort the items after their frequency
        words = []
        for key in word_frequency.keys():
            words.append((key, word_frequency[key]))

        words.sort(key=lambda pair: -pair[1])

        # create the feature columns
        words_to_consider = min(len(words), self._config.get("CONTINUOUS_DATA_CONFIG", {}).get("TEXTUAL", {}).get(
            "MAX_GENERATED_FEATURES", 32))
        column_names = []
        considered_words = []
        for item in words[:words_to_consider]:
            considered_words.append(item[0])
            column_names.append(column_name + "_" + item[0])

        # mark in the mapper the correspondence between words and new columns
        word_embedding = {}
        for word, feature in zip(considered_words, column_names):
            word_embedding[word] = feature

        column_meta_data['word_embedding'] = word_embedding
        # column_meta_data['words_to_map'] = considered_words
        column_meta_data['column_names'] = column_names
        column_meta_data['word_delimiter'] = self._config.get("CONTINUOUS_DATA_CONFIG", {}).get("TEXTUAL`", {}).get(
            "WORD_DELIMITERS",
            "?!|/.,:;'-={}[]()")

        data = pd.DataFrame(0, index=np.arange(data.shape[0]), columns=column_names)

        for item in words[:words_to_consider]:
            word = item[0]
            for line in word_occurence[word]:
                data.iloc[line][column_name + "_" + word] = 1

        self._mapper.set(column_name, column_meta_data)
        return data

    def _get_world_cloud(self, sentence: str, nlp=None, separators: str = None) -> list:
        """
            Converts text format to a standardized format through tokenizing and stemming.
        :param sentence: the string to be analyzed
        :return: the list of tokens
        """
        # self._nlp = spacy.load('en', disable=['parser', 'ner'])
        # self._wordnet_lemmatizer = WordNetLemmatizer()
        if sentence == "nan":
            return []

        if nlp is None:
            nlp = self._nlp

        # tokenize with nltk
        tokens = word_tokenize(sentence)
        clean_sentence = ""
        if separators is None:
            separators = self._config.get("CONTINUOUS_DATA_CONFIG", {}).get("TEXTUAL`", {}).get("WORD_DELIMITERS",
                                                                                                "?!|/.,:;'-={}[]()")

        for word in tokens:
            if not (word in separators):
                clean_sentence += (word.lower() + " ")

        # lemmatize with spacy
        doc = nlp(clean_sentence)

        words = []
        for word in doc:
            lemma = word.lemma_
            if lemma != "-PRON-" and not (lemma in separators):
                words.append(lemma)

        return list(set(words))

    def _determine_cont_discrete(self, data: DataFrame, column_name: str) -> str:
        """
            Determines heuristically if the data withing the DataFrames is continuous or discrete
            If column explicitly marked as continuous or discrete, it will be considered as specified
        :param data: DataFrame containing the data to analyse
        :return: string 'continuous' or 'discrete' depending on the result
        """
        # 1. Check if the column was marked explicitly as continuous or discrete
        if column_name in self._config.get("CONTINUOUS_FEATURES", []):
            return 'continuous'
        elif column_name in self._config.get("CATEGORICAL_FEATURES", []):
            return 'discrete'

        # 2. If not explicitly marked, determine heuristically
        categorical_threshold = self._config.get("CATEGORICAL_THRESHOLD", 0)
        ratio = 1. * data[column_name].nunique() / data[column_name].count()

        if ratio < categorical_threshold:
            return 'discrete'
        else:
            return 'continuous'

    """""""""""""""""""""""""""""""""""""""""""""   Conversion methods    """""""""""""""""""""""""""""""""""""""""""""

    @staticmethod
    def convert(data: DataFrame, mapper: 'Mapper', verbose: bool = True) -> DataFrame:
        """
            Converts new data to a format previously mapped into 'mapper'
        :param verbose: decides if the convert() method will produce any output
        :param data: DataFrame with data to be transformed
        :param mapper: Mapper class containing the rules for transformation.
        :return: converted data in form of DataFrame
        """
        mapper = mapper.get_mapper("Engineer")

        processed_data = DataFrame()

        for column in data.columns:

            interm_data = None
            if column in mapper.get("NOT_PROCESSED", []):
                interm_data = data[[column]]
            else:
                print("Engineer: convert column {}.".format(column)) if verbose else None
                column_data = mapper.get(column, None)
                if column_data is None:
                    raise DataEngineeringException(
                        "Could not convert feature {}. Not found in the mapper dictionary.".format(column))

                # else
                if column_data.get("data_type") == "numeric":
                    interm_data = Engineer._convert_numeric(data.loc[:, [column]], column_data)
                else:  # text
                    interm_data = Engineer._convert_text(data.loc[:, [column]], column_data)

            processed_data = pd.concat([processed_data, interm_data], axis=1)

        return processed_data

    @staticmethod
    def _convert_text(data: DataFrame, info: dict) -> DataFrame:
        """
            Converts textual data as specified in the info dictionary
        :param data:
        :param info:
        :return:
        """
        if info.get("not_process", False):
            return data

        if info.get("distribution") == "discrete":
            # noinspection DuplicatedCode
            if info.get("method") == "one_hot_encode":
                new_data = pd.DataFrame(0, index=np.arange(data.shape[0]), columns=info.get("onehotencoded_names"))
                for i, r in data[[info.get("name")]].iterrows():
                    if str(info.get("name") + "_" + str(r[0])) in new_data.columns:
                        new_data.iloc[i][str(info.get("name") + "_" + str(r[0]))] = 1

                data = new_data

        else:  # continuous
            data = data[[info.get("name")]].fillna(info.get("default_value", ""))

            nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            word_delimiter = info.get('word_delimiter')
            columns = info.get("column_names")
            new_data = pd.DataFrame(0, index=np.arange(data.shape[0]), columns=columns)
            for row in range(data.shape[0]):
                text = data.at[row, info.get("name")]
                tokens = Engineer._get_world_cloud(None, text, nlp, separators=word_delimiter)
                for token in tokens:
                    if not (info.get("word_embedding").get(token) is None):
                        new_data.iloc[row][info.get("word_embedding").get(token)] = 1

            data = new_data

        return data

    @staticmethod
    def _convert_numeric(data: DataFrame, info: dict) -> DataFrame:
        """
            Convert numeric features as specified in the info dictionary
        :param info:
        :return:
        """
        if info.get("not_process", False):
            return data

        if info.get("distribution") == "discrete":
            # noinspection DuplicatedCode
            if info.get("method") == "one_hot_encode":
                new_data = pd.DataFrame(0, index=np.arange(data.shape[0]), columns=info.get("onehotencoded_names"))
                for i, r in data[[info.get("name")]].iterrows():
                    if str(info.get("name") + "_" + str(r[0])) in new_data.columns:
                        new_data.iloc[i][str(info.get("name") + "_" + str(r[0]))] = 1

                data = new_data

        else:  # continuous
            # 1. Capping outliers
            column_name = info.get("name")
            lower_limit = info.get("lower_limit")
            upper_limit = info.get("upper_limit")

            data[column_name][data[column_name] < lower_limit] = lower_limit
            data[column_name][data[column_name] > upper_limit] = upper_limit

            # 2. Fill nans with default
            default_value = info.get("default_value")
            data = data[[column_name]].fillna(default_value)

            # 3. Polynomial features
            polynomial_features = info.get("polynomial_features")
            poly = pp.PolynomialFeatures(polynomial_features, include_bias=False)
            poly_data = poly.fit_transform(data)
            poly_feature_names = poly.get_feature_names([column_name])

            data = DataFrame(poly_data, columns=poly_feature_names)

            # 4. Normalization, standardization
            method = info.get("normalization_method")

            if method == "min_max":
                max_values = info.get("normalization_max")
                min_values = data.min("normalization_min")

                data = (data - min_values) / (max_values - min_values)

            elif method == "z_score":
                mean = info.get('normalization_mean')
                std = info.get('normalization_std')

                data = (data - mean) / std

        return data
