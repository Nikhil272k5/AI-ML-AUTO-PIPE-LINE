import warnings

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from random import randint, random, randrange
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

from ....Exceptions import RandomForestModelException
from ..abstractModel import AbstractModel
from ..modelTypes import RANDOM_FOREST_MODEL
from ..constants import CLASSIFICATION, REGRESSION, AVAILABLE_TASKS


class RandomForestModel(AbstractModel):
    """
        AbstractModel implementation using a random forest actual model.
        The framework used is Sklearn
    """

    def __init__(self, task: str = "", config: dict = None, predicted_name: list = None,
                 dictionary=None):
        """
            Initializes a random forest model.
        :param config: the configuration dictionary: expected to receive the RANDOM_FOREST_CONFIG dictionary
        :param predicted_name: the name of the the predicted column
        :param dictionary:
        """
        AbstractModel.__init__(self)

        if type(dictionary) is dict:  # for internal use;
            self._init_from_dictionary(dictionary)  # load from a dictionary when loading from file the model
            return

        # model metadata
        self._task = task
        self._config = config
        if config is None:
            self._config = {}
        self._predicted_name = predicted_name
        self._criterion = ""

        # created model data
        self._model = None
        self._model_score = None

        # print data
        self._configured = None
        self._n_estimators = None
        self._criterion = None

        # summary data
        self._train_criterion = None
        self._val_criterion = None

    # noinspection DuplicatedCode
    def _model_train(self, X: DataFrame, Y: DataFrame, train_time: int = 600, callbacks: list = None,
                     validation_split: float = 0.2, verbose: bool = True) -> 'AbstractModel':
        """
            Trains the model with the data provided.
        :param validation_split: how much from the data(as percentage) should be used as validation
        :param callbacks: a list of predefined callbacks that get called at every epoch
        :param train_time: time of the training session in seconds: default 10 minutes
        :param X: the independent variables in form of Pandas DataFrame
        :param Y: the dependents(predicted) values in form of Pandas DataFrame
        :param verbose: decides whether or not the model prints intermediary outputs
        :return: the model
        """
        # check once more the predicted names
        if self._predicted_name is None:
            self._predicted_name = list(Y.columns)

        if self._task not in AVAILABLE_TASKS:
            self._task = self._determine_task_type(Y)

        # handle validation
        if validation_split is None:
            x_train = X.to_numpy()
            y_train = Y.to_numpy()

            print("Training on {} samples...".format(len(y_train))) if verbose else None
        else:

            if type(validation_split) != float:
                raise RandomForestModelException("Parameter validation_split should be None or float in range [0,1)")
            if validation_split < 0 or validation_split >= 1:
                validation_split = 0.2
                warnings.warn("RandomForestModel: configured validation percentage is out of bounds; using default "
                              "value 0.2", RuntimeWarning)

            x_train, x_val, y_train, y_val = train_test_split(X.to_numpy(), Y.to_numpy(), test_size=validation_split,
                                                              random_state=randrange(2048))

            print("Training on {} samples. Validating on {}...".format(len(y_train), len(y_val))) if verbose else None

        # prepare for time handling
        seconds_count = 0
        epochs = 0

        start_time = time.time()
        requested_finish = start_time + train_time

        keep_training = True
        best_model = None
        best_score = None

        while keep_training:
            keep_training = False

            epoch_start = time.time()

            model = self._model
            # start a random forest model
            if self._model is None:
                model = self._create_model()
            model.fit(x_train, y_train.ravel())

            # compare to the actual model and update if necessary
            # print(model.score(x_train, y_train))
            if validation_split is None:  # take only training score into consideration
                criterion = model.score(x_train, y_train)
            else:
                criterion = model.score(x_val, y_val)

            if self._task == CLASSIFICATION and (self._model_score is None or self._model_score < criterion) or \
                    self._task == REGRESSION and (self._model_score is None or self._model_score > criterion):
                self._model_score = criterion
                self._model = model

                # data for printing
                self._configured = True
                self._n_estimators = self._model.n_estimators
                self._criterion = self._model.criterion

            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            seconds_count += epoch_duration
            epochs += 1

            time_per_epoch = seconds_count / epochs

            epochs_to_complete = (requested_finish - epoch_end) / time_per_epoch  # how much time
            # available split to the average epoch time
            if epochs_to_complete - int(epochs_to_complete) >= 0.5:
                epochs_to_complete = int(epochs_to_complete) + 1
            else:
                epochs_to_complete = int(epochs_to_complete)

            if epochs_to_complete > 0:
                keep_training = True

            if epochs % 10 == 9:
                expected_finish = epoch_end + epochs_to_complete * time_per_epoch

                # printed format
                date = time.localtime(expected_finish)
                if time.localtime(epoch_end).tm_mday == date.tm_mday:
                    day = ""
                elif time.localtime(epoch_end).tm_mday == date.tm_mday - 1:
                    day = "tomorrow|"
                else:
                    day = "{}/{}/{}|".format(date.tm_mday, date.tm_mon, date.tm_year)

                hour = date.tm_hour
                minute = date.tm_min
                second = date.tm_sec

                if self._task == CLASSIFICATION:
                    loss_name = "accuracy"
                else:
                    loss_name = "loss"

                if not (validation_split is None):
                    criterion_train = self._model.score(x_train, y_train)
                    criterion_val = self._model.score(x_val, y_val)

                    print("Epoch {} - Training {}: {} - Validation {}: {} - ETA: {}{}:{}:{}"
                          .format(epochs, loss_name, criterion_train, loss_name, criterion_val,
                                  day, hour, minute, second)) if verbose else None
                else:
                    criterion_train = self._model.score(x_train, y_train)
                    print("Epoch {} - Training {}: {} - ETA: {}{}:{}:{}".format(epochs, loss_name, criterion_train,
                                                                                day, hour, minute,
                                                                                second)) if verbose else None

        if not (validation_split is None):
            self._train_criterion = self._model.score(x_train, y_train)
            self._val_criterion = self._model.score(x_val, y_val)
        else:
            self._train_criterion = self._model.score(x_train, y_train)

    def _model_predict(self, X: DataFrame, raw_output: bool = False) -> DataFrame:
        """
                Predicts the output of X based on previous learning
            :param X: DataFrame; the X values to be predicted into some Y Value
            :param raw_output: returns the exact output of the model, without rebasing into the initial classes
            :return: DataFrame with the predicted data
        """
        if self._model is None:
            raise RandomForestModelException("Could not call predict before train.")

        data = X.to_numpy()
        pred = self._model.predict(data)

        df = DataFrame(pred, columns=self._predicted_name)

        return df

    def model_type(self) -> str:
        """
                Returns the model type from available model types in file "model_types.py"
            :return: string with the model type
        """
        return RANDOM_FOREST_MODEL

    def _create_model(self):
        """
            Creates the sklearn model for the learning task requested and returns it
        :return: the model created; either RandomForestClassifier or RandomForestRegressor
        """
        if self._task is None:
            return None
        elif self._task == CLASSIFICATION:
            return self._create_classifier()

        elif self._task == REGRESSION:
            return self._create_regressor()
        else:
            return None

    def _create_classifier(self) -> RandomForestClassifier:
        """
            Creates a classifier model
        :return: the model
        """
        config = self._config.get("CLASSIFIER", {})

        self._criterion = config.get("CRITERION", 'gini')
        max_features = config.get("MAX_FEATURES", 'sqrt')
        if max_features == "none":
            max_features = None

        return RandomForestClassifier(
            n_estimators=config.get("N_ESTIMATORS", 100),
            criterion=config.get("CRITERION", 'gini'),
            min_samples_split=config.get("MIN_SAMPLES_SPLIT", 2),
            bootstrap=True,
            max_features=max_features,
            n_jobs=-1,  # using all the processors
            random_state=randint(1, 1024),
            ccp_alpha=random() * 0.4
        )

    def _create_regressor(self) -> RandomForestRegressor:
        """
            Creates a regressor model
        :return: the model
        """
        config = self._config.get("REGRESSOR", {})

        self._criterion = config.get("CRITERION", 'mse')
        max_features = config.get("MAX_FEATURES", 'sqrt')
        if max_features == "none":
            max_features = None

        return RandomForestRegressor(
            n_estimators=config.get("N_ESTIMATORS", 100),
            criterion=config.get("CRITERION", 'mse'),
            min_samples_split=config.get("MIN_SAMPLES_SPLIT", 2),
            bootstrap=True,
            max_features=max_features,
            n_jobs=-1,  # using all the processors
            random_state=randint(1, 1024),
            ccp_alpha=random() * 0.4
        )

    def to_dict(self) -> dict:
        """
                Returns a dictionary representation that encapsulates all the details of the model
            :return: dictionary with 2 mandatory keys : MODEL_TYPE, MODEL_DATA
        """
        # !!! should match _init_from_dictionary loading format
        # get the model data
        model = pickle.dumps(self._model)

        data = {
            "MODEL": model,
            "METADATA": {
                "PREDICTED_NAME": self._predicted_name,
                "CONFIG": self._config,
                "TASK": self._task,
                "CONFIGURED": self._configured,
                "N_ESTIM": self._n_estimators,
                "CRITERION": self._criterion,
                "TRAIN_CRIT": self._train_criterion,
                "VAL_CRIT": self._val_criterion,

                "MODEL_SCORE": self._model_score,

            }
        }

        return {
            "MODEL_TYPE": self.model_type(),
            "MODEL_DATA": data
        }

    def _init_from_dictionary(self, dictionary: dict):
        """
            Initializes the class from a provided dictionary
        :param dictionary: dictionary with the previously saved data
        :return:
        """
        # !!! should match to_dict loading format
        data = dictionary.get("MODEL_DATA")

        mdata = data.get("METADATA")
        model = data.get("MODEL")

        # init the data
        self._predicted_name = mdata.get("PREDICTED_NAME")
        self._config = mdata.get("CONFIG")
        self._task = mdata.get("TASK")
        self._configured = mdata.get("CONFIGURED")
        self._n_estimators = mdata.get("N_ESTIM")
        self._criterion = mdata.get("CRITERION")
        self._train_criterion = mdata.get("TRAIN_CRIT")
        self._val_criterion = mdata.get("VAL_CRIT")
        self._model_score = mdata.get("MODEL_SCORE")

        # init the model
        self._model = pickle.loads(model)

    def _description_string(self) -> str:
        if self._configured is False:
            return "Random Forest - Not configured"
        else:
            task = "Classifier" if self._task == CLASSIFICATION else "Regression"
            return "Random Forest {task} - Estimators: {estimators} | Criterion: {criterion}".format(
                task=task,
                estimators=self._n_estimators,
                criterion=self._criterion
            )

    def get_config(self) -> dict:
        return self._config

    def summary(self) -> dict:
        """
            Returns summary about the deep learning model
        :return: dictionary with summary
        """
        metadata = {
            "N_ESTIMATORS": self._config.get("N_ESTIMATORS", 100),
            "CRITERION": self._config.get("CRITERION", 'mse')
        }

        train_data = {
            "VALIDATION_CRITERION": self._val_criterion,
            "TRAIN_CRITERION": self._train_criterion
        }

        return {
            "MODEL_TYPE": self.model_type(),
            "METADATA": metadata,
            "TRAIN_DATA": train_data
        }

    def eval(self, X: DataFrame, Y: DataFrame, task: str, metric: str, include_train_stats: bool = False):
        """
            Not available for this type of model. Throws warning and returns 0.
        :param X:
        :param Y:
        :param task:
        :param metric:
        :param include_train_stats:
        :return: 0
        """
        warnings.warn("Method eval() of AbstractModel not available for RandomForestModel.", RuntimeWarning)

        return 0

    def get_labels(self) -> list:
        """
            For the moment this method is used only in neural networks for the evolutionary flow.
        :return:
        """
        return []
