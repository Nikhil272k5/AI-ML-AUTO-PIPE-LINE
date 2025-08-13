class ModelSelectionException(Exception):
    """
        Exception for the model selection
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "ModelSelectionException: {}.".format(self._message)


class DeepLearningModelException(Exception):
    """
        Generic exception for the neural network
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "NeuralNetworkModelException: {}.".format(self._message)


class RandomForestModelException(Exception):
    """
        Generic exception for the random forest
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "RandomForestModelException: {}.".format(self._message)


class SvmModelException(Exception):
    """
        Generic exception for the support vector machine
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "SvmModelException: {}.".format(self._message)


class ModelLoaderException(Exception):
    """
        Generic exception for the model loader
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "ModelLoaderException: {}.".format(self._message)


class LearnerException(Exception):
    """
        Generic exception for the learner
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "LearnerException: {}.".format(self._message)


class EvolutionaryModelException(Exception):
    """
        Generic exception for the evolutionary model
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "EvolutionaryModelException: {}.".format(self._message)


class AbstractModelException(Exception):
    """
        Generic exception for the abstract model
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "AbstractModelException: {}.".format(self._message)


