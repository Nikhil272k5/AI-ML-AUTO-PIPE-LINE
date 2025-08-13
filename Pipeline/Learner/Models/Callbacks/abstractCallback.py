from ..abstractModel import AbstractModel
from abc import abstractmethod, ABC


class AbstractCallback(ABC):
    """
        Defines how a callback should behave like.
        A callback has a model and a frequency of calling.
    """

    def __init__(self, lambda_function=None):
        """
            Initializes an AbstractCallback object
        :param lambda_function: function defined by the user that is executed at each f call, with the arguments of f
        """
        self._fun = lambda_function

    @abstractmethod
    def f(self, data: dict):
        """
            Calls the
        :param data: the data that is passed to the callback, as defined in the specialized callback
        :return:
        """

    def __call__(self, data: dict, *args, **kwargs):
        """
            Calls the callback, with the logic to be implemented within the actual implementation
        :param data: the data that is passed to the callback
        :param args: any number of objects as defined in the actual implementation
        :param kwargs: any number of mapped objects as defined in the actual implementation
        :return: None
        """
        return self.f(data)
