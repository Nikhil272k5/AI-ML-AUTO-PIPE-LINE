"""
    Contains all the exceptions thrown in the DataProcessor module
"""


class DataProcessorException(Exception):
    """
        Generic exception for the module
    """

    def __init__(self, message):
        super().__init__(message)


class DataSplittingException(Exception):
    """
        Exception for the splitting logic
    """

    def __init__(self, message):
        super().__init__(message)


class DataEngineeringException(Exception):
    """
        Exception for the data engineering logic
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "DataEngineeringException: {}.".format(self._message)


class MapperException(Exception):
    """
        Exception for the mapper logic
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "MapperException: {}.".format(self._message)
