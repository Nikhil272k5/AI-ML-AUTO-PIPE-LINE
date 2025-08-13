class PipelineException(Exception):
    """
        Exception for the pipeline logic
    """

    def __init__(self, message):
        super().__init__(message)
        self._message = message

    def __repr__(self):
        return "DataEngineeringException: {}.".format(self._message)

