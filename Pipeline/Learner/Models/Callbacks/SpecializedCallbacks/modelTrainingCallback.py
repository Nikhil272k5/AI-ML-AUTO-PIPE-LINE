from ..abstractCallback import AbstractCallback


class ModelTrainingCallback(AbstractCallback):
    """
        Receives data from the current model train session and calls
    the abstract function with it.
    """

    def f(self, data: dict):
        data["type"] = "MODEL_TRAINING"
        if self._fun is not None:
            try:
                self._fun(data)
            except Exception:
                pass

        return data
