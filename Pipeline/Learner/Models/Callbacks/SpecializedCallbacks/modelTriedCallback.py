from ..abstractCallback import AbstractCallback


class ModelTriedCallback(AbstractCallback):
    """
        Defines a callback that receives some dictionary with data from the evolutionary
    model with the current model that it tried at each epoch and calls the abstract function with it.
    """

    def f(self, data: dict):
        data["type"] = "TRIED_MODEL"
        if self._fun is not None:
            try:
                self._fun(data)
            except Exception:
                pass

        return data