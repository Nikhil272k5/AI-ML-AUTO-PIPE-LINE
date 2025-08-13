from ..abstractCallback import AbstractCallback


class EvolutionaryFeedback(AbstractCallback):
    """
        Defines a callback that receives some dictionary with data from the evolutionary
    model at each epoch and calls the abstract function with it.
    """

    def f(self, data: dict):
        data["type"] = "EVOLUTIONARY_FEEDBACK"
        if self._fun is not None:
            try:
                self._fun(data)
            except Exception:
                pass

        return data
