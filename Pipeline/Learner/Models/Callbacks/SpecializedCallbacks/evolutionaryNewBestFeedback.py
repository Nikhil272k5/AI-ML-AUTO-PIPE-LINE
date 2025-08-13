from ..abstractCallback import AbstractCallback


class EvolutionaryNewBestFeedback(AbstractCallback):
    """
        Receives the data for the current best in the evolutionary search and
    returns it to the websocket.
    """

    def f(self, data: dict):
        data["type"] = "EVOLUTIONARY_NEW_BEST_FEEDBACK"
        if self._fun is not None:
            try:
                self._fun(data)
            except Exception:
                pass

        return data
