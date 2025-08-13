from ..abstractCallback import AbstractCallback


class DataConversionCallback(AbstractCallback):
    """
        Defines a callback that, after the data engineering step in pipeline fit, returns a
    preview of the generated data.
    """

    def f(self, data: dict):
        data["type"] = "PROCESSED_DATA_PREVIEW"
        if self._fun is not None:
            try:
                self._fun(data)
            except Exception:
                pass

        return data
