from pandas import DataFrame
from ...Exceptions import DataSplittingException


class Splitter:
    """
        Handles the data split logic.

        Methods:
            - XYsplit: splits the dataset into two data sets based on a y column provided
    """

    @staticmethod
    def XYsplit(data: DataFrame, y_column: str) -> tuple:
        """
            Splits the data in X and Y.
        :param data: dataframe with the dataset
        :param y_column: the name of the predicted column
        :return: tuple like (X,Y), where both are dataframes | None on error
        :exception: DataSplittingException
        """
        if y_column in data.columns:
            y = data[[y_column]]
            x_cols = data.columns.tolist()
            x_cols.remove(y_column)  # removing the y columns from the x subset
            x = data[x_cols]
            return x, y
        else:
            raise DataSplittingException("Cannot split after non-existing Y column {}".format(y_column))
