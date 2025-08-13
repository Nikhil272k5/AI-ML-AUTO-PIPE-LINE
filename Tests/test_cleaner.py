from unittest import TestCase
from pandas import DataFrame, NA
import numpy as np

from Pipeline.DataProcessor.DataCleaning import Cleaner
from Pipeline.Mapper import Mapper


class TestCleaner(TestCase):
    def setUp(self) -> None:
        """
            Sets up a fictive dataset
        :return:
        """
        self._df = DataFrame(np.asarray(
            [
                [NA, 2, 3, 4, 'a', 'b'],
                [NA, 2, 4, 3, 'abc', 'xyz'],
                [NA, NA, 1, 2, NA, NA]
            ]
        ),
            columns=['a', 'b', 'c', 'd', 'e', 'f'])

        self._config = {
            "COLUMNS_TO_REMOVE": ['b'],
            "DO_NOT_REMOVE": ['a'],
            "REMOVE_WHERE_Y_MISSING": True,
            "REMOVE_ROWS": True,
            "ROW_REMOVAL_THRESHOLD": 0.6,
            "REMOVE_COLUMNS": True,
            "COLUMN_REMOVAL_THRESHOLD": 0.6
        }

        self._mapper = Mapper("name")

    def test_clean(self):
        cleaner = Cleaner(self._config)
        df = cleaner.clean(self._df, self._mapper)

        self.assertEqual(['a', 'c', 'd', 'e', 'f'], sorted(df.columns))
        self.assertEqual((2, 5), df.shape)

    def test_convert(self):
        mapper = Mapper("name")
        mapper.set("RemovedColumns", ['a', 'b', 'c'])

        mapper2 = Mapper("name2")
        mapper2.set_mapper(mapper, "Cleaner")

        cleaner = Cleaner({})
        data = cleaner.convert(self._df, mapper2)

        self.assertEqual(['d', 'e', 'f'], sorted(data.columns))
        self.assertEqual((3, 3), data.shape)
