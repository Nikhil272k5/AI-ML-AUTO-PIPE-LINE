from unittest import TestCase
from pandas import DataFrame, NA
import numpy as np

from Pipeline import Splitter
from Pipeline.Exceptions import DataSplittingException


class TestSplitter(TestCase):
    def setUp(self) -> None:
        self._df = DataFrame(np.asarray(
            [
                [NA, 2, 3, 4, 'a', 'b'],
                [NA, 2, 4, 3, 'abc', 'xyz'],
                [NA, NA, 1, 2, NA, NA]
            ]
        ),
            columns=['a', 'b', 'c', 'd', 'e', 'f'])

        self._s = Splitter()

    def test_xysplit(self):
        try:
            self._s.XYsplit(self._df, 'undefined')
            self.fail("Should not be able to split after no existing column.")
        except DataSplittingException:
            pass

        x, y = self._s.XYsplit(self._df, 'e')

        self.assertEqual(['a', 'b', 'c', 'd', 'f'], sorted(x.columns))
        self.assertEqual('e', y.columns[0])
