from unittest import TestCase
import os
import shutil

from Pipeline.Mapper import Mapper


class TestMapper(TestCase):

    def setUp(self) -> None:
        self._mapper = Mapper("m1")
        self._rec_mapper = Mapper("m2")

        self._rec_mapper.set("a", "b")
        self._rec_mapper.set("l", [1,2,3])

        if not os.path.exists("./.tmp_test_mapper_files"):
            os.mkdir("./.tmp_test_mapper_files")

    def tearDown(self) -> None:
        if os.path.exists("./.tmp_test_mapper_files"):
            shutil.rmtree("./.tmp_test_mapper_files")

    def test_get_name(self):
        self.assertEqual(self._mapper.get_name(), "m1")

    def test__get_fields(self):
        self.assertEqual(self._mapper._get_fields(), {})

        self._mapper.set("a", 1)
        self._mapper.set("b", 2)

        self.assertEqual(self._mapper._get_fields(), {"a":1,"b":2})

        self._mapper.set("b", 3)

        self.assertEqual(self._mapper._get_fields(), {"a": 1, "b": 3})


    def test__get_recurrent_mappers(self):
        self.assertEqual(self._mapper._get_recurrent_mappers(), {})

        self._mapper.set_mapper(self._rec_mapper)

        self.assertEqual(self._mapper._get_recurrent_mappers(), {"m2":self._rec_mapper.get_map()})

    def test_set(self):
        pass

    def test_get(self):
        self._mapper.set("a",1)
        self.assertEqual(self._mapper.get("a"),1)

    def test_get_map(self):
        self.assertEqual(self._rec_mapper.get_map(),{"FIELDS":{"a":"b","l":[1,2,3]}, "MAPPERS":{}})

    def test_get_mapper(self):
        self._mapper.set_mapper(self._rec_mapper)
        self.assertNotEqual(self._mapper.get_mapper("m2"), self._rec_mapper)
        self.assertEqual(self._mapper.get_mapper("m2").get_map(), self._rec_mapper.get_map())

    def test_set_mapper(self):
        self._mapper.set_mapper(self._rec_mapper)
        self.assertNotEqual(self._mapper.get_mapper("m2"), self._rec_mapper)
        self.assertEqual(self._mapper.get_mapper("m2").get_map(), self._rec_mapper.get_map())

    def test__init_from_file(self):
        self._rec_mapper.save_to_file("./.tmp_test_mapper_files/mapper")
        mapper = Mapper("name", file="./.tmp_test_mapper_files/mapper")
        self.assertEqual(sorted(list(mapper.get_map().keys())), sorted(list(self._rec_mapper.get_map().keys())))


    def test_save_to_file(self):
        self._rec_mapper.save_to_file("./.tmp_test_mapper_files/mapper")
        self.assertTrue(os.path.exists("./.tmp_test_mapper_files/mapper"))

