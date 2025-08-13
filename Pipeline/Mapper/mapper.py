import pickle

from ..Exceptions import MapperException


class Mapper:
    """
        Maps modifications over a dataset.
        Keeps track of every modification that was made on the raw dataset for future conversion of new data.

        Has a name for reference and a dictionary which has 2 main sub dictionaries, namely "FIELDS" and "MAPPERS".
            - the first one keeps track of non recurring attributes, like "mean_value":20
            - the second one keeps track of recurring Mapper dictionaries, mapped by the Mapper name

        Used in the pipeline by every module to keep track of data changes.
        Each module will be given an empty Mapper which will be handled independent according to the needs of the module,
            and will be than handed to the caller module for recurrent saving.

        In the case of data conversion( for predicting purposes ), each module will be given it's previously created mapper
            in order to correctly transform the data.


        Methods:
            - get_name: returns the name
            - set: sets a value to a key
            - get: retrieves the value of a key
            - get_map: returns the map of (key, value) pairs
            - get_mapper: retrieves a mapper by name
            - set_mapper: sets a mapper using it's name
            - save_to_file: saves to a file
    """

    def __init__(self, name: str, file: str = None, dictionary: dict = None):
        """
            Receives a dictionary of mappings that contains changes from the raw data to the processed data.
        :param file: path to file; if not None the mapper inits itself from a file
        :param dictionary: dictionary of changes
        """

        self._name = name
        if dictionary:
            self._map = dictionary
        else:
            self._map = {
                "FIELDS": {},
                "MAPPERS": {}
            }
            if file:
                self._init_from_file(file)

    def get_name(self) -> str:
        """
            Returns the name of the mapper
        :return: name:str
        """
        return self._name

    def _get_fields(self) -> dict:
        """
            Get the fields dictionary of the current mapper
        :return: reference to fields map
        """
        return self._map.get("FIELDS", {})

    def _get_recurrent_mappers(self) -> dict:
        """
            Get the recurrent mappers' dictionary that this map holds
        :return: reference to recurrent mappers' map
        """
        return self._map.get("MAPPERS", {})

    def set(self, key, value):
        """
            Sets the value of a key for a field
            In case of collisions with previous (key, value) pair, the old one will be overwritten
        :param key: key for which the value has to be set
        :param value: the value to be set
        :return: value
        """
        self._get_fields()[key] = value
        return value

    def get(self, key, default=None):
        """
            Gets the value of a key
        :param default: the default value the method will return if there is no key mapped
        :param key: the key to search for
        :return: the value of the key or default if the key does not exist
        """
        return self._get_fields().get(key, default)

    def get_map(self) -> dict:
        """
            Return the raw map
        :return: map
        """
        return self._map

    def get_mapper(self, name: str, default: dict = {}) -> 'Mapper':
        """
            Returns the mapper with the given name
        :param default: the default value in case the searched mapper is not found
        :param name: the name of the mapper, as saved previously
        :return: mapper instance
        """
        submap = self._get_recurrent_mappers().get(name, default)
        return Mapper(name, dictionary=submap)

    def set_mapper(self, mapper: 'Mapper', name: str = None) -> 'Mapper':
        """
            Sets the dictionary of a mapper within the recurring mappers field.
            If a dictionary with the same name exists it will be overwritten.
        :param name: the name of the saved mapper
        :param mapper: the mapper to add
        :return: the current mapper
        """
        if name is None:
            name = mapper.get_name()
        self._get_recurrent_mappers()[name] = mapper.get_map()
        return self

    def _init_from_file(self, file: str) -> 'Mapper':
        """
            Inits the mapper from a configuration previously saved to file
        :param file: file to load the mapper from
        :return: mapper
        :exception MapperException
        """
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                self._map = data
            return self
        except Exception as e:
            raise MapperException("Could not init from file {}.".format(file))

    def save_to_file(self, file: str) -> 'Mapper':
        """
            Saves the mapper to file
        :param file: path to save file
        :return: current mapper
        """
        import pickle
        with open(file, 'wb') as f:
            pickle.dump(self.get_map(), f)
        return self
