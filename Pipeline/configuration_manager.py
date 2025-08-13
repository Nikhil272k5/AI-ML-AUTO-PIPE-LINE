
def complete_configuration(user_config: dict, default_config: dict) -> dict:
    """
        The method compares the provided user configuration to the default config of the pipeline.
    The output is a complete configuration, similar to the default_configuration. If the user has overwritten
    fields in the default configuration, the user value will be taken.
    :param user_config: the configuration provided by the user
    :param default_config: the default configuration defined for the pipeline
    :return: dictionary with the complete configuration that has the user preferences
    """
    result = {}
    for key in list(default_config.keys()):     # loop through all the required keys
        user_data = user_config.get(key, None)  # what the user should have provided for the key
        default_data = default_config.get(key)

        if user_data is None:   # the user has not provided this data
            result[key] = default_data
        else:                   # the user has provided this data
            if type(default_data) is dict:       # this key contains a sub-dictionary
                if type(user_data) is dict:      # the user has provided the correct data type -> check recursively
                    result[key] = complete_configuration(user_data, default_data)
                else:
                    result[key] = default_data
            else:                                # the user has provided a value, should not be recursively checked
                result[key] = user_data

    return result


