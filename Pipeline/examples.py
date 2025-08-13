"""Pipeline examples

    The file contains examples with each type of flow
"""
from . import Pipeline, load_pipeline, load_model, read_csv


# # FLOW 1 - The fully automation goal ----------------------------------------------------------------------
# # provide a dataset and a custom configuration file (you can also provide parts of the configuration
# #                   you are interested in; those will be merged with the default configuration)

data = read_csv("../Datasets/titanic.csv")  # read the dataset
config = {  # create a dictionary to modify parts of the configuration
    "DATA_PROCESSING": True,
    "DATA_PROCESSING_CONFIG": {

        "DATA_CLEANING": True,
        "DATA_CLEANING_CONFIG": {
            "COLUMNS_TO_REMOVE": ["PassengerId"],
        },

        "PREDICTED_COLUMN_NAME": "Survived",
    },
    "TRAINING": True,
    "TRAINING_CONFIG": {
        "TYPE": "evolutionary",
        "TASK": "classification",
        "TIME": "10s",
        "PREDICTED_COLUMN_NAME": "Survived",

        "EVOLUTIONARY_MODEL_CONFIG": {
              "GENERAL_CRITERION": ""
        }
    }
}
pipeline = Pipeline(config=config)  # create a pipeline with an augmented configuration
model = pipeline.fit(data)  # train and get the model

new_data = read_csv("../Datasets/titanic_new.csv")
converted_data = pipeline.convert(new_data)

prediction = model.predict(converted_data)  # using explicit conversion
# or
prediction = pipeline.predict(new_data)  # for implicit conversion
prediction.to_csv("../Results/titanic_prediction.csv")


# FLOW 2 - The feature engineering flow --------------------------------------------------------------------------------
# provide a dataset and a custom configuration, then process the data

data = read_csv("../Datasets/titanic.csv")  # read the dataset

config = {  # create a dictionary to modify parts of the configuration
    "DATA_PROCESSING": True,
    "DATA_PROCESSING_CONFIG": {

        "PREDICTED_COLUMN_NAME": "Survived",

        "FEATURE_ENGINEERING_CONFIG": {
            "DO_NOT_PROCESS": ["PassengerId"]
        }
    }
}

pipeline = Pipeline(config=config)
processes_data = pipeline.process(data)
processes_data.to_csv("../Datasets/titanic_processed.csv")


# FLOW 3 - The default learning flow -----------------------------------------------------------------------------------
# choose a default model in the custom configuration and train it

data = read_csv("../Datasets/titanic.csv")  # read the dataset
config = {  # create a dictionary to modify parts of the configuration
    "DATA_PROCESSING": True,
    "DATA_PROCESSING_CONFIG": {

        "DATA_CLEANING": True,
        "DATA_CLEANING_CONFIG": {
            "COLUMNS_TO_REMOVE": ["PassengerId"],
        },

        "PREDICTED_COLUMN_NAME": "Survived",
    },
    "TRAINING": True,
    "TRAINING_CONFIG": {
        "TYPE": "default",
        "TIME": "10s",
        "PREDICTED_COLUMN_NAME": "Survived",
        "DEFAULT_MODEL": "neural_network"
    }
}
pipeline = Pipeline(config=config)  # create a pipeline with an augmented configuration
model = pipeline.fit(data)  # train and get the model

new_data = read_csv("../Datasets/titanic_new.csv")
converted_data = pipeline.convert(new_data)

prediction = model.predict(converted_data)  # using explicit conversion
# or
prediction = pipeline.predict(new_data)  # for implicit conversion
prediction.to_csv("../Results/titanic_prediction.csv")


# FLOW 4 - The evolutionary learning flow ------------------------------------------------------------------------------
# it is identical to FLOW 1 since the fully automation goal is achieved through the evolutionary learning method
data = read_csv("../Datasets/titanic.csv")  # read the dataset
config = {  # create a dictionary to modify parts of the configuration
    "DATA_PROCESSING": True,
    "DATA_PROCESSING_CONFIG": {

        "DATA_CLEANING": True,
        "DATA_CLEANING_CONFIG": {
            "COLUMNS_TO_REMOVE": ["PassengerId"],
        },

        "PREDICTED_COLUMN_NAME": "Survived",
    },
    "TRAINING": True,
    "TRAINING_CONFIG": {
        "TYPE": "evolutionary",
        "TASK": "regression",
        "TIME": "10s",
        "PREDICTED_COLUMN_NAME": "Survived"
    }
}
pipeline = Pipeline(config=config)  # create a pipeline with an augmented configuration
model = pipeline.fit(data)  # train and get the model

new_data = read_csv("../Datasets/titanic_new.csv")
converted_data = pipeline.convert(new_data)

prediction = model.predict(converted_data)  # using explicit conversion
# or
prediction = pipeline.predict(new_data)  # for implicit conversion
prediction.to_csv("titanic_prediction.csv")
pipeline.save("../PipelineFiles/pipeline")


# FLOW 5 - The data conversion flow ------------------------------------------------------------------------------------
# based on a saved pipeline that has previously been used to process data, you can convert a similar dataset

pipeline = load_pipeline("../PipelineFiles/pipeline")  # load a pipeline that has previously processed data (so it
#                   has the rules saved)
data = read_csv("../Datasets/titanic.csv")  # read the dataset
converted_data = pipeline.convert(data)
converted_data.to_csv("../Datasets/titanic_converted.csv")


# FLOW 6 - The raw data prediction flow --------------------------------------------------------------------------------
# based on a saved pipeline that has previously been used to process data and learn a model, you can predict raw data

pipeline = load_pipeline("../PipelineFiles/pipeline")  # load a pipeline that has previously processed data and learnt
#            a model (so it has the rules and the model saved)
data = read_csv("../Datasets/titanic.csv")  # read the dataset
prediction = pipeline.predict(data)


# FLOW 7 - The converted data prediction flow --------------------------------------------------------------------------
# same as FLOW 6, but conversion can be done separately

pipeline = load_pipeline("../PipelineFiles/pipeline")  # load a pipeline that has previously processed data and learnt
#            a model (so it has the rules and the model saved)
data = read_csv("../Datasets/titanic.csv")  # read the dataset
converted_data = pipeline.convert(data)

prediction = pipeline.predict(converted_data)


# saving and loading demo ----------------------------------------------------------------------------------------------

pipeline = Pipeline()  # create a pipeline

# ---
#  any pipeline call
# ---

model = pipeline.get_model()  # if the pipeline has been trained

pipeline.save("../PipelineFiles/pipeline", include_model=False)  # save as binary file; you can decide whether the model
                                                                    # should be included or not
model.save("../ModelFiles/model")           # save as binary file as well

pipeline = load_pipeline("../PipelineFiles/pipeline")
model = load_model("../ModelFiles/model")

# both the pipeline and the model will be the same as before saving (the pipeline may lack the model if include_model is
# set to false; this is done for memory purposes)
