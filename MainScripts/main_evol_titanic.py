# this script provides an example on how to use the pipeline with the evolutionary algorithms
# the code is similar to the code used for the other approaches, but all the changes rely in the configuration file
import json

from pandas import read_csv
from Pipeline import Pipeline, load_pipeline
from Pipeline import EvolutionaryFeedback, PipelineFeedback, ModelTriedCallback

pip2 = load_pipeline("/Users/mihai/Desktop/pipeline")
data = read_csv("../Datasets/titanic.csv")




# create a pipeline with the default configuration
config = {
    "DATA_PROCESSING": True,
    "DATA_PROCESSING_CONFIG": {
        "PREDICTED_COLUMN_NAME": "Survived",
    },
    "TRAINING": True,
    "TRAINING_CONFIG": {
        "TYPE": "evolutionary",
        "TASK": "",
        "TIME": "10s",
        "PREDICTED_COLUMN_NAME": "Survived"
    }
}


def print_stats(d):
    print("Stats {}".format(d))


pipeline = Pipeline(config=config)

# fit the data to the pipeline
model = pipeline.fit(data, verbose=False, training_callbacks=[
    # EvolutionaryFeedback(print_stats),
    # PipelineFeedback(print_stats),
    # ModelTriedCallback(print_stats),
])
# summary = model.summary()
# with open('summary.json', 'w') as outfile:
#     json.dump(summary, outfile)
# # save the model for further reusage
# print(summary.get("BEST_MODEL"))
# model_save_file = "./models/titanic_evol_model"
# model.save(model_save_file)
#
# # the pipeline can also be saved for further use in data conversion, training and prediction
# # beware though that when using the saved pipeline for training, a new training session will
# #   begin, despite the model that was saved with the pipeline
# #   the model within the pipeline is used only to make predictions
# #   if one wants to further train the model from the pipeline, it can do so by retrieving the model and
# #       calling train() on it
# pipeline_save_file = "./pipelines/titanic_evol_file"
# pipeline.save(pipeline_save_file)

pred = pipeline.predict(data)
print(pred)

pipeline.save("demo_pipeline")
