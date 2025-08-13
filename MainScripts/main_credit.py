from Pipeline import Pipeline, read_csv

config = {  # create a dictionary to modify parts of the configuration
    "DATA_PROCESSING": True,
    "DATA_PROCESSING_CONFIG": {

        "DATA_CLEANING": True,
        "DATA_CLEANING_CONFIG": {
            "COLUMNS_TO_REMOVE": ['Unnamed: 0'],
        },

        "PREDICTED_COLUMN_NAME": "A16",
    },
    "TRAINING": True,
    "TRAINING_CONFIG": {
        "TYPE": "evolutionary",
        "TASK": "classification",
        "TIME": "4m",
        "PREDICTED_COLUMN_NAME": "A16",

        "EVOLUTIONARY_MODEL_CONFIG": {
            "GENERAL_CRITERION": "BCE",
            "POPULATION_SIZE": 4,

            "NEURAL_NETWORK_EVOL_CONFIG": {
                "LEARNING_RATE_RANGE": [0.00001, 0.0005],
            }
        }
    }
}

pip = Pipeline(config=config)


df = read_csv("../Datasets/credit.csv")

mask = df.A16 == "+"
column_name = 'A16'
df.loc[mask, column_name] = 1

mask = df.A16 == "-"
column_name = 'A16'
df.loc[mask, column_name] = 0

df.to_csv("../Datasets/credit.csv")

df = read_csv("../Datasets/credit.csv")

pip.fit(df)