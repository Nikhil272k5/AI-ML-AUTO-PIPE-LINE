from pandas import read_csv, DataFrame

from Pipeline import Pipeline


data = read_csv("../Datasets/house_train.csv")


config = {
    "DATA_PROCESSING": True,
    "DATA_PROCESSING_CONFIG": {
        "PREDICTED_COLUMN_NAME": "SalePrice",
    },
    "TRAINING": True,
    "TRAINING_CONFIG": {
        "TYPE": "evolutionary",
        "TASK": "regression",
        "TIME": "10s",
        "DEFAULT_MODEL": "neural_network",
        "PREDICTED_COLUMN_NAME": "SalePrice"
    }
}

pipeline = Pipeline(config=config)

model = pipeline.fit(data)

pred = pipeline.predict(data.loc[:, data.columns!="SalePrice"])
eval = model.eval(pipeline.convert(data.loc[:, data.columns!="SalePrice"]),data.loc[:, data.columns=="SalePrice"], "regression", "MSE")

print(1)