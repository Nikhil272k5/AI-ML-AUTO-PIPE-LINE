from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from pandas import concat
from Pipeline import Pipeline, read_csv


from pandas import DataFrame
import numpy as np
a = np.asarray([0,1]).reshape(2,1)
b = np.asarray([[0,1],[0,1]]).reshape(2,2)

# A = DataFrame(a, columns = ["Survived"])
#
# B = DataFrame(b, columns = ["Survived_0", "Survived_1"])
#
# lss = log_loss(A,B, labels=list(range))

data = read_csv("../Datasets/fake_job_postings.csv")
data = data[:10]

config = {
    "DATA_PROCESSING": True,
    "DATA_PROCESSING_CONFIG": {

        "DATA_CLEANING": True,
        "DATA_CLEANING_CONFIG": {
            "COLUMNS_TO_REMOVE": ["job_id"],
        },

        "PREDICTED_COLUMN_NAME": "fraudulent",
    },
    "TRAINING": True,
    "TRAINING_CONFIG": {
        "TYPE": "evolutionary",
        "TASK": "classification",
        "TIME": "0m 40s",
        "PREDICTED_COLUMN_NAME": "fraudulent",

        "EVOLUTIONARY_MODEL_CONFIG": {
              "GENERAL_CRITERION": "BCE"
        }
    }
}

pipeline = Pipeline(config=config)
data = pipeline.process(data)



xtr, xts, ytr, yts = train_test_split(data.loc[:, data.columns != "fraudulent"], data.loc[:, data.columns == "fraudulent"], test_size = 0.2)
train_data = concat([xtr, ytr], axis=1)
test_data = concat([xts, yts], axis=1)

model = pipeline.learn(train_data)
pred = pipeline.predict(xts)

yts.reset_index(inplace=True, drop=True)
diff = pred != yts
s = diff.sum()
print((pred != yts).sum())