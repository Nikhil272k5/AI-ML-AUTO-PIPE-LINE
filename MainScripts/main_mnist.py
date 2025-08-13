import tensorflow as tf
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_tr_flat = []
for i in range(len(x_train)):
    arr = x_train[i].flatten()
    arr = np.append(arr, [y_train[i]])
    x_tr_flat.append(arr)

x_tr_flat = np.asarray(x_tr_flat)

column_names = ["pixel_{}".format(i) for i in range(784)]
column_names.append("digit")

train = pd.DataFrame(x_tr_flat, columns=column_names)

from Pipeline import Pipeline

config = {  # create a dictionary to modify parts of the configuration
    "DATA_PROCESSING": False,

    "TRAINING": True,
    "TRAINING_CONFIG": {
        "TYPE": "evolutionary",
        "TASK": "classification",
        "TIME": "100s",
        "PREDICTED_COLUMN_NAME": "digit",

        "EVOLUTIONARY_MODEL_CONFIG": {
            "GENERAL_CRITERION": "BCE",
            "POPULATION_SIZE": 1,
        }
    }
}

pip = Pipeline(config=config)

pip.fit(train)
print(1)
