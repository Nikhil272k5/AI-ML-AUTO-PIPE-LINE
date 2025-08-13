# this file contains a demo using the container_problem dataset

from Pipeline import Pipeline, Splitter
from pandas import read_csv

data = read_csv("../Datasets/container_problem.csv")

pipeline = Pipeline()

model = pipeline.fit(data)
#
X, Y = Splitter.XYsplit(data, "runtime")
X_conv = pipeline.convert(X)
pred = model.predict(X_conv)
pred.to_csv("Datasets/container_predicted", index=False)
