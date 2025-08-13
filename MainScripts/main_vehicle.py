# this file contains some examples using the vehicle dataset
from pandas import read_csv
from Pipeline import Pipeline, Splitter

data = read_csv("../Datasets/vehicle.csv")
data.head()
pipeline = Pipeline()

X, Y = Splitter.XYsplit(data, "Class")

model = pipeline.fit(data)
pred = model.predict(pipeline.convert(X))

diff = (Y != pred).sum(axis=None)

print("Accuracy {}".format(1-diff["Class"]/len(X)))
