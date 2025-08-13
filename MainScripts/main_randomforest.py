# this file tests the random forest model
from pandas import read_csv
from Pipeline import Pipeline, load_pipeline


data = read_csv("../Datasets/titanic.csv")

pipeline = Pipeline()

pipeline.fit(data)


pred1 = pipeline.predict(pipeline.convert(data).drop("Survived", axis=1))
pipeline.save("tmp_files/pipeline_rf_titanic")

del pipeline

pipeline = load_pipeline("../tmp_files/pipeline_rf_titanic")
pred2 = pipeline.predict(pipeline.convert(data).drop("Survived", axis=1))

print((pred1 != pred2).any())