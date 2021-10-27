"""
Spark Linear Regression
"""
from __future__ import print_function
from os.path import join, dirname
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors


def wrapper():
    """
    wrapper function
    """
    spark = (
        SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp")
        .appName("LinearRegression")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    input_lines = spark.sparkContext.textFile(join(dirname(__file__), "regression.txt"))
    data = input_lines.map(lambda x: x.split(",")).map(
        lambda x: (float(x[0]), Vectors.dense(float(x[1])))
    )
    col_names = ["label", "features"]
    dframe = data.toDF(col_names)
    train_test = dframe.randomSplit([0.5, 0.5])
    training_df = train_test[0]
    test_df = train_test[1]
    lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    model = lir.fit(training_df)
    full_predictions = model.transform(test_df).cache()
    predictions = full_predictions.select("prediction").rdd.map(lambda x: x[0])
    labels = full_predictions.select("label").rdd.map(lambda x: x[0])
    prediction_and_label = predictions.zip(labels).collect()
    for prediction in prediction_and_label:
        print(prediction)
    spark.stop()


if __name__ == "__main__":
    wrapper()
