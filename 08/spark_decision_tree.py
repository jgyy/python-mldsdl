"""
Spark Decision Tree
"""
from os.path import join, dirname
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array


def wrapper():
    """
    Boilerplate Spark stuff
    """
    conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
    sparkc = SparkContext(conf=conf)
    sparkc.setLogLevel("ERROR")

    def binary(ynum):
        """
        Some functions that convert our CSV input data into numerical
        features for each job candidate
        """
        if ynum == "Y":
            return 1
        return 0

    def map_education(degree):
        """
        map education function
        """
        if degree == "BS":
            return 1
        if degree == "MS":
            return 2
        if degree == "PhD":
            return 3
        return 0

    def create_labeled_points(fields):
        """
        Convert a list of raw fields from our CSV file to a
        LabeledPoint that MLLib can use. All data must be numerical...
        """
        years_experience = int(fields[0])
        employed = binary(fields[1])
        previous_employers = int(fields[2])
        education_level = map_education(fields[3])
        top_tier = binary(fields[4])
        interned = binary(fields[5])
        hired = binary(fields[6])
        return LabeledPoint(
            hired,
            array(
                [
                    years_experience,
                    employed,
                    previous_employers,
                    education_level,
                    top_tier,
                    interned,
                ]
            ),
        )

    raw_data = sparkc.textFile(join(dirname(__file__), "PastHires.csv"))
    header = raw_data.first()
    raw_data = raw_data.filter(lambda x: x != header)
    csv_data = raw_data.map(lambda x: x.split(","))
    training_data = csv_data.map(create_labeled_points)
    test_candidates = [array([10, 1, 3, 1, 0, 0])]
    test_data = sparkc.parallelize(test_candidates)
    model = DecisionTree.trainClassifier(
        training_data,
        numClasses=2,
        categoricalFeaturesInfo={1: 2, 3: 4, 4: 2, 5: 2},
        impurity="gini",
        maxDepth=5,
        maxBins=32,
    )
    predictions = model.predict(test_data)
    print("Hire prediction:")
    results = predictions.collect()
    for result in results:
        print(result)
    print("Learned classification tree model:")
    print(model.toDebugString())


if __name__ == "__main__":
    wrapper()
