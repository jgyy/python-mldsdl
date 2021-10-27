"""
spark k means
"""
from math import sqrt
from numpy import array
from numpy.random import uniform, normal, seed
from pyspark.mllib.clustering import KMeans
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import scale


def wrapper():
    """
    Boilerplate Spark stuff
    Load the data; note I am normalizing it with scale() - very important!
    Build the model (cluster the data)
    Print out the cluster assignments
    Evaluate clustering by computing Within Set Sum of Squared Errors
    """
    kmean = 5
    conf = SparkConf().setMaster("local").setAppName("SparkKMeans")
    sparkc = SparkContext(conf=conf)
    sparkc.setLogLevel("ERROR")

    def create_clustered_data(num, kcluster):
        """
        Create fake income/age clusters for N people in k clusters
        """
        seed(10)
        points_per_cluster = float(num) / kcluster
        xdata = []
        for _ in range(kcluster):
            income_centroid = uniform(20000.0, 200000.0)
            age_centroid = uniform(20.0, 70.0)
            for _ in range(int(points_per_cluster)):
                xdata.append(
                    [normal(income_centroid, 10000.0), normal(age_centroid, 2.0)]
                )
        xdata = array(xdata)
        return xdata

    seed(0)
    data = sparkc.parallelize(scale(create_clustered_data(100, kmean)))
    clusters = KMeans.train(data, kmean, maxIterations=10, initializationMode="random")
    result_rdd = data.map(clusters.predict).cache()
    print("Counts by value:")
    counts = result_rdd.countByValue()
    print(counts)
    print("Cluster assignments:")
    results = result_rdd.collect()
    print(results)
    error = lambda point: sqrt(
        sum([x ** 2 for x in (point - clusters.centers[clusters.predict(point)])])
    )
    wssse = data.map(error).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(wssse))


if __name__ == "__main__":
    wrapper()
