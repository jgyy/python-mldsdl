"""
TF-IDF
"""
from os.path import join, dirname
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF


def wrapper():
    """
    wrapper function
    """
    conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
    sparkc = SparkContext(conf=conf)
    sparkc.setLogLevel("ERROR")
    raw_data = sparkc.textFile(join(dirname(__file__), "subset-small.tsv"))
    fields = raw_data.map(lambda x: x.split("\t"))
    documents = fields.map(lambda x: x[3].split(" "))
    document_names = fields.map(lambda x: x[1])
    hashing_tf = HashingTF(100000)
    termf = hashing_tf.transform(documents)
    termf.cache()
    idf = IDF(minDocFreq=2).fit(termf)
    tfidf = idf.transform(termf)
    gettysburg_tf = hashing_tf.transform(["Gettysburg"])
    gettysburg_hash_value = int(gettysburg_tf.indices[0])
    gettysburg_relevance = tfidf.map(lambda x: x[gettysburg_hash_value])
    zipped_results = gettysburg_relevance.zip(document_names)
    print("Best document for Gettysburg is:")
    print(zipped_results.max())


if __name__ == "__main__":
    wrapper()
