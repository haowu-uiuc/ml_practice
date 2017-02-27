from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils

from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# HDFS_ROOT = "hdfs://192.17.103.104:9000/haowu"
HDFS_ROOT = "."
TRAIN_FILE = HDFS_ROOT + "/train.csv"
TEST_FILE = HDFS_ROOT + "/test.csv"
OUTPUT_DIR = HDFS_ROOT + "/nn_results"

# conf = SparkConf().setAppName("Titanic")
# conf = conf.setMaster("local")
sc = SparkContext()
spark = SparkSession \
    .builder \
    .appName("Titanic") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


def parseFeatures(allfeature):
    pclass1 = 1 if float(allfeature[0]) == 1 else 0
    pclass2 = 1 if float(allfeature[0]) == 2 else 0
    pclass3 = 1 if float(allfeature[0]) == 3 else 0
    sex = 1 if allfeature[2] == 'male' else 0
    age = 0 if allfeature[3] is None else float(allfeature[3])
    hasage = 0 if allfeature[3] is None else 1
    sibsp = float(allfeature[4])
    parch = float(allfeature[5])
    fare = 0 if allfeature[7] is None else float(allfeature[7])
    hasfare = 0 if allfeature[7] is None else 1
    hasembarked = 0 if allfeature[9] is None else 1
    embarked_S = 0
    embarked_C = 0
    embarked_Q = 0
    if allfeature[9] is not None:
        if (allfeature[9]) == 'S':
            embarked_S = 1
        elif (allfeature[9] == 'C'):
            embarked_C = 1
        elif (allfeature[9] == 'Q'):
            embarked_Q = 1

    return Vectors.dense([
        pclass1,
        pclass2,
        pclass3,
        sex,
        age,
        hasage,
        sibsp,
        parch,
        fare,
        hasfare,
        hasembarked,
        embarked_S,
        embarked_C,
        embarked_Q
    ])


# Load and parse the data
def parsePoint(line):
    survived = float(line[1])
    feature = parseFeatures(line[2:])
    return LabeledPoint(survived, feature)


def parseTestPoint(line):
    feature = parseFeatures(line[1:])
    return LabeledPoint(0.0, feature)


data = spark.read.csv(TRAIN_FILE, header=True)
parsedData = MLUtils.convertVectorColumnsToML(data.rdd.map(parsePoint).toDF())
(train_data, valid_data) = parsedData.randomSplit([0.7, 0.3])


# Build 4-layer feed forward neural network
# specify layers for the neural network:
# input layer of size 14 (features), two intermediate of size 10 and 8
# and output of size 2 (classes)
layers = [len(train_data.first().features), 30, 30, 2]
# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(
    maxIter=100, layers=layers, blockSize=128, seed=1234)
# train the model
model = trainer.fit(train_data)

# Evaluating the model on training data
result = model.transform(valid_data)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

testdata = spark.read.csv(TEST_FILE, header=True)
test_parsed_data = MLUtils.convertVectorColumnsToML(
    testdata.rdd.map(parseTestPoint).toDF())
test_result = model.transform(test_parsed_data)
test_predictions = test_result.select("prediction").rdd.map(
    lambda x: int(x[0]))
testIdAndPreds = testdata.rdd.map(lambda x: x[0]).zip(test_predictions)

# output (id, pred) of test
result_df = spark.createDataFrame(testIdAndPreds, ['PassengerId', 'Survived'])
result_df.write.csv(OUTPUT_DIR, header=True)

# Save and load model
# model.save(spark, "./pythonSVMWithSGDModel")
# sameModel = SVMModel.load(spark, "./pythonSVMWithSGDModel")
