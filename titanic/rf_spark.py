from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.mllib.tree import RandomForest

# HDFS_ROOT = "hdfs://192.17.103.104:9000/haowu"
HDFS_ROOT = "."
TRAIN_FILE = HDFS_ROOT + "/train.csv"
TEST_FILE = HDFS_ROOT + "/test.csv"
OUTPUT_DIR = HDFS_ROOT + "/rf_results"

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


def getCategoricalFeaturesInfo():
    map = dict()
    map[3] = 2
    return map


# Load and parse the data
def parsePoint(line):
    # values = [float(x) for x in line.split(' ')]
    survived = float(line[1])
    feature = parseFeatures(line[2:])
    return LabeledPoint(survived, feature)


data = spark.read.csv(TRAIN_FILE, header=True)
parsedData = data.rdd.map(parsePoint)
(train_data, valid_data) = parsedData.randomSplit([0.7, 0.3])
train_data = parsedData

# Build the random forest
model = RandomForest.trainClassifier(
    train_data, numClasses=2, numTrees=200,
    categoricalFeaturesInfo=getCategoricalFeaturesInfo(),
    featureSubsetStrategy="auto",
    impurity='gini', maxDepth=10, maxBins=64)

# Evaluating the model on training data
predictions = model.predict(valid_data.map(lambda x: x.features))
labelsAndPreds = valid_data.map(lambda lp: lp.label).zip(predictions)
trainErr = labelsAndPreds.filter(
    lambda (v, p): v != p).count() / float(valid_data.count())
print("Training Error = " + str(trainErr))

testdata = spark.read.csv(TEST_FILE, header=True)
test_predictions = model.predict(
    testdata.rdd.map(lambda x: parseFeatures(x[1:]))).map(lambda x: x)
testIdAndPreds = testdata.rdd.map(lambda p: p[0]).zip(
    test_predictions.map(lambda x: int(x)))

# output (id, pred) of test
result_df = spark.createDataFrame(testIdAndPreds, ['PassengerId', 'Survived'])
result_df.write.csv(OUTPUT_DIR, header=True)

# Save and load model
# model.save(spark, "./pythonSVMWithSGDModel")
# sameModel = SVMModel.load(spark, "./pythonSVMWithSGDModel")
