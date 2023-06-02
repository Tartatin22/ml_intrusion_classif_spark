"""
JOANNY Marion
BAZIRE Martin

"""
import matplotlib.pyplot as plt
#  imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import RandomForestClassifier

import optuna


def objective(trial, train_df, test_df):
    maxDepth = trial.suggest_int("max_depth", 2, 30)
    maxBins = trial.suggest_int("max_bins", 10, 300)

    model = RandomForestClassifier(featuresCol="features", labelCol='attack',
                                   maxBins=maxBins,
                                   maxDepth=maxDepth)

    fitted_model = model.fit(train_df)
    predictions = fitted_model.transform(test_df)
    predictions = predictions.withColumn("prediction", predictions["prediction"].cast(FloatType()))
    predictions = predictions.withColumn("attack", predictions["attack"].cast(FloatType()))
    truth_pred = predictions.select(["attack", "prediction"]).rdd.map(tuple)
    metrics = MulticlassMetrics(truth_pred)
    return metrics.accuracy


#  const
PATH_TRAINING_DATASET = "../NSL-KDD/KDDTrain+.txt"
PATH_TEST_DATASET = "../NSL-KDD/KDDTest+.txt"

columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
           'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
           'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
           'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
           'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
           'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
           'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
           'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level']


def is_attack(c):
    return 0 if c == "normal" else 1


def main():
    #  create spark session
    spark = SparkSession.builder.appName('NSL_KDD_classify').getOrCreate()

    #  concat train and test for preprocess
    df = spark.read.options(delimiter=',').option("header", False).option("inferSchema", True) \
        .csv(PATH_TRAINING_DATASET).union(spark.read.options(delimiter=',')
                                          .option("header", False).option("inferSchema", True).csv(PATH_TEST_DATASET))

    #  add columns names
    for i in range(len(columns)):
        df = df.withColumnRenamed("_c" + str(i), columns[i])

    print("Data preview :")
    df.show(1)

    #  encode attack with boolean flag
    udf_is_attack = udf(is_attack)
    df = df.withColumn("attack", udf_is_attack("attack").cast('int'))

    print("Data with attack column boolean encode :")
    df.show(1)

    #  get text cols
    string_cols = [col[0] for col in df.dtypes if col[1] == "string"]
    string_cols_idx = [col + "_index" for col in string_cols]

    #  fit a string indexer on text cols
    model = StringIndexer(inputCols=string_cols, outputCols=string_cols_idx).fit(df)

    #  inference on string indexer
    df = model.transform(df)

    #  fit a encoder model on indexed cols
    encoded_cols = [col + "_encoded" for col in string_cols_idx]
    model = OneHotEncoder(inputCols=string_cols_idx, outputCols=encoded_cols).fit(df)

    #  inference on encoder
    df = model.transform(df)

    print("data with indexed column encoded :")
    df.show(1)

    #  columns to vectorize
    features_cols = columns
    features_cols.remove("attack")
    for c in string_cols:
        features_cols.remove(c)
    for c in encoded_cols:
        features_cols.append(c)

    #  vectorize model
    model = VectorAssembler(inputCols=features_cols, outputCol='features')

    #  inference on vectorize model
    df = model.transform(df)

    print("data with feature vector :")
    df.show(1)

    #  split train test dataset
    train_df, test_df = df.randomSplit([0.8, 0.2])

    #  tricks to have args
    f = lambda trial: objective(trial, train_df, test_df)

    # Pass func to Optuna studies
    study = optuna.create_study(direction='maximize')
    study.optimize(f, n_trials=20)
    print("best value, best params : ", study.best_value, study.best_params)
    return
    #  ml model
    model = RandomForestClassifier(featuresCol="features", labelCol='attack')

    #  fit model
    fitted_model = model.fit(train_df)

    #  inference
    predictions = fitted_model.transform(test_df)

    #  cast to float
    predictions = predictions.withColumn("prediction", predictions["prediction"].cast(FloatType()))
    predictions = predictions.withColumn("attack", predictions["attack"].cast(FloatType()))

    #  put prediction and truth together in tuple
    truth_pred = predictions.select(["attack", "prediction"]).rdd.map(tuple)

    #  calc metrics
    metrics = MulticlassMetrics(truth_pred)

    print("Confusion matrix : ", metrics.confusionMatrix().toArray())
    print("Accuracy : ", metrics.accuracy)
    print("False positive rate : ", metrics.weightedFalsePositiveRate)
    print("True positive rate : ", metrics.weightedTruePositiveRate)
    return


if __name__ == "__main__":
    main()
