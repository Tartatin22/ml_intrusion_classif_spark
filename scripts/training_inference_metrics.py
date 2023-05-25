from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.ml.classification import RandomForestClassifier

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
    #  create session
    spark = SparkSession.builder.appName('NSL_KDD_classify').getOrCreate()

    stages = []

    #  read csv file using spark.read
    df = spark.read.options(delimiter=',').option("header", False).option("inferSchema", True)\
        .csv(PATH_TRAINING_DATASET).union(spark.read.options(delimiter=',')
                .option("header", False).option("inferSchema", True).csv(PATH_TEST_DATASET))
    df.show(1)
    df.printSchema()
    print("before")

    #  add columns names
    for i in range(len(columns)):
        df = df.withColumnRenamed("_c" + str(i), columns[i])

    #  attack_flag
    udf_is_attack = udf(is_attack)
    df = df.withColumn("label", udf_is_attack("attack").cast('int'))
    columns.remove("attack")

    #  string indexer on all columns
    indexed_columns = [c + "_idx" for c in columns]
    stages.append(StringIndexer(inputCols=columns, outputCols=indexed_columns))

    #  columns to vectorize
    columns_to_vectorize = indexed_columns

    #  encoding cat variable
    for col_to_encode in ["protocol_type", "service", "flag"]:

        idx_col = col_to_encode + "_idx"
        columns_to_vectorize.remove(idx_col)

        vect_col = col_to_encode + "_vect"
        columns_to_vectorize.append(vect_col)

        #  one-hot-encoding
        stages.append(OneHotEncoder(inputCol=idx_col, outputCol=vect_col))

    #  vectorize
    stages.append(VectorAssembler(inputCols=columns_to_vectorize, outputCol='features'))

    #  execute pipeline
    pipeline = Pipeline(stages=stages)

    data = pipeline.fit(df).transform(df)

    data.printSchema()

    model = RandomForestClassifier(
        maxBins=10000,
        featuresCol='features', labelCol='label')

    train_df, test_df = data.randomSplit([0.8, 0.2])

    print("features")
    train_df.select("features").show(1, truncate=False)

    fitted_model = model.fit(train_df)
    predictions = fitted_model.transform(test_df)
    print("columns : ", predictions.columns)
    predictions.select("label", "prediction").show(10)
    return


if __name__ == "__main__":
    main()
