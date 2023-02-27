
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.regression import LinearRegression


conf = SparkConf().setMaster("local[*]").setAppName("Pipeline")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)


data = spark.read.csv('./predictions.csv',header=True,inferSchema=True)

columns_to_drop = ['features', 'scaled']
data = data.drop(*columns_to_drop)



features = ['energy-kcal_100g',
 'energy_100g',
 'fat_100g',
 'saturated-fat_100g',
 'carbohydrates_100g',
 'sugars_100g',
 'proteins_100g',
 'salt_100g',
 'sodium_100g']



assemble = VectorAssembler(inputCols=features, outputCol='features')
assembled_data = assemble.setHandleInvalid("skip").transform(data)
assembled_data.show(5)


scale=StandardScaler(inputCol='features',outputCol='scaled')
scaled=scale.fit(assembled_data)
data_scaled=scaled.transform(assembled_data)
data_scaled.show(5)



cols = ["scaled", "prediction"]
scaled_data = data_scaled.select(*cols)
scaled_data1 = scaled_data.withColumnRenamed("prediction","claster_pred")

#разделим выборку на обучающую и тестовую
train_df, test_df = scaled_data1.randomSplit(weights = [0.70, 0.30], seed = 42)




lr = LinearRegression(featuresCol = 'scaled', labelCol='claster_pred', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))



#метрики линейной регрессии
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# сравним rmse со средним значением 
train_df.describe().show()
lr_test = lr.fit(test_df)

print("Coefficients: " + str(lr_test.coefficients))
print("Intercept: " + str(lr_test.intercept))

trainingSummary_test = lr_test.summary
print("RMSE: %f" % trainingSummary_test.rootMeanSquaredError)
print("r2: %f" % trainingSummary_test.r2)


#Random Forest

RandomForest = RandomForestClassifier(labelCol="claster_pred", featuresCol="scaled",
                        predictionCol='rand_prediction', numTrees=20, maxDepth=3)

model_rand = RandomForest.fit(train_df)

predictions_rand = model_rand.transform(test_df)


# Select (prediction, true label) and compute test error
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="claster_pred", predictionCol="rand_prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions_rand)
print("Test Error = %g" % (1.0 - accuracy))


evaluator_pr = MulticlassClassificationEvaluator(
    labelCol="claster_pred", predictionCol="rand_prediction", metricName="precision")
precision = evaluator.evaluate(predictions_rand)
print(f"Precision = {precision} ")


log_reg = LogisticRegression(featuresCol = 'scaled', labelCol = 'claster_pred', maxIter=10)
lrModel = log_reg.fit(train_df)


# Силуэтные оценки для разного количества кластеров
silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='scaled', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')

for i in range(2,15):
    
    KMeans_algo=KMeans(featuresCol='scaled', k=i)
    
    KMeans_fit=KMeans_algo.fit(scaled_data)
    
    output=KMeans_fit.transform(scaled_data)
    
    
    score=evaluator.evaluate(output)
    
    silhouette_score.append(score)
    
    print(f"Clusters: {i}, Silhouette Score: {score}")

#создадим pipeline для автоматизации работы spark приложения
pipeline = Pipeline(stages=[assemble, scale, lr, RandomForest])

pipelineModel = pipeline.fit(scaled_data)
model = pipelineModel.transform(test_df)

