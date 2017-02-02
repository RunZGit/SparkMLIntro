# Spark Machine Learning Introduction:
## Description
This repo provides two introductory lab on machine learning in spark. Both lab are concentrates on using K-mean algorithm to cluster the iris dataset. One lab uses Scala, the other uses python.
## Requirements
- Hadoop Cluster with Spark Installed
- Spark-Shell or Zeppelin Notebook Installed
## Instructions:
### Imports:
```scala
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.sql.functions._
```
### Load data:
Note, the data needs to be stored on HDFS. "Iris-cleaned.csv" is provided in this repository.
```scala
 val data = sc.textFile("spark_data/Iris-cleaned.csv")
 //Uses the first row as header
 val header = data.first
 //Ignore the first row when load the rest of the data
 val rows = data.filter(l => l != header)
 // Creates the class to hold the data for each row
 case class Iris(index: Int, sepal_length: Double, sepal_width: Double, petal_length: Double, petal_width: Double, species: String)
val allSplit = rows.map(line => line.split(","))
val allData = allSplit.map(p=>Iris(p(0).trim.toInt, p(1).trim.toDouble, p(2).trim.toDouble, p(3).trim.toDouble, p(3).trim.toDouble, p(5).toString))
val allDF = allData.toDF()
```
### Convert data to RDD:
We are converting the DataFrame to RDD rows so we can feed the data into the KMean classifier.
```scala
val rowsRDD = allDF.rdd.map(r=> (r.getInt(0), r.getDouble(1), r.getDouble(2), r.getDouble(3), r.getDouble(4), r.getString(5)))
rowsRDD.cache()
val vectors = allDF.rdd.map(r=> Vectors.dense( r.getDouble(1), r.getDouble(2), r.getDouble(3), r.getDouble(4)))
vectors.cache()
```
### KMeans:
This step is rather simple
```scala
//3 clusters, and 100 iterations
val kMeansModel = KMeans.train(vectors, 3, 100)
```
To see the center of each clusters:
```scala
kMeansModel.clusterCenters.foreach(println)
```
### Prediction:
```scala
val predictions = rowsRDD.map{r => (r._1, kMeansModel.predict(Vectors.dense(r._2, r._3, r._4, r._5) ))}
val predDF = predictions.toDF("index", "CLUSTER")
```
### Result:
```scala
//Join the prediction with our original dataframe using the index.
val t = allDF.join( predDF, "index")
t.filter("CLUSTER = 0").show(100)
t.filter("CLUSTER = 1").show(100)
t.filter("CLUSTER = 2").show(100)
```
