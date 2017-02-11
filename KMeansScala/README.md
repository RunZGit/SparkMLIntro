# Spark Machine Learning with Scala

## Table Of Contents:
  1. [Objective](#Objective) 
  2. [Requirements](#Requirements)  
  3. [Installation](#Installation)
  4. [Example](#Example)  
    a. [Required Imports](#required-imports)  
    b. [Loading Data](#loading-data)  
    c. [Extract the features from dataset using PCA](#extract-the-features-from-dataset-using-pca)   
    d. [RDD Conversion](#rdd-conversion)   
    e. [Performing KMeans Clustering](#performing-kmeans-clustering)  
    f. [Prediction Data](#prediction-data)  
    g. [Results](#results) 
    
## Objective
  - Clustering the Iris dataset
  
## Requirements
- Hadoop Cluster with Spark and scala installed
- Basic Knowledge of Programming
- Zepplein notebook

## Installation
Spark is preinstalled with the standard hadoop clusters. To check if spark is installed on your machine execute the following command:
```bash
spark-shell
```
If spark and/or scala are not installed on your machine, [click this](https://www.tutorialspoint.com/apache_spark/apache_spark_installation.htm) and follow the steps to install spark and/or scala.
If your zeppelin is not enabled, you can enable it by following [this link](http://hortonworks.com/hadoop-tutorial/apache-zeppelin-hdp-2-4/)
Then go to your cluster port: 9995 to access zeppelin.

## Example
### Required Imports
```dep
 %dep

// IMPORTANT! 
// This step/paragraph must be executed FIRST; if you have already executed other commands/paragraphs, 
//   please click "Interpreter" in the menu above and restart the "spark" interpreter and then run this paragraph
//   before any other one.

z.reset()
z.load("com.databricks:spark-csv_2.11:1.4.0")   // Spark CSV package
```
Make sure this two blocks of imports are called seperately, since they are in different programing languages
```scala
import org.apache.spark.ml.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.clustering.KMeans
```

### Loading Data
Note, the data needs to be stored on HDFS /user/zeppelin directory. "Iris-cleaned.csv" is provided in this repository.
```scala
val df = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true")       // Use first line of all files as header
    .option("inferSchema", "true")  // Automatically infer data types
    .load("Data.csv") // Read all Iris data
```

### Extract the features from dataset using PCA
Sometimes, data scientists uses principal components analysis(PCA) to reduce the dimentionality of the data. This allows us to visulize the data. This lab will only show how to find the points of PCAs, it is up to you if you want to figure out how to print the data.
```scala
val assembler = new VectorAssembler()
  .setInputCols(Array("column name",....,"column name"))
  .setOutputCol("features")

val feature_vector = assembler.transform(df.select("column name",....,"column name"))

val pca = new PCA()
  .setInputCol("features")
  .setOutputCol("pcaFeatures")
  .setK(10) //number of principle components
  .fit(output)
  
val pcaDF = pca.transform(output)
val result = pcaDF.select("pcaFeatures")
result.show(false)
```

### RDD Conversion
We are converting the DataFrame to RDD rows so we can feed the data into the KMean classifier.
```scala
//Converting the entrie data to RDD
val rowsRDD = df.rdd.map(r => (r.getInt(0), r.getDouble(1), ....., r.getString(n)))
//Converting the features that are used for the input of KMean
val vectors = df.rdd.map(r => Vectors.dense( r.getDouble(1), ....., r.getDouble(n)))
```

### Performing KMeans Clustering
This step is rather simple, in order to trian the classifier, all you need to do is to instruct it how many clusters and how many iterations there are.
```scala
  // n clusters, and x iterations
  val kMeansModel = KMeans.train(vectors, n, x)
```
To see the center of each clusters:
```scala
println("Final Centers: ")
 kMeansModel.clusterCenters.foreach(println)
```

### Prediction Data
```scala
val predictions = rowsRDD.map{r => (r._1, kMeansModel.predict(Vectors.dense(r._2, r._3, r._4, r._5) ))}
val predDF = predictions.toDF("index", "CLUSTER")
```

### Results
```scala
  // Join the prediction with our original dataframe using the index.
  val t = df.join( predDF, "index")
  t.filter("CLUSTER = 0").show(100)
  t.filter("CLUSTER = 1").show(100)
  t.filter("CLUSTER = 2").show(100)
```

