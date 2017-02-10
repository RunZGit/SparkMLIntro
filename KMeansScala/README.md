
# Spark Machine Learning with Scala

## Table Of Contents:
  1. [Objective](#Objective) 
  2. [Requirements](#Requirements)  
  3. [Installation](#Installation)
  4. [What is Spark?](#What is Spark?)
  5. [Who and What is Spark Used For?](#Who and What is Spark Used For?)
  6. [Spark MLlib](#Spark MLlib)
  7. [KMeans](#KMeans)
  8. [Example](#Example)
    a. [Required Imports](#required-imports)  
    b. [Loading Data](#loading-data)  
    c. [RDD Conversion](#rdd-conversion)  
    d. [Performing KMeans Clustering](#performing-kmeans-clustering)  
    e. [Prediction Data](#prediction-data)  
    f. [Results](#results)  

## Objective
From this introduction, a student should be able to do simple clustering with quantitative data using scala.

## Requirements
- Hadoop Cluster with Spark and scala installed
- Basic Knowledge of Programming

## Installation
Spark is preinstalled with the standard hadoop clusters. To check if spark is installed on your machine execute the following command:
```bash
spark-shellversion
```
If there is an error or no output then spark is not installed.

To check if scala is installed on your machine, execute the following command:
```bash
scala-version
```

If there is an error or no output then scala is not installed.
If spark and/or scala are not installed on your machine, [click this](https://www.tutorialspoint.com/apache_spark/apache_spark_installation.htm) and follow the steps to install spark and/or scala.

## What is Spark?
On the Apache Spark homepage, Spark claims to be 100x faster than standard Hadoop than MapReduce when performing in-memory computations and 10x faster than Hadoop when performing on-disk computations.

What does this mean? Spark, much like MapReduce, works by distrubuting data across a cluster and processes it parallel; however, unlike your standard MapReduce, most of the data processing occurs in-memory rather than on-disk.

To achieve this, Spark internally maintains what is called Resilient Distributed Datasets (RDDs) which are read-only data stored on a cluster. The RDDs are stored on the cluster in a fault-tolerant. This new data structure was developed to overcome the MapReduce linear dataflow. That is, a typical MapReduce program will read data from disk, run the Map Phase, run the Reduce Phase, and store the Reduced results on disk. [More Information](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf)

Spark natively supports Java, Scala, Python, and R. Unlike Hadoop, Spark does not rely on a Streaming API to work with languages other than Java. Furthermore, Spark supports interactive shells for Scala, Python, and R.

Similar to Hadoop, many powerful libraries utilizes Spark's computation engine to perform data analytics. These libraries include Spark SQL, Spark Streaming, MLlib (Machine Learning Library), and GraphX (Graphing Libarary).

Lastly, Spark supports many different distributive computing setups such as Hadoop, HDFS, Cassandra, HBase, and Mesos.

## Who and What is Spark Used For?
  - [eBay](https://spark.apache.org/powered-by.html)
    - Spark Core for transaction logging, aggregation, and analytics
  - [VideoAmp](https://spark.apache.org/powered-by.html)
    - Intelligent video ads targetting specific online and television viewers
  - [MyFitnessPal](https://spark.apache.org/powered-by.html)
    - Clean-up user specified food data to identify high-quality food items
    - Recommendation engine for recipes and foods.
  - [IBM](http://www.ibmbigdatahub.com/blog/what-spark)
    - IBM SPSS: Spark MLlib algorithms are invoked from IBM SPSS Modeler workflows
    - IBM BigSQL: Spark is used to access data from HDFS, S3, HBase, and other NoSQL databases using IBM BigSQL. Spark RDD is returned to IBM BigSQL for processing.
    - IBM InfoSphere Streams: Spark transformation, action, and MLlib functions can be added to existing Stream application for improved data analytics.
    - IBM Cloudant: Spark analyzes data collected on IBM Bluemix.
  - [Uber](https://www.qubole.com/blog/big-data/apache-spark-use-cases/)
    - Spark is used with Kafka and HDFS in a continuous Extract, Transform, Load pipeline. Uber terabytes of raw userdata into structured data to perform more complicated analytics.
  - [Pinterest](https://www.qubole.com/blog/big-data/apache-spark-use-cases/)
    - Spark Streaming is leverage to perform real-time analytics on how users are engaging with Pins. This allows Pinterest to make relevant information navigating the site.
  - [Yahoo](https://www.datanami.com/2014/03/06/apache_spark_3_real-world_use_cases/)
    - Spark MLlib is used to customize Yahoo's homepage news feed for each user.
    - Spark is also used with Hive to allow Yahoo to query advertisement user data for analytics.

## Spark MLlib
Spark MLlib is the machine learning package of spark. The package has numerous algorithms built in including:
- logistic regression
- SVMs
- k-mean
- decision trees
- ensembles of trees(Random Forests and Gradient-Boosted Trees)
- and many more
You can view more about these [here](https://spark.apache.org/docs/latest/mllib-guide.html).

## KMeans
K-Means clustering is a very popular ML algorithm in [unsupervised learning](https://www.mathworks.com/discovery/unsupervised-learning.html). It is able to group similar data into k groups. The algorithm initially create k random points in the hyperspace. Then each point is clustered based on which cluster center is the closest to the point by the euclidean distance metric. You can then choose the midpoint of each of those clusters and repeat the process again using those new points. This is done until you a specified termination criteria. The result will return a local minimum of the points clustring.

## Example
### Required Imports
```scala
  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.mllib.clustering.KMeans
  import org.apache.spark.sql.functions._
```

### Loading Data
Note, the data needs to be stored on HDFS. "Iris-cleaned.csv" is provided in this repository.
```scala
  val data = sc.textFile("spark_data/Iris-cleaned.csv")

  // Uses the first row as header
  val header = data.first

  // Ignore the first row when load the rest of the data
  val rows = data.filter(l => l != header)

  // Creates the class to hold the data for each row
  case class Iris(index: Int, sepal_length: Double, sepal_width: Double,
     petal_length: Double, petal_width: Double, species: String)

  val allSplit = rows.map(line => line.split(","))
  val allData = allSplit.map(p => Iris(p(0).trim.toInt, p(1).trim.toDouble,
    p(2).trim.toDouble, p(3).trim.toDouble, p(3).trim.toDouble, p(5).toString))
  val allDF = allData.toDF()
```

### RDD Conversion
We are converting the DataFrame to RDD rows so we can feed the data into the KMean classifier.
```scala
  val rowsRDD = allDF.rdd.map(r => (r.getInt(0), r.getDouble(1), r.getDouble(2),
    r.getDouble(3), r.getDouble(4), r.getString(5)))
  rowsRDD.cache()
  val vectors = allDF.rdd.map(r => Vectors.dense( r.getDouble(1), r.getDouble(2),
    r.getDouble(3), r.getDouble(4)))
  vectors.cache()
```

### Performing KMeans Clustering
This step is rather simple, in order to trian the classifier, all you need to do is to instruct it how many clusters and how many iterations there are.
```scala
  // 3 clusters, and 100 iterations
  val kMeansModel = KMeans.train(vectors, 3, 100)
```
To see the center of each clusters:
```scala
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
  val t = allDF.join( predDF, "index")
  t.filter("CLUSTER = 0").show(100)
  t.filter("CLUSTER = 1").show(100)
  t.filter("CLUSTER = 2").show(100)
```
