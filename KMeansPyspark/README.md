# Spark Machine Learning with Scala

## Table Of Contents:
  1. [Objective](#objective)
  2. [Example](#example)  
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

## Background

### Required Imports
In order to perform Kmeans with spark you need to use the following imports:
```python
  from pyspark.mllib.clustering import KMeans, KMeansModel
  from numpy import array
  import random
  import numpy as np
```
Numpy is a python package which gives helpful tools to create and manipulate arrays. KMeans and KMeansModel are packages from Pyspark machine learning

### Loading Data
Here we are loading the data to analyze. Notice that it needs to be located on HDFS.
```python
  data = sc.textFile("hdfsdirectory/input.txt")
```

### RDD Conversion
Now we are taking the inputfile and parsing the data out of it line by line. You can define a lambda inline like this or define a local function.
```python
  parsedData = data.map(lambda line: array([float(x) for x in line.split(',')]))
```

### Performing KMeans Clustering
This step is rather simple. Here we are specifying that we want to train a KMeans algorithm on the parsedData with 3 clusters and a max of 10 iterations.
```python
  // 3 clusters, and 10 iterations
  clusters = KMeans.train(parsedData, 3, maxIterations=10, initializationMode="random")
```

### Prediction Data
Here is how you can define a mapper to parse a file:
```python
  def realMapper(line):
    mline = line.split(",")
    return mline[4].encode("utf-8")
  # use the mapper to get the data
  real = data2.map(realMapper).collect()
```
Lets say real is the correct clustering of the data. 
Now, to get the prediction of the data from your KMeans algorithm. This is how you do it:
```python
  rdd = clusters.predict(parsedData)
  pred = rdd.collect()
```

### Results
Now you can create a numpy array of prediction vs real data and print it like this:
```python
  data = np.empty((len(pred), 2))

  data[:,0] = real
  data[:,1] = pred

  print data
```
