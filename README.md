# Spark Machine Learning Introduction:
## Table Of Contents:
  1. [Objective](#Objective) 
  2. [What is Spark?](#What-is-Spark?)
  3. [Who and What is Spark Used For?](#Who-and-What-is-Spark-Used-For?)
  4. [Spark MLlib](#Spark-MLlib)
  5. [KMeans](#KMeans)
  6. [Get Started](#Getting-Started)

## Objective
From this introduction, a student should be able to do simple clustering with quantitative data using spark.

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
- K-Means
- Decision trees
- Ensembles of trees(Random Forests and Gradient-Boosted Trees)
- and many more
You can view more about these [here](https://spark.apache.org/docs/latest/mllib-guide.html).

## KMeans
K-Means clustering is a very popular ML algorithm in [unsupervised learning](https://www.mathworks.com/discovery/unsupervised-learning.html). It is able to group similar data into k groups. The algorithm initially create k random points in the hyperspace. Then each point is clustered based on which cluster center is the closest to the point by the euclidean distance metric. You can then choose the midpoint of each of those clusters and repeat the process again using those new points. This is done until you a specified termination criteria. The result will return a local minimum of the points clustring.

## Getting Started 
  - [Scala Lab](https://github.com/RunZGit/SparkMLIntro/tree/master/KMeansScala)
  - [Python Lab](https://github.com/RunZGit/SparkMLIntro/tree/master/KMeansPyspark)
