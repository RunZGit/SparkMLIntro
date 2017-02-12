# Spark Machine Learning with Pyspark

## Table Of Contents:
  1. [Objective](#objective)
  2. [Requirements](#requirements)
  3. [Installation](#installation)
  4. [Example](#example)   
    a. [Required Imports](#required-imports)    
    b. [Loading Data](#loading-data)  
    c. [RDD Conversion](#rdd-conversion)  
    d. [Performing KMeans Clustering](#performing-kmeans-clustering)  
    e. [Prediction Data](#prediction-data)  
    f. [Real Data](#real-data)  
    f. [Results](#results)    
  5. [Tasks](#tasks)  

## Objective
From this introduction, a student should be able to do simple clustering with quantitative data using pyspark.

## Requirements
- Hadoop Cluster with pyspark installed
- Basic Knowledge of Programming

## Installation
PySpark comes standard with most Spark installation. If you have Ambari installed, both Spark and Spark2 can be installed via the Ambari Web UI.

Otherwise, please ensure that Java 7+ is installed. To check if Java is installed, run the following command:
```
  $ java -version
```
If the command does not run, then Java is not installed. If the version outputted is not `1.7` or above, then please upgrade your Java version.

We are now ready to install Spark. Download Spark [here](https://spark.apache.org/downloads.html) and extract the Spark tar file with the following command:
```
  $ tar xvf <spark.tar>
```

It is suggested that you move the Spark files from your downloads to some standard directory such as `/usr/local/spark`. You can do so by running the following command:
```
  cd <spark_download_location>
  mv <spark_binary> /usr/local/spark
```

Now we need to update the `~/.bashrc` file by adding the path the Spark binary location. To do so, run the following command:
```
  export PATH = $PATH:/usr/local/spark/bin
```

Spark should now be installed. To verify the installation, run the command `pyspark`. You should get a Python REPL with Spark integration.

Installation Adapted from [TutorialPoint](https://www.tutorialspoint.com/apache_spark/apache_spark_installation.htm)

## Example
Here is an example of perfoming KMeans clustering with pyspark on a small dataset of datapoints:
```
1,1,0
2,2,0
3,2,0
4,1,0
10,1,1
11,11,1
```
The first two columns are the x-y coordinates while the third column is the classification.
### Required Imports
In order to perform Kmeans with Spark you need to use the following imports:
```python
  from pyspark.mllib.clustering import KMeans, KMeansModel
  from numpy import array
  import random
  import numpy as np
```
Numpy is a python package which gives helpful tools to create and manipulate arrays. KMeans and KMeansModel are packages from Pyspark machine learning.

### Loading Data
Here we are loading the data to analyze. Notice that it needs to be located on HDFS.
```python
  data = sc.textFile("hdfsdirectory/input.txt")
```

### RDD Conversion
Now we are taking the inputfile and parsing the data out of it line by line. You can define a define a local function like this or use a lambda function.
```python
  def mapper(line):
    mline = line.split(",")
    return (int(mline[0]),int(mline[1]))
  
  parsedData = data.map(mapper)
  
  # or 
  
  parsedDate = data.map(lambda line: tuple(line.split(",")[0:2])
```

### Performing KMeans Clustering
This step is rather simple. Here we are specifying that we want to train a KMeans algorithm on the parsedData with 2 clusters, since we have two classes to classify, and a max of 10 iterations.
```python
  // 2 clusters, and 10 iterations
  clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode="random")
```

### Prediction Data
Now we are going to get the prediction of our original data from the KMeans model we just created:
```python
  rdd = clusters.predict(parsedData)
  pred = rdd.collect()
``` 

### Real Data
Lets get the real cluster labels from our data set:
```python
  def realMapper(line):
    mline = line.split(",")
    return int(mline[3])
  real = data.map(realMapper).collect()
```

### Results
Now you can create a numpy array of prediction vs real data and print it like this:
```python
  data = np.empty((len(pred), 2))

  data[:,0] = real
  data[:,1] = pred

  print data
```
here are two example outputs:
```
0,0
0,0
0,0
0,0
1,1
1,1
```

```
0,1
0,1
0,1
0,1
1,0
1,0
```
As you can see the outputs are different. This is because the Kmeans model chooses how to label the clusters, so they could be different from what you choose to be your labels.

## Tasks
1. [Iris Task](https://github.com/RunZGit/SparkMLIntro/tree/master/KMeansPyspark/IrisData)
2. [Stock Data Task](https://github.com/RunZGit/SparkMLIntro/tree/master/KMeansPyspark/StockProblem)
