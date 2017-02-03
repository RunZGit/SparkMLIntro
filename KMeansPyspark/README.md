# Spark Machine Learning with Scala

## Table Of Contents:
  1. [Example](#example)  
    a. [Required Imports](#required-imports)  
    b. [Loading Data](#loading-data)  
    c. [RDD Conversion](#rdd-conversion)  
    d. [Performing KMeans Clustering](#performing-kmeans-clustering)  
    e. [Prediction Data](#prediction-data)  
    f. [Results](#results)  

## Example
### Required Imports
```python
  from pyspark.mllib.clustering import KMeans, KMeansModel
  from numpy import array
  from math import sqrt
  import random
  import numpy as np
```

### Loading Data
Note, the data needs to be stored on HDFS. "iris.txt" is provided in this repository.
```python
  data = sc.textFile("/tmp/pysparkinput/iris.txt")
```

### RDD Conversion
We are converting the DataFrame to RDD rows so we can feed the data into the KMean classifier.
```python
  parsedData = data.map(lambda line: array([float(x) for x in line.split(',')]))
```

### Performing KMeans Clustering
This step is rather simple
```python
  // 3 clusters, and 10 iterations
  clusters = KMeans.train(parsedData, 3, maxIterations=10, initializationMode="random")
```

### Prediction Data
The first step is to import the additional textFile "iris2.txt" and get all the true mappings.
```python  
  data2 = sc.textFile("/tmp/pysparkinput/iris2.txt")
```
Next, define a mapper to get the names from the file
```python
  def realMapper(line):
    mline = line.split(",")
    return mline[4].encode("utf-8")
  # use the mapper to get the data
  real = data2.map(realMapper).collect()
```
Now, to get the prediction
```python
  rdd = clusters.predict(parsedData)
  pred = rdd.collect()
```

### Results
Create a numpy array of prediction vs real data and print it
```python
  data = np.empty((len(pred), 2), dtype='|S20')

  data[:,0] = real
  data[:,1] = pred

  print data
```
