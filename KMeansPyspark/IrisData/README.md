# PySpark Machine Learning Iris Dataset Task
## Iris Dataset
Now that you know how to perform Kmeans with Pyspark, lets try doing it with a small dataset. The "Hello World" dataset of the Machine Learning community is called the Iris Dataset.
There are three types of irises in the data set: Setosa, Versicolor, and Virginica. Here is an example line of the dataset:
```
5.1,3.5,1.4,0.2,Iris-setosa
```
Your task is to create a Kmeans clustering model of the Iris Dataset contianed in the iris.txt file. Use 3 clusters.
The output of your algorithm should be a two column print out of a numpy array with the predicted clusters on one side and the real clusters on the other. 

## Hints
- Use two mappers, one for getting the numerical data and the other for getting the classification
- You can use numpy to concatenate the lists of the predicted clusters and real clusters
