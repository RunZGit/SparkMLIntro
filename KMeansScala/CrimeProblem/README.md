# Kmeans on Crime   
You are given the crimes committed of 50 US states in 1973 per 100,000 residents. The data contains the murder, assault, and rape rate as well as urbanpopulation.   
The columns of the dataset looks like this
```
State |	crime_cluster |	Murder | Assault | UrbanPop | Rape

```
Your task is to cluster the cities into 4 clusters based on the crime rate, a sample clustring result is in the data for you to compare your clustering output. Print out the result of the cluster centers as well as the prediction. You also need to provide a mean squared error of the model.


## HINT
- It is okay if the spark clustering result is not ideal, since it does not supports.  
- For mean squared error, look up computCost() function in this [link](https://spark.apache.org/docs/latest/mllib-clustering.html)
