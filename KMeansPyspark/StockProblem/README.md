# PySpark Machine Learning Stock Market Task
## Stock Market
Suppose you are hired to predict stock market fluxuations by a big name company. Your data looks like this:
```
2015-09-16:SPY,UP,2.76708,-3.28704,200.18,91.5775,81.572,84,73.2035,79.5918
```
You realize that one way to make the big bucks is to predict the UP/Down (in the 2nd column of the dataset) fluxuation of stocks would be to use K-Means.
Your task is to write a PySpark script in which it will create a Kmeans Model to predict stock price using the provided spykmeans.txt file.
You should output the accuracy of your model (total amount correctly predicted/total number of data points).

## Hints
- Work with tuples rather than lists
- You can use numpy to concatenate the lists of the predicted clusters and real clusters
