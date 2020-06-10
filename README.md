# Project Documentation
Welcome! This is Dawei Huang's project documentation for UChicago Ethics Project: Interpetability of Machine Learning. NOTE THE INTRODUCTION IS NOT MY ABSTRACT. The introduction just includes more information that I was not able to fit into my abstract.

## Introduction
The interpretability of machine learning algorithms must be accessed since the legitimacy of implementation must rely on whether or not humans have a good understanding of the decisions promoted by such algorithms given certain inputs. While some ML models based on linear regressions and decision trees are more readily interpretable, others like neural networks and random forst algorithms are rather ambiguous due to the complexity of its inner mechanisms. The reference of the latter two models as "black box" models is a nod to the difficulty in unraveling its mechanisms and thus would be much less interpretable to human investigation. 

For the purposes of this project, we will explore one avenue of ML interpretability which states that a model is interpretable in which the contributions of each feature(variable input) could be properly accessed. Of course, there exists already exists model-agnostic methods such as Sharpley Values and Partial Dependency Plots (PDP) that, in a broad sense, measures how marginal changes in a variable relative to other variables effects the output of any "black-box" algorithm. Understandibly these model-agnostic accessment methods are very computationally expensive especially for very large datasets so it would be advantageous in some cases to use an alternative model-agnostic method that are much less expensive computationally at the cost of assessment accuracy.

Following this specific definition of ML interpretability, I have developed a randomGen model-agnostic model that accesses the contribution of each variable input to the final output. For each variable column input, for example, I will randomly select an entry to be randomly altered (bounded by the variable column minimum and maximum), run the new inputs through the black-box algorithm, and then use cosine similarity to measure how significantly the new output deviates from the initial output. A graph can be created for each random variable of # random changes vs deviation from the initial output. Note that a variable is more significant to the model algorithm if it experiences a much steeper deviation from the initial outpuut following a sequence of random changes. The linear regression slopes of each line plot are given but should only be used for consideration.

Keep in mind that this randomGen model-agnostic method is by no means a definitive assessment on the weight of each variable on the model as this model is mainly dependent on the generation of psuedo-random data. Therefore it would be prudent to run randomGen a few times to get a better sense of each variable's weight and to correct for any possible outlying extreme graphs. We expect that randomGen is most successful in isolating variables that significantly have a higher weight than other variables.

This approach is inspired by the Monte Carlo Method for Approximating Pi where randomly generated points are used to approximate Pi. Check out here: https://en.wikipedia.org/wiki/Monte_Carlo_method.

## Project.ipynb
In this jupyter notebook file, we will be testing this randomGen model-agnostic model on both interpretable and black box models. We will be frequently comparing our randomGen method with the Shapley Value method because they both assess the weights/contribution of each variable to the output. In addition we will be making note of our observations here as well.

## How to run randMethod.py
Here is an example on how we use randomGen to access the weights of variables in a linear regression:
```
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#data = pd.read_csv('airfoil_self_noise.dat')
data = pd.read_table("airfoil_self_noise.dat")


X = data.loc[:, data.columns != 'Soundpressure']
y = data['Soundpressure']
    
randMethod.createGraph(X,y,list(X.columns),300, LinearRegression(), False) 
```
![Image of Graph](https://github.com/huangdaweiUCHICAGO/ethicsProject/blob/master/misc/graph.PNG)

Graph without smoothing
```
randMethod.createGraph(X,y,list(X.columns),100, RandomForestRegressor(),False) #without smoothing 
```
![Image of Graph](https://github.com/huangdaweiUCHICAGO/ethicsProject/blob/master/misc/graph1.PNG)

Graph with smoothing
```
randMethod.createGraph(X,y,list(X.columns),100, RandomForestRegressor(),True) #with smoothing 
```
![Image of Graph](https://github.com/huangdaweiUCHICAGO/ethicsProject/blob/master/misc/graph2.PNG)

createGraph(X,y,cols,i,func):
* X: dataframe representing input variables and input data
* y: dataframe representing ideal output
* cols: which variables should be displayed on the graph
* i: # of random change iterations for each variable category in X
* func: machine learning algorithm (could be an interpretable or black box algorithm)
* smooth: (True/False) smooths out the graph for aesthetic purposes

## Relevant Files/Links
* randMethod.py: Contains the code to the RandomGen model agnostic method. "import randMethod" in order to get access to the relevant functions in the file.
* project.ipynb: Contains examples and possible applications of the RandomGen model agnostic method on different kinds of dataSets. Refer to this file for more information on how to run my code. Also includes insights that I gathered while running my code. 
* abstract.pdf: Abstract (1 page) summarizing motivation, methods, main insights, and conclusions of my project
* projectProposal: My project proposal.
* risk_factors_cervical_cancer.csv: Risks Factors -> Cervical Cancer Diagnosis database
* hour.csv: Environmental Conditions -> # of bikes rented per hour
* Link to 3-minute Video:
