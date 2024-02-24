<H3>Name:Aruru Sai Bandhavi</H3>
<H3>Register No: 212221240006</H3>
<H3>Experiment 2</H3>
<H3>Date:24-02-2024</H3>
<h1 align =center>Implementation of Exact Inference Method of Bayesian Network</h1>

## Aim:
To implement the inference Burglary P(B| j,⥗m) in alarm problem by using Variable Elimination method in Python.

## Algorithm:

Step 1: Define the Bayesian Network structure for alarm problem with 5 random variables, Burglary,Earthquake,John Call,Mary Call and Alarm.<br>
Step 2: Define the Conditional Probability Distributions (CPDs) for each variable using the TabularCPD class from the pgmpy library.<br>
Step 3: Add the CPDs to the network.<br>
Step 4: Initialize the inference engine using the VariableElimination class from the pgmpy library.<br>
Step 5: Define the evidence (observed variables) and query variables.<br>
Step 6: Perform exact inference using the defined evidence and query variables.<br>
Step 7: Print the results.<br>

## Program :
```
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
class BayesClassifier:
  def __init__(self):
    self.clf = GaussianNB()
  def fit(self, X, y):
    self.clf.fit(X, y)
  def predict(self, X):
    return self.clf.predict(X)
ir = load_iris()
X_train, X_test, y_train, y_test = train_test_split(ir.data, ir.target,test_size=0.33, random_state = 33)
clf = BayesClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accu = accuracy_score(y_test, y_pred)
print("Accuracy:",accu*100)
```


## Output :
![image](https://github.com/Saibandhavi75/Ex2---AAI/assets/94208895/d359f6f0-cc6d-4e99-9211-f901126f861d)


## Result :
Thus, Bayesian Inference was successfully determined using Variable Elimination Method

