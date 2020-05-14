from sklearn import tree
from sklearn.datasets import load_iris
import graphviz
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus
import graphviz

train = pd.read_csv('trainDataset.csv')
test = pd.read_csv('testDataset.csv')
train=train.replace('child',0).replace('teenage',1).replace('adult',2)
train=train.replace('male',1).replace('female',0)
test=test.replace('child',0).replace('teenage',1).replace('adult',2)
test=test.replace('male',1).replace('female',0)
y_train = train.pop('Survived')
reg = DecisionTreeClassifier(criterion="gini")
reg.fit(train, y_train)

dot_data = tree.export_graphviz(reg,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render()
