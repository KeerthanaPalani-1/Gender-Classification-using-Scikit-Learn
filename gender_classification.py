from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

#In order to identify the classifier with the highest level of precision, it is imperative to incorporate the numpy package along with the accuracy_score function.
import numpy as np
from sklearn.metrics import accuracy_score

#training data
#[height, weight, shoe size]
X = [[181,80,44], [177,70,43], [160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,65,40],[171,75,42],[181,85,43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

#test data
test_data = [[190, 70, 43],[154, 75, 38],[181,65,40]]
test_labels = ['male','female','male']

#Gender Classification using 7 classification models of machine learning
#LogisticRegression
clf = LogisticRegression()
clf = clf.fit(X,Y)
prediction_lr = clf.predict(test_data)
print ("Logistic Regression Model Result: ",prediction_lr)

#K-Nearest Neighbors (KNN)
clf = KNeighborsClassifier()
clf = clf.fit(X,Y)
prediction_knn = clf.predict(test_data)
print ("K-Nearest Neighbors  Model Result: ",prediction_knn)

#Support Vector Machines (SVM)
clf = SVC()
clf = clf.fit(X,Y)
prediction_svc = clf.predict(test_data)
print ("Support Vector Machines Model Result: ",prediction_svc)

#DecisionTree 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
prediction_tree = clf.predict(test_data)
print ("Decision Tree Model Result: ",prediction_tree)

#RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(X,Y)
prediction_rf = clf.predict(test_data)
print ("Random Forest Model Result: ",prediction_rf)

#Naive Bayes
clf = GaussianNB()
clf = clf.fit(X,Y)
prediction_NB = clf.predict(test_data)
print ("Naive Bayes Model Result: ",prediction_NB)

#Gradient Boosting
clf = GradientBoostingClassifier()
clf = clf.fit(X,Y)
prediction_GB = clf.predict(test_data)
print ("Gradient Boosting Model Result: ",prediction_GB)

#accuracy scores
lr_acc=accuracy_score(prediction_lr,test_labels)  
knn_acc=accuracy_score(prediction_knn,test_labels)
svc_acc = accuracy_score(prediction_svc,test_labels)
tree_acc = accuracy_score(prediction_tree,test_labels)
rf_acc = accuracy_score(prediction_rf,test_labels)
nv_acc=accuracy_score(prediction_NB,test_labels)
gb_acc=accuracy_score(prediction_GB,test_labels)

#finding accurate model
classifiers = ['Logistic Regression', 'K-Nearest Neighbors', 'SVC','Decision Tree', 'Random Forest','Naive Bayes','Gradient Boosting']
accuracy = np.array([lr_acc, knn_acc, svc_acc, tree_acc, rf_acc, nv_acc, gb_acc])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc] + ' is the accurate classifier for this gender classification')


