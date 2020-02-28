# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:27:13 2019

@author: hari4
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset 
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

#Cleaning the given text data
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
    
#Importing Bag of words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
x_mtx = cv.fit_transform(corpus).toarray()

y_vctr = dataset.iloc[:, -1].values

#Train Test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_mtx, y_vctr, test_size=0.20, random_state=0)

# Running all Classsifier models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifier_models = [LogisticRegression(random_state=0), 
                     KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2), 
                     SVC(kernel='linear', random_state=0), SVC(kernel='rbf', random_state=0), 
                     GaussianNB(), DecisionTreeClassifier(criterion="entropy", random_state=0),
                     RandomForestClassifier(n_estimators=40, criterion="entropy", random_state=0)]

y_prdc_results = [model.fit(x_train, y_train).predict(x_test) for model in classifier_models]

# importing confusion matrix
from sklearn.metrics import confusion_matrix, classification

cm_results = [confusion_matrix(y_test, y_prdc_results[i]) for i in range(0, len(classifier_models))]

acc_results = [classification.accuracy_score(y_test, y_prdc_results[i]) for i in range(0, len(classifier_models))]
prec_results = [classification.precision_score(y_test, y_prdc_results[i]) for i in range(0, len(classifier_models))]
recal_results = [classification.recall_score(y_test, y_prdc_results[i]) for i in range(0, len(classifier_models))]
f1_results = [classification.f1_score(y_test, y_prdc_results[i]) for i in range(0, len(classifier_models))]
x_label = ["LR", "KNN", "SVC(L)", "SVC(NL)", "NB", "DT", "RF"]

#Visualization of Results
plt.ylim(0, max(acc_results))
plt.bar(x_label, acc_results)
plt.title("Histogram view of accuracy")
plt.xlabel("Classification Models")
plt.ylabel("Accuracy")
plt.show()

plt.ylim(0, max(prec_results))
plt.bar(x_label, prec_results)
plt.title("Histogram view of precision")
plt.xlabel("Classification Models")
plt.ylabel("Precision")
plt.show()

plt.ylim(0, max(recal_results))
plt.bar(x_label, recal_results)
plt.title("Histogram view of recall")
plt.xlabel("Classification Models")
plt.ylabel("Recall")
plt.show()

plt.ylim(0, max(f1_results))
plt.bar(x_label, f1_results)
plt.title("Histogram view of F1 score")
plt.xlabel("Classification Models")
plt.ylabel("F1 Score")
plt.show()