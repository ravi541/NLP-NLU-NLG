# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:43:56 2026

@author: Raviteja
"""

#Importing libarires
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset =pd.read_csv(r"C:\Users\Raviteja\Downloads\Restaurant_Reviews.tsv",delimiter='\t',quoting=3)

# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corupus=[]

for i in range(0,1000):
    review=re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corupus.append(review)
    
    
# Creating the bag of words model

from sklearn.feature_extraction.text import CountVectorizer

cv= CountVectorizer()
X=cv.fit_transform(corupus).toarray()

y=dataset.iloc[:,1].values
    

# Splitting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred= classifier.predict(X_test)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(X_train,y_train)
bias

variance = classifier.score(X_test, y_test)
