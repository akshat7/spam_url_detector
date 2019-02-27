from functools import wraps
from subprocess import call

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_spli
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from collections import Counter
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import random
import math
import sys
import os

def accuracy_statistics(y_true, y_pred):
    
    '''
        Note - Condition is present means that URL is bad.

        Definition 1. A true positive test result is one that detects the condition when the
        condition is present.

        Definition 2. A true negative test result is one that does not detect the condition when
        the condition is absent.

        Definition 3. A false positive test result is one that detects the condition when the
        condition is absent.

        Definition 4. A false negative test result is one that does not detect the condition when
        the condition is present. 
    '''
    
    tp = tn = fp = fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 'good' and y_pred[i] == 'good':
            tn += 1
        elif y_true[i] == 'bad' and y_pred[i] == 'bad':
            tp += 1
        elif y_true[i] == 'good' and y_pred[i] == 'bad':
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn
        

def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')   #get tokens after splitting by slash
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')  #get tokens after splitting by dash
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')  #get tokens after splitting by dot
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))    #remove redundant tokens
    return allTokens

def TL():
    allurls = 'data4L.csv' #path to our all urls file
    allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False) #reading file
    allurlsdata = pd.DataFrame(allurlscsv)  #converting to a dataframe

    allurlsdata = np.array(allurlsdata) #converting it into an array
    random.shuffle(allurlsdata) #shuffling

    y = [d[1] for d in allurlsdata] #all labels 
    corpus = [d[0] for d in allurlsdata]    #all urls corresponding to a label (either good or bad)
    vectorizer = TfidfVectorizer(tokenizer=getTokens)   #get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)    #get the X vector
#     print X

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   #split into training and testing set 80/20 ratio

#     lgs = LogisticRegression()  #using logistic regression
#     lgs = KNeighborsClassifier()
#     lgs = svm.SVC()
    lgs = tree.DecisionTreeClassifier()
#     lgs = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(50,2), random_state=1)

    lgs.fit(X_train, y_train)
    print(lgs.score(X_test, y_test))    #pring the score. It comes out to be 98%
    y_pred = lgs.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print "True Negatives: ", tn
    print "True Positives: ", tp
    print "False Negatives: ", fn
    print "False Positives: ", fp
    print len(y_test), len(y_pred)
    return vectorizer, lgs

vectorizer, lgs  = TL()
# checking some random URLs. The results come out to be expected. The first two are okay and the last four are malicious/phishing/bad

X_predict = ['akshat.','google.com/search=faizanahad','pakistanifacebookforever.com/getpassword.php/','www.radsport-voggel.de/wp-admin/includes/log.exe','ahrenhei.without-transfer.ru/nethost.exe','www.itidea.it/centroesteticosothys/img/_notes/gum.exe']

X_predict = vectorizer.transform(X_predict)

y_Predict = lgs.predict(X_predict)

print(y_Predict)    #printing predicted values

