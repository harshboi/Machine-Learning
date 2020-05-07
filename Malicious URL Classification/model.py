#!/usr/bin/python

import json, sys, getopt, os

#######################################################################################################
# For running the model, type to command line:
# python3 model.py train.json classify.json
#######################################################################################################

import json
with open(sys.argv[1], "r", errors='ignore') as file:
    corpus = json.load(file)

with open(sys.argv[2], "r", errors='ignore') as file:
    test_corpus = json.load(file)

max_alexa_rank = 0
for i in range(len(corpus)):
    max_alexa_rank = max(int(corpus[i]['alexa_rank']) if corpus[i]['alexa_rank'] is not None else 0, max_alexa_rank)
max_alexa_rank += 1

import numpy as np
import sys

data = []
labels = []
test_data = []
for i in range(len(corpus)):
    data.append([0 if corpus[i]['alexa_rank'] == None else int(corpus[i]['alexa_rank']), int(corpus[i]['default_port']), 
                 int(corpus[i]['domain_age_days']), int(corpus[i]['port']), int(corpus[i]['num_domain_tokens']),
                 int(corpus[i]['num_path_tokens'])])
    labels.append(int(corpus[i]['malicious_url']))

for i in range(len(test_corpus)):
    test_data.append([0 if test_corpus[i]['alexa_rank'] == None else int(test_corpus[i]['alexa_rank']), int(test_corpus[i]['default_port']), 
                int(test_corpus[i]['domain_age_days']), int(test_corpus[i]['port']), int(test_corpus[i]['num_domain_tokens']),
                int(test_corpus[i]['num_path_tokens'])])
data = np.asarray(data)
labels = np.asarray(labels)

test_data = np.asarray(test_data)


for i in range(len(data)):
    for j in range(len(data[i])):
        if (data[i][j] < 0):
#             print(i)
            data[i][j] = 0

for i in range(len(test_data)):
    for j in range(len(test_data[i])):
        if (test_data[i][j] < 0):
#             print(i)
            test_data[i][j] = 0


import sklearn
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression().fit(data[:-200], labels[:-200])
print("Logistic Regression training score", reg.score(data[:-200], labels[:-200]))
# Note this is the test set, the last 200 elements on which I did not train the dataset
print("Logistic regression test score", reg.score(data[-200:], labels[-200:]))


from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor

# A simple four layer neural network that again only trains on all examples but the last 200
model = Sequential()
model.add(Dense(100, input_shape=(6,), kernel_initializer='normal', activation='relu'))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization( momentum=0.99, epsilon=0.001))
model.add(Dense(2, activation='softmax', kernel_initializer='normal'))
# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())
# Other losses explored - binary_crossentropy  mean_squared_error (both performed poorly)

# the model has a test set of size 200 elements and another validation set which is 25% of the dataset
model.fit(data[:-200],labels[:-200],epochs=200, validation_split=0.25)

# Cell used for calculating the test set score
count = 0
ans = model.predict_classes(data[-200:])
for i in range(len(ans)):
    if ans[i] == labels[len(labels)-200+i]:
        count += 1
print("Test accuracy is", count/len(ans))

# A naive Bayes model for the same dataset
# Note the model performs poorly if the alexa rank is set to 0 for examples without an alexa rank
# Also the naive bayes approach here is flawed as this model requires a frequency table (columns denoting port numbers) instead of values that the features take
from sklearn.naive_bayes import GaussianNB, MultinomialNB

for i in range(len(data)):
    data[i][0] = max_alexa_rank if data[i][0] == 0 else data[i][0]

gnb = GaussianNB()
y_pred = gnb.fit(data[:-200], labels[:-200]).predict(data[-200:])

# print("Number of mislabeled points out of a total %d points : %d" % (200, (labels[-200:] != y_pred).sum()))

print("Naive Bayes train set accuracy is", gnb.score(data[-200:], labels[-200:]))
print("Naive Bayes test set accuracy is", gnb.score(data[:-200], labels[:-200]))

from sklearn.metrics import roc_auc_score
print("AUC-ROC score for naive bayes is", roc_auc_score(labels[-200:], y_pred))

test_ans = model.predict_classes(test_data)
import pdb
# print(test_ans)
count = 0
for i in range(len(test_ans)): count += 1 if test_ans[i] == 0 else 0
print("Number of clean URLs ",count)

results = open("results.txt","w")
for i in range(len(test_corpus)):
    results.write(test_corpus[i]['url'] + ", " + str(test_ans[i]) + "\n")
results.close() 