from sklearn.svm import LinearSVC
import sklearn
import sys, pickle
import collect
import numpy as np
import gzip, pickle, random, time, sys, clustering, os
import scipy, sklearn

import numpy as np
import random_sampling as SRS
import optimalAlloc as opt

#import equalAlloc as eq
import OPT_A4_VAR as KS4
import OPT_A4_STD as KS41
import OPT_A3 as KS3
import OPT_A2 as KS2

from sklearn.datasets import load_svmlight_file
from sklearn import svm
from datetime import date
from sklearn.svm import LinearSVC
from sets import Set
from copy import copy
from collections import Counter
from sklearn.externals import joblib
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression




#words = ['a', 'b', 'c', 'a']

num = 3

CUT_OFF = 300
nTotals = range(50, 101, 60)
K = [5]
nInit = [0]
nStep = [10]
X = 100

# stratified_outfile = "stratified_"+date.today().strftime("%d_%m_%y")+"_"+str(num)+".data"
stratified_outfile = 'data/'+sys.argv[1] + '_strat.dat'


def get_labels(labels):
    result = []
    for file_labels in labels:
        result.append(np.array(file_labels))
    return np.array(result)

def get_data():

    ## where data is stored
    dataDir = "../../../../sshaar/feature_extractor/"
    
    files = os.listdir(dataDir)
    print (files)

    features = None
    names = []
    labels = []

    t = True

    for fname in files:
        if 'feats.pkl' not in fname:
            continue
        else:
            ## changes name to have filenames and labels instead
            fname_names = fname.strip('feats.pkl') + 'filenames.pkl'
            fname_labels = fname.strip('feats.pkl') + 'labels.pkl'

            nef = pickle.load(open(dataDir + fname))
            print ("nef shape", nef.shape, fname)

            ## tries to shape correctly. WHAT DOES IT DO?
            try:
                nef = np.reshape(nef, (nef.shape[0], nef.shape[2]))
            except:
                print ("loading failed", fname)
                continue
            
            ## accumulates features
            if t:
                features = nef
                t = False
            else:
                features = np.append(features, nef, axis=0)

            ## loads names and labels of the files
            names += pickle.load(open(dataDir + fname_names))
            labels += pickle.load(open(dataDir + fname_labels))
        print ("loaded", fname)
    return (features, names, labels)


## Separate data with unique labels
def separateLabels(X, Y, labels, X_T, Y_T):

    print ("training shape: ", X.shape, Y.shape)
    print ("testing shape: ", X_T.shape, Y_T.shape)
    
    unique_labels = list(Set(labels))
    results = []
    c = 0
    w = 0
    ul_i = 0

    ## counts number of instances for each unique labels
    u_labels = Counter(labels).keys() # equals to list(set(words))
    counts = Counter(labels).values() # counts the elements' frequency
    coupled = zip(u_labels, counts) # couples

    ## sorts according to count of each unique label
    sortedCouples = sorted(coupled, key=lambda tup: tup[1])

    #print (sortedCouples)

    ## picks 2 unique lables with highest counts
    unique_labels = [sortedCouples[len(sortedCouples) -2][0], sortedCouples[len(sortedCouples)-3][0]]
    print unique_labels

    ## stores labels for unique values
    trainLabels = []
    testLabels = []


    print len(unique_labels)
    ## for each element in uniquelabels, train a svm
    while (ul_i < len(unique_labels)):
        ul = unique_labels[ul_i]

        ## positive if label matches
        Y_ = np.array(map(lambda l: ul==l, Y))
        Y_Ti = np.array(map(lambda l: ul==l, Y_T))
        print (ul, "positives : ", sum(Y_), sum(Y_Ti))

        ## adds mapped labels
        trainLabels.append(Y_)
        testLabels.append(Y_Ti)
        ul_i += 1

    ## turns into numpy arrays
    trainLabels = np.array(trainLabels)
    testLabels = np.array(testLabels)


    return (trainLabels, testLabels)



random_outfile = "data/random_"+date.today().strftime("%d_%m_%y")+"_"+str(num)+".data"
stratified_outfile = "data/stratified_"+date.today().strftime("%d_%m_%y")+"_"+str(num)+".data"


# test_scores = map(model.decision_function, test_matrix)

## runs random sampling on models
def run_random_sampling(models, test_matrix, test_tl):

    ## runs simple random sampling with all loaded classifiers
    count = 0
    for model in models:
        results = SRS.main(nTotals, X, model, test_matrix, test_tl[count])
        count += 1
        pickle.dump(results, open(random_outfile, "a"))
    return results


## runs stratified sampling on models
def run_stratified_sampling(models, test_matrix, test_tl):

    num_models = len(models)
    k2, k3, k4, k41 = [], [], [], []
    optim, equal, ops_optim = [], [], []

    for k in K:
        clusters = clustering.KMeanss(models, k, test_matrix, test_tl)
        data = {"vectors" : test_matrix, "true_labels":test_tl, "class_labels":0}


        print ("OPTIMAL")
        d = opt.main(k, nTotals, nInit, nStep, data, clusters, X, num_models)
        optim.append(d)
        print ("___________________________________________________")

        print ("OPT_A2")
        d = KS2.main(k, nTotals, nInit, nStep, data, clusters, X, num_models)
        k2.append(d)
        print ("___________________________________________________")

        # print "EQUAL"
        # d = eq.main(k, nTotals, nInit, nStep, data, clusters, X)
        # equal.append(d)
        # print "___________________________________________________"

        print ("OPT_A3")
        d = KS3.main(k, nTotals, nInit, nStep, data, clusters, X, num_models)
        k3.append(d)
        print ("___________________________________________________")

        print ("OPT_A4 threshold")
        d = KS4.main(k, nTotals, nInit, nStep, data, clusters, X, num_models)
        k4.append(d)
        print ("___________________________________________________")

        print ("OPT_A4 std")
        d = KS41.main(k, nTotals, nInit, nStep, data, clusters, X, num_models)
        k41.append(d)
        print ("___________________________________________________")

        print ("OPTIMAL OPS")
        d = opt.main(k, nTotals, nInit, nStep, data, clusters, X, num_models)
        ops_optim.append(d)
        print ("___________________________________________________")


    return (k2, k3, k4, k41, optim, equal, ops_optim)








def main():

    ## loads data
    (features_, names, labels_) = get_data()

    features = []
    labels_ = np.array(labels_)

    ## shuffles data
    shuffInd = sklearn.utils.shuffle(range(len(labels_)))

    features = features_[shuffInd]
    labels = labels_[shuffInd]

    ## splits data into training and testing at CUTOFF
    X_training = features[:CUT_OFF, :]
    Y_training = labels[:CUT_OFF]

    X_testing = features[CUT_OFF:, :]
    Y_testing = labels[CUT_OFF:]

    print "what's this?", X_testing.shape, X_training.shape, len(list(X_training.shape))

    (trainLabels, testLabels) = separateLabels(X_training, Y_training, labels, X_testing,Y_testing)

    print ("separated training shape, ", trainLabels.shape)
    print ("separated test shape, ", testLabels.shape)
    

    models = []
    for clf in range (trainLabels.shape[0]):
        print ("Start training classifier ", clf)
        ## alterantes between linear svm and logistic regression
        if (1):
            ## trains linear svm
            linear_SVC = SVC(kernel = 'linear', probability=True)
            linear_SVC.fit(X_training, trainLabels[clf])
            models.append(linear_SVC)
        else:
            ## trains logistic regression
            logit = LogisticRegression()
            logit.fit(X_training, trainLabels[clf])
            models.append(logit)

        print ("Finished training classifier ", clf)
            

    #print ("Number of svms: ", my_svms)


    run_random_sampling(models, X_testing, testLabels)
    print ("\n\n")
    strat = run_stratified_sampling(models, X_testing, testLabels)

    

   


main()
