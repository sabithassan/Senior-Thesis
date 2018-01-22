import pickle
from sklearn.svm import LinearSVC
import sklearn
import numpy as np
import os
from copy import copy
from collections import Counter
from sets import Set
from sklearn.cluster import KMeans

## reads features of file
def get_labels(labels):
    result = []
    for file_labels in labels:
        result.append(np.array(file_labels))
    return np.array(result)

def get_data():
    files = os.listdir(os.curdir)

    features = None
    names = []
    labels = []

    t = True

    for fname in files:
        if 'feats.pkl' not in fname:
            continue
        else:
            fname_names = fname.strip('feats.pkl') + 'filenames.pkl'
            fname_labels = fname.strip('feats.pkl') + 'labels.pkl'

            nef = pickle.load(open(fname))
            print nef.shape, fname
            nef = np.reshape(nef, (nef.shape[0], nef.shape[2]))
            if t:
                features = nef
                t = False
            else:
                features = np.append(features, nef, axis=0)
            names += pickle.load(open(fname_names))
            labels += pickle.load(open(fname_labels))
        print "loaded", fname
    return (features, names, labels)



from sklearn import svm
import numpy as np
import math
import pickle
import random
import matplotlib.pyplot as plt

## file names
speakerFiles = {}

## for training data
speakerVecs = []
speakerLabels = []
allVecs = []
allLabels = []

## for verification data
vSpeakerVecs = []
vSpeakerLabels = []
vAllVecs = []
vAllLabels = []

## keeping track globally
allInd = []

# computes FAR error
def computeFAR(labels):
    return 100*sum(labels)/float(len(labels))


#computes FRR error
def computeFRR (labels):
    return 100*(len(labels) - sum(labels))/float(len(labels))


#trains SVM 
def createSVM(kern, vectors, labels):

    clf = svm.SVC(kernel = kern, probability = True)
    clf.fit(vectors, labels)

    return clf


#trains SVM with probability
def createSVM2(kern, vectors, labels):
    
    clf = svm.SVC(kernel = kern, probability = True)
    clf.fit(vectors, labels)

    return clf

## returns difference between most likely and second most likely
def getMargin(classProbs):
    
    sortedProbs = np.sort(classProbs)
    flipped = sortedProbs[::-1]
    if (len(sortedProbs)<2):
        print "There should be at least two classes"

    return flipped[0]-flipped[1]


#returns sample with lowest margin difference
def updateMargin (classifier, vectors):

    global allInd
    probs = classifier.predict_proba(vectors)
    min_diff = 999999
    ind = -1

    ## checks all data for sample with lowest margin difference
    for x in range (len(probs)):
        classProbs = probs[x]
        margin = getMargin(classProbs)
        if (margin < min_diff):
            if x not in allInd:
                min_diff = margin
                ind = x
                
    if (ind == -1):
        print "INDEX IS -1, something wrong"
    allInd.append(ind)
    return ind

## gets sample based on least confidence
def updateLeastConf(classifier, vectors):
    global allInd
    probs = classifier.predict_proba(vectors)
    max_diff = -9999999
    for x in range (len(probs)):
        classProbs = probs[x]
        sortedProbs = np.sort(classProbs)
        flipped = sortedProbs[::-1]
        diff = 1- flipped[0]
        if (diff > max_diff):
            if x not in allInd:
                max_diff = diff
                ind = x
                
    if (ind == -1):
        print "INDEX IS -1, something wrong"
    allInd.append(ind)
    return ind

## gets sample based on maximum entropy
def updateEntropy(classifier, vectors):
    global allInd
    probs = classifier.predict_proba(vectors)
    print probs.shape
    max_diff = -9999999
    for x in range (len(probs)):
        classProbs = probs[x]
        sumProbs = 0
        ## calculates entropy:
        for prob in classProbs:
            if (not (prob == 0)):
                sumProbs += prob*math.log(prob)
        sumProbs = -sumProbs
        
        if (sumProbs > max_diff):
            if x not in allInd:
                max_diff = sumProbs
                ind = x
                
    if (ind == -1):
        print "INDEX IS -1, something wrong entropy"
    allInd.append(ind)
    return ind


## updates sample based on maxmin
def updateMaxMin(classifier, vectors):
    return



## trains active learner
def trainActive(kern, vectors, labels, initial, others, measure):
    global allInd
    allInd = []
    
    print "training active learner, ", measure
    ## chooses inital number of samples from available data
    training_samples = random.sample(range(0, len(vectors)), initial)
    training_vecs = vectors[training_samples]
    training_labels = labels[training_samples]
    active_classifier = createSVM2(kern, training_vecs, training_labels)

    print "len vec", len(training_vecs), "len labels", len(training_labels)
    ## repeatedly adds sample
    for x in range (others):
        print "updating :", x
        ## gets the sample and adds to data
        if (measure == "margin"):
            sample = updateMargin(active_classifier, vectors)
        elif (measure == "least_conf"):
            sample = updateLeastConf(active_classifier, vectors)
        elif (measure == "entropy"):
            sample = updateEntropy(active_classifier, vectors)
        else:
              print "Measure not known"
              return active_classifier
              
        # print "sample to be updated", sample
        training_vecs = np.append(training_vecs, np.array([vectors[sample]]), axis = 0)
        training_labels = np.append(training_labels,labels[sample])
        ## updates classifier after each sample
        active_classifier = createSVM2(kern, training_vecs, training_labels)
    return active_classifier


## trains active learner using stratified sampling
def trainActiveStratified(kern, vectors, labels, initial, others, measure, K):
    print "training active learner with stratified, ", measure
    
    kmeans = KMeans(n_clusters=K, random_state=0).fit(vectors)
    clusterVecs = {}
    clusterLabels = {}

    ## initiate clusters
    for x in range (K):
        clusterVecs [x] = []
        clusterLabels [x] = []

    ## map vectors and labels to clusters
    for x in range (len(kmeans.labels_)):
        clusterVecs[kmeans.labels_[x]].append(vectors[x])
        clusterLabels[kmeans.labels_[x]].append(labels[x])

    ## convert arrays into np arrays
    for x in range (K):
        clusterVecs[x] = np.array(clusterVecs[x])
        clusterLabels[x] = np.array(clusterLabels[x])


    training_vecs = np.array([])
    training_labels = np.array([])
    ## allocate initial samples using stratified sampling
    for x in range (K):
        samples = random.sample(range(0, len(clusterVecs[x])), initial/K)
        xSamples = clusterVecs[x][samples]
        xLabels = clusterLabels[x][samples]
        if (x == 0):
            training_vecs = xSamples
            print training_vecs.shape
        else:
            training_vecs = np.append(training_vecs, xSamples, axis = 0)
        training_labels = np.append(training_labels, xLabels)

    #training_labels = np.array(training_labels)
    print "initial done"
    print "labels shape", training_labels.shape
    print "train data shape", training_vecs.shape
    
    global allInd
    allInd = []

    active_classifier = createSVM2(kern, training_vecs, training_labels)

    print "len vec", len(training_vecs), "len labels", len(training_labels)
    nrem = others
    ## repeatedly adds sample
    while (nrem > 0):
        for k in range (K):
            #for x in range (others/K):
            print "updating :", k
            ## gets the sample and adds to data
            if (measure == "margin"):
                sample = updateMargin(active_classifier, clusterVecs[k])
            elif (measure == "least_conf"):
                sample = updateLeastConf(active_classifier, clusterVecs[k])
            elif (measure == "entropy"):
                sample = updateEntropy(active_classifier, clusterVecs[k])
            else:
                  print "Measure not known"
                  return active_classifier
                  
            # print "sample to be updated", sample
            training_vecs = np.append(training_vecs, np.array([clusterVecs[k][sample]]), axis = 0)
            training_labels = np.append(training_labels,clusterLabels[k][sample])
            ## updates classifier after each sample
            active_classifier = createSVM2(kern, training_vecs, training_labels)
        nrem -= K
        print "remaining", nrem
    return active_classifier



## trains passive learner
def trainPassive(kern, vectors, labels, resource):

    print "Training passive learner"
    ## chooses samples according to resource
    training_samples = random.sample(range(0, len(vectors)), resource)
    training_vecs = vectors[training_samples]
    training_labels = labels[training_samples]

    return createSVM(kern, training_vecs, training_labels)




# adds bias to decision value of SVM and returns prediction
def predict(vectors, clf, T):

    labels = clf.decision_function(vectors)
    labels = np.subtract(labels, T)
    ## if distance > 0, class 1, else 0
    f = lambda x : 1 if x >= 0 else 0
    ff = np.vectorize(f)
    decisions = ff(labels)
    
    return decisions


def main ():

    CUT_OFF = 5000
    (features_, names, labels_) = get_data()
    
    features = []
    labels_ = np.array(labels_)
    print labels_

    shuffInd = sklearn.utils.shuffle(range(len(labels_)))
    #for i in range(len(shuffInd)):
    #    features.append(features_[shuffInd[i]])
    #    labels.append(labels_[shuffInd[i]])
    print "feat 1 before shape", features_[0].shape
    features = features_[shuffInd]
    labels = labels_[shuffInd]



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

    ## picks 2 unique lables with highest counts
    unique_labels = [sortedCouples[len(sortedCouples) -1][0], sortedCouples[len(sortedCouples)-2][0]]
    print unique_labels


    print len(unique_labels)
    ## for each element in uniquelabels, train a svm
    while (ul_i < 1):
        ul = unique_labels[ul_i]

        labels = np.array(map(lambda l: ul==l, labels))
        #Y_T = np.array(map(lambda l: ul==l, Y_T))
        ul_i += 1

        #print type(features)
    print "feat shape", features.shape
    print "label shape", labels.shape
    print "feat 1 shape", features[0].shape


    trainingFeats = features[:CUT_OFF, :]
    trainingLabels = labels[:CUT_OFF]
    #print trainingLabels

    testFeats = features[CUT_OFF:]
    testLabels = labels[CUT_OFF:]


    
    N = 600
    NINIT = 200
    NOTHER = N-NINIT
    K = 10
    #global trainingFeats, trainingLabels, testFeats, testLabels
    ## trains active learner and passive learner
    #print "data length", len(tAllVecs)
    
    
    ## trains different active classifiers based on measures of uncertainty
    active_classifier_entropy_S = trainActiveStratified("linear", trainingFeats, trainingLabels, NINIT, NOTHER, "entropy", K)
    active_classifier_entropy = trainActive("linear", trainingFeats, trainingLabels, NINIT, NOTHER, "entropy")
    #active_classifier_least = trainActive("linear", trainingFeats, trainingLabels, NINIT, NOTHER, "least_conf")
    #active_classifier_margin = trainActive("linear", trainingFeats, trainingLabels, NINIT, NOTHER, "margin")

    passive_classifier = trainPassive("linear", trainingFeats, trainingLabels, N)
    passive_score = passive_classifier.score(testFeats, testLabels)

    
    ## scores of different active learners
    active_score_entropy = active_classifier_entropy.score(testFeats, testLabels)
    active_score_entropy_S = active_classifier_entropy_S.score(testFeats, testLabels)
    #active_score_least = active_classifier_least.score(testFeats, testLabels)
    #active_score_margin = active_classifier_margin.score(testFeats, testLabels)

    print "passive learner score", passive_score
    print "parameters", N, NINIT, NOTHER, K
    print "active learner entropy score", active_score_entropy
    print "active learner entropy STRATIFIED score", active_score_entropy_S
    #print "active learner least confident score", active_score_least
    #print "active learner margin score", active_score_margin

main()
